import os
import math
import time
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import sys
import json
import torch
from functools import partial

from ray.air.checkpoint import Checkpoint

from .pretrainer import PreTrainer
from common.logging import logger


from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe, LLaMAModelPipe, LLaMAModel
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron import get_timers
from megatron.checkpointing import save_checkpoint
from megatron.training import setup_model_and_optimizer, build_train_valid_test_data_iterators, setup_teacher_model, train
from megatron.training import evaluate_and_print_results
from megatron.global_vars import get_current_device, get_current_device_index
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.compression.compress import redundancy_clean

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    args = get_args()
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


from contextlib import nullcontext
class MegatronDeepspeedHabanaPreTrainer(PreTrainer):
    def __init__(self, config):
        self.config = config
        self.starting_step = 0
        self.mode = "ddp"
    
    def _train_valid_test_datasets_provider(self, train_val_test_num_samples):
        """Build train, valid, and test datasets."""
        args = get_args()

        print_rank_0('> building train, validation, and test datasets '
                    'for GPT ...')
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            train_data_prefix=args.train_data_path,
            valid_data_prefix=args.valid_data_path,
            test_data_prefix=args.test_data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            use_seq_len_plus_one_tokens=args.use_seq_len_plus_one_tokens)
        print_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds
   
    def _model_provider(self, pre_process=True, post_process=True, parallel_output=True):
        """Build the model."""

        print_rank_0('building {} model ...'.format(self.config.get("model_type")))
        see_memory_usage(f"Before Building Model", force=True)

        args = get_args()
        model = None
        with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                remote_device=None if args.remote_device == 'none' else args.remote_device,
                                config_dict_or_path=args.deepspeed_config,
                                enabled=args.zero_stage == 3,
                                mpu=mpu):
            if args.deepspeed and not args.no_pipeline_parallel:

                # verify --deepspeed_activation_checkpointing
                # mandatory! otherwise the model uses fork() mapping to Megatron's RNGStatesTrackerSingleton
                # while GPTModelPipe uses DS checkpoint activations that uses DS's RNGStatesTracker
                if args.checkpoint_activations and args.checkpoint_activations_granularity == "full":
                    assert args.deepspeed_activation_checkpointing, \
                        "Flag --deepspeed_activation_checkpointing is mandatory when using GPTModelPipe" \
                        " with checkpoint activations granularity full."
                if self.config.get("model_type") == "gpt" :
                    model = GPTModelPipe(
                        num_tokentypes=0,
                        parallel_output=parallel_output,
                    )
                elif self.config.get("model_type") == "llama":
                    model = LLaMAModelPipe(
                        num_tokentypes=0,
                        parallel_output=True
                    )
                # This is a hack to give us a reference to get_batch_pipe from within training.py
                # We need to call model.set_batch_fn after deepspeed.initialize
                model._megatron_batch_fn = self._get_batch_pipe

                # Predompute the attention mask and store it in args. This avoids having to
                # pipeline it as an activation during training. The mask is constant, and thus
                # we can reuse it.
                current_device = get_current_device()
                attention_mask = torch.tril(torch.ones(
                    (1, args.seq_length, args.seq_length), device=current_device)).view(
                        1, 1, args.seq_length, args.seq_length)

                # Convert attention mask to binary:
                attention_mask = (attention_mask < 0.5)
                if args.fp16:
                    attention_mask = attention_mask.half()
                elif args.bf16:
                    attention_mask = attention_mask.bfloat16()

                if args.mask_tensor_adding:
                    args.attn_mask = attention_mask * -10000.0
                else:
                    args.attn_mask = attention_mask.to(torch.bool)

            else:
                if self.config.get("model_type") == "gpt" :
                    assert args.position_embedding_type != PositionEmbeddingType.alibi, \
                        "GPTModel doesn't yet support ALiBi positional encoding"
                    model = GPTModel(
                        num_tokentypes=0,
                        parallel_output=parallel_output,
                        pre_process=pre_process,
                        post_process=post_process
                    )
                elif self.config.get("model_type") == "llama":
                    model = LLaMAModel(
                        num_tokentypes=0,
                        parallel_output=True,
                        pre_process=pre_process,
                        post_process=post_process
                    )

        see_memory_usage(f"After Building Model", force=True)
        return model
       
    def prepare(self,
        model, 
        tokenizer=None,
        dataset=None,
        optimizer=None,
        accelerator=None):
        
        args = get_args()
        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        # TODO: Make it back torch.DoubleTensor once supporting float64
        start_time_tensor = torch.FloatTensor([_TRAIN_START_TIME]).to(get_current_device())

        torch.distributed.all_reduce(start_time_tensor,
                                    op=torch.distributed.ReduceOp.MIN,
                                    async_op=args.use_hpu)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
            time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')

        timers = get_timers()

        if args.deepspeed:
            if isinstance(args.deepspeed_config, dict) :
                args.deepspeed_configuration = args.deepspeed_config
            else:
                args.deepspeed_configuration = json.load(
                    open(args.deepspeed_config, 'r', encoding='utf-8'))
            if "curriculum_learning" in args.deepspeed_configuration and \
                "enabled" in args.deepspeed_configuration["curriculum_learning"]:
                args.curriculum_learning = args.deepspeed_configuration[ \
                    "curriculum_learning"]["enabled"]
            if args.curriculum_learning and not args.no_pipeline_parallel:
                from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                    import CurriculumScheduler
                args.curriculum_scheduler = CurriculumScheduler( \
                    args.deepspeed_configuration["curriculum_learning"])
            if "compression_training" in args.deepspeed_configuration:
                args.compression_training = True
            if args.universal_checkpoint:
                args.deepspeed_configuration["checkpoint"] = {"load_universal": True}
            # Clear deepspeed_config to force deepspeed to take config from args.deepspeed_configuration at initialize()
            args.deepspeed_config = None

        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup').start()
        self.model, self.optimizer, self.lr_scheduler = setup_model_and_optimizer(self._model_provider, teacher=False)
        timers('model-and-optimizer-setup').stop()
        print_datetime('after model, optimizer, and learning rate '
                    'scheduler are built')

        # Data stuff.
        timers('train/valid/test-data-iterators-setup').start()
        if args.virtual_pipeline_model_parallel_size is not None:
            all_data_iterators = [
                build_train_valid_test_data_iterators(self._train_valid_test_datasets_provider)
                for _ in range(len(model))
            ]
            self.train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
            self.valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
            self.test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]
        else:
            self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator \
                = build_train_valid_test_data_iterators(
                    self._train_valid_test_datasets_provider)
        timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')

    def _get_batch(self, data_iterator):
        """Generate a batch"""
        args = get_args()
        tokenizer = get_tokenizer()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        if not args.use_seq_len_plus_one_tokens:
            labels = torch.roll(tokens_, shifts=-1, dims=1)
            labels[:, -1] = -1
            tokens = tokens_
        else:
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
            labels = labels,
            dummy_sample= None,)

        tokens[tokens == -1] = 0
        labels[labels == -1] = 0

        return tokens, labels, loss_mask, attention_mask, position_ids

    def _get_batch_pipe(self, data):
        """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
        args = get_args()
        tokenizer = get_tokenizer()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        if not args.use_seq_len_plus_one_tokens:
            labels = torch.roll(tokens_, shifts=-1, dims=1)
            if labels is not None:
                labels[:, -1] = -1
                tokens = tokens_
        else:
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()

        if labels is None:
            logger.warning(f"labels is empty, skip")
            return

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss,
            labels = labels,
            dummy_sample = None,
            )
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0


        if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
            # seqlen-based curriculum learning
            # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
            tokens = tokens[:, :args.curriculum_seqlen].contiguous()
            position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
            labels = labels[:, :args.curriculum_seqlen].contiguous()
            loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

        return (tokens, position_ids, attention_mask), (labels, loss_mask)

            
    def _data_post_process(self, data, data_sampler_state_dict):
        args = get_args()
        if args.data_efficiency_curriculum_learning:
            if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
                args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
                current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
                if current_seqlen < args.seq_length:
                    data['text'] = data['text'][:, :(current_seqlen+1)].contiguous()
            elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
                args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
                current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
                if current_seqlen < args.seq_length:
                    orig_num_token = torch.numel(data['text'])
                    reshape_len = (data['text'].size()[1] // (current_seqlen+1)) * (current_seqlen+1)
                    data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen+1),
                        data['text'][:, -(current_seqlen+1):]), 0).contiguous()
                    num_row = math.ceil(orig_num_token / (current_seqlen+1))
                    num_row = min(num_row, data['text'].size()[0])
                    if num_row > 1 and num_row % 2 != 0:
                        num_row -= 1
                    data['text'] = data['text'][:num_row, :].contiguous()
            else:
                args.data_efficiency_curriculum_learning_seqlen_type = None
        return data

    def _loss_func(self, loss_mask, moe_loss, mos_loss, output_tensor):
        args = get_args()
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        
        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])
        if args.mos or args.kd:
            # assert max(args.num_experts) >= 1
            loss = loss + moe_loss + mos_loss
            if args.mos:
                return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'mos loss': mos_loss}
            elif args.kd:
                return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'kd loss': mos_loss}
            print_rank_0('>>> total loss: {}, lm loss {}, kd loss {}'.format(loss, averaged_loss[0], mos_loss))
        else:
            if max(args.num_experts) <= 1:
                return loss, {'lm loss': averaged_loss[0]}
            else:
                loss = loss + moe_loss
                return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}
 
    def _calculate_mos_loss(self, args, stu_output, teacher_model, tokens, position_ids, attention_mask):
        mos_loss = 0
        alpha = args.kd_alpha_ce
        beta = args.kd_beta_ce
        kd_temp = args.kd_temp
        
        if teacher_model:
            with torch.no_grad():
                if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
                    assert args.curriculum_seqlen is not None
                    curriculum_seqlen = args.curriculum_seqlen
                    tokens = tokens[:, :curriculum_seqlen].contiguous()
                    position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                    attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
                    # No need to truncate labels as we do not need it for the teacher logits
                tea_output, *tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
                assert stu_output.size() == tea_output.size(), 'teacher and student output should match in size. Student: {}, Teacher: {}, CL seq length {}'.format(stu_output.size(), tea_output.size(), args.curriculum_seqlen)

            student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
            tea_logits = F.softmax(tea_output / kd_temp, dim=2) # The target logits is expected to be probabilities. If we use log_softmax, then we need to set target_log to true when initializing the KLDivLoss.

            mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)

            mos_loss = mos_loss.div(args.seq_length) * beta
        return mos_loss
        
    def _forward_step_func(self, data_iterator, model, teacher_model=None):
        """Forward step."""
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator').start()
        tokens, labels, loss_mask, attention_mask, position_ids = self._get_batch(
            data_iterator)
        timers('batch-generator').stop()

        if args.mos or args.kd:
            # The forward func can return either the loss or the logits, depending on whether passing in the labels or not.
            stu_output, *other_losses = model(tokens, position_ids, attention_mask)
            if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
                assert args.curriculum_seqlen is not None
                labels = labels[:, :args.curriculum_seqlen].contiguous()
            output_tensor = mpu.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
        else:
            output_tensor, *other_losses = model(tokens, position_ids, attention_mask,
                                                labels=labels)
        if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
            loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

        moe_losses = []
        for moe_loss in other_losses:
            if moe_loss is not None:
                moe_losses.append(moe_loss)
        moe_loss = sum(moe_losses) * args.moe_loss_coeff

        mos_loss = 0
        if args.mos or args.kd:
            assert model.training
            mos_loss = self._calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask)
        
        # Output_tensor stores the standard loss, loos_func calculates the total loss.
        return output_tensor, partial(self._loss_func, loss_mask, moe_loss, mos_loss)


    def train(self) :
        args = get_args()
        timers = get_timers()
        teacher_model = None
        if args.mos or args.kd: # Set up teacher model
            teacher_model = setup_teacher_model(args, self._model_provider)

        # Print setup timing.
        print_rank_0('done with setup ...')
        timers.log(['model-and-optimizer-setup', 'train/valid/test-data-iterators-setup'])

        iteration = args.iteration

        if args.do_valid and args.do_pretrain_validation:
            prefix = 'evaluation on val data for the initial checkpoint weights'
            evaluate_and_print_results(prefix, self._forward_step_func,
                                    self.valid_data_iterator, self.model,
                                    iteration, False)

        if args.do_train and args.train_iters > 0:
            print_rank_0('training ...')
            iteration = train(self._forward_step_func,
                            self.model, self.optimizer, self.lr_scheduler,
                            self.train_data_iterator, self.valid_data_iterator, 
                            teacher_model=teacher_model)
            training_prefix = 'the end of training'
            print_datetime('after training is done')
        else:
            training_prefix = 'skipping training'
            print_rank_0('skipping training ...')

        if args.do_valid:
            prefix = ' '.join([training_prefix, 'for val data'])
            evaluate_and_print_results(prefix, self._forward_step_func,
                                    self.valid_data_iterator, self.model,
                                    iteration, False)

        # Clean the model and do evaluation again
        if args.compression_training and model:
            model = [redundancy_clean(model[0], args.deepspeed_config, mpu)]
            if args.do_valid:
                prefix = ' '.join([training_prefix,
                                'and after model cleaning for val data'])
                evaluate_and_print_results(prefix, self._forward_step_func,
                                        self.valid_data_iterator, self.model,
                                        iteration, False)

        if args.save and (iteration != args.iteration or args.universal_checkpoint):
            save_checkpoint(iteration, self.model, self.optimizer, self.lr_scheduler)

        if args.do_test:
            # Run on test data.
            prefix = ' '.join([training_prefix, 'for test data'])
            evaluate_and_print_results(prefix, self._forward_step_func,
                                    self.test_data_iterator, self.model,
                                    0, True)
