import os
import math
import time
from torch import nn
import torch.nn.functional as F
from datetime import datetime
import sys
import json
import torch
import transformers
from functools import partial

from ray.air.checkpoint import Checkpoint

from .pretrainer import PreTrainer
from common.logging import logger


from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, GPTModelPipe
from megatron.model.rotary_pos_embedding import RotaryEmbedding
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_num_microbatches
from megatron import update_num_microbatches
from megatron.checkpointing import save_checkpoint
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import calc_params_l2_norm
from megatron.training import setup_model_and_optimizer, build_train_valid_test_data_iterators, setup_teacher_model
from megatron.training import train_step, evaluate_and_print_results, training_log, save_checkpoint_and_time
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.compression.compress import redundancy_clean

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


from contextlib import nullcontext
class MegatronDeepspeedPreTrainer(PreTrainer):
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
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            train_data_prefix=args.train_data_path,
            valid_data_prefix=args.valid_data_path,
            test_data_prefix=args.test_data_path,
            data_cache_path=args.data_cache_path)
        print_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds

   
    def _model_provider(self, pre_process=True, post_process=True):
        """Build the model."""
        print_rank_0('building GPT model ...')
        see_memory_usage(f"Before Building Model", force=True)

        args = get_args()
        config = core_transformer_config_from_args(args)
        with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                remote_device=None if args.remote_device == 'none' else args.remote_device,
                                config_dict_or_path=args.deepspeed_config,
                                enabled=args.zero_stage == 3,
                                mpu=mpu):
            if args.deepspeed and not args.no_pipeline_parallel:
                model = GPTModelPipe(
                    config=config,
                    num_tokentypes=0,
                    parallel_output=True
                )
                # This is a hack to give us a reference to get_batch_pipe from within training.py
                # We need to call model.set_batch_fn after deepspeed.initialize
                model._megatron_batch_fn = self._get_batch_pipe

                # Predompute the attention mask and store it in args. This avoids having to
                # pipeline it as an activation during training. The mask is constant, and thus
                # we can reuse it.
                attention_mask = torch.tril(torch.ones(
                    (1, args.seq_length, args.seq_length), device=get_accelerator().current_device_name())).view(
                        1, 1, args.seq_length, args.seq_length)

                # Convert attention mask to binary:
                attention_mask = (attention_mask < 0.5)
                if args.fp16:
                    attention_mask = attention_mask.half()
                elif args.bf16:
                    attention_mask = attention_mask.bfloat16()

                # Attention mask must be bool.
                args.attn_mask = attention_mask.to(torch.bool)

                # For prertaining, since sequence length is fixed, cache rotary embedding in args, to avoid communicating around
                if args.use_rotary_position_embeddings:
                    rotary_dim = args.hidden_size // args.num_attention_heads \
                        if args.kv_channels is None else args.kv_channels

                    if args.rotary_percent < 1.0:
                        rotary_dim = int(rotary_dim * args.rotary_percent)

                    # partial rotary embeddings, which is better than full rotary
                    # Wang and Komatsuzaki et al
                    # https://github.com/kingoflolz/mesh-transformer-jax/
                    rotary_pos_emb = RotaryEmbedding(rotary_dim)(args.seq_length).to(
                        get_accelerator().current_device_name())
                    if args.fp16:
                        rotary_pos_emb = rotary_pos_emb.half()
                    elif args.bf16:
                        rotary_pos_emb = rotary_pos_emb.bfloat16()
                    args.rotary_pos_emb = rotary_pos_emb

            else:
                model = GPTModel(
                    config=config,
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
        # Set pytorch JIT layer fusion options and warmup JIT functions.
        args = get_args()
        if get_accelerator().device_name() == 'cuda':
            set_jit_fusion_options()

        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = get_accelerator().DoubleTensor([_TRAIN_START_TIME])
        #TODO: fix the hang issue of function all_reduce
        #torch.distributed.all_reduce(start_time_tensor,
        #                            op=torch.distributed.ReduceOp.MIN)
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
                args.curriculum_learning_legacy = args.deepspeed_configuration[ \
                    "curriculum_learning"]["enabled"]
            if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
                from deepspeed.runtime.data_pipeline.curriculum_scheduler \
                    import CurriculumScheduler
                args.curriculum_scheduler = CurriculumScheduler( \
                    args.deepspeed_configuration["curriculum_learning"])
            if "compression_training" in args.deepspeed_configuration:
                args.compression_training = True

        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
        self.model, self.optimizer, self.opt_param_scheduler = setup_model_and_optimizer(
            self._model_provider, ModelType.encoder_or_decoder, teacher=False, data_post_process=self._data_post_process,
            build_train_valid_test_datasets_provider=self._train_valid_test_datasets_provider)
        timers('model-and-optimizer-setup').stop()
        print_datetime('after model, optimizer, and learning rate '
                    'scheduler are built')

        # Data stuff.
        timers('train/valid/test-data-iterators-setup', log_level=0).start(
            barrier=True)
        if args.virtual_pipeline_model_parallel_size is not None:
            all_data_iterators = [
                build_train_valid_test_data_iterators(
                    self._train_valid_test_datasets_provider)
                for _ in range(len(model))
            ]
            self.train_data_iterator = [data_iterators[0]
                                for data_iterators in all_data_iterators]
            self.valid_data_iterator = [data_iterators[1]
                                for data_iterators in all_data_iterators]
            self.test_data_iterator = [data_iterators[2]
                                for data_iterators in all_data_iterators]
        else:
            self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator \
                = build_train_valid_test_data_iterators(
                    self._train_valid_test_datasets_provider)
        if args.data_efficiency_curriculum_learning:
            if args.deepspeed_dataloader is not None:
                # We use args to pass the deepspeed_dataloader because adding
                # output to setup_model_and_optimizer will break the API for other
                # cases. We clear args.deepspeed_dataloader after updating
                # train_data_iterator because args will be saved in checkpoint and
                # attempting to save the whole deepspeed_dataloader will lead to
                # "AttributeError: Can't pickle local object...".
                self.train_data_iterator = iter(args.deepspeed_dataloader)
                args.deepspeed_dataloader = None
            else:
                self.train_data_iterator = None
        timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')
        args.teacher_model = None
        if args.mos or args.kd: # Set up teacher model
            args.teacher_model = setup_teacher_model(args, model_provider)

        # Print setup timing.
        print_rank_0('done with setup ...')
        timers.log(['model-and-optimizer-setup',
                    'train/valid/test-data-iterators-setup'], barrier=True)

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
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        return tokens, labels, loss_mask, attention_mask, position_ids

    def _get_batch_pipe(self, data):
        """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
        args = get_args()
        tokenizer = get_tokenizer()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
            # seqlen-based curriculum learning
            # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
            tokens = tokens[:, :args.curriculum_seqlen].contiguous()
            position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
            if labels is not None:
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
                if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
                    assert args.curriculum_seqlen is not None
                    curriculum_seqlen = args.curriculum_seqlen
                    tokens = tokens[:, :curriculum_seqlen].contiguous()
                    position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                    attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
                    # No need to truncate labels as we do not need it for the teacher logits
                tea_output, tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
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
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self._get_batch(
            data_iterator)
        timers('batch-generator').stop()

        if args.data_efficiency_curriculum_learning:
            args.curriculum_seqlen = tokens.size()[1]
            if hasattr(args, 'data_efficiency_curriculum_learning_seqlen_type') and \
                args.data_efficiency_curriculum_learning_seqlen_type == 'seqlen_reshape':
                args.data_efficiency_curriculum_learning_numel = torch.numel(tokens)

        if args.mos or args.kd:
            # The forward func can return either the loss or the logits, depending on whether passing in the labels or not.
            stu_output, other_losses = model(tokens, position_ids, attention_mask)
            if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
                assert args.curriculum_seqlen is not None
                labels = labels[:, :args.curriculum_seqlen].contiguous()
            output_tensor = tensor_parallel.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
        else:
            output_tensor, other_losses = model(tokens, position_ids, attention_mask,
                                                labels=labels)
        if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
            loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

        moe_losses = []
        for moe_loss in other_losses:
            if moe_loss is not None:
                moe_losses.append(moe_loss)
        moe_loss = sum(moe_losses) * args.moe_loss_coeff

        mos_loss = 0
        if args.mos or args.kd:
            assert model.training
            if args.teacher_forward and args.teacher_model is not None:
                mos_loss = self._calculate_mos_loss(args, stu_output,
                    args.teacher_model[0], tokens, position_ids, attention_mask)

        # Output_tensor stores the standard loss, loos_func calculates the total loss.
        return output_tensor, partial(self._loss_func, loss_mask, moe_loss, mos_loss)

    def train(self) :
        args = get_args()
        timers = get_timers()

        # Write args to tensorboard
        write_args_to_tensorboard()

        if args.random_ltd:
            # random-ltd requires different randomness on each rank
            import random
            random.seed(args.seed + torch.distributed.get_rank())

        # Turn on training mode which enables dropout.
        for model_module in self.model:
            model_module.train()

        # Tracking loss.
        total_loss_dict = {}

        # Iterations.
        iteration = args.iteration

        # Translate args to core configuration
        config = core_transformer_config_from_args(args)
        if not args.deepspeed:
            config.grad_scale_func = self.optimizer.scale_loss
        config.timers = timers

        timers('interval-time', log_level=0).start(barrier=True)
        print_datetime('before the start of training step')
        report_memory_flag = True
        if args.random_ltd:
            assert self.model[0].random_ltd_enabled()
            args.random_ltd_layer_num = self.model[0].random_ltd_scheduler.get_random_ltd_layer_num()
            
        while iteration < args.train_iters and (args.train_tokens is None or \
            args.consumed_train_tokens < args.train_tokens):
            update_num_microbatches(args.consumed_train_samples)
            if args.deepspeed:
                # inform deepspeed of any batch size changes
                global_batch_size = mpu.get_data_parallel_world_size() * \
                                    args.micro_batch_size * \
                                    get_num_microbatches()
                self.model[0].set_train_batch_size(global_batch_size)

            if args.curriculum_learning_legacy and not args.no_pipeline_parallel:
                args.curriculum_seqlen = args.curriculum_scheduler.update_difficulty( \
                        args.iteration + 1)
            args.curr_iteration = iteration
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(self._forward_step_func,
                        self.train_data_iterator,
                        self.model,
                        self.optimizer,
                        self.opt_param_scheduler,
                        config)
            iteration += 1
            args.iteration = iteration
            new_samples = mpu.get_data_parallel_world_size() * \
                                        args.micro_batch_size * \
                                        get_num_microbatches()
            args.consumed_train_samples += new_samples
            # This actual_seq_length is used for actual consumed tokens calculation, flops calculation, and logging.
            args.actual_seq_length = args.seq_length
            if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
                args.actual_seq_length = args.curriculum_seqlen
            if args.random_ltd:
                args.random_ltd_reserved_length = self.model[0].random_ltd_scheduler.get_current_seq()
                if args.random_ltd_reserved_length < args.actual_seq_length:
                    args.actual_seq_length = (args.actual_seq_length * (args.num_layers - args.random_ltd_layer_num) + args.random_ltd_reserved_length * args.random_ltd_layer_num) // args.num_layers
            if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
                if hasattr(args, 'data_efficiency_curriculum_learning_numel'):
                    act_mbsz = args.data_efficiency_curriculum_learning_numel / args.curriculum_seqlen
                    act_token = act_mbsz * args.actual_seq_length
                    args.consumed_train_tokens += mpu.get_data_parallel_world_size() * \
                            get_num_microbatches() * act_token
                else:
                    args.consumed_train_tokens += new_samples * args.actual_seq_length
            else:
                args.consumed_train_tokens += new_samples * args.actual_seq_length
            
            # Logging.
            if args.deepspeed:
                if hasattr(self.model[0].optimizer, 'cur_scale'):
                    loss_scale = self.model[0].optimizer.cur_scale
                else:
                    loss_scale = None
            else:
                loss_scale = self.optimizer.get_loss_scale().item()
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(self.model)
            report_memory_flag = training_log(loss_dict, total_loss_dict,
                                            self.optimizer.param_groups[0]['lr'],
                                            iteration, loss_scale,
                                            report_memory_flag, skipped_iter,
                                            grad_norm, params_norm, num_zeros_in_grad,
                                            self.model, self.optimizer)

            # Autoresume
            if args.adlr_autoresume and \
            (iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(iteration, self.model, self.optimizer,
                                                self.opt_param_scheduler)

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0 and \
            args.do_valid:
                prefix = 'iteration {}'.format(iteration)
                evaluate_and_print_results(prefix, self._forward_step_func,
                                        self.valid_data_iterator, self.model,
                                        iteration, None,
                                        config, False)

            # Checkpointing
            saved_checkpoint = False
            if args.exit_signal_handler:
                signal_handler = get_signal_handler()
                if any(signal_handler.signals_received()):
                    save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                            self.opt_param_scheduler)
                    print_datetime('exiting program after receiving SIGTERM.')
                    sys.exit()

            if args.save and args.save_interval and \
            iteration % args.save_interval == 0:
                save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                        self.opt_param_scheduler)
                saved_checkpoint = True

            # Exiting based on duration
            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = get_accelerator().IntTensor(
                    [train_time > args.exit_duration_in_mins])
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    if not saved_checkpoint:
                        save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                                self.opt_param_scheduler)
                    print_datetime('exiting program after {} minutes'.format(train_time))
                    sys.exit()

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                if args.save and not saved_checkpoint:
                    save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                            self.opt_param_scheduler)
                torch.distributed.barrier()
                print_datetime('exiting program at iteration {}'.format(iteration))
                sys.exit()
        
        print_datetime('after training is done')
        # Clean the model
        if args.compression_training:
            self.model = [redundancy_clean(self.model[0], args.deepspeed_config, mpu)]

        if args.save and iteration != 0:
            save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler)

        return iteration


 