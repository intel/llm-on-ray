import torch

from megatron import get_args, print_rank_0
from megatron.core import mpu
from megatron.data.data_samplers import build_pretraining_data_loader
from deepspeed.accelerator import get_accelerator

from common.dataprocesser import DataProcesser

class MegatronProcesser(DataProcesser):
    def prepare(self, tokenizer, dataset, **kwargs):
        args = get_args()

        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

        print_rank_0('> building train, validation, and test datasets ...')
        iteration = kwargs.get("step", 0)
        if iteration:
            # passed value is starting step
            iteration -= 1
            args.consumed_train_samples = iteration * args.global_batch_size
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

        # Data loader only on rank 0 of each model parallel group.
        if args.use_dataset_only or mpu.get_tensor_model_parallel_rank() == 0:

        # Build datasets.
            train_ds, valid_ds, test_ds = dataset

            # Build dataloders.
            train_dataloader = build_pretraining_data_loader(
                train_ds, args.consumed_train_samples)
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples)
            test_dataloader = build_pretraining_data_loader(test_ds, 0)

        return train_dataloader, valid_dataloader, test_dataloader
