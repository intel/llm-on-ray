from megatron import get_args, print_rank_0
from megatron.training import build_train_valid_test_datasets, update_train_iters
from megatron.data import gpt_dataset

from common.dataset import Dataset


class MegatronDataset(Dataset):
    def __call__(self, config):
        def _train_valid_test_datasets_provider(train_val_test_num_samples):
            """Build train, valid, and test datasets."""
            args = get_args()
            print_rank_0("> building train, validation, and test datasets " "for GPT ...")
            train_ds, valid_ds, test_ds = gpt_dataset.build_train_valid_test_datasets(
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
                data_cache_path=args.data_cache_path,
            )
            print_rank_0("> finished creating GPT datasets ...")

            return train_ds, valid_ds, test_ds

        args = get_args()
        update_train_iters(args)
        datasets = build_train_valid_test_datasets(_train_valid_test_datasets_provider)
        print_rank_0(datasets)
        return datasets
