#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from megatron import get_args, print_rank_0
from megatron.training import build_train_valid_test_datasets, update_train_iters
from megatron.data import gpt_dataset

from llm_on_ray.common.dataset import Dataset


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
