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
from megatron.core import mpu
from megatron.data.data_samplers import build_pretraining_data_loader

from llm_on_ray.common.dataprocesser import DataProcesser


class MegatronProcesser(DataProcesser):
    def prepare(self, tokenizer, dataset, **kwargs):
        args = get_args()

        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

        print_rank_0("> building train, validation, and test datasets ...")
        iteration = kwargs.get("step", 0)
        if iteration:
            # passed value is starting step
            iteration -= 1
            args.consumed_train_samples = iteration * args.global_batch_size
            args.consumed_valid_samples = (
                (args.iteration // args.eval_interval) * args.eval_iters * args.global_batch_size
            )

        # Data loader only on rank 0 of each model parallel group.
        if args.use_dataset_only or mpu.get_tensor_model_parallel_rank() == 0:
            # Build datasets.
            train_ds, valid_ds, test_ds = dataset

            # Build dataloders.
            train_dataloader = build_pretraining_data_loader(train_ds, args.consumed_train_samples)
            valid_dataloader = build_pretraining_data_loader(valid_ds, args.consumed_valid_samples)
            test_dataloader = build_pretraining_data_loader(test_ds, 0)

        return train_dataloader, valid_dataloader, test_dataloader
