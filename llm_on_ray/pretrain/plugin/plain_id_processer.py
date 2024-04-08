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

import torch
import transformers

from llm_on_ray.common.dataprocesser import DataProcesser


class PlainIDProcesser(DataProcesser):
    def prepare(self, tokenizer, datasets):
        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 1)
        self.config.get("per_device_eval_batch_size", 1)

        def label(examples):
            examples["input_ids"] = examples["tokens"].copy()
            examples["labels"] = examples["tokens"].copy()
            return examples

        train_datasets = [
            dataset["train"].map(label, remove_columns=["tokens"]) for dataset in datasets
        ]
        train_dataloaders = [
            torch.utils.data.DataLoader(
                train_dataset,
                shuffle=False,
                collate_fn=transformers.default_data_collator,
                batch_size=per_device_train_batch_size,
            )
            for train_dataset in train_datasets
        ]
        return train_dataloaders, None
