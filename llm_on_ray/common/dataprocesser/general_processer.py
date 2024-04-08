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

from itertools import chain

import numpy as np
import torch
import datasets
import transformers

from llm_on_ray.common.dataprocesser import DataProcesser

RESPONSE_KEY = "### Response:"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"


class DataCollatorForCompletionOnlyLM(transformers.DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is not None:
                response_token_ids_end_idx = response_token_ids_start_idx + 1

                # Make pytorch loss function ignore all tokens up through the end of the response key
                labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


class GeneralProcesser(DataProcesser):
    def prepare(self, tokenizer, dataset):
        per_device_train_batch_size = self.config.get("per_device_train_batch_size")
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size")
        max_length = self.config.get("max_length")
        custom_chat_template = self.config.get("custom_chat_template")
        model_default_chat_template = self.config.get("model_default_chat_template")

        group = self.config.get("group")
        block_size = self.config.get("block_size")
        shuffle = self.config.get("shuffle")
        tokenizer.pad_token = tokenizer.eos_token

        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names

        def tokenize_function(examples):
            if self.config.get("is_base_model"):
                if custom_chat_template:
                    tokenizer.chat_template = custom_chat_template
                    new_tokenizer = tokenizer.apply_chat_template(
                        examples,
                        tokenize=False,
                        max_length=max_length,
                    )
                else:
                    tokenizer.chat_template = self.config.get("default_chat_template")
                    new_tokenizer = tokenizer.apply_chat_template(
                        examples,
                        tokenize=False,
                        max_length=max_length,
                    )
            else:
                if model_default_chat_template:
                    tokenizer.chat_template = model_default_chat_template
                    new_tokenizer = tokenizer.apply_chat_template(
                        examples,
                        tokenize=False,
                        max_length=max_length,
                    )
                else:
                    new_messages = [
                        {
                            "role": "user",
                            "content": "instruction: "
                            + examples["instruction"]
                            + " context: "
                            + examples["context"],
                        },
                        {"role": "assistant", "content": "response: " + examples["response"]},
                    ]

                    new_tokenizer = tokenizer.apply_chat_template(
                        new_messages,
                        tokenize=False,
                        max_length=max_length,
                    )
            return new_tokenizer

        tokenized_datasets = dataset.map(
            tokenize_function,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Tokenize dataset",
        )

        if group:

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
                # customize this part to your needs.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                load_from_cache_file=False,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        train_dataset = tokenized_datasets["train"]
        train_dataloader_params = {
            "shuffle": shuffle,
            "collate_fn": data_collator,
            "batch_size": per_device_train_batch_size,
            "pin_memory": True,
        }
        train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_params)

        eval_dataloader = None
        if "validation" in tokenized_datasets:
            eval_dataset = tokenized_datasets["validation"]
            eval_dataloader_params = {
                "shuffle": shuffle,
                "collate_fn": data_collator,
                "batch_size": per_device_eval_batch_size,
            }
            eval_dataloader = torch.utils.data.DataLoader(eval_dataset, **eval_dataloader_params)
        return train_dataloader, eval_dataloader
