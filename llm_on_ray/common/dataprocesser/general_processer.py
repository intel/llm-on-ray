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

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction: "
INPUT_KEY = "Input: "
RESPONSE_KEY = "### Response: "
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
)
TEXT_COLUMN_NAME = "text"

SLIMORCA_PROMPT_DICT = {
    "prompt_with_input": ("### System: {system} \n" "### User: {user} \n### Assistant: {gpt}"),
    "prompt_without_input": ("### System: {system} \n" "### Assistant: {gpt}"),
}


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


class ChatDataPreprocess(DataProcesser):
    base_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"""

    def __init__(self, config):
        super().__init__(config)
        self.prompt_template = self.base_template
        self.user = "### Instruction:\n"
        self.assistant = "### Response:\n"
        self.end = "### End\n"

    def create_data(self, examples):
        if self.config.get("gpt_base_model"):
            instruction = examples["instruction"]
            response = examples["response"]
            context = examples.get("context")
            if not instruction:
                raise ValueError(f"Expected an instruction in: {examples}")
            if not response:
                raise ValueError(f"Expected a response in: {examples}")
            if context:
                new_messages = PROMPT_WITH_INPUT_FORMAT.format(
                    instruction=instruction, response=response, input=context
                )
            else:
                new_messages = PROMPT_NO_INPUT_FORMAT.format(
                    instruction=instruction, response=response
                )
        else:
            new_messages = [
                {
                    "role": "system",
                    "content": INTRO_BLURB + "\n",
                },
                {
                    "role": "user",
                    "content": examples["instruction"]
                    + "\n"
                    + INPUT_KEY
                    + examples["context"]
                    + "\n",
                },
                {"role": "assistant", "content": examples["response"] + "\n"},
            ]

        return new_messages

    def tokenize_func(self, tokenizer, message):
        if self.config.get("gpt_base_model"):
            return tokenizer(
                message, add_special_tokens=False, max_length=self.config.get("max_length")
            )
        else:
            if self.config.get("chat_template") is not None:
                tokenizer.chat_template = self.config.get("chat_template")
            elif tokenizer.chat_template is not None:
                pass
            else:
                tokenizer.chat_template = self.config.get("default_chat_template")

            new_tokenizer = tokenizer.apply_chat_template(
                message,
                tokenize=False,
            )
            return tokenizer(
                new_tokenizer, add_special_tokens=False, max_length=self.config.get("max_length")
            )

    def tokenize_dataset(self, tokenizer, dataset):
        group = self.config.get("group")
        block_size = self.config.get("block_size")
        tokenizer.pad_token = tokenizer.eos_token

        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names

        tokenized_datasets = dataset.map(
            lambda examples: self.tokenize_func(tokenizer, self.create_data(examples)),
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

        return tokenized_datasets

    def prepare_dataloader(self, tokenizer, dataset):
        per_device_train_batch_size = self.config.get("per_device_train_batch_size")
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size")
        shuffle = self.config.get("shuffle")

        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            mlm=False,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )

        tokenized_datasets = self.tokenize_dataset(tokenizer, dataset)

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


class SlimOrcaDataPreprocess(ChatDataPreprocess):
    def __init__(self, config):
        super().__init__(config)
        self.default_system = "You are a helpful, respectful and honest assistant."

    def create_data(self, data):
        examples = {}
        conv = data["conversations"]
        # system
        if conv[0]["from"] != "system":
            examples["system"] = self.default_system
            start = 0
        elif conv[0]["from"] == "system" and conv[0]["value"] == "":
            examples[conv[0]["from"]] = self.default_system
            start = 1
        else:
            examples[conv[0]["from"]] = conv[0]["value"]
            start = 1

        for j in range(start, len(conv) - 1, 2):
            examples[conv[j]["from"]] = conv[j]["value"]
            examples[conv[j + 1]["from"]] = conv[j + 1]["value"]

        if self.config.get("gpt_base_model"):
            if examples["human"]:
                return PROMPT_WITH_INPUT_FORMAT.format(
                    instruction=examples["system"],
                    response=examples["gpt"],
                    input=examples["human"],
                )
            else:
                return PROMPT_NO_INPUT_FORMAT.format(
                    instruction=examples["system"], response=examples["gpt"]
                )
        else:
            new_messages = [
                {"role": "system", "content": INTRO_BLURB + "\n"},
                {
                    "role": "user",
                    "content": examples["system"] + "\n" + INPUT_KEY + examples["human"] + "\n",
                },
                {"role": "assistant", "content": examples["gpt"] + "\n"},
            ]
            return new_messages


class OpenOrcaDataPreprocess(ChatDataPreprocess):
    def __init__(self, config):
        super().__init__(config)
        self.default_system = "You are an AI assistant. You will be given a task. You must generate a detailed and long answer."

    def create_data(self, examples):
        if self.config.get("gpt_base_model"):
            if not examples["system"]:
                examples["system"] = self.default_system

            if examples["question"]:
                return PROMPT_WITH_INPUT_FORMAT.format(
                    instruction=examples["system"],
                    response=examples["chosen"],
                    input=examples["question"],
                )
            else:
                return PROMPT_NO_INPUT_FORMAT.format(
                    instruction=examples["system"], response=examples["chosen"]
                )
        else:
            new_messages = [
                {"role": "system", "content": INTRO_BLURB + "\n"},
                {
                    "role": "user",
                    "content": examples["system"] + "\n" + INPUT_KEY + examples["question"] + "\n",
                },
                {"role": "assistant", "content": examples["chosen"] + "\n"},
            ]
            return new_messages
