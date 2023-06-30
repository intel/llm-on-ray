import math
import time
from itertools import chain

import numpy as np
import torch
import datasets
import transformers

from .dataprocesser import DataProcesser
from ..logging import logger

INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)
TEXT_COLUMN_NAME = "text"
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
        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 1)
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 1)
        group = self.config.get("group", False)
        shuffle = self.config.get("shuffle", False)
        tokenizer.pad_token = tokenizer.eos_token

        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names

        if TEXT_COLUMN_NAME not in column_names:
            def prompt(rec):
                instruction = rec["instruction"]
                response = rec["response"]
                context = rec.get("context")
                if not instruction:
                    raise ValueError(f"Expected an instruction in: {rec}")
                if not response:
                    raise ValueError(f"Expected a response in: {rec}")
                if context:
                    rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
                else:
                    rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
                return rec 

            dataset = dataset.map(
                prompt,
                load_from_cache_file=False,
                desc="Prompt",
            )
            column_names += [TEXT_COLUMN_NAME]

        max_length = self.config.get("max_length", 1024)
        def tokenize_function(examples):
            return tokenizer(examples[TEXT_COLUMN_NAME], max_length=max_length)

        tokenized_datasets = dataset.map(
            tokenize_function,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Tokenize dataset",
        )

        if group:
            block_size = self.config.get("block_size", 1024)
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
            default_data_collator=transformers.default_data_collator

        else:
            default_data_collator = DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
            )

        train_dataset = tokenized_datasets["train"]
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            collate_fn=default_data_collator, 
            batch_size=per_device_train_batch_size
        )

        eval_dataloader = None
        if "validation" in tokenized_datasets:
            eval_dataset = tokenized_datasets["validation"]
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, 
                collate_fn=default_data_collator, 
                batch_size=per_device_eval_batch_size
            )
        return train_dataloader, eval_dataloader
