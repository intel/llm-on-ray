import math
import time
from itertools import chain
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
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
DEFAULT_SEED = 42

# This is a training prompt that does not contain an input string.  The instruction by itself has enough information
# to respond.  For example, the instruction might ask for the year a historic figure was born.
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

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
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

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

class DataCollatorForCompletionOnlyLM(transformers.DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
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

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

class Dolly2Processer(DataProcesser):
    def prepare(self, tokenizer, dataset):
        preprocessing_num_workers = self.config.get("preprocessing_num_workers", 4)
        batched = self.config.get("batched", True)
        batch_size = self.config.get("batch_size", 1000)

        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 1)
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 1)
        shuffle = self.config.get("shuffle", False)
        max_length = self.config.get("max_length")
        test_size = self.config.get("test_size")
        seed = self.config.get("seed")

        if max_length is None:
            max_length = tokenizer.model_max_length

        dataset = dataset["train"]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
        
        def _add_text(rec):
            instruction = rec["instruction"]
            response = rec["response"]
            context = rec.get("context")

            if not instruction:
                raise ValueError(f"Expected an instruction in: {rec}")

            if not response:
                raise ValueError(f"Expected a response in: {rec}")

            # For some instructions there is an input that goes along with the instruction, providing context for the
            # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
            # some piece of information from it.  The response is that information to extract.  In other cases there is
            # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
            # born.
            if context:
                rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
            else:
                rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
            return rec

        dataset = dataset.map(
            _add_text,
            num_proc=preprocessing_num_workers,
        )

        def preprocess_batch(batch: Dict[str, List], tokenizer: transformers.AutoTokenizer, max_length: int) -> dict:
            return tokenizer(
                batch["text"],
                max_length=max_length,
                truncation=True,
            )

        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=batched,
            batch_size = batch_size,
            num_proc=preprocessing_num_workers,
            remove_columns=["instruction", "context", "response", "text", "category"],
        )

        dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)

        default_data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
        )

        if test_size is None or test_size == 0:
            train_dataset = dataset
            eval_dataset = None
        else:
            split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
            train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=shuffle, 
            collate_fn=default_data_collator, 
            batch_size=per_device_train_batch_size
        )
        if eval_dataset is not None:
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, 
                collate_fn=default_data_collator, 
                batch_size=per_device_eval_batch_size
            )
        else:
            eval_dataloader = None
        return train_dataloader, eval_dataloader