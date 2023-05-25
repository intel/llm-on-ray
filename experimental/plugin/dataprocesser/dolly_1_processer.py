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

RESPONSE_KEY = "### Response:\n"

class DataCollatorForCompletionOnlyLM(transformers.DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                if np.array_equal(response_token_ids, batch["labels"][i, idx : idx + len(response_token_ids)]):
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                raise RuntimeError("Could not find response key token IDs")

            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


class Dolly1Processer(DataProcesser):
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

        def preprocess_batch(batch: Dict[str, List], tokenizer: transformers.AutoTokenizer, max_length: int) -> dict:
            return tokenizer(
                batch["text"],
                max_length=max_length,
                truncation=True,
            )

        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            num_proc=preprocessing_num_workers,
            batched=True,
            batch_size=batch_size,
            remove_columns=["instruction", "input", "output", "text"],
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


        default_data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
        )
        
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