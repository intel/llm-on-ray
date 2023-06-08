import math
import time
from itertools import chain

import torch
import transformers

from .dataprocesser import DataProcesser
from ..logging import logger

class GeneralProcesser(DataProcesser):
    def prepare(self, tokenizer, dataset):
        preprocessing_num_workers = self.config.get("preprocessing_num_workers", 4)
        overwrite_cache = self.config.get("overwrite_cache", True)
        batched = self.config.get("batched", True)

        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 1)
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 1)
        shuffle = self.config.get("shuffle", False)

        column_names = dataset["train"].column_names
        def tokenize_function(examples):
            examples["all"] = examples["instruction"] + examples["output"]
            return tokenizer(examples["all"])

        tokenized_datasets = dataset.map(
            tokenize_function,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Tokenize dataset",
        )

        train_dataset = tokenized_datasets["train"]
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            collate_fn=transformers.default_data_collator, 
            batch_size=per_device_train_batch_size
        )

        eval_dataloader = None
        if "validation" in lm_datasets:
            eval_dataset = lm_datasets["validation"]
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, 
                collate_fn=transformers.default_data_collator, 
                batch_size=per_device_eval_batch_size
            )
        return train_dataloader, eval_dataloader
