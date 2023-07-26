import math
import time
from itertools import chain

import torch
import transformers

from .dataprocesser import DataProcesser
from ..logging import logger

class RMDataProcesser(DataProcesser):

    def prepare(self, tokenizer, dataset):

        block_size = self.config.get("block_size")
        
        
        if block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
            block_size = 1024
        else:
            if block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(block_size, tokenizer.model_max_length)
        
        def tokenize_function(examples):
            
            tokenizer.pad_token = tokenizer.eos_token
            chosen = tokenizer(
                examples["prompt"] + examples["chosen"],
                max_length=block_size,
                truncation=True,
                padding="max_length",
            )
            
            examples["chosen_input_ids"] = chosen["input_ids"]
            examples["chosen_attention_mask"] = chosen["attention_mask"]


            rejected = tokenizer(
                examples["prompt"] + examples["rejected"],
                max_length=block_size,
                truncation=True,
                padding="max_length",
            )
            examples["rejected_input_ids"] = rejected["input_ids"]
            examples["rejected_attention_mask"] = rejected["attention_mask"]

            # Column 0 always contains the preferred response.
            examples["labels"] = 0

            return examples

        preprocessing_num_workers = self.config.get("preprocessing_num_workers", 1)
        overwrite_cache = self.config.get("overwrite_cache", True)

        lm_datasets = dataset.map(
            tokenize_function,
            num_proc=preprocessing_num_workers,
            remove_columns=["chosen", "rejected", "prompt"],
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["test"]

        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 2)
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 4)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            collate_fn=transformers.default_data_collator, 
            batch_size=per_device_train_batch_size
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, 
            collate_fn=transformers.default_data_collator, 
            batch_size=per_device_eval_batch_size
        )
        return train_dataloader, eval_dataloader