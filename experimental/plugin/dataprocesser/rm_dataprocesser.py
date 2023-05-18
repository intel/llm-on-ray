import math
import time
from itertools import chain

import torch
import transformers

from .default_dataprocesser import DefaultDataProcesser

class RMDataProcesser(DefaultDataProcesser):

    def prepare_dataset(self, dataset, tokenizer):

        block_size = self.config.get("block_size")
        if block_size is None:
            block_size = 1024
        else:
            if block_size > tokenizer.model_max_length:
                pass
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
            remove_columns=["chosen", "rejected", "prompt", "response"],
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