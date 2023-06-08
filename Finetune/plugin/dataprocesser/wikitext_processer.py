import math
import time
from itertools import chain

import torch
import transformers

from .dataprocesser import DataProcesser
from ..logging import logger

class WikitextProcesser(DataProcesser):
    def prepare(self, tokenizer, dataset):
        preprocessing_num_workers = self.config.get("preprocessing_num_workers", 4)
        overwrite_cache = self.config.get("overwrite_cache", True)
        batched = self.config.get("batched", True)
        batch_size = self.config.get("batch_size", 1000)
        block_size = self.config.get("block_size")

        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 1)
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 1)
        shuffle = self.config.get("shuffle", False)

        column_names = dataset["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=batched,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Tokenize dataset",
        )
        
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

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=batched,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        train_dataset = lm_datasets["train"]
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