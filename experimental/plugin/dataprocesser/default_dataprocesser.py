import math
import time
from itertools import chain

import torch
import transformers

from .dataprocesser import DataProcesser

class DefaultDataProcesser(DataProcesser):
    def prepare(self, model, tokenizer, dataset, optimizer):
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
        train_dataloader, eval_dataloader = self.prepare_dataset(dataset, tokenizer)
        lr_scheduler = self.get_lr_scheduler(optimizer, train_dataloader)
        return model, tokenizer, train_dataloader, eval_dataloader, optimizer, lr_scheduler

    def get_lr_scheduler(self, optimizer, train_dataloader):
        overrode_max_train_steps = False
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        max_train_steps  = self.config.get("max_train_steps")
        num_train_epochs = self.config.get("num_train_epochs", 1)
        lr_scheduler_type = self.config.get("lr_scheduler_type", "linear")

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        num_warmup_steps = self.config.get("num_warmup_steps", 0)
        lr_scheduler = transformers.get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        return lr_scheduler

    def prepare_dataset(self, dataset, tokenizer):
        column_names = dataset["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        preprocessing_num_workers = self.config.get("preprocessing_num_workers", 1)
        overwrite_cache = self.config.get("overwrite_cache", True)

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        block_size = self.config.get("block_size")
        if block_size is None:
            block_size = tokenizer.model_max_length
            # if block_size > 1024:
            #     logger.warning(
            #         f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            #         "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            #     )
            block_size = 1024
        else:
            if block_size > tokenizer.model_max_length:
                # logger.warning(
                #     f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                #     f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                # )
                pass
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
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

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