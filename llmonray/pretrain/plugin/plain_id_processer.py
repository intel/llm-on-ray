import torch
import transformers

from llmonray.common.dataprocesser import DataProcesser


class PlainIDProcesser(DataProcesser):
    def prepare(self, tokenizer, datasets):
        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 1)
        self.config.get("per_device_eval_batch_size", 1)

        def label(examples):
            examples["input_ids"] = examples["tokens"].copy()
            examples["labels"] = examples["tokens"].copy()
            return examples

        train_datasets = [
            dataset["train"].map(label, remove_columns=["tokens"]) for dataset in datasets
        ]
        train_dataloaders = [
            torch.utils.data.DataLoader(
                train_dataset,
                shuffle=False,
                collate_fn=transformers.default_data_collator,
                batch_size=per_device_train_batch_size,
            )
            for train_dataset in train_datasets
        ]
        return train_dataloaders, None
