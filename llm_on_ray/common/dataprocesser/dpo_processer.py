import datasets
import torch
from typing import Dict, List, Any

from llm_on_ray import common
from llm_on_ray.common.dataprocesser import DataProcesser

IGNORE_INDEX = -100


class DPOProcesser(DataProcesser):
    def return_prompt_and_responses(self, samples) -> Dict[str, str]:
        pass

    def tokenize_dataset(self, tokenizer, dataset):
        pass


class DPOIntelOrcaProcesser(DPOProcesser):

    def return_prompt_and_responses(self, samples) -> Dict[str, str]:
        return {
            "prompt": " ".join(
                [
                    system + question
                    for system, question in zip(samples["system"], samples["question"])
                ]
            ),
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    def tokenize_dataset(self, tokenizer, dataset):
        common.logger.info("DPOIntelOrcaProcesser tokenize_dataset")

        tokenizer.pad_token = tokenizer.eos_token
        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names
        common.logger.info(dataset)
        common.logger.info("column_names")
        common.logger.info(column_names)

        raw_datasets = dataset.map(
            self.return_prompt_and_responses,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Tokenize dataset",
        )
        train_dataset = raw_datasets["train"]
        common.logger.info("raw_datasets train")
        common.logger.info(train_dataset)

        column_names = train_dataset.column_names
        common.logger.info("column_names 1")
        common.logger.info(column_names)

        def preprocess_function(examples):
            prompts = {p.strip() for p in examples["prompt"]}
            chosens = {c.strip() for c in examples["chosen"]}
            rejects = {r.strip() for r in examples["rejected"]}

            examples = {
                "prompt": [],
                "chosen": [],
                "rejected": [],
                "chosen_response_only": [],
                "rejected_response_only": [],
                "chosen_input_ids": [],
                "chosen_attention_mask": [],
                "chosen_labels": [],
                "rejected_input_ids": [],
                "rejected_attention_mask": [],
                "rejected_labels": [],
                "prompt_input_ids": [],
                "prompt_attention_mask": []}

            for prompt, chosen, reject in zip(prompts, chosens, rejects):
                prompt_tokens = tokenizer.tokenize(prompt, return_tensors="pt")

                if len(prompt_tokens) > self.config["max_prompt_length"]:
                    prompt_tokens = prompt_tokens[: self.config["max_prompt_length"]]

                prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
                prompt_mask = [1] * len(prompt_ids)

                max_resp = self.config["max_length"] - len(prompt_ids)
                chosen_tokens = tokenizer.tokenize(chosen)
                chosen_tokens = chosen_tokens[: max_resp - 1]
                chosen_tokens.append(tokenizer.eos_token)
                chosen_ids = tokenizer.convert_tokens_to_ids(chosen_tokens)
                chosen_mask = [1] * len(chosen_ids)

                reject_tokens = tokenizer.tokenize(reject)
                reject_tokens = reject_tokens[:max_resp - 1]
                reject_tokens.append(tokenizer.eos_token)
                reject_ids = tokenizer.convert_tokens_to_ids(reject_tokens)
                reject_mask = [1] * len(reject_ids)

                chosen_input_ids = prompt_ids + chosen_ids
                chosen_attention_mask = prompt_mask + chosen_mask
                chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids

                reject_input_ids = prompt_ids + reject_ids
                reject_attention_mask = prompt_mask + reject_mask
                reject_labels = [IGNORE_INDEX] * len(prompt_ids) + reject_ids

                # padding
                input_len = len(chosen_input_ids)
                if self.config["pad_max"]:
                    pad_len = self.config["max_length"] - input_len
                    chosen_input_ids = chosen_input_ids + [0] * pad_len
                    chosen_labels = chosen_labels + [-100] * pad_len
                    chosen_attention_mask = chosen_attention_mask + [0] * pad_len
                    assert len(chosen_input_ids) == self.config["max_length"]

                input_len = len(reject_input_ids)
                if self.config["pad_max"]:
                    pad_len = self.config["max_length"] - input_len
                    reject_input_ids = reject_input_ids + [0] * pad_len
                    reject_labels = reject_labels + [-100] * pad_len
                    reject_attention_mask = reject_attention_mask + [0] * pad_len
                    assert len(reject_input_ids) == self.config["max_length"]

                examples["prompt"].append(prompt)
                examples["chosen"].append(prompt + chosen)
                examples["rejected"].append(prompt + reject)
                examples["chosen_response_only"].append(chosen)
                examples["rejected_response_only"].append(reject)

                examples["chosen_input_ids"].append(chosen_input_ids)
                examples["chosen_attention_mask"].append(chosen_attention_mask)
                examples["chosen_labels"].append(chosen_labels)

                examples["rejected_input_ids"].append(reject_input_ids)
                examples["rejected_attention_mask"].append(reject_attention_mask)
                examples["rejected_labels"].append(reject_labels)

                examples["prompt_input_ids"].append(prompt_ids)
                examples["prompt_attention_mask"].append(prompt_mask)

            return examples

        if train_dataset is not None:
            # Create train feature from dataset
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

        eval_dataset = raw_datasets["validation"]
        column_names = eval_dataset.column_names

        common.logger.info("raw_datasets validation")
        common.logger.info(eval_dataset)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )
        common.logger.info("train_dataset")
        common.logger.info(train_dataset)
        common.logger.info("eval_dataset")
        common.logger.info(eval_dataset)
        tokenized_datasets = {"train": train_dataset, "validation": eval_dataset}
        return tokenized_datasets