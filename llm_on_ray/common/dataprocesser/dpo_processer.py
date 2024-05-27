import datasets
import torch
from typing import Dict, List, Any

from llm_on_ray import common
from llm_on_ray.common.dataprocesser import DataProcesser

IGNORE_INDEX = -100


class DPOProcesser(DataProcesser):
    def return_prompt_and_responses(self, samples) -> Dict[str, str]:
        pass

    def preprocess_function(self, examples, tokenizer):
        pass

    def tokenize_dataset(self, tokenizer, dataset):
        pass


class DPOIntelOrcaProcesser(DPOProcesser):
    def return_prompt_and_responses(self, samples) -> Dict[str, str]:
        common.logger.info("DPOIntelOrcaProcesser return_prompt_and_responses")

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

    def preprocess_function(self, examples, tokenizer):
        common.logger.info("DPOIntelOrcaProcesser preprocess_function")
        common.logger.info(examples)

        for prompt in examples["prompt"]:
            prompts = {p.strip() for p in prompt}
        chosens = {c.strip() for c in examples["chosen"]}
        rejects = {r.strip() for r in examples["rejected"]}
        common.logger.info("prompts")
        common.logger.info(prompts)

        examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        for prompt, chosen, reject in zip(prompts, chosens, rejects):
            prompt_tokens = tokenizer.tokenize(prompt, return_tensors="pt")

            if len(prompt_tokens) > self.config["max_prompt_length"]:
                prompt_tokens = prompt_tokens[: self.config["max_prompt_length"]]

            prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)

            max_resp = self.config["max_length"] - len(prompt_ids)
            chosen_tokens = tokenizer.tokenize(chosen)
            chosen_tokens = chosen_tokens[: max_resp - 1]
            chosen_tokens.append(tokenizer.eos_token)
            chosen_ids = tokenizer.convert_tokens_to_ids(chosen_tokens)

            reject_tokens = tokenizer.tokenize(reject)
            reject_tokens = reject_tokens[: max_resp - 1]
            reject_tokens.append(tokenizer.eos_token)
            reject_ids = tokenizer.convert_tokens_to_ids(reject_tokens)

            chosen_input_ids = prompt_ids + chosen_ids

            reject_input_ids = prompt_ids + reject_ids

            # padding
            input_len = len(chosen_input_ids)
            if self.config["pad_max"]:
                pad_len = self.config["max_length"] - input_len
                chosen_input_ids = chosen_input_ids + [0] * pad_len
                assert len(chosen_input_ids) == self.config["max_length"]

            input_len = len(reject_input_ids)
            if self.config["pad_max"]:
                pad_len = self.config["max_length"] - input_len
                reject_input_ids = reject_input_ids + [0] * pad_len
                assert len(reject_input_ids) == self.config["max_length"]

            examples["prompt"].append(prompt)
            examples["chosen"].append(prompt + chosen)
            examples["rejected"].append(prompt + reject)

        common.logger.info("examples")
        common.logger.info(examples)
        return examples

    def tokenize_dataset(self, tokenizer, dataset):
        common.logger.info("DPOProcesser tokenize_dataset")
        tokenizer.pad_token = tokenizer.eos_token
        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names
        common.logger.info(dataset)

        raw_datasets = dataset.map(
            self.return_prompt_and_responses,
            remove_columns=column_names,
        )
        train_dataset = raw_datasets["train"]
        common.logger.info("raw_datasets train")
        common.logger.info(train_dataset)

        column_names = train_dataset.column_names

        if train_dataset is not None:
            # Create train feature from dataset
            train_dataset = train_dataset.map(
                lambda examples: self.preprocess_function(examples, tokenizer),
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

        eval_dataset = raw_datasets["validation"]
        common.logger.info("raw_datasets validation")
        common.logger.info(eval_dataset)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                lambda examples: self.preprocess_function(examples, tokenizer),
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )
        common.logger.info("train_dataset")
        common.logger.info(train_dataset)
        common.logger.info("eval_dataset")
        common.logger.info(eval_dataset)

        return train_dataset, eval_dataset
