#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from typing import Dict

IGNORE_INDEX = -100


class DPOIntelOrcaProcesser:
    @staticmethod
    def tokenize_dataset(config, tokenizer, dataset):
        tokenizer.pad_token = tokenizer.eos_token
        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names

        def return_prompt_and_responses(samples) -> Dict[str, str]:
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

        raw_datasets = dataset.map(
            return_prompt_and_responses,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Tokenize dataset",
        )
        train_dataset = raw_datasets["train"]
        column_names = train_dataset.column_names

        """
        Copied from https://github.com/intel/intel-extension-for-transformers/blob/5ba5fa8048b63bec8a3be8a7122a3db8344ad065/
        intel_extension_for_transformers/neural_chat/examples/finetuning/dpo_pipeline/dpo_clm.py#L308
        """

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
                "prompt_attention_mask": [],
            }

            for prompt, chosen, reject in zip(prompts, chosens, rejects):
                prompt_tokens = tokenizer.tokenize(prompt, return_tensors="pt")

                if len(prompt_tokens) > config["Dataset"]["max_prompt_length"]:
                    prompt_tokens = prompt_tokens[: config["Dataset"]["max_prompt_length"]]

                prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
                prompt_mask = [1] * len(prompt_ids)

                max_resp = config["Dataset"]["max_length"] - len(prompt_ids)
                chosen_tokens = tokenizer.tokenize(chosen)
                chosen_tokens = chosen_tokens[: max_resp - 1]
                chosen_tokens.append(tokenizer.eos_token)
                chosen_ids = tokenizer.convert_tokens_to_ids(chosen_tokens)
                chosen_mask = [1] * len(chosen_ids)

                reject_tokens = tokenizer.tokenize(reject)
                reject_tokens = reject_tokens[: max_resp - 1]
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
                if config["Dataset"]["pad_max"]:
                    pad_len = config["Dataset"]["max_length"] - input_len
                    chosen_input_ids = chosen_input_ids + [0] * pad_len
                    chosen_labels = chosen_labels + [-100] * pad_len
                    chosen_attention_mask = chosen_attention_mask + [0] * pad_len
                    assert len(chosen_input_ids) == config["Dataset"]["max_length"]

                input_len = len(reject_input_ids)
                if config["Dataset"]["pad_max"]:
                    pad_len = config["Dataset"]["max_length"] - input_len
                    reject_input_ids = reject_input_ids + [0] * pad_len
                    reject_labels = reject_labels + [-100] * pad_len
                    reject_attention_mask = reject_attention_mask + [0] * pad_len
                    assert len(reject_input_ids) == config["Dataset"]["max_length"]

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

        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )
        tokenized_datasets = {"train": train_dataset, "validation": eval_dataset}

        return tokenized_datasets


class DPOFuneTuning:
    def __init__(self, config):
        self.config = config

    def get_model(self):
        # load policy model
        model = AutoModelForCausalLM.from_pretrained(
            self.config["General"]["base_model"],
            config=self.config,
            low_cpu_mem_usage=True,
            use_auth_token=True if self.config["General"]["config"]["use_auth_token"] else None,
        )
        model.config.use_cache = False
        return model

    def get_model_ref(self):
        # load reference model
        model_ref = AutoModelForCausalLM.from_pretrained(
            self.config["General"]["base_model"],
            config=self.config,
            low_cpu_mem_usage=True,
            use_auth_token=True if self.config["General"]["config"]["use_auth_token"] else None,
        )
        model_ref.config.use_cache = False
        return model_ref

    def dpo_train(self, training_args, tokenized_datasets, tokenizer):
        from trl import DPOTrainer

        lora_config = self.config["General"].get("lora_config", None)
        return DPOTrainer(
            self.get_model(),
            self.get_model_ref() if lora_config is not None else None,
            args=training_args,
            beta=self.config["Training"].get("beta"),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"]
            if tokenized_datasets.get("validation") is not None
            else None,
            tokenizer=tokenizer,
            peft_config=LoraConfig(**lora_config) if lora_config is not None else None,
            max_length=self.config["Dataset"].get("max_length"),
            max_prompt_length=self.config["Dataset"].get("max_prompt_length"),
            force_use_ref_model=True if lora_config is not None else False,
        )


class GaudiDPOFuneTuning(DPOFuneTuning):
    def dpo_train(self, training_args, gaudi_config, tokenized_datasets, tokenizer):
        from optimum.habana.trl import GaudiDPOTrainer as DPOTrainer

        lora_config = self.config["General"].get("lora_config", None)
        return DPOTrainer(
            self.get_model(),
            self.get_model_ref() if lora_config is not None else None,
            args=training_args,
            gaudi_config=gaudi_config,
            beta=self.config["Training"].get("beta"),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"]
            if tokenized_datasets.get("validation") is not None
            else None,
            tokenizer=tokenizer,
            peft_config=LoraConfig(**lora_config) if lora_config is not None else None,
            max_length=self.config["Dataset"].get("max_length"),
            max_prompt_length=self.config["Dataset"].get("max_prompt_length"),
        )
