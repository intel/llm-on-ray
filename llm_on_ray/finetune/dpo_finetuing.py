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
import torch
import transformers
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from typing import Dict

from itertools import chain

from llm_on_ray.finetune.data_process import DPOIntelOrcaProcessor
from llm_on_ray.finetune.finetuning import Finetuning

IGNORE_INDEX = -100


class DPOFineTuning(Finetuning):
    def tokenize_dataset(self, config: Dict, tokenizer, dataset):
        processor = DPOIntelOrcaProcessor(config, tokenizer)

        for key in dataset:
            prompts = processor.make_prompt(dataset[key])
            dataset[key] = datasets.Dataset.from_dict(prompts)

        train_dataset = dataset["train"]
        column_names = list(train_dataset.features)
        (
            processor.tokenize_by_neural_chat
            if config["Dataset"].get("data_preprocess_type", "neural_chat") == "neural_chat"
            else processor.tokenize
        )
        if train_dataset is not None:
            # Create train feature from dataset
            train_dataset = train_dataset.map(
                processor.tokenize,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

        eval_dataset = dataset.get("validation")

        if eval_dataset is not None:
            column_names = eval_dataset.column_names
            eval_dataset = eval_dataset.map(
                processor.tokenize,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )
        tokenized_datasets = {"train": train_dataset, "validation": eval_dataset}

        return tokenized_datasets

    def load_model(self, config: Dict):
        model_name = config["General"]["base_model"]
        model_dtype = self.convert_dtype(config["Training"].get("mixed_precision", "no"))
        model_config = config["General"].get("config", {})
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=model_dtype, **model_config
        )

        egc = config["General"].get("enable_gradient_checkpointing", False)
        if egc:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        model.to(dtype=model_dtype, device=torch.device(config["Training"]["device"]))

        return model

    def load_model_ref(self, config: Dict):
        model_name = config["General"]["base_model"]
        model_dtype = self.convert_dtype(config["Training"].get("mixed_precision", "no"))
        model_config = config["General"].get("config", {})

        # load reference model
        model_ref = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=model_dtype, **model_config
        )

        model_ref.config.use_cache = False
        model_ref.to(dtype=model_dtype, device=torch.device(config["Training"]["device"]))

        return model_ref

    def get_trainer(self, config: Dict, model, tokenizer, tokenized_dataset, data_collator):
        device = config["Training"]["device"]
        lora_config = config["General"].get("lora_config", None)

        if device in ["cpu", "gpu"]:
            from transformers import Trainer, TrainingArguments
            from trl import DPOTrainer

            training_args = self.convert_to_training_args(TrainingArguments, config)

            trainer = DPOTrainer(
                model,
                self.load_model_ref(config) if lora_config is not None else None,
                args=training_args,
                beta=config["Training"].get("beta"),
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"]
                if tokenized_dataset.get("validation") is not None
                else None,
                tokenizer=tokenizer,
                peft_config=LoraConfig(**lora_config) if lora_config is not None else None,
                max_length=config["Dataset"].get("max_length"),
                max_prompt_length=config["Dataset"].get("max_prompt_length"),
            )
        elif device in ["hpu"]:
            from optimum.habana.trl import GaudiDPOTrainer as DPOTrainer
            from optimum.habana.transformers import GaudiTrainingArguments
            from optimum.habana import GaudiConfig

            # If gaudi_config_name is provided, load gaudi_config from huggingface model hub(https://huggingface.co/Habana), otherwise use default gaudi_config
            gaudi_config_name = config["General"].get("gaudi_config_name", None)
            if gaudi_config_name is not None:
                gaudi_config = GaudiConfig.from_pretrained(gaudi_config_name)
            else:
                gaudi_config = GaudiConfig()
                gaudi_config.use_fused_adam = True
                gaudi_config.use_fused_clip_norm = True

            training_args = self.convert_to_training_args(GaudiTrainingArguments, config)
            trainer = DPOTrainer(
                model,
                self.load_model_ref(config) if lora_config is not None else None,
                args=training_args,
                gaudi_config=gaudi_config,
                beta=config["Training"].get("beta"),
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"]
                if tokenized_dataset.get("validation") is not None
                else None,
                tokenizer=tokenizer,
                peft_config=LoraConfig(**lora_config) if lora_config is not None else None,
                max_length=config["Dataset"].get("max_length"),
                max_prompt_length=config["Dataset"].get("max_prompt_length"),
            )

        return training_args, trainer
