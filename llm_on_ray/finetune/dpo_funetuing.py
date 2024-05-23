from transformers import AutoModelForCausalLM


class DPOFuneTuning:
    def __init__(self, config):
        self.config = config

    def get_model(self):
        # load policy model
        model = AutoModelForCausalLM.from_pretrained(
            self.config["General"]["base_model"],
            config=self.config,
            low_cpu_mem_usage=True,
            use_auth_token=True if self.config["config"]["base_model"] else None
        )
        model.config.use_cache = False
        return model

    def get_model_ref(self):
        # load reference model
        model_ref = AutoModelForCausalLM.from_pretrained(
            self.config["General"]["base_model"],
            config=self.config,
            low_cpu_mem_usage=True,
            use_auth_token=True if self.config["config"]["base_model"] else None
        )
        model_ref.config.use_cache = False
        return model_ref

    def dpo_train(self, training_args, tokenized_datasets, tokenizer):
        from intel_extension_for_transformers.transformers.dpo_trainer import DPOTrainer
        peft_config = self.config["General"].get("lora_config", None)
        return DPOTrainer(
            self.get_model(),
            self.get_model_ref() if peft_config is not None else None,
            args=training_args,
            beta=self.config["Training"].get("beta"),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_length=self.config["Dataset"].get("max_length"),
        )


class GaudiDPOFuneTuning(DPOFuneTuning):
    def dpo_train(self, training_args, tokenized_datasets, tokenizer):
        from intel_extension_for_transformers.transformers.dpo_trainer import GaudiDPOTrainer as DPOTrainer

        peft_config = self.config["General"].get("lora_config", None)
        return DPOTrainer(
            self.get_model(),
            self.get_model_ref() if peft_config is not None else None,
            args=training_args,
            beta=self.config["Training"].get("beta"),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_length=self.config["Dataset"].get("max_length"),
        )
