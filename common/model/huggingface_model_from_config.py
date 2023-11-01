import transformers

from .model import Model
from peft import get_peft_model, LoraConfig

class HuggingFaceModelFromConfig(Model):
    def __call__(self, config):
        name = config.get("name")
        model_config = config.get("config", {})
        auto_config = None
        if name is not None:
            auto_config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=name, **model_config)
        else:
            auto_config = transformers.AutoConfig.for_model(**model_config)
        model = transformers.AutoModelForCausalLM.from_config(auto_config)
        lora_config = config.get("lora_config")
        if lora_config:
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config) 
        return model
