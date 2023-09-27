import transformers

from .model import Model
from peft import get_peft_model, LoraConfig

class HuggingFaceModelForCausalLM(Model):
    def __call__(self, config):
        name = config.get("name")
        model_config = config.get("config", {})
        model = transformers.AutoModelForCausalLM.from_pretrained(name, **model_config)
        lora_config = config.get("lora_config")
        if lora_config:
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
        return model
