import transformers

from .model import Model
from peft import get_peft_model, LoraConfig
import deltatuner


class HuggingFaceModelForCausalLM(Model):
    def __call__(self, config):
        name = config.get("name")
        model_dtype = config.get("dtype")
        model_config = config.get("config", {})
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=model_dtype, **model_config
        )

        lora_config = config.get("lora_config", None)
        if lora_config:
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)
            deltatuner_config = config.get("deltatuner_config", None)
            if deltatuner_config:
                model = deltatuner.optimize(model, **deltatuner_config)

        gradient_checkpointing = config.get("gradient_checkpointing")
        if gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        return model
