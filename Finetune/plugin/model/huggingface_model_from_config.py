import transformers

from .model import Model

class HuggingFaceModelFromConfig(Model):
    def __call__(self, config):
        name = config.get("name")
        model_config = config.get("config", {})
        if name is not None:
            config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=name, **model_config)
        else:
            config = transformers.AutoConfig.for_model(**model_config)
        return transformers.AutoModelForCausalLM.from_config(config)
