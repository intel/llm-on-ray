import transformers

from common.model.model import Model

# for huggingface model weight random initialization
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

        return model
