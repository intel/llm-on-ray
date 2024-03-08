import transformers

from llm_on_ray.common.tokenizer import Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    def __call__(self, config):
        name = config.get("name")
        load_config = config.get("config", {})
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, **load_config)
        return tokenizer
