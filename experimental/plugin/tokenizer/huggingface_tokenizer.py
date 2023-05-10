from .tokenizer import Tokenizer
import transformers

class HuggingFaceTokenizer(Tokenizer):
    def __call__(self, config):
        name = config.get("name")
        if name is None:
            raise ValueError("Tokenizer config error, config should contain a tokenizer name")
        load_config = config.get("config", {})
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, **load_config)
        return tokenizer