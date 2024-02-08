from llmonray.common.tokenizer.tokenizer import Tokenizer


class _EmptyTokenizer:
    def __init__(self, max_token_id):
        self.max_token_id = max_token_id

    def __len__(self):
        return self.max_token_id


class EmptyTokenizer(Tokenizer):
    def __call__(self, config):
        tokenizer_config = config.get("config")
        return _EmptyTokenizer(**tokenizer_config)
