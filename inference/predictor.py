import re
import torch
from transformers import AutoTokenizer, StoppingCriteriaList, TextIteratorStreamer
from inference.inference_config import InferenceConfig
from utils import max_input_len, StoppingCriteriaSub

class Predictor:
    def __init__(self, infer_conf: InferenceConfig) -> None:
        self.infer_conf = infer_conf
        self.tokenizer = AutoTokenizer.from_pretrained(infer_conf.model_description.tokenizer_name_or_path)
        self.device = torch.device(infer_conf.device)
        prompt = infer_conf.model_description.prompt
        stop_words = prompt.stop_words
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def tokenize_inputs(self, text):
        if self.device.type == "hpu":
            input_tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=max_input_len(input_token_len),
            )
        else:
            input_tokens = self.tokenizer(
                text, return_tensors="pt", padding=True
            )
        return input_tokens.input_ids.to(device=self.device)

    def configure_tokenizer(self, model_name):
        model = self.model
        tokenizer = self.tokenizer
        if re.search("llama", model.config.architectures[0], re.IGNORECASE):
            # unwind broken decapoda-research config
            model.generation_config.pad_token_id = 0
            model.generation_config.bos_token_id = 1
            model.generation_config.eos_token_id = 2

        if (
            hasattr(model.generation_config, "pad_token_id")
            and model.generation_config.pad_token_id is not None
            and not "chatglm" in model_name
        ):
            tokenizer.pad_token_id = model.generation_config.pad_token_id
        if (
            hasattr(model.generation_config, "eos_token_id")
            and model.generation_config.eos_token_id is not None
            and not "chatglm" in model_name
        ):
            tokenizer.eos_token_id = model.generation_config.eos_token_id
        if (
            hasattr(model.generation_config, "bos_token_id")
            and model.generation_config.bos_token_id is not None
        ):
            tokenizer.bos_token_id = model.generation_config.bos_token_id

        if tokenizer.pad_token_id is None:
            model.generation_config.pad_token_id = (
                tokenizer.pad_token_id
            ) = tokenizer.eos_token_id

        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        if not model.config.is_encoder_decoder:
            tokenizer.padding_side = "left"

        if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    def generate(self, prompt, **config):
        pass

    def streaming_generate(self, prompt, streamer, **config):
        pass

    def get_streamer(self):
        return TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True)
