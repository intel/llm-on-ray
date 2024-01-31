import re
import torch
from inference_config import InferenceConfig, DEVICE_CPU

class Predictor:

  def __init__(self, inferCfg: InferenceConfig) -> None:
     self.inferCfg = inferCfg

  def configure_tokenizer(self, model_name, tokenizer):
    model = self.model
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
  
  def generate(self, inputs, **config):
    pass

  def streaming_generate(self, inputs, streamer, **config):
    pass

  @staticmethod
  def get_torch_dtype(inferenceConfig: InferenceConfig, hf_config) -> torch.dtype:
    '''
    return torch default dtype, a.k.a float32, if it's cpu only inference without ipex because
    bfloat16 is too slow and float16 is not supported in CPU
    '''
    if hf_config is None or Predictor.is_cpu_without_ipex(inferenceConfig):
        return torch.get_default_dtype()
    if hasattr(hf_config, 'torch_dtype'):
        t = hf_config.torch_dtype
        if t:
            return t
    if hasattr(hf_config, '__getitem__'):
        t = hf_config['torch_dtype']
        if t:
            return t
    return torch.get_default_dtype()

  @staticmethod
  def is_cpu_without_ipex(inferenceConfig: InferenceConfig) -> bool:
      return (not inferenceConfig.ipex.enabled) and inferenceConfig.device == DEVICE_CPU
