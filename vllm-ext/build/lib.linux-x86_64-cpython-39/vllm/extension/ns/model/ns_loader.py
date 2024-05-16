import importlib
from typing import Optional
import torch
from torch import nn
from vllm.config import CacheConfig, DeviceConfig, LoRAConfig, ModelConfig, ParallelConfig, SchedulerConfig, VisionLanguageConfig
from vllm.model_executor.model_loader.utils import set_default_torch_dtype

# extend get_model_loader to load ns model
vllm_loader = importlib.import_module('vllm.model_executor.model_loader.loader')

class NSModelLoaderV2(vllm_loader.DefaultModelLoader):

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig,
                   cache_config: CacheConfig) -> nn.Module:
        if (model_config.quantization is None) or (model_config.quantization != 'ns'):
            raise ValueError(f"Model {model_config.model} is not a NS model")
        
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                vllm_model = vllm_loader._initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config, cache_config)
            # initialize native inference engine
            vllm_model.model.init_inference_engine(model_config, parallel_config, scheduler_config)
            vllm_model.model.load_weights([0])
            weights_generator = self._get_weights_iterator(model_config.model,
                                           model_config.revision,
                                           fall_back_to_pt=getattr(
                                               vllm_model,
                                               "fall_back_to_pt_during_load",
                                               True))
            
            filtered_weights = [(name, weight) for name, weight in weights_generator if (name == "lm_head.weight" or name == "model.embed_tokens.weight")]
            vllm_model.load_weights(filtered_weights)

        return vllm_model.eval()
