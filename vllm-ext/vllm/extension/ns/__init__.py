# extension for integrate with neural-speed
import importlib
from typing import Optional
import os

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import get_quant_config

from vllm.logger import init_logger

from inference_engine import Model as IE_Model

logger = init_logger(__name__)

# parameters for interacting with native code, same variables defined in native vllm_cont_batch.cpp
#==========================================================================================
_KV_CACHE_MARK_YES          = -1
_KV_CACHE_MARK_NO           = 0
_KV_CACHE_LAST_DIM          = 2
_KV_CACHE_CPY_PARAMS_SIZE   = 5
_KV_CACHE_ELEMENT_USED      = 4

# global model object to store the model
#==========================================================================================
_IE_MODEL: IE_Model = None

# global request_id to internal

# allow ns quantization
#==========================================================================================
vllm_config = importlib.import_module('vllm.config')
old_verify_quantization = vllm_config.ModelConfig._verify_quantization

def _verify_quntization(self):
    if self.quantization is not None and self.quantization == "ns":
        os.environ["NS_QUANTIZATION"] = "1"
        return
    return old_verify_quantization(self)

vllm_config.ModelConfig._verify_quantization = _verify_quntization

## add new loadformat ns
# FORMAT_DICT = {e.name : e.value for e in vllm_config.LoadFormat}
# FORMAT_DICT.update({"NS": "ns"})
# LF = enum.Enum("LoadFormat", FORMAT_DICT)

# vllm_config.LoadFormat = LF

# register ns quant config, 
# TODO add quant_ns_config.json under huggingface cache folder like below,
# {
# "quantization_config": {
#     "quant_method": "ns",
#     "weight_dtype": "int4",
#     "alg": "sym",
#     "group_size": 128,
#     "scale_dtype": "fp32",
#     "compute_dtype": "int8",
#     "version": "v1"
# }
# }
quant = importlib.import_module('vllm.model_executor.layers.quantization')

from vllm.extension.ns.quantization.cpu_ns_config import NSQuantConfig

quant.QUANTIZATION_METHODS["ns"] = NSQuantConfig

logger.info("__ns extension: add ns quantization config, %s", NSQuantConfig.__name__)

# use ns model loader for ns
#==========================================================================================
vllm_loader = importlib.import_module('vllm.model_executor.model_loader.loader')
old_get_model_loader = vllm_loader.get_model_loader

from vllm.extension.ns.model.ns_loader import NSModelLoaderV2
from vllm.config import LoadConfig, ModelConfig

def get_model_loader_ns(load_config: LoadConfig) -> vllm_loader.BaseModelLoader:
    if os.environ["NS_QUANTIZATION"] == "1":
        return NSModelLoaderV2(load_config)
    return old_get_model_loader(load_config)

vllm_loader.get_model_loader = get_model_loader_ns

def _get_linear_method_ns(model_config: vllm_config.ModelConfig,
        load_config: LoadConfig) -> Optional["LinearMethodBase"]:
    """Get the (maybe quantized) linear method."""
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config, load_config)
        linear_method = quant_config.get_linear_method()
    return linear_method

vllm_loader._get_linear_method = _get_linear_method_ns

def _get_quantization_config(
    model_config: ModelConfig,
    load_config: LoadConfig) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    if model_config.quantization is not None:
        return get_quant_config(model_config, load_config)
    return None

vllm_loader._get_quantization_config = _get_quantization_config

# reload to make above changes take effect
model_loader = importlib.import_module('vllm.model_executor.model_loader')
importlib.reload(model_loader)

logger.info("__ns extension: use ns model loader for ns model, %s", NSModelLoaderV2.__name__)

# register ns model
# from vllm.model_executor.models import ModelRegistry

# replace LlamaModel in models/llama.py with our NSLLamaModel
#==========================================================================================
from vllm.extension.ns.model.ns_model import NSLLamaModel
# ModelRegistry.register_model("LlamaForCausalLM", NSLLamaModel)

vllm_llama = importlib.import_module('vllm.model_executor.models.llama')
vllm_llama.LlamaModel = NSLLamaModel

logger.info("__ns extension: replace LlamaModel with ns LLamaModel, %s", NSLLamaModel.__name__)

# use our CPUCacheEngine for ns
#==========================================================================================
cpu_worker = importlib.import_module('vllm.worker.cpu_worker')

from vllm.extension.ns.kv_cache.ns_cache import NSCPUCacheEngine
cpu_worker.CPUCacheEngine = NSCPUCacheEngine

def get_cache_block_size_bytes(self) -> int:
    """Return the size in bytes of a single KV cache block.
    """
    return NSCPUCacheEngine.get_cache_block_size(self.cache_config, self.model_config, self.parallel_config, self.scheduler_config)

cpu_worker.CPUWorker.get_cache_block_size_bytes = get_cache_block_size_bytes

logger.info("__ns extension: use ns cache engine for ns, %s", NSCPUCacheEngine.__name__)

# use our execute_model method to do some conversion and pass more parameters
#==========================================================================================
cpu_model_runner = importlib.import_module('vllm.worker.cpu_model_runner')

from vllm.extension.ns.model.ns_model import execute_model
cpu_model_runner.CPUModelRunner.execute_model = execute_model

logger.info("__ns extension: replace execute_model in cpu_model_runner, %s", execute_model.__name__)

# use extended BlockSpaceManagerV1 to allocate and free native slot and kv caches
#==========================================================================================
interfaces = importlib.import_module("vllm.core.interfaces")

def get_block_space_manager_class(version: str):
    version = version.lower()

    if version == "v1":
        from vllm.extension.ns.kv_cache.ns_cache import NSBlockSpaceManagerV1
        return NSBlockSpaceManagerV1

    # TODO: v2 is to be supported
    # if version == "v2":
    #     from vllm.core.block_manager_v2 import BlockSpaceManagerV2
    #     return BlockSpaceManagerV2

    raise ValueError(f"Unknown version {version=}")

interfaces.BlockSpaceManager.get_block_space_manager_class = get_block_space_manager_class

logger.info("__ns extension: replace BlockSpaceManager.get_block_space_manager_class in vllm.core.interfaces with %s", get_block_space_manager_class.__name__)
