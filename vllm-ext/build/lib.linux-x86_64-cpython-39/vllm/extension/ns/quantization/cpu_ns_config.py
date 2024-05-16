from typing import List, Dict, Any, Optional
import torch
from torch.nn.modules import Module
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase)
from vllm.model_executor.layers.linear import LinearMethodBase


class NSQuantConfig(QuantizationConfig):
    def __init__(self, quant_method="ns", weight_dtype="int4", group_size="32", scale_dtype="fp32", compute_dtype="int8",
                 alg="sym", version="v1"):
        self.quant_method = quant_method
        self.weight_dtype = weight_dtype
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.compute_dtype = compute_dtype
        self.alg = alg
        self.version = version

    def get_name(self) -> str:
        return "ns"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.int]
    
    # useless
    def get_min_capability(self) -> int:
        pass

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quant_ns_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NSQuantConfig":
        return cls(**config["quantization_config"])

    def get_linear_method(self) -> LinearMethodBase:
        return NSLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []
    
    def get_quant_method(self, layer: Module) -> Optional[QuantizeMethodBase]:
        return None
    
    
class NSLinearMethod(LinearMethodBase):
    def __init__(self, quant_config) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        pass

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass