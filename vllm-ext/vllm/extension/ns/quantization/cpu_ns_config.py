#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List, Dict, Any, Optional
import os

from pathlib import Path

import torch
from torch.nn.modules import Module

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.model_loader.weight_utils import get_lock
from vllm.logger import init_logger

import huggingface_hub
from huggingface_hub import snapshot_download

from tqdm.auto import tqdm

import shutil

logger = init_logger(__name__)


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


# copy quant_ns_config to model snapshot folder if any
def copy_quant_ns_config(model_config: ModelConfig, load_config: LoadConfig):
    # only handle ns config here
    if model_config.quantization != "ns":
        return
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = NSQuantConfig.get_config_filenames()

    # If the quantization config is not found, skip
    if not possible_config_filenames:
        logger.warn(
            "ns quantization config file name not specified for model %s. Will use default ns config",
            model_name_or_path,
        )
        return
    quant_config_path = os.path.join(hf_folder, possible_config_filenames[0])
    if os.path.exists(quant_config_path):
        logger.info("Found ns quantization config file %s", quant_config_path)
        return
    downloaded_model_name = Path(
        hf_folder
    ).parent.parent.name  # e.g., hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/
    src_file = os.path.join(
        Path(__file__).parent.absolute(), downloaded_model_name, possible_config_filenames[0]
    )
    if not os.path.exists(src_file):
        logger.warn(
            "no ns quantization config file found in %s. Will use default ns config", src_file
        )
        return
    shutil.copy(src_file, quant_config_path)
    logger.info("Copyed ns quantization config file to %s", quant_config_path)


class NSQuantConfig(QuantizationConfig):
    def __init__(
        self,
        quant_method="ns",
        weight_dtype="int8",
        group_size="128",
        scale_dtype="fp32",
        compute_dtype="int8",
        alg="sym",
        version="v1",
    ):
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

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        pass

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass
