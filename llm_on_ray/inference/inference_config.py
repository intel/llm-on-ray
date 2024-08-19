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

import os
from pydantic import BaseModel, validator, ConfigDict
from pydantic_yaml import parse_yaml_raw_as
from typing import List, Dict, Union

PRECISION_BF16 = "bf16"
PRECISION_FP32 = "fp32"

DEVICE_CPU = "cpu"
DEVICE_HPU = "hpu"
DEVICE_GPU = "gpu"
DEVICE_CUDA = "cuda"


class Prompt(BaseModel):
    intro: str = ""
    human_id: str = ""
    bot_id: str = ""
    stop_words: List[str] = []


class ModelConfig(BaseModel):
    trust_remote_code: bool = False
    use_auth_token: Union[str, None] = None
    load_in_4bit: bool = False
    torch_dtype: Union[str, None] = None
    revision: Union[str, None] = None


class Ipex(BaseModel):
    enabled: bool = False
    precision: str = "bf16"

    @validator("precision")
    def _check_precision(cls, v: str):
        if v:
            assert v in [PRECISION_BF16, PRECISION_FP32]
        return v


class Vllm(BaseModel):
    enabled: bool = False
    max_num_seqs: int = 256
    precision: str = "bf16"
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    max_seq_len_to_capture: int = 8192
    response_role: str = "assistant"
    lora_modules: Union[str, None] = None

    @validator("precision")
    def _check_precision(cls, v: str):
        if v:
            assert v in [PRECISION_BF16, PRECISION_FP32]
        return v


# for IPEX-LLM model
class IpexllmModelConfig(BaseModel):
    load_in_low_bit: str = ""

    @validator("load_in_low_bit")
    def _check_load_in_low_bit(cls, v: str):
        if v:
            assert v in ["sym_int4", "asym_int4", "sym_int5", "asym_int5", "sym_int8"]
        return v


# for HPU model
class HpuModelConfig(BaseModel):
    # Enable HPU graph runtime
    use_hpu_graphs: bool = True
    # Enable Torch compile, only works for llama
    torch_compile: bool = False
    # Point to an HQT config json file
    quant_config: Union[str, None] = None


# for non-streaming response
class ModelGenerateResult(BaseModel):
    text: Union[str, List[str]] = ""
    input_length: Union[int, None] = None
    generate_length: Union[int, None] = None


class ModelDescription(BaseModel):
    model_id_or_path: Union[str, None] = None
    tokenizer_name_or_path: Union[str, None] = None
    config: ModelConfig = ModelConfig()
    prompt: Prompt = Prompt()
    chat_processor: Union[str, None] = None

    gpt_base_model: bool = False

    quantized_model_id_or_path: Union[str, None] = None
    quantization_type: Union[str, None] = None

    peft_model_id_or_path: Union[str, None] = None
    peft_type: Union[str, None] = None

    ipexllm: bool = False
    ipexllm_config: IpexllmModelConfig = IpexllmModelConfig()

    # prevent warning of protected namespaces
    # DO NOT TOUCH
    model_config = ConfigDict(protected_namespaces=())  # type: ignore

    # specify model_loader and input_processor
    input_processor: str = "AutoProcessor"
    model_loader: str = "AutoModel"

    chat_model_with_image: bool = False
    chat_template: Union[str, None] = None
    default_chat_template: str = "llm_on_ray/inference/models/templates/default_template.jinja"

    @validator("quantization_type")
    def _check_quant_type(cls, v: str):
        if v:
            assert v in ["ipex_smoothquant", "ipex_weightonly", "llamacpp"]
        return v

    @validator("peft_type")
    def _check_perftype(cls, v: str):
        if v:
            assert v in ["lora", "adalora"]
        return v


class AutoscalingConfig(BaseModel):
    min_replicas: int = 1
    initial_replicas: int = 1
    max_replicas: int = 1
    target_ongoing_requests: float = 1.0
    metrics_interval_s: float = 10.0
    look_back_period_s: float = 30.0
    smoothing_factor: float = 1.0
    upscaling_factor: Union[float, None] = None
    downscaling_factor: Union[float, None] = None
    downscale_delay_s: float = 600.0
    upscale_delay_s: float = 30.0


class InferenceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    name: str = "default"
    route_prefix: Union[str, None] = None
    dynamic_max_batch_size: int = 8
    num_replicas: Union[int, None] = None
    max_ongoing_requests: int = 100
    autoscaling_config: Union[AutoscalingConfig, None] = None
    cpus_per_worker: int = 24
    gpus_per_worker: int = 0
    hpus_per_worker: int = 0
    deepspeed: bool = False
    vllm: Vllm = Vllm()
    workers_per_group: int = 2
    device: str = DEVICE_CPU
    ipex: Ipex = Ipex()
    hpu_model_config: HpuModelConfig = HpuModelConfig()
    model_description: ModelDescription = ModelDescription()

    # prevent warning of protected namespaces
    # DO NOT TOUCH
    model_config = ConfigDict(protected_namespaces=())  # type: ignore

    @validator("host")
    def _check_host(cls, v: str):
        if v:
            assert v in ["0.0.0.0", "127.0.0.1"]
        return v

    @validator("port")
    def _check_port(cls, v: int):
        assert v > 0 & v < 65535
        return v

    @validator("device")
    def _check_device(cls, v: str):
        if v:
            assert v.lower() in [DEVICE_CPU, DEVICE_GPU, DEVICE_CUDA, DEVICE_HPU]
        return v.lower()

    @validator("workers_per_group")
    def _check_workers_per_group(cls, v: int):
        if v:
            assert v > 0
        return v


all_models: Dict[str, InferenceConfig] = {}
base_models: Dict[str, InferenceConfig] = {}
_models: Dict[str, InferenceConfig] = {}

_cur = os.path.dirname(os.path.abspath(__file__))
_models_folder = _cur + "/models"
for model_file in os.listdir(_models_folder):
    file_path = _models_folder + "/" + model_file
    if os.path.isdir(file_path):
        continue
    with open(file_path, "r") as f:
        m: InferenceConfig = parse_yaml_raw_as(InferenceConfig, f)
        _models[m.name] = m

all_models = _models.copy()
