import os
from pydantic import BaseModel, validator, ConfigDict
from pydantic_yaml import parse_yaml_raw_as
from typing import List, Dict

IPEX_PRECISION_BF16 = 'bf16'
IPEX_PRECISION_FP32 = 'fp32'

DEVICE_CPU = "cpu"
DEVICE_HPU = "hpu"
DEVICE_XPU = "xpu"
DEVICE_CUDA = "cuda"

class Prompt(BaseModel):
    intro: str = ""
    human_id: str = ""
    bot_id: str = ""
    stop_words: List[str] = []

class ModelConfig(BaseModel):
    trust_remote_code: bool = False
    use_auth_token: str = None
    load_in_4bit: bool = False

class Ipex(BaseModel):
    enabled: bool = True
    precision: str = 'bf16'

    @validator('precision')
    def _check_precision(cls, v: str):
        if v:
            assert v in [IPEX_PRECISION_BF16, IPEX_PRECISION_FP32]
        return v

# for bigdl model
class BigDLModelConfig(BaseModel):
    load_in_low_bit: str = ""

    @validator("load_in_low_bit")
    def _check_load_in_low_bit(cls, v: str):
        if v:
            assert v in ["sym_int4", "asym_int4", "sym_int5", "asym_int5", "sym_int8"]
        return v

class ModelDescription(BaseModel):
    model_id_or_path: str = None
    bigdl: bool = False
    tokenizer_name_or_path: str = None
    chat_processor: str = None
    gpt_base_model: bool = False
    quantized_model_id_or_path: str = None
    quantization_type: str = None
    peft_model_id_or_path: str = None
    peft_type: str = None
    # only effective when device is hpu
    use_hpu_graphs: bool = True
    prompt: Prompt = Prompt()
    config: ModelConfig = ModelConfig()
    bigdl_config: BigDLModelConfig = BigDLModelConfig()

    # prevent warning of protected namespaces
    # DO NOT TOUCH
    model_config = ConfigDict(protected_namespaces=())
    
    @validator('quantization_type')
    def _check_quant_type(cls, v: str):
        if v:
            assert v in ["ipex_smoothquant", 'ipex_weightonly', 'llamacpp']
        return v

    @validator('peft_type')
    def _check_perftype(cls, v: str):
        if v:
            assert v in ["lora", "adalora", "deltatuner"]
        return v

class InferenceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    name: str = None
    route_prefix: str = None
    cpus_per_worker: int = 24
    gpus_per_worker: int = 0
    hpus_per_worker: int = 0
    deepspeed: bool = False
    workers_per_group: int = 2
    device: str = DEVICE_CPU
    ipex: Ipex = Ipex()
    model_description: ModelDescription = ModelDescription()

    # prevent warning of protected namespaces
    # DO NOT TOUCH
    model_config = ConfigDict(protected_namespaces=())

    @validator('host')
    def _check_host(cls, v: str):
        if v:
            assert v in ["0.0.0.0", "127.0.0.1"]
        return v

    @validator('port')
    def _check_port(cls, v: int):
        assert v > 0 & v < 65535
        return v

    @validator('device')
    def _check_device(cls, v: str):
        if v:
            assert v.lower() in [DEVICE_CPU, DEVICE_XPU, DEVICE_CUDA, DEVICE_HPU]
        return v.lower()

    @validator('workers_per_group')
    def _check_workers_per_group(cls, v: int):
        if v:
            assert v > 0
        return v

all_models : Dict[str, InferenceConfig] = {}
base_models : Dict[str, InferenceConfig] = {}
_models : Dict[str, InferenceConfig] = {}

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

_gpt2_key = "gpt2"
_gpt_j_6b = "gpt-j-6b"
base_models[_gpt2_key] = _models[_gpt2_key]
base_models[_gpt_j_6b] = _models[_gpt_j_6b]
