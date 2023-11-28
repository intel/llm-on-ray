import os
from pydantic import BaseModel, validator, ConfigDict
from pydantic_yaml import parse_yaml_raw_as
from typing import List, Dict

class Prompt(BaseModel):
    intro: str = ""
    human_id: str = ""
    bot_id: str = ""
    stop_words: List[str] = []

class ModelConfig(BaseModel):
    trust_remote_code: bool = False
    use_auth_token: str = None

class ModelDescription(BaseModel):
    model_id_or_path: str = None
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
    port: int = 8000
    name: str = None
    route_prefix: str = None
    precision: str = 'bf16'
    cpus_per_worker: int = 24
    gpus_per_worker: int = 0
    deepspeed: bool = False
    workers_per_group: int = 2
    ipex: bool = False
    device: str = "cpu"
    model_description: ModelDescription = ModelDescription()

    # prevent warning of protected namespaces
    # DO NOT TOUCH
    model_config = ConfigDict(protected_namespaces=())

    @validator('port')
    def _check_port(cls, v: int):
        assert v > 0 & v < 65535
        return v

    @validator('device')
    def _check_device(cls, v: str):
        if v:
            assert v in ['cpu', 'xpu', 'cuda', 'hpu']
        return v

    @validator('precision')
    def _check_precision(cls, v: str):
        if v:
            assert v in ['bf16', 'fp32']
        return v

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
    with open(_models_folder + "/" + model_file, "r") as f:
        m: InferenceConfig = parse_yaml_raw_as(InferenceConfig, f)
        _models[m.name] = m

env_model = "MODEL_TO_SERVE"
if env_model in os.environ:
    all_models[os.environ[env_model]] = _models[os.environ[env_model]]
else:
    # all_models["gpt-j-6B-finetuned-52K"] = gpt_j_finetuned_52K
    all_models = _models.copy()

_gpt2_key = "gpt2"
_gpt_j_6b = "gpt-j-6b"
base_models[_gpt2_key] = _models[_gpt2_key]
base_models[_gpt_j_6b] = _models[_gpt_j_6b]
