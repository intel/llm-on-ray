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

from transformers import StoppingCriteria
import torch
from inference.inference_config import InferenceConfig, DEVICE_CPU
from inference.api_openai_backend.openai_protocol import ChatMessage
from typing import Dict, Any, List, Union
from enum import Enum


def get_deployment_actor_options(infer_conf: InferenceConfig):
    _ray_env_key = "env_vars"
    # OMP_NUM_THREADS will be set by num_cpus, so not set in env
    _predictor_runtime_env_ipex = {
        "KMP_BLOCKTIME": "1",
        "KMP_SETTINGS": "1",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",
        "MALLOC_CONF": "oversize_threshold:1,background_thread:true,\
            metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000",
    }
    runtime_env: Dict[str, Any] = {_ray_env_key: {}}
    if infer_conf.ipex.enabled:
        runtime_env[_ray_env_key].update(_predictor_runtime_env_ipex)
    if infer_conf.deepspeed:
        runtime_env[_ray_env_key]["DS_ACCELERATOR"] = infer_conf.device
    # now PredictorDeployment itself is a worker, we should require resources for it
    ray_actor_options: Dict[str, Any] = {"runtime_env": runtime_env}
    if infer_conf.device == "cpu":
        ray_actor_options["num_cpus"] = infer_conf.cpus_per_worker
    elif infer_conf.device == "cuda":
        ray_actor_options["num_gpus"] = infer_conf.gpus_per_worker
    elif infer_conf.device == "hpu":
        ray_actor_options["resources"] = {"HPU": infer_conf.hpus_per_worker}
    else:
        # TODO add xpu
        pass
    return ray_actor_options


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            length = 1 if len(stop.size()) == 0 else stop.size()[0]
            if torch.all((stop == input_ids[0][-length:])).item():
                return True
        return False


class PromptFormat(Enum):
    CHAT_FORMAT = 1
    PROMPTS_FORMAT = 2
    INVALID_FORMAT = 3


# used in inference with Gaudi
def max_input_len(input_text_length):
    if input_text_length <= 128:
        return 128
    elif input_text_length <= 512:
        return 512
    elif input_text_length <= 2048:
        return 2048
    else:
        print("Max support length is 4096")
        return 4096


def get_torch_dtype(infer_conf: InferenceConfig, hf_config) -> torch.dtype:
    """
    return torch default dtype, a.k.a float32, if it's cpu only inference without
    ipex because bfloat16 is too slow and float16 is not supported in CPU
    """
    if hf_config is None or is_cpu_without_ipex(infer_conf):
        return torch.get_default_dtype()
    if hasattr(hf_config, "torch_dtype"):
        t = hf_config.torch_dtype
        if t:
            return t
    if hasattr(hf_config, "__getitem__"):
        t = hf_config["torch_dtype"]
        if t:
            return t
    return torch.get_default_dtype()


def is_cpu_without_ipex(infer_conf: InferenceConfig) -> bool:
    return (not infer_conf.ipex.enabled) and infer_conf.device == DEVICE_CPU


def get_prompt_format(input: Union[List[str], List[dict], List[ChatMessage]]):
    chat_format = True
    prompts_format = True
    for item in input:
        if isinstance(item, str) or isinstance(item, list):
            chat_format = False
        elif isinstance(item, dict) or isinstance(item, ChatMessage):
            prompts_format = False
        else:
            chat_format = False
            prompts_format = False
            break
    if chat_format:
        return PromptFormat.CHAT_FORMAT
    if prompts_format:
        return PromptFormat.PROMPTS_FORMAT
    return PromptFormat.INVALID_FORMAT


def module_import(module_name, clazz, **clazzs_kwargs):
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, clazz)


def module_import_and_init(module_name, clazz, **clazzs_kwargs):
    import importlib

    module = importlib.import_module(module_name)
    class_ = getattr(module, clazz)
    return class_(**clazzs_kwargs)
