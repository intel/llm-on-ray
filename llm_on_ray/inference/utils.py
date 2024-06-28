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
import pathlib
from transformers import StoppingCriteria, TextStreamer, AutoConfig
from ray.util.queue import Queue
import torch
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from llm_on_ray.inference.inference_config import (
    InferenceConfig,
    DEVICE_CPU,
    DEVICE_HPU,
    PRECISION_BF16,
    PRECISION_FP32,
)
from llm_on_ray.inference.api_openai_backend.openai_protocol import ChatMessage


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
    if infer_conf.vllm.enabled:
        runtime_env[_ray_env_key]["OMP_PROC_BIND"] = "true"
    # now PredictorDeployment itself is a worker, we should require resources for it
    ray_actor_options: Dict[str, Any] = {"runtime_env": runtime_env}
    if infer_conf.device == "cpu":
        ray_actor_options["num_cpus"] = infer_conf.cpus_per_worker
    elif infer_conf.device == "cuda":
        ray_actor_options["num_gpus"] = infer_conf.gpus_per_worker
    elif infer_conf.device == "hpu" and not infer_conf.deepspeed:
        # when using deepspeed on hpu, deployment actor does not need HPU
        ray_actor_options["resources"] = {"HPU": infer_conf.hpus_per_worker}
    else:
        # TODO add xpu
        pass
    return ray_actor_options


def get_max_seq_length(config: AutoConfig):
    config = config.to_dict()
    # chatglm2, bloom, chatglm3
    if "seq_length" in config:
        return config["seq_length"]
    # qwen2, llama-2, llama, dolly, gptneox, qwen, qwen1.5, opt, phi
    if "max_position_embeddings" in config:
        return config["max_position_embeddings"]
    # baichuan, baichuan2
    if "model_max_length" in config:
        return config["model_max_length"]
    # gptj
    if "n_positions" in config:
        return config["n_positions"]
    # mpt
    if "max_seq_len" in config:
        return config["max_seq_len"]
    # chatglm
    if "max_sequence_length" in config:
        return config["max_sequence_length"]
    # whisper
    if "max_length" in config:
        return config["max_length"]

    print("Not found max seq length, setting to default 512")
    return 512


class RayTextIteratorStreamer(TextStreamer):
    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
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


def decide_torch_dtype(infer_conf: InferenceConfig, hf_config=None):
    """
    Decide torch dtype based on user config and model config.
    This function modifies `torch_dtype` in infer_conf.model_description.config.
    """
    # First, create torch_dtype if it does not exist
    if not hasattr(infer_conf.model_description.config, "torch_dtype"):
        infer_conf.model_description.config.torch_dtype = None

    if infer_conf.model_description.config.torch_dtype:
        # respect user config
        if infer_conf.model_description.config.torch_dtype == PRECISION_BF16:
            infer_conf.model_description.config.torch_dtype = torch.bfloat16
        elif infer_conf.model_description.config.torch_dtype == PRECISION_FP32:
            infer_conf.model_description.config.torch_dtype = torch.float32
        return
    elif hf_config is None:
        # default to float32 if hf_config is not supplied
        infer_conf.model_description.config.torch_dtype = torch.get_default_dtype()
    # if hf_config contains recommended torch_dtype, use it
    elif hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype:
        infer_conf.model_description.config.torch_dtype = hf_config.torch_dtype
    elif hasattr(hf_config, "__getitem__") and "torch_dtype" in hf_config:
        infer_conf.model_description.config.torch_dtype = hf_config["torch_dtype"]

    # if using hpu
    if infer_conf.device == DEVICE_HPU:
        # if using deepspeed, we should use bfloat16
        # TODO if quantization is enabled, we should use bfloat16
        if infer_conf.deepspeed:
            infer_conf.model_description.config.torch_dtype = torch.bfloat16
    elif is_cpu_without_ipex(infer_conf):
        # cpu without ipex use default float32
        infer_conf.model_description.config.torch_dtype = torch.get_default_dtype()


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


def parse_jinja_file(chat_template: Union[str, None]):
    if chat_template is None:
        return None

    try:
        # Get the absolute path of the provided chat template
        jinja_path = os.path.abspath(chat_template)

        # If the user specifies a jinja file, the absolute path to jinja_path exists.
        # If jinja_path does not exist, it means that the user did not specify jinja and the default jinja is used.
        if not os.path.exists(jinja_path):
            jinja_path = str(
                pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
                / chat_template
            )

        with open(jinja_path, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"File {jinja_path} not found.")
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")
