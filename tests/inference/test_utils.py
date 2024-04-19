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

import pytest
import torch

from llm_on_ray.inference.utils import (
    get_deployment_actor_options,
    StoppingCriteriaSub,
    max_input_len,
    decide_torch_dtype,
    is_cpu_without_ipex,
)
from llm_on_ray.inference.inference_config import InferenceConfig, DEVICE_CPU, DEVICE_HPU


# Mock the InferenceConfig for testing
@pytest.fixture
def mock_infer_conf():
    infer_conf = InferenceConfig()
    infer_conf.ipex.enabled = False
    infer_conf.deepspeed = False
    infer_conf.device = DEVICE_CPU
    infer_conf.cpus_per_worker = 1
    infer_conf.gpus_per_worker = 0
    infer_conf.hpus_per_worker = 0
    return infer_conf


def test_get_deployment_actor_options_cpu(mock_infer_conf):
    actor_options = get_deployment_actor_options(mock_infer_conf)
    assert actor_options["num_cpus"] == mock_infer_conf.cpus_per_worker
    assert "num_gpus" not in actor_options
    assert "resources" not in actor_options


# Add more tests for different configurations of InferenceConfig


def test_stopping_criteria_sub():
    stopping_criteria = StoppingCriteriaSub(stops=[torch.tensor([1, 2, 3])])
    input_ids = torch.tensor([[1, 2, 3]])
    scores = torch.tensor([0.0])
    assert stopping_criteria(input_ids, scores) is True

    input_ids = torch.tensor([[1, 2, 4]])
    assert stopping_criteria(input_ids, scores) is False


# Add more tests for different stopping criteria


def test_max_input_len():
    assert max_input_len(100) == 128
    assert max_input_len(200) == 512
    assert max_input_len(1000) == 2048
    assert max_input_len(5000) == 4096


# Add more tests for edge cases


def test_is_cpu_without_ipex(mock_infer_conf):
    assert is_cpu_without_ipex(mock_infer_conf) is True

    mock_infer_conf.ipex.enabled = True
    assert is_cpu_without_ipex(mock_infer_conf) is False


def test_decide_torch_dtype_cpu_without_ipex(mock_infer_conf):
    decide_torch_dtype(mock_infer_conf)

    assert mock_infer_conf.model_description.config.torch_dtype == torch.get_default_dtype()


def test_decide_torch_dtype_respect_user_config(mock_infer_conf):
    mock_infer_conf.model_description.config.torch_dtype = torch.float16
    mock_infer_conf.device = DEVICE_CPU

    decide_torch_dtype(mock_infer_conf)

    assert mock_infer_conf.model_description.config.torch_dtype == torch.float16


def test_decide_torch_dtype_hpu_with_deepspeed(mock_infer_conf):
    mock_infer_conf.device = DEVICE_HPU
    mock_infer_conf.deepspeed = True

    decide_torch_dtype(mock_infer_conf)

    assert mock_infer_conf.model_description.config.torch_dtype == torch.bfloat16


def test_decide_torch_dtype_with_hf_config_torch_dtype(mock_infer_conf):
    mock_infer_conf.device = DEVICE_CPU
    mock_infer_conf.ipex.enabled = True
    hf_config = {"torch_dtype": torch.float16}

    decide_torch_dtype(mock_infer_conf, hf_config)

    assert mock_infer_conf.model_description.config.torch_dtype == torch.float16


def test_decide_torch_dtype_with_hf_config_torch_dtype_as_attribute(mock_infer_conf):
    class HFConfig:
        def __init__(self):
            self.torch_dtype = torch.float16

    mock_infer_conf.device = DEVICE_CPU
    mock_infer_conf.ipex.enabled = True
    hf_config = HFConfig()

    decide_torch_dtype(mock_infer_conf, hf_config)

    assert mock_infer_conf.model_description.config.torch_dtype == torch.float16


if __name__ == "__main__":
    pytest.main(["-v", __file__])
