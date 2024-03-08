import pytest
import torch

from inference.utils import (
    get_deployment_actor_options,
    StoppingCriteriaSub,
    max_input_len,
    get_torch_dtype,
    is_cpu_without_ipex,
)
from inference_config import InferenceConfig, DEVICE_CPU


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


def test_get_torch_dtype_cpu_without_ipex(mock_infer_conf):
    hf_config = None
    dtype = get_torch_dtype(mock_infer_conf, hf_config)
    assert dtype == torch.get_default_dtype()


# Add more tests for different configurations and hf_config values


def test_is_cpu_without_ipex(mock_infer_conf):
    assert is_cpu_without_ipex(mock_infer_conf) is True

    mock_infer_conf.ipex.enabled = True
    assert is_cpu_without_ipex(mock_infer_conf) is False


# Add more tests for different configurations
