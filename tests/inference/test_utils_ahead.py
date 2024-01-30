import pytest
import torch
from utils import (
    get_deployment_actor_options,
    StoppingCriteriaSub,
    max_input_len,
    get_torch_dtype,
    is_cpu_without_ipex,
)
from inference_config import InferenceConfig, DEVICE_CPU


# Mock the InferenceConfig for testing
# @pytest.fixture
class MockInferenceConfig(InferenceConfig):
    def __init__(
        self,
        ipex_enabled: bool,
        ipex_precision: str,
        vllm_enabled: bool,
        vllm_precision: str,
        **kwargs,
    ):
        # Call the constructor of the base class (InferenceConfig)
        super().__init__(**kwargs)
        self.ipex.enabled = ipex_enabled
        self.ipex.precision = ipex_precision
        self.vllm.enabled = vllm_enabled
        self.vllm.precision = vllm_precision


@pytest.fixture
def mock_infer_conf(
    ipex_enabled=False,
    deepspeed=False,
    cpus_per_worker=1,
    gpus_per_worker=0,
    hpus_per_worker=0,
    device_type=None,
):
    infer_conf = InferenceConfig()
    infer_conf.ipex.enabled = ipex_enabled
    infer_conf.deepspeed = deepspeed
    infer_conf.device = DEVICE_CPU
    infer_conf.cpus_per_worker = cpus_per_worker
    infer_conf.gpus_per_worker = gpus_per_worker
    infer_conf.hpus_per_worker = hpus_per_worker
    return infer_conf


def test_get_deployment_actor_options_cpu(mock_infer_conf):
    actor_options = get_deployment_actor_options(mock_infer_conf)
    assert "runtime_env" in actor_options

    if mock_infer_conf.device == "cpu":
        assert "num_cpus" in actor_options
        assert actor_options["num_cpus"] == mock_infer_conf.cpus_per_worker
    elif mock_infer_conf.device == "cuda":
        assert "num_gpus" in actor_options
        assert actor_options["num_gpus"] == mock_infer_conf.gpus_per_worker
    elif mock_infer_conf.device == "hpu":
        assert "resources" in actor_options
        assert "HPU" in actor_options["resources"]
        assert actor_options["resources"]["HPU"] == mock_infer_conf.hpus_per_worker

    if mock_infer_conf.ipex.enabled:
        assert "_predictor_runtime_env_ipex" in actor_options["runtime_env"]
        ipex_env_vars = actor_options["runtime_env"]["_predictor_runtime_env_ipex"]
        assert "KMP_BLOCKTIME" in ipex_env_vars
        assert "KMP_SETTINGS" in ipex_env_vars
        assert "KMP_AFFINITY" in ipex_env_vars
        assert "MALLOC_CONF" in ipex_env_vars

    # if mock_infer_conf.deepspeed:
    #     assert "DS_ACCELERATOR" in actor_options["runtime_env"][_ray_env_key]


# Add more tests for different configurations of InferenceConfig


def test_stopping_criteria_sub():
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    stopping_criteria = StoppingCriteriaSub(stops=[torch.tensor([1, 2, 3])], encounters=1)
    assert stopping_criteria(input_ids, scores) is True

    stopping_criteria.stops = [torch.tensor([7, 8, 9])]
    assert stopping_criteria(input_ids, scores) is False

    stopping_criteria.stops = []
    assert stopping_criteria(input_ids, scores) is False

    empty_input_ids = torch.tensor([])
    assert stopping_criteria(empty_input_ids, scores) is False

    stopping_criteria.stops = [torch.tensor([6])]
    assert stopping_criteria(input_ids, scores) is False


# Add more tests for different stopping criteria


def test_max_input_len():
    assert max_input_len(100) == 128
    assert max_input_len(200) == 512
    assert max_input_len(1000) == 2048
    assert max_input_len(5000) == 4096


# Add more tests for edge cases


def test_get_torch_dtype(mock_infer_conf):
    # mock_config.vllm.enabled = True
    # Case 2: When hf_config is None and it's CPU only inference with vllm
    mock_config = MockInferenceConfig(vllm_enabled=True)
    dtype = get_torch_dtype(mock_config, hf_config=None)
    assert dtype == torch.get_default_dtype()

    # Case 1: When hf_config is None and it's CPU only inference without ipex
    dtype = get_torch_dtype(mock_infer_conf, hf_config=None)
    assert dtype == torch.get_default_dtype()

    # mock_config.ipex.enabled = True
    # Case 2: When hf_config is None and it's CPU only inference with ipex
    mock_config = MockInferenceConfig(ipex_enabled=True)
    dtype = get_torch_dtype(mock_config, hf_config=None)
    assert dtype == torch.get_default_dtype()

    # Case 3: When hf_config has torch_dtype attribute without ipex
    mock_config = MockInferenceConfig(ipex_enabled=False)

    class HFConfigWithTorchDtype:
        torch_dtype = torch.float16

    hf_config = HFConfigWithTorchDtype()
    dtype = get_torch_dtype(mock_config, hf_config=hf_config)
    assert dtype == torch.float32

    # Case 4: When hf_config has torch_dtype attribute with ipex
    mock_config = MockInferenceConfig(ipex_enabled=True)

    class HFConfigWithTorchDtype:
        torch_dtype = torch.float16

    hf_config = HFConfigWithTorchDtype()
    dtype = get_torch_dtype(mock_config, hf_config=hf_config)
    assert dtype == torch.float16

    # Case 5: When hf_config has torch_dtype key in __getitem__ without ipex
    mock_config = MockInferenceConfig(ipex_enabled=False)

    class HFConfigWithGetItem:
        def __getitem__(self, key):
            if key == "torch_dtype":
                return torch.bfloat16

    hf_config = HFConfigWithGetItem()
    dtype = get_torch_dtype(mock_config, hf_config=hf_config)
    assert dtype == torch.float32

    # Case 6: When hf_config has torch_dtype key in __getitem__ with ipex
    mock_config = MockInferenceConfig(ipex_enabled=True)

    class HFConfigWithGetItem:
        def __getitem__(self, key):
            if key == "torch_dtype":
                return torch.bfloat16

    hf_config = HFConfigWithGetItem()
    dtype = get_torch_dtype(mock_config, hf_config=hf_config)
    assert dtype == torch.bfloat16

    # Case 7: When hf_config doesn't have torch_dtype attribute or key
    class HFConfigWithoutTorchDtype:
        pass

    hf_config = HFConfigWithoutTorchDtype()
    dtype = get_torch_dtype(mock_infer_conf, hf_config=hf_config)
    assert dtype == torch.get_default_dtype()

    hf_config = None
    dtype = get_torch_dtype(mock_infer_conf, hf_config)
    assert dtype == torch.get_default_dtype()


# Add more tests for different configurations and hf_config values


def test_is_cpu_without_ipex(mock_infer_conf):
    assert is_cpu_without_ipex(mock_infer_conf) is True

    mock_infer_conf.ipex.enabled = True
    assert is_cpu_without_ipex(mock_infer_conf) is False


# Add more tests for different configurations
