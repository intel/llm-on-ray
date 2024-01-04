import pytest

from inference_config import InferenceConfig
from pydantic_yaml import parse_yaml_raw_as

from vllm_predictor import VllmPredictor

# CONFIG_FILE="../inference/models/gpt-j-6b.yaml"
CONFIG_FILE="../inference/models/gpt2.yaml"

@pytest.fixture
def infer_conf():
    with open(CONFIG_FILE, "r") as f:
        infer_conf = parse_yaml_raw_as(InferenceConfig, f)
    return infer_conf

@pytest.fixture
def prompt():
    return "The future of AI is"

@pytest.fixture
def config():
    return {
        "temperature": 0.8,
        "top_p": 0.95
    }

def test_generate(infer_conf, prompt, config):

    vllm_predictor = VllmPredictor(infer_conf)
    vllm_predictor.generate(prompt, **config)

def test_streaming_generate(infer_conf, prompt, config):
    ...
