import pytest

from inference_config import InferenceConfig
from pydantic_yaml import parse_yaml_raw_as

from vllm_predictor import VllmPredictor

# gpt2 is not supported due to "gelu_new is unsupported on CPU"
CONFIG_FILE="../inference/models/llama-2-7b-chat-hf.yaml"

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

@pytest.mark.skip(reason="tested")
def test_generate(infer_conf, prompt, config):
    vllm_predictor = VllmPredictor(infer_conf)
    output = vllm_predictor.generate(prompt, **config)
    print(f"\n"
          f"Prompt: {prompt}\n"
          f"Output: {output}"
          f"\n")

def test_streaming_generate(infer_conf, prompt, config):
    vllm_predictor = VllmPredictor(infer_conf)
    streamer = vllm_predictor.get_streamer()
    vllm_predictor.streaming_generate(prompt, streamer, **config)

