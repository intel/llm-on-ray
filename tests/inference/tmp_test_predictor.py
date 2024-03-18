import pytest

from inference_config import InferenceConfig
from pydantic_yaml import parse_yaml_raw_as

from predictor import Predictor

CONFIG_FILE = "../inference/models/gpt2.yaml"
model_name = "gpt2"


@pytest.fixture
def infer_conf():
    with open(CONFIG_FILE, "r") as f:
        infer_conf = parse_yaml_raw_as(InferenceConfig, f)
    return infer_conf


@pytest.fixture
def prompt():
    return "The future of AI is"


@pytest.fixture
def text():
    return "The future of AI is"


@pytest.fixture
def config():
    return {"temperature": 0.8, "top_p": 0.95}


# @pytest.mark.skip(reason="tested")
def test_generate(infer_conf, prompt, config):
    predictor = Predictor(infer_conf)
    output = predictor.generate(prompt, **config)
    print(f"\n" f"Prompt: {prompt}\n" f"Output: {output}" f"\n")


def test_streaming_generate(infer_conf, prompt, config):
    predictor = Predictor(infer_conf)
    streamer = predictor.get_streamer()
    predictor.streaming_generate(prompt, streamer, **config)


def test_tokenize_inputs(infer_conf, text):
    predictor = Predictor(infer_conf)
    tokens = predictor.tokenize_inputs(text)
    print(f"\n" f"Text: {text}\n" f"Tokens: {tokens}" f"\n")


def test_configure_tokenizer(infer_conf, model_name):
    predictor = Predictor(infer_conf)
    predictor.configure_tokenizer(model_name)
