import pytest
from transformers import AutoTokenizer
import pprint

from benchmark_serving import sample_requests_IPEX


def download_dataset(dataset: str):
    import subprocess

    subprocess.run(["wget", dataset])


@pytest.mark.parametrize(
    "input_tokens, max_new_tokens, num_requests",
    [
        ("64", 128, 10),
        ("32", 128, 20),
        ("32", None, 30),
    ],
)
def test_sample_requests_IPEX(input_tokens, max_new_tokens, num_requests):
    import os.path

    if not os.path.isfile("prompt.json"):
        download_dataset(
            "https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json"
        )

    sampled_requests = sample_requests_IPEX(
        dataset_path="prompt.json",
        input_tokens=input_tokens,
        max_new_tokens=max_new_tokens,
        num_requests=num_requests,
        tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b"),
    )

    pprint.pprint(sampled_requests)

    _, prompt_len, _ = sampled_requests[0]
    assert len(sampled_requests) == num_requests
    assert prompt_len == int(input_tokens)
