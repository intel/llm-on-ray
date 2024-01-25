import subprocess
import pytest
import os


# @pytest.fixture
def script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p):
    current_working_directory = os.getcwd()
    print(current_working_directory)

    config_path = "../inference/models/" + model_name + ".yaml"
    print(config_path)
    cmd = ["python", "../inference/serve.py", "--config_file", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result)

    cmd1 = [
        "python",
        "../examples/inference/api_server_openai/query_http_requests.py",
        "--model_name",
        model_name,
    ]

    if streaming_response:
        cmd1.append("--streaming_response")

    if max_new_tokens is not None:
        cmd1.extend(["--max_new_tokens", str(max_new_tokens)])

    if temperature is not None:
        cmd1.extend(["--temperature", str(temperature)])

    if top_p is not None:
        cmd1.extend(["--top_p", str(top_p)])

    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    print(result1)


model_names = ["gpt2", "llama-2-7b-chat-hf", "neural-chat-7b-v3-1"]
streaming_responses = [False, True]
max_new_tokens_values = [None, 128, 256, 1024, 2048]
temperature_values = [None, 0.8, 0.6, 0.4, 0.2]
top_p_values = [None, 0.1, 0.3, 0.5, 0.7, 0.9]


@pytest.mark.parametrize(
    "model_name,streaming_response,max_new_tokens,temperature,top_p",
    [
        (model_name, streaming_response, max_new_tokens, temperature, top_p)
        for model_name in model_names
        for streaming_response in streaming_responses
        for max_new_tokens in max_new_tokens_values
        for temperature in temperature_values
        for top_p in top_p_values
    ],
)
def test_script(model_name, streaming_response, max_new_tokens, temperature, top_p):
    script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p)
