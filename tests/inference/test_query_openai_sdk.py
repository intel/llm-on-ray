import subprocess
import pytest


# @pytest.fixture
def script_with_args(api_base, model_name, streaming_response, max_new_tokens, temperature, top_p):
    config_path = "../inference/models/" + model_name + ".yaml"
    print(config_path)
    cmd = ["python", "../inference/serve.py", "--config_file", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result)
    cmd1 = [
        "python",
        "../examples/inference/api_server_openai/query_openai_sdk.py",  # 请替换为您的脚本文件名
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


request_api_bases = ["http://localhost:8000/v1"]
model_names = ["gpt2", "llama-2-7b-chat-hf", "neural-chat-7b-v3-1"]
streaming_responses = [False, True]
max_new_tokens_values = [None, 128, 1024]
temperature_values = [None, 0.8, 0.6]
top_p_values = [None, 0.7, 0.9]


@pytest.mark.parametrize(
    "api_base,model_name,streaming_response,max_new_tokens,temperature,top_p",
    [
        (api_base, model_name, streaming_response, max_new_tokens, temperature, top_p)
        for api_base in request_api_bases
        for model_name in model_names
        for streaming_response in streaming_responses
        for max_new_tokens in max_new_tokens_values
        for temperature in temperature_values
        for top_p in top_p_values
    ],
)
def test_script(api_base, model_name, streaming_response, max_new_tokens, temperature, top_p):
    script_with_args(api_base, model_name, streaming_response, max_new_tokens, temperature, top_p)
