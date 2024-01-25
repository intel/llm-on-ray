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


@pytest.mark.parametrize(
    "model_name,streaming_response,max_new_tokens,temperature,top_p",
    [
        ("gpt2", False, None, None, None),
        ("neural-chat-7b-v3-1", False, 100, None, None),
    ],
)
def test_script(model_name, streaming_response, max_new_tokens, temperature, top_p):
    script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p)
