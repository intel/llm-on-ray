import subprocess
import pytest


def script_with_args(api_base, model_name, streaming_response, max_new_tokens, temperature, top_p):
    config_path = "../inference/models/" + model_name + ".yaml"

    cmd_serve = ["python", "../inference/serve.py", "--config_file", config_path]

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    assert "Error" not in result_serve.stderr

    cmd_openai = [
        "python",
        "../examples/inference/api_server_openai/query_openai_sdk.py",
        "--model_name",
        model_name,
    ]

    if streaming_response:
        cmd_openai.append("--streaming_response")

    if max_new_tokens is not None:
        cmd_openai.extend(["--max_new_tokens", str(max_new_tokens)])

    if temperature is not None:
        cmd_openai.extend(["--temperature", str(temperature)])

    if top_p is not None:
        cmd_openai.extend(["--top_p", str(top_p)])

    result_openai = subprocess.run(cmd_openai, capture_output=True, text=True)

    assert "Error" not in result_openai.stderr

    assert isinstance(result_openai.stdout, str)

    assert len(result_openai.stdout) > 0


request_api_bases = ["http://localhost:8000/v1"]
model_names = ["gpt2"]
streaming_responses = [False, True]
max_new_tokens_values = [None, 128]
temperature_values = [None, 0.8]
top_p_values = [None, 0.7]


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
