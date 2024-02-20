import subprocess
import pytest


def script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p):
    config_path = "../.github/workflows/config/" + model_name + ".yaml"

    cmd_serve = ["python", "../inference/serve.py", "--config_file", config_path]

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    # Ensure there are no errors in the serve script execution
    assert "Error" not in result_serve.stderr

    cmd_http = [
        "python",
        "../examples/inference/api_server_openai/query_http_requests.py",
        "--model_name",
        model_name,
    ]

    if streaming_response:
        cmd_http.append("--streaming_response")

    if max_new_tokens is not None:
        cmd_http.extend(["--max_new_tokens", str(max_new_tokens)])

    if temperature is not None:
        cmd_http.extend(["--temperature", str(temperature)])

    if top_p is not None:
        cmd_http.extend(["--top_p", str(top_p)])

    result_http = subprocess.run(cmd_http, capture_output=True, text=True)
    # Ensure there are no errors in the http query script execution
    assert "Error" not in result_http.stderr

    assert result_http.returncode == 0

    assert isinstance(result_http.stdout, str)

    assert len(result_http.stdout) > 0


model_names = ["gpt2"]
streaming_responses = [False, True]
max_new_tokens_values = [None, 128, 8192]
temperature_values = [None, 0.8]
top_p_values = [None, 0.7]


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
