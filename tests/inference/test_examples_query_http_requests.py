import subprocess
import pytest
import os


def script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    config_path = "../.github/workflows/config/" + model_name + "-ci.yaml"

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

    assert isinstance(result_http.stdout, str)

    assert len(result_http.stdout) > 0


@pytest.mark.parametrize(
    "model_name,streaming_response,max_new_tokens,temperature,top_p",
    [
        (model_name, streaming_response, max_new_tokens, temperature, top_p)
        for model_name in ["gpt2"]
        for streaming_response in [False, True]
        for max_new_tokens in [None, 128]
        for temperature in [None, 0.8]
        for top_p in [None, 0.7]
    ],
)
def test_script(model_name, streaming_response, max_new_tokens, temperature, top_p):
    script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p)
