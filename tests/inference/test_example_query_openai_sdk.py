import subprocess
import pytest
import os

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_KEY"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"


def script_with_args(api_base, model_name, streaming_response, max_new_tokens, temperature, top_p):
    # Other OpenAI SDK tests
    if api_base != "http://localhost:8000/v1":
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_BASE_URL"] = api_base

    current_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(
        current_path, "../../.github/workflows/config/" + model_name + "-ci.yaml"
    )

    os.path.join(current_path, "../../inference/serve.py")

    cmd_serve = ["llm_on_ray-serve", "--config_file", config_path]

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    # Print the output of subprocess.run for checking if output is expected
    print(result_serve)

    # Ensure there are no errors in the serve script execution
    assert "Error" not in result_serve.stderr

    example_openai_path = os.path.join(
        current_path, "../../examples/inference/api_server_openai/query_openai_sdk.py"
    )

    cmd_openai = [
        "python",
        example_openai_path,
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

    # Print the output of subprocess.run for checking if output is expected
    print(result_openai)

    # Ensure there are no errors in the OpenAI API query script execution
    assert "Error" not in result_openai.stderr

    assert isinstance(result_openai.stdout, str)

    assert len(result_openai.stdout) > 0


# Parametrize the test function with different combinations of parameters
@pytest.mark.parametrize(
    "api_base,model_name,streaming_response,max_new_tokens,temperature,top_p",
    [
        (api_base, model_name, streaming_response, max_new_tokens, temperature, top_p)
        for api_base in ["http://localhost:8000/v1"]
        for model_name in ["gpt2"]
        for streaming_response in [False, True]
        for max_new_tokens in [None, 128]
        for temperature in [None, 0.8]
        for top_p in [None, 0.7]
    ],
)
def test_script(api_base, model_name, streaming_response, max_new_tokens, temperature, top_p):
    script_with_args(api_base, model_name, streaming_response, max_new_tokens, temperature, top_p)
