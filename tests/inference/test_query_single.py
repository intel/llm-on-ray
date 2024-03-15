import subprocess
import pytest
import os

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_KEY"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"


def script_with_args(
    base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k
):
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

    # Returncode should be 0 when there is no exception
    assert result_serve.returncode == 0

    example_query_single_path = os.path.join(
        current_path, "../../examples/inference/api_server_simple/query_single.py"
    )

    cmd_openai = [
        "python",
        example_query_single_path,
        "--model_endpoint",
        base_url + "/" + model_name,
    ]

    if streaming_response:
        cmd_openai.append("--streaming_response")

    if max_new_tokens is not None:
        cmd_openai.extend(["--max_new_tokens", str(max_new_tokens)])

    if temperature is not None:
        cmd_openai.extend(["--temperature", str(temperature)])

    if top_p is not None:
        cmd_openai.extend(["--top_p", str(top_p)])

    if top_k is not None:
        cmd_openai.extend(["--top_k", str(top_k)])

    result_query_single = subprocess.run(cmd_openai, capture_output=True, text=True)

    # Print the output of subprocess.run for checking if output is expected
    print(result_query_single)

    # Ensure there are no errors in the OpenAI API query script execution
    assert "Error" not in result_query_single.stderr

    # Returncode should be 0 when there is no exception
    assert result_query_single.returncode == 0


# Parametrize the test function with different combinations of parameters
# TODO: more models and combinations will be added and tested.
@pytest.mark.parametrize(
    "base_url,model_name,streaming_response,max_new_tokens,temperature,top_p, top_k",
    [
        (base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k)
        for base_url in ["http://localhost:8000/v1"]
        for model_name in ["gpt2"]
        for streaming_response in [False, True]
        for max_new_tokens in [None, 128]
        for temperature in [None, 0.8]
        for top_p in [None, 0.7]
        for top_k in [None, 5]
    ],
)
def test_script(
    base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k
):
    script_with_args(
        base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k
    )
