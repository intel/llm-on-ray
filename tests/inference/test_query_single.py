import subprocess
import pytest
import os

os.environ["no_proxy"] = "localhost,127.0.0.1"


def start_serve(model_name):
    current_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(
        current_path, "../../.github/workflows/config/" + model_name + "-ci.yaml"
    )

    cmd_serve = ["llm_on_ray-serve", "--config_file", config_path, "--simple"]

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    # Ensure there are no errors in the serve script execution
    assert result_serve.returncode == 0, print(
        "\n" + "Serve error stderr message: " + "\n", result_serve.stderr
    )

    # Print the output of subprocess.run for checking if output is expected
    print("\n" + "Serve message: " + "\n", result_serve.stdout)

    # Ensure there are no errors in the serve script execution
    assert "Error" not in result_serve.stderr


def script_with_args(
    base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k
):
    current_path = os.path.dirname(os.path.abspath(__file__))

    os.path.join(current_path, "../../.github/workflows/config/" + model_name + "-ci.yaml")

    example_query_single_path = os.path.join(
        current_path, "../../examples/inference/api_server_simple/query_single.py"
    )

    cmd_single = [
        "python",
        example_query_single_path,
        "--model_endpoint",
        base_url + model_name,
    ]

    if streaming_response:
        cmd_single.append("--streaming_response")

    if max_new_tokens is not None:
        cmd_single.extend(["--max_new_tokens", str(max_new_tokens)])

    if temperature is not None:
        cmd_single.extend(["--temperature", str(temperature)])

    if top_p is not None:
        cmd_single.extend(["--top_p", str(top_p)])

    if top_k is not None:
        cmd_single.extend(["--top_k", str(top_k)])

    result_query_single = subprocess.run(cmd_single, capture_output=True, text=True)

    # Print the output of subprocess.run for checking if output is expected
    print(result_query_single)

    # Ensure there are no errors in the OpenAI API query script execution
    assert "Error" not in result_query_single.stderr

    # Returncode should be 0 when there is no exception
    assert result_query_single.returncode == 0


executed_models = {}


# Parametrize the test function with different combinations of parameters
# TODO: more models and combinations will be added and tested.
@pytest.mark.parametrize(
    "base_url,model_name,streaming_response,max_new_tokens,temperature,top_p, top_k",
    [
        (base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k)
        for base_url in ["http://localhost:8000/"]
        for model_name in ["gpt2"]
        for streaming_response in [None]
        for max_new_tokens in [None]
        for temperature in [None]
        for top_p in [None]
        for top_k in [None]
    ],
)
def test_script(
    base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k
):
    global executed_models

    # Check if this modelname has already executed start_serve
    if model_name not in executed_models:
        start_serve(model_name)
        # Mark this modelname has already executed start_serve
        executed_models[model_name] = True

    script_with_args(
        base_url, model_name, streaming_response, max_new_tokens, temperature, top_p, top_k
    )
