import subprocess
import pytest

# Config matrix
models_array = ["gpt2", "gpt2 gpt-j-6b", "gpt2 bloom-560m falcon-7b"]
model_endpoint_array = ["http://127.0.0.1:8000", None]
streaming_response_array = [True, False]
max_new_tokens_array = [10, None]
temperature_array = [None]
top_p_array = [0.6, None]
top_k_array = [5, None]


# Parametrize the test function with different combinations of parameters
@pytest.mark.parametrize(
    "model_endpoint, streaming_response, max_new_tokens, temperature, top_p, top_k",
    [
        (models, model_endpoint, streaming_response, max_new_tokens, temperature, top_p, top_k)
        for models in models_array
        for model_endpoint in model_endpoint_array
        for streaming_response in streaming_response_array
        for max_new_tokens in max_new_tokens_array
        for temperature in temperature_array
        for top_p in top_p_array
        for top_k in top_k_array
    ],
)
def test_script(
    models, model_endpoint, streaming_response, max_new_tokens, temperature, top_p, top_k
):
    # Validate model endpoint and get port
    tmp_list = model_endpoint_array[0].split(":")
    assert len(tmp_list) == 3, "Invalid URL, model endpoint should be like http://127.0.0.1:8000"

    port = int(tmp_list[2])
    assert port > 0 and port < 65535, "Invalid Port, it should be 0~65535"

    # Run serve.py to activate all models
    cmd_serve = ["python", "../inference/serve.py", "--models", models, "--port", port, "--simple"]

    if streaming_response:
        cmd_serve.append["--streaming_response"]
    if max_new_tokens is not None:
        cmd_serve.append["--max_new_tokens", max_new_tokens]
    if temperature is not None:
        cmd_serve.append["--temperature", temperature]
    if top_p is not None:
        cmd_serve.append["--top_p", top_p]
    if top_k is not None:
        cmd_serve.append["--top_k", top_k]

    for it in models.split(" "):
        model = it.strip()
        assert len(model) > 0, "Invalid empty model."
        if model_endpoint is not None:
            cmd_serve.append["--model_endpoint", model_endpoint + "/" + model]
        else:
            cmd_serve.append["--model_endpoint", "http://127.0.0.1:8000" + "/" + model]
        result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)
        # TODO: Find a better way to assert the result, like checking processes etc.
        assert "Error" not in result_serve.stderr
