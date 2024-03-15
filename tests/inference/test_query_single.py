import subprocess
import pytest

# Config matrix
# models_array = ["gpt2", "gpt2 gpt-j-6b", "gpt2 bloom-560m", "falcon-7b"]
# model_endpoint_array = ["http://127.0.0.1:8000", None]
# streaming_response_array = [True, False]
# max_new_tokens_array = [10, None]
# temperature_array = [0.7, None]
# top_p_array = [0.6, None]
# top_k_array = [5, None]


# Parametrize the test function with different combinations of parameters
@pytest.mark.parametrize(
    "models, model_endpoint, streaming_response, max_new_tokens, temperature, top_p, top_k",
    [
        (models, model_endpoint, streaming_response, max_new_tokens, temperature, top_p, top_k)
        for models in ["gpt2"]
        for model_endpoint in ["http://127.0.0.1:8000"]
        for streaming_response in [False]
        for max_new_tokens in [10]
        for temperature in [0.7]
        for top_p in [0.6]
        for top_k in [5]
    ],
)
def test_script(
    models, model_endpoint, streaming_response, max_new_tokens, temperature, top_p, top_k
):
    # Validate model endpoint and get port
    tmp_list = ["http://127.0.0.1:8000"][0].split(":")
    assert len(tmp_list) == 3, "Invalid URL, model endpoint should be like http://127.0.0.1:8000"

    port = int(tmp_list[2])
    assert port > 0 and port < 65535, "Invalid Port, it should be 0~65535"

    # Run serve.py to activate all models
    cmd_serve = [
        "python",
        "../inference/serve.py",
        "--models " + str(models),
        "--port " + str(port),
        "--simple",
    ]

    if streaming_response:
        cmd_serve.append("--streaming_response")
    if max_new_tokens is not None:
        cmd_serve.append("--max_new_tokens " + str(max_new_tokens))
    if temperature is not None:
        cmd_serve.append("--temperature " + str(temperature))
    if top_p is not None:
        cmd_serve.append("--top_p " + str(top_p))
    if top_k is not None:
        cmd_serve.append("--top_k " + str(top_k))

    for it in models.split(" "):
        model = it.strip()
        assert len(model) > 0, "Invalid empty model."
        if model_endpoint is not None:
            cmd_serve.append("--model_endpoint " + model_endpoint + "/" + model)
        else:
            cmd_serve.append("--model_endpoint " + "http://127.0.0.1:8000" + "/" + model)
        result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)
        assert "Error" not in result_serve.stderr
        assert result_serve == 0
        print("Asserted no erros in the result log, which is:")
        print(result_serve.stderr)
