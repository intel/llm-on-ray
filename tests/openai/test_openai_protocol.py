import openai
import pytest
import os
import subprocess
from openai import OpenAI

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_KEY"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
openai_base_url = os.environ["OPENAI_BASE_URL"]
openai_api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)


def start_serve(model_name):
    print("start_serve")
    current_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(
        current_path, "../../.github/workflows/config/" + model_name + "-ci.yaml"
    )
    os.path.join(current_path, "../../inference/serve.py")
    cmd_serve = ["llm_on_ray-serve", "--config_file", config_path]

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    # Ensure there are no errors in the serve script execution
    assert result_serve.returncode == 0, print(
        "\n" + "Serve error stderr message: " + "\n", result_serve.stderr
    )

    # Print the output of subprocess.run for checking if output is expected
    print("\n" + "Serve message: " + "\n", result_serve.stdout)


def models(openai_testing_model):  # noqa: F811
    models = client.models.list()
    assert len(models.data) == 1, "Only the test model should be returned"
    assert models.data[0].id == openai_testing_model, "The test model id should match"


def completions(openai_testing_model):  # noqa: F811
    completion = client.completions.create(
        model=openai_testing_model,
        prompt="Hello world",
        top_p=0.1,
        max_tokens=2,
    )
    assert completion.model == openai_testing_model
    assert completion.model
    # assert completion.choices[0].text == "Hello world"


def chat(openai_testing_model):  # noqa: F811
    # create a chat completion
    chat_completion = client.chat.completions.create(
        model=openai_testing_model,
        messages=[{"role": "user", "content": "Hello world"}],
        top_p=1,
    )
    assert chat_completion
    assert chat_completion.usage
    assert chat_completion.id
    assert isinstance(chat_completion.choices, list)
    assert chat_completion.choices[0].message.content


def completions_bad_request(openai_testing_model):  # noqa: F811
    with pytest.raises(openai.BadRequestError) as exc_info:
        client.completions.create(
            model=openai_testing_model,
            prompt="Hello world",
            temperature=-0.1,
        )
    assert "temperature" in str(exc_info.value)


def chat_bad_request(openai_testing_model):  # noqa: F811
    with pytest.raises(openai.BadRequestError) as exc_info:
        client.chat.completions.create(
            model=openai_testing_model,
            messages=[{"role": "user", "content": "Hello world"}],
            temperature=-0.1,
        )
    assert "temperature" in str(exc_info.value)


def completions_stream(openai_testing_model):  # noqa: F811
    i = 0
    for completion in client.completions.create(
        model=openai_testing_model, prompt="Hello world", stream=True, top_p=1
    ):
        i += 1
        assert completion
        assert completion.id
        assert isinstance(completion.choices, list)
        assert isinstance(completion.choices[0].text, str)
    assert i > 4


def chat_stream(openai_testing_model):  # noqa: F811
    i = 0
    for chat_completion in client.chat.completions.create(
        model=openai_testing_model,
        messages=[{"role": "user", "content": "Hello world"}],
        stream=True,
        temperature=0.4,
        frequency_penalty=0.02,
        top_p=1,
    ):
        if i == 0:
            assert chat_completion
            assert chat_completion.id
            assert isinstance(chat_completion.choices, list)
            assert chat_completion.choices[0].delta.role
        else:
            assert chat_completion
            assert chat_completion.id
            assert isinstance(chat_completion.choices, list)
            assert chat_completion.choices[0].delta == {} or hasattr(
                chat_completion.choices[0].delta, "content"
            )
        i += 1
    assert chat_completion
    assert chat_completion.id
    assert isinstance(chat_completion.choices, list)
    assert chat_completion.choices[0].delta == {} or hasattr(
        chat_completion.choices[0].delta, "content"
    )
    print(dir(chat_completion.choices[0]))
    assert chat_completion.choices[0].model_fields["finish_reason"]
    assert i > 4


def completions_stream_bad_request(openai_testing_model):  # noqa: F811
    with pytest.raises(openai.APIError) as exc_info:
        for _ in client.completions.create(
            model=openai_testing_model,
            prompt="Hello world",
            stream=True,
            temperature=-0.1,
        ):
            pass
    assert "temperature" in str(exc_info.value)


def chat_stream_bad_request(openai_testing_model):  # noqa: F811
    with pytest.raises(openai.APIError) as exc_info:
        for _chat_completion in client.chat.completions.create(
            model=openai_testing_model,
            messages=[{"role": "user", "content": "Hello world"}],
            stream=True,
            temperature=-0.1,
        ):
            pass
    assert "temperature" in str(exc_info.value)


executed_models = {}


# Parametrize the test function with different combinations of parameters
@pytest.mark.parametrize(
    "model,test_func",
    [
        (model, test_func)
        for model in ["gpt2"]
        for test_func in [
            "models",
            # "completions",
            # "completions_stream",
            "chat",
            "chat_stream",
            # "chat_bad_request",
            # "chat_stream_bad_request"
            # "completions_bad_request",
            # "completions_stream_bad_request",
        ]
    ],
)
def test_openai(model, test_func):
    global executed_models

    # Check if this modelname has already executed start_serve
    if model not in executed_models:
        start_serve(model)
        # Mark this modelname has already executed start_serve
        executed_models[model] = True
    eval(test_func + "('" + model + "')")
