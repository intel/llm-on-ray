#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import openai
import pytest
import os
from openai import OpenAI
from basic_set import start_serve

openai_base_url = os.environ["OPENAI_BASE_URL"]
openai_api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)


def models(openai_testing_model):  # noqa: F811
    models = client.models.list()
    assert len(models.data) == 1, "Only the test model should be returned"
    assert models.data[0].id == openai_testing_model, "The test model id should match"


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


def chat_bad_request(openai_testing_model):  # noqa: F811
    client.chat.completions.create(
        model=openai_testing_model,
        messages=[{"role": "user", "content": "Hello world"}],
        temperature=-0.1,
    )
    # with pytest.raises(openai.OpenAIError) as exc_info:
    try:
        client.chat.completions.create(
            model=openai_testing_model,
            messages=[{"role": "user", "content": "Hello world"}],
            temperature=-0.1,
        )
        # response = client.completions.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt="Say this is a test"
        # )
    except openai.InvalidRequestError as exc_info:
        print(exc_info)
        # assert "temperature" in str(exc_info.value)


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
    assert chat_completion.choices[0].model_fields["finish_reason"]
    assert i > 4


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
            "chat",
            "chat_stream",
            # "chat_bad_request",
            # "chat_stream_bad_request"
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
