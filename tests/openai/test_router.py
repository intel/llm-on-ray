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

import json

import pytest
from starlette.requests import Request
from starlette.responses import Response

from llm_on_ray.inference.api_openai_backend.router_app import (
    ChatCompletionRequest,
)

from llm_on_ray.inference.api_openai_backend.openai_protocol import (
    ModelResponse,
    ErrorResponse,
    ChatMessage,
)

from llm_on_ray.inference.api_openai_backend.router_app import (
    _chat_completions_wrapper,
)


async def fake_generator():
    for _ in range(10):
        yield ModelResponse(num_generated_tokens=1, generated_text="abcd")
    yield ModelResponse(num_generated_tokens=1, generated_text="abcd", finish_reason="stop")


async def fake_generator_with_error():
    for _ in range(5):
        yield ModelResponse(num_generated_tokens=1, generated_text="abcd")
    yield ModelResponse(
        error=ErrorResponse(message="error", internal_message="error", code=500, type="error"),
        finish_reason="error",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("generator", [fake_generator])
async def test_chat_completions_stream(generator):
    generator = _chat_completions_wrapper(
        "1",
        ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="test")],
        ),
        Response(),
        generator(),
    )
    count = 0
    had_done = False
    had_finish = False
    async for x in generator:
        count += 1
        assert x.startswith("data: ")
        assert x.endswith("\n\n")
        if x.strip() == "data: [DONE]":
            assert had_finish
            had_done = True
        elif had_done:
            raise AssertionError()
        else:
            dct = json.loads(x[6:].strip())
            assert "id" in dct
            assert "object" in dct
            assert "choices" in dct
            if count == 1:
                assert dct["choices"][0]["delta"]["role"] == "assistant"
            elif dct["choices"][0]["finish_reason"]:
                assert dct["choices"][0]["finish_reason"] == "stop"
                had_finish = True
            else:
                assert dct["choices"][0]["delta"]["content"] == "abcd"
    assert had_done
    assert had_finish
    assert count == 14
    assert dct["usage"]["completion_tokens"] == 11


@pytest.mark.asyncio
@pytest.mark.parametrize("generator", [fake_generator_with_error])
async def test_chat_completions_stream_with_error(generator):
    generator = _chat_completions_wrapper(
        "1",
        ChatCompletionRequest(model="test", messages=[ChatMessage(role="user", content="test")]),
        Response(),
        generator(),
    )
    count = 0
    had_error = False
    had_done = False
    async for x in generator:
        count += 1
        assert x.startswith("data: ")
        assert x.endswith("\n\n")
        if x.strip() == "data: [DONE]":
            had_done = True
        elif had_done:
            raise AssertionError()
        else:
            dct = json.loads(x[6:].strip())
            if "error" in dct:
                had_error = True
                assert dct["error"]["message"] == "error"
            elif had_error:
                raise AssertionError()
            else:
                assert "id" in dct
                assert "object" in dct
                assert "choices" in dct
                if count == 1:
                    assert dct["choices"][0]["delta"]["role"] == "assistant"
                elif not had_error:
                    assert dct["choices"][0]["delta"]["content"] == "abcd"
    assert dct["error"]
    assert had_error
    assert count == 8
