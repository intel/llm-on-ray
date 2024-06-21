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

import subprocess
import pytest
import os
from basic_set import start_serve
import requests
from llm_on_ray.inference.api_simple_backend.simple_protocol import (
    SimpleRequest,
    SimpleModelResponse,
)


executed_models = []


# Parametrize the test function with different combinations of parameters
# TODO: more models and combinations will be added and tested.
@pytest.mark.parametrize(
    "prompt,streaming_response,max_new_tokens,temperature,top_p, top_k",
    [
        (
            prompt,
            streaming_response,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        )
        for prompt in ["Once upon a time", ""]
        for streaming_response in [None, True, "error"]
        for max_new_tokens in [None, 128, "error"]
        for temperature in [None]
        for top_p in [None]
        for top_k in [None]
    ],
)
def test_script(prompt, streaming_response, max_new_tokens, temperature, top_p, top_k):
    global executed_models

    # Check if this modelname has already executed start_serve
    if "gpt2" not in executed_models:
        start_serve("gpt2", simple=True)
        # Mark this modelname has already executed start_serve
        executed_models.append("gpt2")
    config = {}
    if max_new_tokens:
        config["max_new_tokens"] = max_new_tokens
    if temperature:
        config["temperature"] = temperature
    if top_p:
        config["top_p"] = top_p
    if top_k:
        config["top_k"] = top_k

    try:
        sample_input = SimpleRequest(text=prompt, config=config, stream=streaming_response)
    except ValueError as e:
        print(e)
        return
    outputs = requests.post(
        "http://localhost:8000/gpt2",
        proxies={"http": None, "https": None},  # type: ignore
        json=sample_input.dict(),
        stream=streaming_response,
    )

    outputs.raise_for_status()

    simple_response = SimpleModelResponse.from_requests_response(outputs)
    if streaming_response:
        for output in simple_response.iter_content(chunk_size=1, decode_unicode=True):
            print(output, end="", flush=True)
        print()
    else:
        print(simple_response.text, flush=True)
