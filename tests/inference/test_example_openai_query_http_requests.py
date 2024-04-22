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


def script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p):
    current_path = os.path.dirname(os.path.abspath(__file__))

    example_http_path = os.path.join(
        current_path, "../../examples/inference/api_server_openai/query_http_requests.py"
    )

    cmd_http = [
        "python",
        example_http_path,
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
    assert result_http.returncode == 0, print(
        "\n" + "Http error stderr message: " + "\n", result_http.stderr
    )

    # Print the output of subprocess.run for checking if output is expected
    print("\n" + "Model in Http output message: " + "\n", result_http.stdout)

    assert isinstance(result_http.stdout, str), print("\n" + "Http output is nor string" + "\n")

    assert len(result_http.stdout) > 0, print("\n" + "Http output length is 0" + "\n")


executed_models = []


@pytest.mark.parametrize(
    "model_name,streaming_response,max_new_tokens,temperature,top_p",
    [
        (model_name, streaming_response, max_new_tokens, temperature, top_p)
        for model_name in ["bloom-560m", "opt-125m", "gpt2"]
        for streaming_response in [False, True]
        for max_new_tokens in [None, 128]
        for temperature in [None, 0.8]
        for top_p in [None, 0.7]
    ],
)
def test_script(model_name, streaming_response, max_new_tokens, temperature, top_p):
    global executed_models

    # Check if this modelname has already executed start_serve
    if model_name not in executed_models:
        start_serve(model_name)
        # Add this modelname for already executed start_serve
        executed_models.append(model_name)
    script_with_args(model_name, streaming_response, max_new_tokens, temperature, top_p)
