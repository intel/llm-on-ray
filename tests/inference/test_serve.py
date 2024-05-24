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


# Parametrize the test function with different combinations of parameters
@pytest.mark.parametrize(
    "config_file, models, port, simple, keep_serve_termimal,list_model_ids",
    [
        (
            config_file,
            models,
            port,
            simple,
            keep_serve_termimal,
            list_model_ids,
        )
        for config_file in ["../.github/workflows/config/gpt2-ci.yaml", None]
        for models in ["gpt2", "llama-2-7b-chat-hf"]
        for port in [8000]
        for simple in [False]
        for keep_serve_termimal in [False]
        for list_model_ids in [False, True]
    ],
)
def test_script(
    config_file,
    models,
    port,
    simple,
    keep_serve_termimal,
    list_model_ids,
):
    cmd_serve = ["llm_on_ray-serve"]
    if list_model_ids:
        cmd_serve.append("--list_model_ids")
    else:
        if config_file is not None:
            cmd_serve.append("--config_file")
            cmd_serve.append(str(config_file))
        elif models is not None:
            cmd_serve.append("--models")
            cmd_serve.append(str(models))
        if port is not None:
            cmd_serve.append("--port")
            cmd_serve.append(str(port))
        if simple:
            cmd_serve.append("--simple")
        if keep_serve_termimal:
            cmd_serve.append("--keep_serve_termimal")

    print(cmd_serve)
    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)
    if list_model_ids:
        # Check if the model IDs are listed
        assert "gpt2" in result_serve.stdout
        assert "gpt2.yaml" in result_serve.stdout

    assert "Error" not in result_serve.stderr
    assert result_serve.returncode == 0
    print("Output of stderr:")
    print(result_serve.stderr)
