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
    "config_file, models, simple, keep_serve_termimal",
    [
        (
            config_file,
            models,
            simple,
            keep_serve_termimal,
        )
        for config_file in ["../.github/workflows/config/gpt2-ci.yaml"]
        for models in ["gpt2"]
        for simple in [False]
        for keep_serve_termimal in [False]
    ],
)
def test_script(
    config_file,
    models,
    simple,
    keep_serve_termimal,
):
    cmd_serve = ["python", "../llm_on_ray/inference/serve.py"]
    if config_file is not None:
        cmd_serve.append("--config_file")
        cmd_serve.append(str(config_file))
    if models is not None:
        cmd_serve.append("--models")
        cmd_serve.append(str(models))
    if simple:
        cmd_serve.append("--simple")
    if keep_serve_termimal:
        cmd_serve.append("--keep_serve_termimal")

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    assert "Error" not in result_serve.stderr
    assert result_serve.returncode == 0
    print("Output of stderr:")
    print(result_serve.stderr)
