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
import subprocess
from openai import OpenAI


def start_serve(model_name):
    print("start_serve")
    current_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(
        current_path, "../.github/workflows/config/" + model_name + "-ci.yaml"
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
