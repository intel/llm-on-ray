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

import os
import pytest
import subprocess
import re

figures_name = [
    "choice1_vllm_peak_throughput.png",
    "choice2_average_latency_per_token_compare.png",
    "choice3_tokens_32_20.png",
]


def script_with_args(choice):
    global figures_name
    current_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(current_path, "../../benchmarks/run_benchmark.sh")
    visualize_script = os.path.join(current_path, "../../benchmarks/benchmark_visualize.py")
    save_dir = os.path.join(current_path, "../../benchmarks/figures")

    cmd_bench = ["bash", benchmark_script, str(choice), "test"]
    cmd_visual = [
        "python",
        visualize_script,
        "--choice",
        str(choice),
        "--run-mode",
        "test",
    ]
    result_bench = subprocess.run(cmd_bench, capture_output=True, text=True)
    assert "Error" not in result_bench.stderr
    assert result_bench.returncode == 0
    print("Output of result_bench:", result_bench)
    result_visual = subprocess.run(cmd_visual, capture_output=True, text=True)
    print(result_visual)
    assert "Error" not in result_visual.stderr
    assert result_visual.returncode == 0
    print("Output of stderr:", result_visual.stderr)
    bs = re.search(r"bs:\s+(\[.*?\])", result_visual.stdout)
    if bs:
        bs = bs.group(1)
    iter = re.search(r"iter:\s+(\[.*?\])", result_visual.stdout)
    if iter:
        iter = iter.group(1)
    assert (bs == "[1, 2, 4]") or (iter == "[1, 2]")
    if choice in [1, 2, 3]:
        file_path = os.path.join(save_dir, figures_name[choice - 1])
        assert os.path.exists(file_path)


@pytest.mark.parametrize(
    "choice",
    [(choice) for choice in [1, 2, 3, 4]],
)
def test_script(choice):
    script_with_args(choice)
