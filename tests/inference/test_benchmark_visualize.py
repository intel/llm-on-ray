import os
import pytest
import subprocess

figures_name = [
    "choice1_vllm_peak_throughput.png",
    "choice2_average_latency_per_token_compare.png",
    "choice3_tokens_32_20.png",
]


def script_with_args(choice, benchmark_dir, save_dir):
    global figures_name
    current_path = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(current_path, "../../benchmarks/run_benchmark.sh")
    visualize_script = os.path.join(current_path, "../../benchmarks/benchmark_visualize.py")
    save_dir = os.path.join(current_path, "../../", save_dir)

    cmd_bench = ["bash", benchmark_script, choice, "test"]
    cmd_visual = ["python", visualize_script, "--choice", choice, "--benchmark-dir", benchmark_dir]
    subprocess.run(cmd_bench, capture_output=True, text=True)
    result_visual = subprocess.run(cmd_visual, capture_output=True, text=True)
    print(result_visual)
    if choice in [1, 2, 3]:
        file_path = os.path.join(save_dir, figures_name[choice - 1])
        assert os.path.exists(file_path)


@pytest.mark.parametrize(
    "choice,save_dir",
    [
        (choice, benchmark_dir, save_dir)
        for choice in [1, 2, 3, 4]
        for benchmark_dir in ["benchmarks/results_test"]
        for save_dir in ["benchmarks/figures"]
    ],
)
def test_script(choice, benchmark_dir, save_dir):
    script_with_args(choice, benchmark_dir, save_dir)
