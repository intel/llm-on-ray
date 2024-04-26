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

import argparse
import matplotlib.pyplot as plt
import string
import re

marks = {}
marks["bs_mark"] = r"bs:"
marks["iter_mark"] = r"iter:"
marks["input_tokens_length_mark"] = r"input_tokens_length:"
marks["prompts_num_mark"] = r"num_prompts:"
marks["total_time_mark"] = r"Total time:"
marks["prompt_length_mark"] = r"Prompt Length (Min/Med/Max):"
marks["request_throughput_mark"] = r"Request Throughput (QPS):"
marks["input_token_throughput_mark"] = r"Input Token Throughput:"
marks["output_token_throughput_mark"] = r"output Token Throughput:"
marks["latency_per_req_mark"] = r"Average latency per Request:"
marks["latency_per_token_mark"] = r"Average latency per Token:"
marks["latency_first_token_mark"] = r"Average latency for First Tokens:"
marks["latency_next_token_mark"] = r"Average latency for Next Tokens:"


def extract_metric(file_name, mark_name):
    with open(file_name, "r") as file:
        logs = file.read()
    extract_value = re.findall(marks[mark_name] + r"\s+([\d.]+)", logs)
    print(f"extract {mark_name}： {extract_value}")
    if mark_name in ["bs_mark", "iter_mark", "input_tokens_length_mark", "prompts_num_mark"]:
        extract_value = list(map(int, extract_value))
    else:
        extract_value = list(map(float, extract_value))

    return extract_value


def get_avg_metric(metric, num_iter, per_iter_len):
    avg_metric = []
    for i in range(0, per_iter_len):
        index = i
        average = 0
        num = 0
        while num < num_iter:
            average += metric[num * per_iter_len + index]
            num += 1
        avg_metric.append(average / num_iter)
    return avg_metric


def get_title_label(mark_name):
    title = marks[mark_name].strip(string.punctuation)
    label = title
    if "Throughput" in label:
        label += " (tokens/s)"
    elif "latency" in label:
        label += " (s)"
    return title, label


def plot_compare_metric(bs, metric_vllm, metric_llmonray, mark_name, save_path):
    plt.plot(bs, metric_vllm, color="red", label="VLLM")
    plt.plot(bs, metric_llmonray, label="without VLLM")
    plt.xticks(bs)

    plt.xlabel("bs")
    title, label = get_title_label(mark_name)
    plt.ylabel(label)
    plt.legend()
    plt.title(title)

    plt.savefig(save_path)
    plt.close()


def plot_vllm_peak_throughput(bs, output_Token_Throughput, mark_name, save_path):
    plt.plot(bs, output_Token_Throughput, color="red", label="VLLM")
    plt.xticks(bs)
    plt.xlabel("bs")
    title, label = get_title_label(mark_name)
    plt.ylabel(label)
    plt.legend()

    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def plot_latency_throughput(concurrency, latency, throughput, save_path):
    fig, ax1 = plt.subplots()

    for i in range(len(latency)):
        latency[i] *= 1000
    ax1.plot(latency, throughput, color="tab:blue")
    mark = []
    for i in concurrency:
        mark.append(f"bs={i}")
    for i in range(len(mark)):
        plt.text(latency[i], throughput[i], mark[i])
    ax1.set_xlabel("Average latency for Next Tokens (ms)")
    ax1.set_ylabel("Output Tokens Throughput (tokens/sec)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Values of Throughput and Next Token Latency corresponding to different bs")
    plt.savefig(save_path)
    plt.close()


def main(args):
    choice = args.choice
    if 1 in choice:
        # get the peak output throughput of llm-on-ray with vllm
        print("draw the output token throughput of llm-on-ray with vllm")
        log_file_vllm = "benchmarks/logs/1_result.txt"
        bs = extract_metric(log_file_vllm, "bs_mark")

        mark_name = "output_token_throughput_mark"
        save_path = "benchmarks/figures/1_vllm_peak_throughput.png"
        output_Token_Throughput = extract_metric(log_file_vllm, mark_name)
        plot_vllm_peak_throughput(bs, output_Token_Throughput, mark_name, save_path)
    if 2 in choice:
        # compare output token throughput(average latency per token) between llm-on-ray with vllm and llm-on-ray
        print("draw vllm vs llmonray(output token throughput/average latency per token)")
        log_file_vllm = "benchmarks/logs/2_result_vllm.txt"
        log_file_llmonray = "benchmarks/logs/2_result_llmonray.txt"
        bs = extract_metric(log_file_vllm, "bs_mark")

        mark_name = "output_token_throughput_mark"
        vllm_output_Token_Throughput = extract_metric(log_file_vllm, mark_name)
        llmonray_output_Token_Throughput = extract_metric(log_file_llmonray, mark_name)
        save_path = "benchmarks/figures/2_output_token_throughput_compare.png"
        plot_compare_metric(
            bs, vllm_output_Token_Throughput, llmonray_output_Token_Throughput, mark_name, save_path
        )

        mark_name = "latency_per_token_mark"
        save_path = "benchmarks/figures/2_average_latency_per_token_compare.png"
        vllm_average_latency_per_token = extract_metric(log_file_vllm, mark_name)
        llmonray_average_latency_per_token = extract_metric(log_file_llmonray, mark_name)
        plot_compare_metric(
            bs,
            vllm_average_latency_per_token,
            llmonray_average_latency_per_token,
            mark_name,
            save_path,
        )
    if 3 in choice:
        # latency vs throughput tradeoff for various number of requests
        print("draw latency vs throughput tradeoff for various number of requests")
        log_file = "benchmarks/logs/3_result.txt"
        iters = extract_metric(log_file, "iter_mark")
        num_prompts = extract_metric(log_file, "prompts_num_mark")
        latency_next_token = extract_metric(log_file, "latency_next_token_mark")
        output_throughput = extract_metric(log_file, "output_token_throughput_mark")
        print("iter: ", iters)
        print("num prompt: ", num_prompts)
        num_iter = len(iters)
        per_iter_len = int(len(num_prompts) / num_iter)
        avg_latency_next_token = get_avg_metric(latency_next_token, num_iter, per_iter_len)
        avg_output_throughput = get_avg_metric(output_throughput, num_iter, per_iter_len)
        print(avg_latency_next_token)
        print(avg_output_throughput)
        save_path = "benchmarks/figures/3_latency_throughput.png"
        plot_latency_throughput(
            num_prompts[:per_iter_len], avg_latency_next_token, avg_output_throughput, save_path
        )
    if 4 in choice:
        # get the latency of llm-on-Ray with vllm
        print("get the latency of llm-on-ray with vllm")
        log_file = "benchmarks/logs/4_result.txt"
        iters = extract_metric(log_file, "iter_mark")
        input_tokens_length_li = extract_metric(log_file, "input_tokens_length_mark")
        latency_first_token = extract_metric(log_file, "latency_first_token_mark")
        latency_next_token = extract_metric(log_file, "latency_next_token_mark")
        print("iter: ", iters)
        print("input_tokens_length: ", input_tokens_length_li)
        print("latency_first_token: ", latency_first_token)
        print("latency_next_token: ", latency_next_token)
        num_iter = len(iters)
        per_iter_len = int(len(input_tokens_length_li) / num_iter)
        avg_latency_first_token = get_avg_metric(latency_first_token, num_iter, per_iter_len)
        avg_latency_next_token = get_avg_metric(latency_next_token, num_iter, per_iter_len)
        print("Results: ")
        print(f"input_tokens_length: {input_tokens_length_li[:per_iter_len]}")
        print(f"avg_latency_first_token: {avg_latency_first_token}")
        print(f"avg_latency_next_token: {avg_latency_next_token}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving latency and throughput."
    )
    parser.add_argument(
        "--choice",
        nargs="*",
        default=[1, 2, 3, 4],
        type=int,
        help="Which type of figure to draw. [1: the peak throughput of llm-on-ray, 2: llm-on-ray with vllm vs llm-on-ray, 3: latency_throughput, 4: get the latecy of llm-on-ray with vllm]",
    )
    args = parser.parse_args()
    main(args)
