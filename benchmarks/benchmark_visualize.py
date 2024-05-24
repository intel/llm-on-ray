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
import json
import os

marks = {}
marks["prompts_num_mark"] = r"num_prompts"
marks["total_time_mark"] = r"benchmark_time"
marks["min_prompt_len_mark"] = r"min_prompt_len"
marks["med_prompt_len_mark"] = r"med_prompt_len"
marks["max_prompt_len_mark"] = r"max_prompt_len"
marks["request_throughput_mark"] = r"throughput_requests_per_s"
marks["input_token_throughput_mark"] = r"input_throughput_tokens_per_s"
marks["output_token_throughput_mark"] = r"output_throughput_tokens_per_s"
marks["latency_per_req_mark"] = r"avg_latency"
marks["latency_per_token_mark"] = r"avg_per_token_latency"
marks["latency_first_token_mark"] = r"avg_first_token_latency"
marks["latency_next_token_mark"] = r"avg_next_token_latency"


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
    if "throughput" in label:
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
    print("generated successfully")


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
    print("generated successfully")


def plot_latency_throughput(concurrency, num_replica, latency, throughput, save_path):
    fig, ax1 = plt.subplots()

    for i in range(len(latency)):
        latency[i] *= 1000
    ax1.plot(latency, throughput, color="tab:blue")
    mark = []
    for i in concurrency:
        mark.append(f"bs={int(i/num_replica)}")
    for i in range(len(mark)):
        plt.text(latency[i], throughput[i], mark[i])
    ax1.set_xlabel("Average latency for Next Tokens (ms)")
    ax1.set_ylabel("Output Tokens Throughput (tokens/sec)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Values of Throughput and Next Token Latency corresponding to different bs")
    plt.savefig(save_path)
    plt.close()
    print("generated successfully")


def extract_metric_choice_1_2(bs_dirs, mark_name):
    bs = []
    metric_value = []
    bs_listdir = os.listdir(bs_dirs)
    bs_listdir.sort(key=lambda x: (len(x), x))
    for bs_dir in bs_listdir:
        bs.append(int(bs_dir.split("_")[-1]))
        res_path = os.path.join(bs_dirs, bs_dir)
        # get the latest summary log file
        res_listdir = os.listdir(res_path)
        res_listdir = [res_i for res_i in res_listdir if "summary" in res_i]
        res_listdir.sort(key=lambda x: (len(x), x))
        log_file = os.path.join(res_path, res_listdir[-1])
        with open(log_file) as f:
            log_content = json.load(f)
        metric_value.append(float(log_content[marks[mark_name]]))
    return bs, metric_value


def extract_metric_choice_3(iter_dirs):
    iters = []
    num_prompts = []
    latency_next_token = []
    output_throughput = []
    iter_listdir = os.listdir(iter_dirs)
    iter_listdir.sort(key=lambda x: (len(x), x))
    for iter_dir in iter_listdir:
        prompt_dirs = os.path.join(iter_dirs, iter_dir)
        iters.append(int(iter_dir.split("_")[-1]))
        prompt_listdir = os.listdir(prompt_dirs)
        prompt_listdir.sort(key=lambda x: (len(x), x))
        for prompt_dir in prompt_listdir:
            num_prompts.append(int(prompt_dir.split("_")[-1]))
            res_path = os.path.join(prompt_dirs, prompt_dir)
            # get the latest summary log file
            res_listdir = os.listdir(res_path)
            res_listdir = [res_i for res_i in res_listdir if "summary" in res_i]
            res_listdir.sort(key=lambda x: (len(x), x))
            log_file = os.path.join(res_path, res_listdir[-1])
            with open(log_file) as f:
                log_content = json.load(f)
            latency_next_token.append(float(log_content[marks["latency_next_token_mark"]]))
            output_throughput.append(float(log_content[marks["output_token_throughput_mark"]]))
    return iters, num_prompts, latency_next_token, output_throughput


def extract_metric_choice_4(iter_dirs):
    iters = []
    input_tokens_length_li = []
    latency_first_token = []
    latency_next_token = []
    iter_listdir = os.listdir(iter_dirs)
    iter_listdir.sort(key=lambda x: (len(x), x))
    for iter_dir in iter_listdir:
        iters.append(int(iter_dir.split("_")[-1]))
        token_dirs = os.path.join(iter_dirs, iter_dir)
        for token_dir in os.listdir(token_dirs):
            input_tokens_length_li.append(int(token_dir.split("_")[-2]))
            res_path = os.path.join(token_dirs, token_dir)
            # get the latest summary log file
            res_listdir = os.listdir(res_path)
            res_listdir = [res_i for res_i in res_listdir if "summary" in res_i]
            res_listdir.sort(key=lambda x: (len(x), x))
            log_file = os.path.join(res_path, res_listdir[-1])
            with open(log_file) as f:
                log_content = json.load(f)
            latency_first_token.append(float(log_content[marks["latency_first_token_mark"]]))
            latency_next_token.append(float(log_content[marks["latency_next_token_mark"]]))

    return iters, input_tokens_length_li, latency_first_token, latency_next_token


def main(args):
    choice = args.choice
    benchmark_dir = args.benchmark_dir
    save_dir = args.save_dir
    current_path = os.path.dirname(os.path.abspath(__file__))
    if benchmark_dir is None:
        if args.run_mode == "benchmark":
            benchmark_dir = os.path.join(current_path, "results")
        elif args.run_mode == "test":
            benchmark_dir = os.path.join(current_path, "results_test")
        else:
            print(
                f"Invalid run_mode, expected value 'test' or 'benchmark', but got {args.run_mode}."
            )
            exit(1)
    if save_dir is None:
        save_dir = os.path.join(current_path, "figures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if 1 in choice:
        # get the peak output throughput of llm-on-ray with vllm
        print("draw the output token throughput of llm-on-ray with vllm")
        choice_dir = os.path.join(benchmark_dir, "choice_1")

        mark_name = "output_token_throughput_mark"
        bs, output_Token_Throughput = extract_metric_choice_1_2(choice_dir, mark_name)
        print("bs: ", bs)
        print("output_Token_throughput: ", output_Token_Throughput)

        save_figure_path = os.path.join(save_dir, "choice1_vllm_peak_throughput.png")
        plot_vllm_peak_throughput(bs, output_Token_Throughput, mark_name, save_figure_path)
    if 2 in choice:
        # compare output token throughput(average latency per token) between llm-on-ray with vllm and llm-on-ray
        print("draw vllm vs llmonray(output token throughput/average latency per token)")
        choice_dir = os.path.join(benchmark_dir, "choice_2")
        vllm_dir = os.path.join(choice_dir, "vllm")
        wo_vllm_dir = os.path.join(choice_dir, "wo_vllm")

        mark_name = "output_token_throughput_mark"
        bs, vllm_output_Token_Throughput = extract_metric_choice_1_2(vllm_dir, mark_name)
        _, wo_vllm_output_Token_Throughput = extract_metric_choice_1_2(wo_vllm_dir, mark_name)
        print("bs: ", bs)
        print("vllm_output_Token_Throughput: ", vllm_output_Token_Throughput)
        print("wo_vllm_output_Token_Throughput: ", wo_vllm_output_Token_Throughput)

        save_figure_path = os.path.join(save_dir, "choice2_output_token_throughput_compare.png")
        plot_compare_metric(
            bs,
            vllm_output_Token_Throughput,
            wo_vllm_output_Token_Throughput,
            mark_name,
            save_figure_path,
        )

        mark_name = "latency_per_token_mark"
        save_figure_path = os.path.join(save_dir, "choice2_average_latency_per_token_compare.png")
        bs, vllm_average_latency_per_token = extract_metric_choice_1_2(vllm_dir, mark_name)
        _, wo_vllm_average_latency_per_token = extract_metric_choice_1_2(wo_vllm_dir, mark_name)
        print("vllm_average_latency_per_token: ", vllm_average_latency_per_token)
        print("wo_vllm_average_latency_per_token: ", wo_vllm_average_latency_per_token)
        plot_compare_metric(
            bs,
            vllm_average_latency_per_token,
            wo_vllm_average_latency_per_token,
            mark_name,
            save_figure_path,
        )
    if 3 in choice:
        # latency vs throughput tradeoff for various number of requests
        choice_dir = os.path.join(benchmark_dir, "choice_3")
        token_dirs = os.listdir(choice_dir)
        print("draw latency vs throughput tradeoff for various number of requests")
        for token_res in token_dirs:
            iter_dirs = os.path.join(choice_dir, token_res)
            save_figure_path = os.path.join(save_dir, "choice3_" + token_res)
            iters, num_prompts, latency_next_token, output_throughput = extract_metric_choice_3(
                iter_dirs
            )

            print("iter: ", iters)
            print("num prompt: ", num_prompts)
            print("latency_next_token: ", latency_next_token)
            print("output_throughput: ", output_throughput)
            num_iter = len(iters)
            per_iter_len = int(len(num_prompts) / num_iter)
            avg_latency_next_token = get_avg_metric(latency_next_token, num_iter, per_iter_len)
            avg_output_throughput = get_avg_metric(output_throughput, num_iter, per_iter_len)
            print("avg_latency_next_token: ", avg_latency_next_token)
            print("avg_output_throughput: ", avg_output_throughput)
            plot_latency_throughput(
                num_prompts[:per_iter_len],
                args.num_replica,
                avg_latency_next_token,
                avg_output_throughput,
                save_figure_path,
            )
    if 4 in choice:
        # get the latency of llm-on-Ray with vllm
        choice_dir = os.path.join(benchmark_dir, "choice_4")
        print("get the latency of llm-on-ray with vllm")
        (
            iters,
            input_tokens_length_li,
            latency_first_token,
            latency_next_token,
        ) = extract_metric_choice_4(choice_dir)
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
    parser.add_argument(
        "--num-replica",
        default=1,
        type=int,
        help="The number of replicas that respond to requests at the same time.",
    )
    parser.add_argument(
        "--run-mode",
        default="benchmark",
        type=str,
        help="Which run mode is used to generate the results, benchmark or test?",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        type=str,
        help="The directory of benchmark results.",
    )
    parser.add_argument("--save-dir", default=None, type=str, help="The directory to save figures.")

    args = parser.parse_args()
    main(args)
