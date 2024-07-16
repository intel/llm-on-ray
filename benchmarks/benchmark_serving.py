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
#
# Copyright 2024 The LLM-on-Ray Authors.
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
# ===========================================================================
#
# This file is inspired by https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
#

import argparse
import asyncio
import json
from pathlib import Path
import random
import time
from tqdm import tqdm
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

from llm_on_ray.inference.inference_config import all_models
import copy

# (prompt str, output str, prompt len, output len, request latency, latencies list)
latency_tracking: List[Tuple[Optional[str], Optional[str], int, int, float, List[float]]] = []


def sample_requests_ShareGPT(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizer,
    min_input_tokens_len: int,
    max_input_tokens_len: int,
    min_output_tokens_len: int,
    max_output_tokens_len: int,
    max_length: int,
) -> List[Tuple[str, int, int]]:
    """
    Sample requests from a dataset of ShareGPT format.

    Args:
        dataset_path (str): The path to the dataset file.
        num_requests (int): The number of requests to sample.
        tokenizer (PreTrainedTokenizer): The tokenizer used to tokenize the prompts and completions.

    Returns:
        List[Tuple[str, int, int]]: A list of tuples containing the sampled requests. Each tuple
        consists of the prompt, the length of the prompt, and the length of the completion.

    """
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        # Prune too short sequences.
        if (min_input_tokens_len is not None and prompt_len < min_input_tokens_len) or (
            min_output_tokens_len is not None and output_len < min_output_tokens_len
        ):
            continue
        # Prune too long sequences.
        if (max_input_tokens_len is not None and prompt_len > max_input_tokens_len) or (
            max_output_tokens_len is not None and output_len > max_output_tokens_len
        ):
            continue
        if max_length is not None and prompt_len + output_len > max_length:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def sample_requests_IPEX(
    dataset_path: str,
    input_tokens: str,
    model_type: str,
    max_new_tokens: int,
    num_requests: int,
    tokenizer: PreTrainedTokenizer,
) -> List[Tuple[str, int, int]]:
    """
    Sample requests from a dataset of IPEX format.

    Args:
        dataset_path (str): The path to the dataset.
        input_tokens (str): The input tokens.
        max_new_tokens (int): The maximum number of new tokens.
        num_requests (int): The number of requests to sample.
        tokenizer (PreTrainedTokenizer): The tokenizer.

    Returns:
        List[Tuple[str, int, int]]: The sampled requests, each represented as a tuple of (prompt, prompt_len, output_len).
    """
    with open(dataset_path) as f:
        prompt_pool = json.load(f)

    if str(input_tokens) in prompt_pool[model_type]:
        prompt = prompt_pool[model_type][input_tokens]
    else:
        raise ValueError(f'Invalid input_tokens to index from dataset "{dataset_path}"!')

    prompt_len = len(tokenizer(prompt).input_ids)
    output_len = prompt_len if not max_new_tokens else max_new_tokens

    # Duplicate prompt to generate samples
    sampled_requests = [(prompt, prompt_len, output_len)] * num_requests

    return sampled_requests


# Sample requests from synthetic prompts with input and output length following gaussian distribution
def sample_requests_synthesis(
    tokenizer: PreTrainedTokenizer,
    input_len_mean: int,
    input_len_stddev: int,
    output_len_mean: int,
    output_len_stddev: int,
    num_requests: int,
) -> List[Tuple[str, int, int]]:
    """
    Sample requests from random generated prompts.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer.
        input_len_mean (int): The input length mean.
        input_len_stddev (int): The input length standard deviation.
        output_len_mean (int): The output length mean.
        output_len_stddev (int): The output length standard deviation.
        num_requests (int): The number of requests to sample.

    Returns:
        List[Tuple[str, int, int]]: The sampled requests, each represented as a tuple of (prompt, prompt_len, output_len).
    """
    sampled_requests = []
    for _ in range(num_requests):
        prompt_len = int(np.random.normal(input_len_mean, input_len_stddev))
        output_len = int(np.random.normal(output_len_mean, output_len_stddev))

        # generate random id list for the prompt having length prompt_len
        def gen_prompt_ids(prompt_len):
            ids = []
            for _ in range(prompt_len):
                ids.append(random.choice(list(tokenizer.get_vocab().values())).value)
            return ids

        # Generte random prompt from tokenizer's vocabulary
        prompt = tokenizer.decode(gen_prompt_ids(prompt_len), return_tensors="pt")
        sampled_requests.append((prompt, prompt_len, output_len))
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Asynchronously generates requests based on the input_requests and request_rate.

    Args:
        input_requests (List[Tuple[str, int, int]]): A list of input requests, where each request is a tuple
            containing a string, an integer, and another integer.
        request_rate (float): The rate at which requests should be generated. If set to float("inf"),
            requests will be generated without any delay.

    Yields:
        Tuple[str, int, int]: A request tuple containing a string, an integer, and another integer.

    """
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    api_url: str,
    model_name: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    config: dict,
    tokenizer: PreTrainedTokenizer,
    track_token_latency: bool = True,
    track_input_output: bool = False,
    progress_bar: tqdm = None,
    simple: bool = False,
    vllm_engine: bool = False,
) -> None:
    """
    Sends a request to the specified API URL with the given prompt and configuration.

    Args:
        api_url (str): The URL of the API to send the request to.
        prompt (str): The prompt text.
        prompt_len (int): The length of the prompt text.
        output_len (int): The desired length of the output.
        config (dict): The configuration for the request.
        progress_bar (tqdm, optional): A progress bar to update during the request. Defaults to None.
    """
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}

    # Use sample output_len if max_new_tokens not specified
    temp_config = copy.deepcopy(config)
    if "max_new_tokens" not in temp_config:
        temp_config["max_new_tokens"] = output_len
    if simple:
        pload = {
            "text": prompt,
            "config": temp_config,
            "stream": track_token_latency,
        }
    else:
        pload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": f"{prompt}"},
            ],
            "stream": track_token_latency,
            "max_tokens": temp_config["max_new_tokens"]
            if "max_new_tokens" in temp_config
            else None,
            "temperature": temp_config["temperature"] if "temperature" in temp_config else None,
            "top_p": temp_config["top_p"] if "top_p" in temp_config else None,
        }
        if vllm_engine:
            pload.update({"ignore_eos": True})

    token_latencies_per_request: List[float] = []

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []

                start_ts = time.perf_counter()

                async for chunk, _ in response.content.iter_chunks():
                    end_ts = time.perf_counter()
                    latency = end_ts - start_ts
                    if track_token_latency:
                        token_latencies_per_request.append(latency)
                    start_ts = end_ts
                    chunks.append(chunk)
                    print(chunk.decode("utf-8") + "|", end="", flush=True)
                print("Token Latencies:", token_latencies_per_request)
                # print(len(chunks), len(token_latencies_per_request))
            # Decode the response
            if simple:
                response_text = b"".join(chunks).decode("utf-8")
                if args.track_token_latency:
                    generate_len = len(tokenizer.encode(response_text))
                else:
                    if vllm_engine:
                        length_name = "num_generated_tokens"
                    else:
                        length_name = "generate_length"
                    try:
                        response_content = json.loads(response_text)
                        if isinstance(response_content, list):
                            generate_len = response_content[0][length_name]
                        else:
                            generate_len = response_content[length_name]
                    except Exception:
                        generate_len = None
            else:
                if args.track_token_latency:
                    response_content = chunks[-2].decode("utf-8")
                    response_content = json.loads(response_content.split("data: ")[1])
                    generate_len = response_content["usage"]["completion_tokens"]
                    response_text = b"".join(chunks).decode("utf-8")
                else:
                    response_text = b"".join(chunks).decode("utf-8")
                    try:
                        generate_len = json.loads(response_text)["usage"]["completion_tokens"]
                    except Exception:
                        generate_len = None
            expected_output_len = temp_config["max_new_tokens"]
            if vllm_engine and generate_len != expected_output_len:
                print(
                    f"Warning: the actual generated length is {generate_len}, which is different from the expected output length({expected_output_len})."
                )
            if progress_bar:
                progress_bar.update()
            break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time

    prompt_str = prompt if track_input_output else None
    output_str = response_text if track_input_output else None

    if generate_len is not None:
        latency_tracking.append(
            (
                prompt_str,
                output_str,
                prompt_len,
                generate_len,
                request_latency,
                token_latencies_per_request,
            )
        )


async def benchmark(
    api_url: str,
    model_name: str,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    config: dict,
    tokenizer: PreTrainedTokenizer,
    track_token_latency: bool = False,
    track_input_output: bool = False,
    progress: bool = False,
    simple: bool = False,
    vllm_engine: bool = False,
) -> None:
    """
    Benchmark the API by sending multiple requests asynchronously.

    Args:
        api_url (str): The URL of the API.
        input_requests (List[Tuple[str, int, int]]): A list of input requests, where each request is a tuple
            containing the prompt, prompt length, and output length.
        request_rate (float): The rate at which requests should be sent, in requests per second.
        config (dict): Configuration parameters for sending requests.
        progress (bool): Whether to display a progress bar.

    Returns:
        None
    """
    tasks: List[asyncio.Task] = []
    progress_bar = tqdm(total=len(input_requests)) if progress else None
    async for request in get_request(input_requests, request_rate):
        tasks.append(
            asyncio.create_task(
                send_request(
                    api_url,
                    model_name,
                    *request,
                    config,
                    tokenizer,
                    track_token_latency,
                    track_input_output,
                    progress_bar,
                    simple,
                    vllm_engine,
                )
            )
        )

    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    """
    Main function for running the benchmark serving.

    Args:
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.simple:
        route_prefix = all_models[args.model_name].route_prefix
        api_url = args.model_endpoint_base + route_prefix
    else:
        api_url = args.model_endpoint_base + "/v1/chat/completions"

    tokenizer_name_or_path = all_models[args.model_name].model_description.tokenizer_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=args.trust_remote_code
    )

    if args.dataset_format == "ShareGPT":
        input_requests = sample_requests_ShareGPT(
            args.dataset,
            args.num_prompts,
            tokenizer,
            args.min_input_tokens_len,
            args.max_input_tokens_len,
            args.min_output_tokens_len,
            args.max_output_tokens_len,
            args.max_length,
        )
    elif args.dataset_format == "IPEX":
        input_requests = sample_requests_IPEX(
            args.dataset,
            args.input_tokens,
            args.model_type,
            args.max_new_tokens,
            args.num_prompts,
            tokenizer,
        )
    elif args.dataset_format == "Synthesis":
        input_requests = sample_requests_synthesis(
            tokenizer,
            args.input_len_mean,
            args.input_len_stddev,
            args.output_len_mean,
            args.output_len_stddev,
            args.num_prompts,
        )

    config: Dict[str, Union[int, float]] = {}

    if args.max_new_tokens:
        config["max_new_tokens"] = int(args.max_new_tokens)
    if args.temperature:
        config["temperature"] = float(args.temperature)
    if args.top_p:
        config["top_p"] = float(args.top_p)
    if args.top_k:
        config["top_k"] = float(args.top_k)
    config["do_sample"] = args.do_sample
    # In order to align with vllm test parameters
    if args.vllm_engine:
        config["ignore_eos"] = True

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(
            api_url,
            args.model_name,
            input_requests,
            args.request_rate,
            config,
            tokenizer,
            args.track_token_latency,
            args.track_input_output,
            args.progress,
            args.simple,
            args.vllm_engine,
        )
    )

    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.3f} s")

    # Compute prompts statistics
    prompt_lens = [prompt_len for _, _, prompt_len, _, _, _ in latency_tracking]
    min_prompt_len = np.min(prompt_lens)
    med_prompt_len = int(np.median(prompt_lens))
    max_prompt_len = np.max(prompt_lens)
    print(f"Prompt Length (Min/Med/Max): {min_prompt_len} / {med_prompt_len} / {max_prompt_len}")

    # Compute the throughput statistics
    input_num_tokens = sum(prompt_len for _, _, prompt_len, _, _, _ in latency_tracking)
    output_num_tokens = sum(output_len for _, _, _, output_len, _, _ in latency_tracking)

    throughput_requests_per_s = args.num_prompts / benchmark_time
    input_throughput_tokens_per_s = input_num_tokens / benchmark_time
    output_throughput_tokens_per_s = output_num_tokens / benchmark_time
    print(f"Request Throughput (QPS): {throughput_requests_per_s:.3f} requests/s")
    print(f"Input Token Throughput: {input_throughput_tokens_per_s:.3f} tokens/s")
    print(f"output Token Throughput: {output_throughput_tokens_per_s:.3f} tokens/s")

    # Compute the latency statistics
    avg_latency = np.mean([latency for _, _, _, _, latency, _ in latency_tracking])
    print(f"Average latency per Request: {avg_latency:.3f} s")

    avg_per_token_latency = np.mean(
        [
            latency / output_len if output_len != 0 else latency
            for _, _, _, output_len, latency, _ in latency_tracking
        ]
    )
    print(f"Average latency per Token: {avg_per_token_latency:.3f} s")

    if args.track_token_latency and latency_tracking:
        avg_first_token_latency = np.mean(
            [latencies[0] for _, _, _, _, _, latencies in latency_tracking if latencies != []]
        )
        avg_next_token_latency = np.mean(
            [
                np.mean(latencies[1:])
                for _, _, _, _, _, latencies in latency_tracking
                if latencies[1:] != []
            ]
        )

        print(f"Average latency for First Tokens: {avg_first_token_latency:.3f} s")
        print(f"Average latency for Next Tokens: {avg_next_token_latency:.3f} s")

    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{args.results_dir} is not a directory")

        print(f"Saving results to {results_dir} ...")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_name_summary = "summary_" + timestamp + ".json"
        file_name_latency_tracking = "latency_tracking_" + timestamp + ".json"

        with open(results_dir / file_name_summary, "w") as f:
            json.dump(
                {
                    "num_prompts": args.num_prompts,
                    "benchmark_time": float(f"{benchmark_time:.3f}"),
                    "min_prompt_len": int(min_prompt_len),
                    "med_prompt_len": int(med_prompt_len),
                    "max_prompt_len": int(max_prompt_len),
                    "throughput_requests_per_s": float(f"{throughput_requests_per_s:.3f}"),
                    "input_throughput_tokens_per_s": float(f"{input_throughput_tokens_per_s:.3f}"),
                    "output_throughput_tokens_per_s": float(
                        f"{output_throughput_tokens_per_s:.3f}"
                    ),
                    "avg_latency": float(f"{avg_latency:.3f}"),
                    "avg_per_token_latency": float(f"{avg_per_token_latency:.3f}"),
                    "avg_first_token_latency": float(f"{avg_first_token_latency:.3f}")
                    if args.track_token_latency
                    else None,
                    "avg_next_token_latency": float(f"{avg_next_token_latency:.3f}")
                    if args.track_token_latency
                    else None,
                },
                f,
            )

        with open(results_dir / file_name_latency_tracking, "w") as f:
            json.dump(latency_tracking, f)

        print(
            f'Results saved to "{results_dir / file_name_summary}" and "{results_dir / file_name_latency_tracking}"'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving latency and throughput."
    )
    parser.add_argument(
        "--model-endpoint-base",
        default="http://127.0.0.1:8000",
        type=str,
        help="Model endpoint base url. Default is http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name used to extract tokenizer from configuration file.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset.")
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["ShareGPT", "IPEX", "Synthesis"],
        required=True,
        help="Dataset format, should be one of {ShareGPT, IPEX, Synthesis}.",
    )
    parser.add_argument(
        "--input-tokens",
        default="32",
        type=str,
        help="Input tokens length, used when --dataset-format=IPEX. Default is 32.",
    )
    parser.add_argument(
        "--min-input-tokens-len",
        default=4,
        type=int,
        help="Limit the minimum length of input tokens, used when --dataset-format=ShareGPT.",
    )
    parser.add_argument(
        "--max-input-tokens-len",
        default=1024,
        type=int,
        help="Limit the maximum length of input tokens, used when --dataset-format=ShareGPT.",
    )
    parser.add_argument(
        "--min-output-tokens-len",
        default=4,
        type=int,
        help="Limit the minimum length of output tokens, used when --dataset-format=ShareGPT.",
    )
    parser.add_argument(
        "--max-output-tokens-len",
        default=None,
        type=int,
        help="Limit the minimum length of output tokens, used when --dataset-format=ShareGPT.",
    )
    parser.add_argument(
        "--max-length",
        default=2048,
        type=int,
        help="Limit the maximum length including input and output tokens, used when --dataset-format=ShareGPT.",
    )
    parser.add_argument(
        "--model-type",
        default="llama",
        type=str,
        help="Model type used to select input prompt, should be one of \
            {gpt-j, llama, gpt-neox, opt, falcon-rw-1b, falcon, bloom, chatglm, codegen, \
            gptbigcode, baichuan, baichuan2, t5, mistral, mpt, qwen, mixtral, stablelm}, \
            and need to correspond to --model_name. Default is llama",
    )
    parser.add_argument(
        "--input-len-mean",
        type=int,
        default=32,
        help="The mean of input length for synthetic generation, used when --dataset-format=Synthesis. Default is 32.",
    )

    parser.add_argument(
        "--input-len-stddev",
        type=int,
        default=8,
        help="The standard deviation of input length for synthetic generation, used when --dataset-format=Synthesis. Default is 8.",
    )

    parser.add_argument(
        "--output-len-mean",
        type=int,
        default=128,
        help="The mean of output length for synthetic generation, used when --dataset-format=Synthesis. Default is 128.",
    )

    parser.add_argument(
        "--output-len-stddev",
        type=int,
        default=32,
        help="The standard deviation of output length for synthetic generation, used when --dataset-format=Synthesis, Default is 32.",
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times. Default is inf.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when downloading tokenizer.",
    )

    parser.add_argument(
        "--max-new-tokens",
        default=None,
        help="The maximum numbers of tokens to generate. dataset sample output length is used when not specified.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="The value used to modulate the next token probabilities.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="If set to float < 1, only the smallest set of most probable tokens \
            with probabilities that add up to `Top p` or higher are kept for generation.",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=None,
        help="The number of highest probability vocabulary tokens to keep \
            for top-k-filtering.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether or not to use sampling; use greedy decoding otherwise.",
    )
    parser.add_argument(
        "--vllm-engine",
        action="store_true",
        help="If set, parameter ignore_eos will be True to generate the completion.",
    )
    parser.add_argument(
        "--progress", action="store_true", help="Whether to display a progress bar."
    )
    parser.add_argument(
        "--track-token-latency",
        action="store_true",
        help="Whether to track token latency in the benchmark.",
    )
    parser.add_argument(
        "--track-input-output",
        action="store_true",
        help="Whether to track input prompt and output response in the benchmark.",
    )
    parser.add_argument(
        "--results-dir", default=None, help="The directory to output the benchmark results."
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Whether to request OpenAI-compatible API for specific model or simple endpoint.",
    )
    args = parser.parse_args()
    main(args)
