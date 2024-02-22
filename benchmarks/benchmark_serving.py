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
# This file is adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
#

import argparse
import asyncio
import json
import random
import time
from tqdm import tqdm
from typing import AsyncGenerator, Dict, List, Tuple, Union

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

from inference.inference_config import all_models

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

# TODO: (request id, prompt len, output len, latencies list)
TOKEN_LATENCY_PER_REQUEST: List[Tuple[int, int, int, List[int]]] = []


def sample_requests_ShareGPT(
    dataset_path: str, num_requests: int, tokenizer: PreTrainedTokenizer
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
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def sample_requests_IPEX(
    dataset_path: str,
    input_tokens: str,
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

    # Only sample from gpt-j subset prompts for now
    model_type = "gpt-j"
    if str(input_tokens) in prompt_pool[model_type]:
        prompt = prompt_pool[model_type][input_tokens]
    else:
        raise ValueError(f'Invalid input_tokens to index from dataset "{dataset_path}"!')

    prompt_len = len(tokenizer(prompt).input_ids)
    output_len = prompt_len if not max_new_tokens else max_new_tokens

    # Duplicate prompt to generate samples
    sampled_requests = [(prompt, prompt_len, output_len)] * num_requests

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
    prompt: str,
    prompt_len: int,
    output_len: int,
    config: dict,
    progress_bar: tqdm = None,
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
    if "max_new_tokens" in config:
        output_len = config["max_new_tokens"]
    else:
        config["max_new_tokens"] = output_len

    pload = {
        "text": prompt,
        "config": config,
        "stream": False,
    }

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            # Decode the response
            b"".join(chunks).decode("utf-8")
            if progress_bar:
                progress_bar.update()
            break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def benchmark(
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    config: dict,
    progress: bool,
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
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(api_url, prompt, prompt_len, output_len, config, progress_bar)
        )
        tasks.append(task)
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

    route_prefix = all_models[args.model_name].route_prefix
    api_url = args.model_endpoint_base + route_prefix

    tokenizer_name_or_path = all_models[args.model_name].model_description.tokenizer_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=args.trust_remote_code
    )

    if args.dataset_format == "ShareGPT":
        input_requests = sample_requests_ShareGPT(args.dataset, args.num_prompts, tokenizer)
    elif args.dataset_format == "IPEX":
        input_requests = sample_requests_IPEX(
            args.dataset, args.input_tokens, args.max_new_tokens, args.num_prompts, tokenizer
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

    benchmark_start_time = time.perf_counter()
    asyncio.run(benchmark(api_url, input_requests, args.request_rate, config, args.progress))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")

    # Compute prompts statistics
    prompt_lens = [prompt_len for prompt_len, _, _ in REQUEST_LATENCY]
    min_prompt_len = np.min(prompt_lens)
    med_prompt_len = int(np.median(prompt_lens))
    max_prompt_len = np.max(prompt_lens)
    print(f"Prompt Length (Min/Med/Max): {min_prompt_len} / {med_prompt_len} / {max_prompt_len}")

    # Compute the throughput statistics
    total_num_tokens = sum(prompt_len + output_len for prompt_len, output_len, _ in REQUEST_LATENCY)
    print(
        f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s, "
        f"{total_num_tokens / benchmark_time:.2f} tokens/s"
    )

    # Compute the latency statistics
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")

    avg_per_token_latency = np.mean(
        [latency / (prompt_len + output_len) for prompt_len, output_len, latency in REQUEST_LATENCY]
    )
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving latency and throughput."
    )
    parser.add_argument(
        "--model_endpoint_base",
        default="http://127.0.0.1:8000",
        type=str,
        help="model endpoint base url",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name used to extract tokenizer from configuration file",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset.")
    parser.add_argument(
        "--dataset-format",
        type=str,
        choices=["ShareGPT, IPEX"],
        required=True,
        help="Dataset format, should be one of [ShareGPT, IPEX].",
    )
    parser.add_argument(
        "--input-tokens",
        default="32",
        type=str,
        help="input tokens length, used when --dataset-format=IPEX",
    )

    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="trust remote code from huggingface"
    )

    parser.add_argument(
        "--max-new-tokens",
        default=None,
        help="The maximum numbers of tokens to generate. dataset sample output length is used when not specified.",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        help="The value used to modulate the next token probabilities.",
    )
    parser.add_argument(
        "--top_p",
        default=None,
        help="If set to float < 1, only the smallest set of most probable tokens \
            with probabilities that add up to `Top p` or higher are kept for generation.",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        help="The number of highest probability vocabulary tokens to keep \
            for top-k-filtering.",
    )
    parser.add_argument(
        "--progress", action="store_true", help="Whether to display a progress bar."
    )

    args = parser.parse_args()
    main(args)
