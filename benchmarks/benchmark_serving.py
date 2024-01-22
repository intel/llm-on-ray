# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py

import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, Dict, List, Tuple, Union

import aiohttp
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer

from inference.inference_config import all_models

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str, num_requests: int, tokenizer: PreTrainedTokenizer
) -> List[Tuple[str, int, int]]:
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


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
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
    api_url: str, prompt: str, prompt_len: int, output_len: int, config: dict
) -> None:
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
            output = b"".join(chunks).decode("utf-8")
            print(f"\n=== Request =====\n{pload}\n================\n")
            print(f"=== Response ===\n{output}\n================\n")
            break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def benchmark(
    api_url: str, input_requests: List[Tuple[str, int, int]], request_rate: float, config: dict
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(api_url, prompt, prompt_len, output_len, config))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    route_prefix = all_models[args.model_name].route_prefix
    api_url = args.model_endpoint_base + route_prefix

    tokenizer_name_or_path = all_models[args.model_name].model_description.tokenizer_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=args.trust_remote_code
    )

    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

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
    asyncio.run(benchmark(api_url, input_requests, args.request_rate, config))
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
        "--max_new_tokens",
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

    args = parser.parse_args()
    main(args)
