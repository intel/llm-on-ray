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

import requests
import argparse
from typing import Dict, Union
from llm_on_ray.inference.api_simple_backend.simple_protocol import (
    SimpleRequest,
    SimpleModelResponse,
)

parser = argparse.ArgumentParser(
    description="Example script to query with single request", add_help=True
)
parser.add_argument(
    "--model_endpoint",
    default="http://127.0.0.1:8000",
    type=str,
    help="Deployed model endpoint.",
)
parser.add_argument(
    "--streaming_response",
    default=False,
    action="store_true",
    help="Whether to enable streaming response.",
)
parser.add_argument(
    "--max_new_tokens", default=128, help="The maximum numbers of tokens to generate."
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
prompt = "Once upon a time,"
config: Dict[str, Union[int, float]] = {}
if args.max_new_tokens:
    config["max_new_tokens"] = int(args.max_new_tokens)
if args.temperature:
    config["temperature"] = float(args.temperature)
if args.top_p:
    config["top_p"] = float(args.top_p)
if args.top_k:
    config["top_k"] = float(args.top_k)

sample_input = SimpleRequest(text=prompt, config=config, stream=args.streaming_response)

proxies = {"http": None, "https": None}
outputs = requests.post(
    args.model_endpoint,
    proxies=proxies,  # type: ignore
    json=sample_input.dict(),
    stream=args.streaming_response,
)

outputs.raise_for_status()

simple_response = SimpleModelResponse.from_requests_response(outputs)
if args.streaming_response:
    for output in simple_response.iter_content(chunk_size=1, decode_unicode=True):
        print(output, end="", flush=True)
    print()
else:
    print(simple_response.text, flush=True)
