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

import asyncio
import aiohttp
import argparse
from typing import Dict, Union
from llm_on_ray.inference.api_simple_backend.simple_protocol import (
    SimpleRequest,
    SimpleModelResponse,
)

parser = argparse.ArgumentParser(
    description="Example script to query with multiple requests", add_help=True
)
parser.add_argument(
    "--model_endpoint",
    default="http://127.0.0.1:8000",
    type=str,
    help="Deployed model endpoint.",
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

config: Dict[str, Union[int, float]] = {}
if args.max_new_tokens:
    config["max_new_tokens"] = int(args.max_new_tokens)
if args.temperature:
    config["temperature"] = float(args.temperature)
if args.top_p:
    config["top_p"] = float(args.top_p)
if args.top_k:
    config["top_k"] = float(args.top_k)


async def send_query(session, endpoint, req):
    async with session.post(endpoint, json=req.dict()) as resp:
        return await resp.text()


prompts = [
    "Once upon a time,",
    "Hi my name is Lewis and I like to",
    "My name is Mary, and my favorite",
    "My name is Clara and I am",
    "My name is Julien and I like to",
    "Today I accidentally",
    "My greatest wish is to",
    "In a galaxy far far away",
    "My best talent is",
]

config1 = {"max_new_tokens": 20}
config2 = {"max_new_tokens": 10}

configs = [config1] * 5 + [config2] * (len(prompts) - 5)

reqs = [SimpleRequest(text=prompt, config=config) for prompt, config in zip(prompts, configs)]


async def send_all_query(endpoint, reqs):
    async with aiohttp.ClientSession() as session:
        tasks = [send_query(session, endpoint, req) for req in reqs]
        responses = await asyncio.gather(*tasks)
        print("\n--------------\n".join(responses))
        print("\nTotal responses:", len(responses))


asyncio.run(send_all_query(args.model_endpoint, reqs))
