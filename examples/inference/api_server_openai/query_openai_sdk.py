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
from openai import OpenAI
import os

parser = argparse.ArgumentParser(
    description="Example script to query with openai sdk", add_help=True
)
parser.add_argument("--model_name", default="gpt2", type=str, help="The name of model to request")
parser.add_argument(
    "--streaming_response",
    default=False,
    action="store_true",
    help="Whether to enable streaming response",
)
parser.add_argument(
    "--max_new_tokens", default=None, help="The maximum numbers of tokens to generate"
)
parser.add_argument(
    "--temperature", default=None, help="The value used to modulate the next token probabilities"
)
parser.add_argument(
    "--top_p",
    default=None,
    help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`Top p` or higher are kept for generation",
)

args = parser.parse_args()

if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.environ["OPENAI_API_KEY"]
else:
    openai_api_key = "not_needed"

if "OPENAI_BASE_URL" in os.environ:
    openai_base_url = os.environ["OPENAI_BASE_URL"]
elif openai_api_key == "not_needed":
    openai_base_url = "http://localhost:8000/v1"
else:
    openai_base_url = "https://api.openai.com/v1"

client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)


def stream_chat():
    for chunk in client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": "What is better sorting algorithm than quick sort?"}],
        stream=True,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content


def chunk_chat():
    output = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "user", "content": "What is better sorting algorithm than quick sort?"},
        ],
        stream=False,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    for chunk in [output]:
        try:
            content = chunk.choices[0].message.content
            if content is not None:
                yield content
        except Exception:
            print(chunk)


if args.streaming_response:
    for i in stream_chat():
        print(i, end="", flush=True)
else:
    for i in chunk_chat():
        print(i, end="", flush=True)
print("")
