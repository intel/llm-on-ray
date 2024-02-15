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

import json
import requests
import argparse

parser = argparse.ArgumentParser(
    description="Example script to query with http requests", add_help=True
)
parser.add_argument(
    "--request_api_base",
    default="http://localhost:8000/v1",
    type=str,
    help="Deployed model endpoint url",
)
parser.add_argument("--model_name", default="gpt2", type=str, help="The name of model to request")
parser.add_argument(
    "--streaming_response",
    default=False,
    action="store_true",
    help="Whether to enable streaming response",
)
parser.add_argument(
    "--max_new_tokens", default=256, help="The maximum numbers of tokens to generate"
)
parser.add_argument(
    "--temperature", default=0.2, help="The value used to modulate the next token probabilities"
)
parser.add_argument(
    "--top_p",
    default=0.7,
    help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`Top p` or higher are kept for generation",
)
parser.add_argument(
    "--input_text",
    default="Tell me a long story with many words.",
    help="question to ask model",
)
args = parser.parse_args()

s = requests.Session()
url = f"{args.request_api_base}/chat/completions"

body = {
    "model": args.model_name,
    "messages": [
        {"role": "user", "content": args.input_text},
    ],
    "stream": args.streaming_response,
    "max_tokens": args.max_new_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
}

proxies = {"http": None, "https": None}
response = s.post(url, json=body, proxies=proxies, stream=args.streaming_response)  # type: ignore
for chunk in response.iter_lines(decode_unicode=True):
    try:
        if chunk is not None:
            if args.streaming_response:
                # Get data from reponse chunk
                chunk_data = chunk.split("data: ")[1]
                if chunk_data != "[DONE]":
                    # Get message choices from data
                    choices = json.loads(chunk_data)["choices"]
                    # Pick content from first choice
                    content = choices[0]["delta"].get("content", "")
                    print(content, end="", flush=True)
            else:
                choices = json.loads(chunk)["choices"]
                content = choices[0]["message"].get("content", "")
                print(content, end="", flush=True)
    except Exception as e:
        print("chunk content: ", chunk)
        raise e
print()
