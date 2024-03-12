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
import os
from collections.abc import MutableMapping
import base64

if "WORKDIR" not in os.environ or not os.environ["WORKDIR"]:
    os.environ["WORKDIR"] = os.getcwd()

parser = argparse.ArgumentParser(
    description="Example script to query with Image through http requests", add_help=True
)
parser.add_argument(
    "--request_api_base",
    default="http://localhost:8000/v1/chat/completions",
    type=str,
    help="Deployed model endpoint url",
)
parser.add_argument(
    "--model_name", default="fuyu-8b", type=str, help="The name of model to request"
)
parser.add_argument(
    "--streaming_response",
    default=False,
    action="store_true",
    help="Whether to enable streaming response",
)
parser.add_argument(
    "--max_new_tokens", default=128, help="The maximum numbers of tokens to generate"
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
    "--image_path",
    default="https://www.adept.ai/images/blog/fuyu-8b/twitter_graph.png",
    help="input image for LLM to analyze.",
)
parser.add_argument(
    "--input_text",
    default="Hi! What is in this image?",
    help="input a question you want to ask about this image.",
)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ================================= #

args = parser.parse_args()
sess = requests.Session()

text_prompt = args.input_text
img_path = args.image_path

if os.path.exists(img_path):
    # Getting the base64 string
    base64_image = encode_image(img_path)
    image = f"data:image/png;base64,{base64_image}"
else:
    image = img_path

invoke_url = f"{args.request_api_base}"
headers = {
    "Content-Type": "application/json",
}
payload = {
    "model": args.model_name,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                },
            ],
        },
    ],
    "stream": args.streaming_response,
    "max_tokens": args.max_new_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
}

proxies: MutableMapping = {"http": None, "https": None}
response = sess.post(invoke_url, headers=headers, json=payload, proxies=proxies)

for idx, chunk in enumerate(response.iter_lines(decode_unicode=True)):
    if chunk is not None:
        if args.streaming_response:
            # Get data from reponse chunk
            import re

            chunk_data = re.sub("^data: ", "", chunk)
            if chunk_data != "[DONE]":
                # Get message choices from data
                choices = json.loads(chunk_data)["choices"]
                # Pick content from first choice
                content = choices[0]["delta"].get("content", "")
                print(content, end="", flush=True)
        else:
            try:
                choices = json.loads(chunk)["choices"]
                content = choices[0]["message"].get("content", "")
                print(content, end="", flush=True)
            except Exception:
                print(chunk)
print("")
