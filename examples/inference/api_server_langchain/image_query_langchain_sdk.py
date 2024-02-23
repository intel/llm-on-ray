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
import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


parser = argparse.ArgumentParser(
    description="Example script to query with langchain sdk", add_help=True
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
    "--image_path",
    default="https://www.adept.ai/images/blog/fuyu-8b/twitter_graph.png",
    help="input image for LLM to analyze.",
)
parser.add_argument(
    "--input_text",
    default="Hi! What is in this image?",
    help="input a question you want to ask about this image.",
)
parser.add_argument("--max_tokens", default=256, help="The maximum numbers of tokens to generate")

# ==================================== #
args = parser.parse_args()

text_prompt = args.input_text
img_path = args.image_path

if os.path.exists(img_path):
    # Getting the base64 string
    base64_image = encode_image(img_path)
    image = f"data:image/png;base64,{base64_image}"
else:
    image = img_path

llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    model_name=args.model_name,
    openai_api_key="not_needed",
    streaming=args.streaming_response,
    max_tokens=args.max_tokens,
)

message = [
    HumanMessage(
        content=[
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {"url": image},
            },
        ]
    )
]


if args.streaming_response:
    output = llm.stream(message)
    for chunk in output:
        print(chunk.content, end="", flush=True)
else:
    output = llm.invoke(message)
    print(output)
