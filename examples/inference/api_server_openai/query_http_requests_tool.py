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
    "--max_new_tokens", default=128, help="The maximum numbers of tokens to generate"
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

s = requests.Session()
url = f"{args.request_api_base}/chat/completions"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

messages = [
    [
        {"role": "user", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What's the weather like in Boston today?"},
    ],
]

proxies = {"http": None, "https": None}

for message in messages:
    print(f"User: {message[1]['content']}")
    print("Assistant:", end=" ", flush=True)

    body = {
        "model": args.model_name,
        "messages": message,
        "tools": tools,
        "tool_choice": "auto",
        "stream": args.streaming_response,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    response = s.post(url, json=body, proxies=proxies, stream=args.streaming_response)  # type: ignore
    output_tool_call = []
    for chunk in response.iter_lines(decode_unicode=True):
        try:
            if chunk is not None and chunk != "":
                if args.streaming_response:
                    # Get data from reponse chunk
                    chunk_data = chunk.split("data: ")[1]
                    if chunk_data != "[DONE]":
                        # Get message choices from data
                        choices = json.loads(chunk_data)["choices"]
                        # Pick content from first choice
                        content = choices[0]["delta"].get("content", "")
                        if content is not None and content != "":
                            print(content, end="", flush=True)
                        tool_calls = choices[0]["delta"].get("tool_calls", "")
                        if tool_calls is not None and tool_calls != "":
                            print("tool_calls:", tool_calls, end="", flush=True)
                            output_tool_call = tool_calls
                else:
                    choices = json.loads(chunk)["choices"]
                    content = choices[0]["message"]["content"]
                    if content is not None:
                        print(content, end="", flush=True)
                    tool_calls = choices[0]["message"]["tool_calls"]
                    if tool_calls is not None:
                        print("tool_calls:", tool_calls, end="", flush=True)
                        output_tool_call = tool_calls
        except Exception as e:
            print("chunk content: ", chunk)
            raise e
    assert output_tool_call[0]["function"]["name"] == "get_current_weather"
    print()
