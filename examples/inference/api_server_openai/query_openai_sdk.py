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

client = OpenAI()
# # List all models.
models = client.models.list()
print(models.data, "\n")

# Note: not all arguments are currently supported and will be ignored by the backend.
chat_completion = client.chat.completions.create(
    model=args.model_name,
    messages=[
        {"role": "assistant", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a long story with many words."},
    ],
    stream=args.streaming_response,
    max_tokens=args.max_new_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
)
print(chat_completion)
