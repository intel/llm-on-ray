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

import os
import json
import requests

s = requests.Session()

api_base = os.getenv("ENDPOINT_URL")
if api_base is None:
    api_base = "http://localhost:8000/v1"
url = f"{api_base}/chat/completions"

model_name = os.getenv("MODEL_TO_SERVE", "gpt2")
body = {
  "model": model_name,
  "messages": [
    {"role": "assistant", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a long story with many words."}
  ],
  "temperature": 1.0,
  "stream": True,
}

proxies = { "http": None, "https": None}
with s.post(url, json=body, proxies=proxies) as response:
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk is not None:
            try:
                # Get data from reponse chunk
                chunk_data = chunk.split("data: ")[1]

                # Get message choices from data
                choices = json.loads(chunk_data)["choices"]

                # Pick content from first choice
                content = choices[0]["delta"]["content"]

                print(content, end="", flush=True)
            except json.decoder.JSONDecodeError:
                # Chunk was not formatted as expected
                pass
            except KeyError:
                # No message was contained in the chunk
                pass
            except:
                pass
    print("")