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

import openai
import os

# List all models.
models = openai.Model.list()
print(models)

# Note: not all arguments are currently supported and will be ignored by the backend.
model_name = os.getenv("MODEL_TO_SERVE", "gpt2")
chat_completion = openai.ChatCompletion.create(
    model=model_name,
    messages=[
      {"role": "assistant", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a long story with many words."}
    ],
    temperature=0.7,
    stream=False,
)
print(chat_completion)