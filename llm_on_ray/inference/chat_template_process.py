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
from typing import List, Union

from llm_on_ray.inference.api_openai_backend.openai_protocol import ChatMessage


class ChatTemplatePreprocess:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def get_prompt(self, input: List, is_mllm=False):
        """Generate response based on input."""
        if self.predictor.infer_conf.model_description.chat_template is not None:
            self.predictor.tokenizer.chat_template = (
                self.predictor.infer_conf.model_description.chat_template
            )
        elif self.predictor.tokenizer.chat_template is None:
            self.predictor.tokenizer.chat_template = (
                self.predictor.infer_conf.model_description.default_chat_template
            )

        if is_mllm:
            if isinstance(input, List):
                if isinstance(input, list) and input and isinstance(input[0], ChatMessage):
                    messages = []
                    for chat_message in input:
                        message = {
                            "role": chat_message.role,
                            "content": chat_message.content,
                        }
                        messages.append(message)
                    texts, images = self._extract_messages(messages)
                elif isinstance(input, list) and input and isinstance(input[0], dict):
                    texts, images = self._extract_messages(input)
                elif isinstance(input, list) and input and isinstance(input[0], list):
                    texts, images = [self._extract_messages(p) for p in input]

                image = self._prepare_image(images)
                prompt = self.predictor.tokenizer.apply_chat_template(texts, tokenize=False)
                return prompt, image
        else:
            if isinstance(input, list) and input and isinstance(input[0], dict):
                prompt = self.predictor.tokenizer.apply_chat_template(input, tokenize=False)
            elif isinstance(input, list) and input and isinstance(input[0], list):
                prompt = [
                    self.predictor.tokenizer.apply_chat_template(t, tokenize=False)
                    for t in input
                ]
            elif isinstance(input, list) and input and isinstance(input[0], ChatMessage):
                messages = []
                for chat_message in input:
                    message = {"role": chat_message.role, "content": chat_message.content}
                    messages.append(message)
                prompt = self.predictor.tokenizer.apply_chat_template(
                    messages, tokenize=False
                )
            elif isinstance(input, list) and input and isinstance(input[0], str):
                prompt = input
            elif isinstance(input, str):
                prompt = input
            else:
                raise TypeError(
                    f"Unsupported type {type(input)} for text. Expected dict or list of dicts."
                )
        return prompt

    def _extract_messages(self, messages):
        texts, images = [], []
        for message in messages:
            if message["role"] == "user" and isinstance(message["content"], list):
                texts.append({"role": "user", "content": message["content"][0]["text"]})
                images.append(
                    {"role": "user", "content": message["content"][1]["image_url"]["url"]}
                )
            else:
                texts.append(message)
        return texts, images

    def _prepare_image(self, messages: list):
        """Prepare image from history messages."""
        from PIL import Image
        import requests
        from io import BytesIO
        import base64
        import re

        # prepare images
        images: List = []
        if isinstance(messages[0], List):
            for i in range(len(messages)):
                for msg in messages[i]:
                    msg = dict(msg)
                    content = msg["content"]
                    if "url" not in content:
                        continue
                    is_data = len(re.findall("^data:image/.+;base64,", content["url"])) > 0
                    if is_data:
                        encoded_str = re.sub("^data:image/.+;base64,", "", content["url"])
                        images[i].append(Image.open(BytesIO(base64.b64decode(encoded_str))))
                    else:
                        images[i].append(Image.open(requests.get(content["url"], stream=True).raw))
        elif isinstance(messages[0], dict):
            for msg in messages:
                msg = dict(msg)
                content = msg["content"]
                if "url" not in content:
                    continue
                is_data = len(re.findall("^data:image/.+;base64,", content["url"])) > 0
                if is_data:
                    encoded_str = re.sub("^data:image/.+;base64,", "", content["url"])
                    images.append(Image.open(BytesIO(base64.b64decode(encoded_str))))
                else:
                    images.append(Image.open(requests.get(content["url"], stream=True).raw))

        return images