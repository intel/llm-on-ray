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
import unittest

import transformers
from datasets import Dataset
from transformers import AutoTokenizer
from llm_on_ray.common.dataprocesser.general_processer import (
    ChatDataPreprocess,
    SlimOrcaDataPreprocess,
)


class TestTokenizeFunction(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.config = {
            "gpt_base_model": True,
            "max_length": 512,
            "trust_remote_code": False,
            "chat_template": "Below is an instruction that describes a task. Write a response that appropriately "
            "completes the request\n {% if messages[0]['role'] == 'system' %}{{ raise_exception("
            "'System role not supported') }}{% endif %}{% for message in messages %}{% if (message["
            "'role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles "
            "must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] "
            "== 'user' %}{{ '### Instruction: ' + message['content'] }}{% elif message['role'] == "
            "'assistant' %}{{ '### Response: '  + message['content'] }}{% endif %}{% endfor %}{{'### "
            "End \n'}}",
        }
        self.processer = SlimOrcaDataPreprocess(self.config)
        examples = {
            "conversations": [
                {"from": "system", "value": "Test system", "weight": None},
                {"from": "human", "value": "Test human", "weight": 0},
                {"from": "gpt", "value": "Test gpt.", "weight": 1},
            ]
        }

        self.ds = Dataset.from_dict(examples)

    def test_tokenize_function_with_gpt_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

        # Verify the format of the result
        expected_result = (
            "### System: Test system \n" "### User: Test human \n" "### Assistant: Test gpt."
        )

        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(self.ds))

        self.assertEqual(expected_result, self.tokenizer.decode(result["input_ids"]))

    def test_tokenize_function_with_custom_chat_template(self):
        # Verify the format of the result
        expected_result = (
            "<|im_start|>system\n"
            "Test system\n"
            "<|im_end|><|im_start|>user\n"
            "Test human\n"
            "<|im_end|><|im_start|>assistant\n"
            "Test gpt.\n"
            "<|im_end|>"
        )

        print(expected_result)
        # Set custom chat template
        self.config["chat_template"] = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'"
            "+ message['content'] + '<|im_end|>'}}{% endfor %}"
        )

        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(self.ds))
        self.assertEqual(expected_result, self.tokenizer.decode(result["input_ids"]))

    def test_tokenize_function_with_default_chat_template(self):
        # Verify the format of the result
        expected_result = (
            "### System: Test system\n" "### User: Test human\n" "### Assistant: Test gpt.\n"
        )
        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(self.ds))
        self.assertEqual(expected_result, self.tokenizer.decode(result["input_ids"]))

    def test_tokenize_function_with_tokenizer_chat_template(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        chat_example = [
            {
                "role": "system",
                "content": "Test system\n",
            },
            {
                "role": "user",
                "content": "Test human\n",
            },
            {
                "role": "assistant",
                "content": "Test gpt.\n",
            },
        ]

        # Verify the format of the result
        expected_result = self.tokenizer.apply_chat_template(
            chat_example, tokenize=True, max_length=self.config.get("max_length")
        )

        self.config["chat_template"] = None
        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(self.ds))
        self.assertEqual(expected_result, result["input_ids"])


if __name__ == "__main__":
    unittest.main()
