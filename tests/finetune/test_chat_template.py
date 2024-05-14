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
from transformers import AutoTokenizer
from llm_on_ray.common.dataprocesser.general_processer import ChatDataPreprocess


class TestTokenizeFunction(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.config = {
            "gpt_base_model": True,
            "max_length": 512,
            "trust_remote_code": False,
            "chat_template": "{% if messages[0]['role'] == 'system' %}"
                             "{% set loop_messages = messages[1:] %}"
                             "{% set system_message = messages[0]['content'] %}"
                             "{% else %}"
                             "{% set loop_messages = messages %}"
                             "{% set system_message = false %}"
                             "{% endif %}"
                             "{% for message in loop_messages %}"
                             "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
                             "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                             "{% endif %}"
                             "{% if loop.index0 == 0 and system_message %}"
                             "{{ system_message }}"
                             "{% endif %}"
                             "{% if message['role'] == 'user' %}"
                             "{{ '### Instruction: ' + message['content'] + eos_token }}"
                             "{% elif message['role'] == 'assistant' %}"
                             "{{ '### Response:'  + message['content'] + eos_token }}"
                             "{% endif %}{% endfor %}"
                             "{{'### End \n'}}",
        }
        self.processer = ChatDataPreprocess(self.config)

    def test_tokenize_function_with_gpt_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

        examples = {
            "instruction": "Test instruction",
            "response": "Test response",
            "context": "Test context",
        }

        # Verify the format of the result
        expected_result = (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n"
            "\n"
            "### Instruction: \n"
            "Test instruction\n"
            "\n"
            "Input: \n"
            "Test context\n"
            "\n"
            "### Response: \n"
            "Test response\n"
            "\n"
            "### End"
        )

        print(self.processer.create_data(examples))
        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(examples))
        print(self.tokenizer.decode(result["input_ids"]))

        self.assertEqual(expected_result, self.tokenizer.decode(result["input_ids"]))

    def test_tokenize_function_with_custom_chat_template(self):
        examples = {
            "instruction": "Test instruction",
            "response": "Test response",
            "context": "Test context",
        }

        # Verify the format of the result
        expected_result = (
            "<|im_start|>user\n"
            "Test instruction\n"
            "\n"
            "Input: Test context\n"
            "\n"
            "<|im_end|><|im_start|>assistant\n"
            "Test response\n"
            "\n"
            "<|im_end|>"
        )

        print(expected_result)
        # Set custom chat template
        self.config["chat_template"] = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'"
            "+ message['content'] + '<|im_end|>'}}{% endfor %}"
        )

        self.config["gpt_base_model"] = False
        print(self.processer.create_data(examples))
        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(examples))
        print(self.tokenizer.decode(result["input_ids"]))
        self.assertEqual(expected_result, self.tokenizer.decode(result["input_ids"]))

    def test_tokenize_function_with_default_chat_template(self):
        examples = {
            "instruction": "Test instruction",
            "response": "Test response",
            "context": "Test context",
        }

        # Verify the format of the result
        expected_result = (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request\n"
            "### Instruction: Test instruction\n"
            "\n"
            "Input: Test context\n"
            "\n"
            "### Response: Test response\n"
            "\n"
            "### End \n"
        )
        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(examples))
        self.assertEqual(expected_result, self.tokenizer.decode(result["input_ids"]))

    def test_tokenize_function_with_tokenizer_chat_template(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        examples = {
            "instruction": "Test instruction",
            "response": "Test response",
            "context": "Test context",
        }

        chat_example = [
            {
                "role": "user",
                "content": "Test instruction\n\nInput: Test context\n\n",
            },
            {
                "role": "assistant",
                "content": "Test response\n\n",
            },
        ]

        # Verify the format of the result
        expected_result = self.tokenizer.apply_chat_template(
            chat_example, tokenize=False, max_length=self.config.get("max_length")
        )

        self.config["chat_template"] = None
        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_func(self.tokenizer, self.processer.create_data(examples))
        self.assertEqual(expected_result, self.tokenizer.decode(result["input_ids"]))


if __name__ == "__main__":
    unittest.main()
