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
from llm_on_ray.common.dataprocesser.general_processer import GeneralProcesser


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
        self.processer = GeneralProcesser(self.config)

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
            "### Instruction:\n"
            "Test instruction\n"
            "\n"
            "Input:\n"
            "Test context\n"
            "\n"
            "### Response:\n"
            "Test response\n"
            "\n"
            "### End"
        )

        result = self.processer.tokenize_function(examples, self.tokenizer)
        self.assertEqual(self.tokenizer.decode(result["input_ids"]), expected_result)

    def test_tokenize_function_with_custom_chat_template(self):
        examples = {
            "instruction": "Test instruction",
            "response": "Test response",
            "context": "Test context",
        }

        # Verify the format of the result
        expected_result = (
            "<|im_start|>user\n"
            "###Instruction:\n"
            "Test instruction\n"
            "\n"
            "###context:\n"
            "Test context\n"
            "\n"
            "<|im_end|><|im_start|>assistant\n"
            "Test response\n"
            "\n"
            "<|im_end|>"
        )
        # Set custom chat template
        self.config["custom_chat_template"] = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'"
            "+ message['content'] + '<|im_end|>'}}{% endfor %}"
        )

        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_function(examples, self.tokenizer)
        self.assertEqual(self.tokenizer.decode(result["input_ids"]), expected_result)

    def test_tokenize_function_with_chat_template(self):
        examples = {
            "instruction": "Test instruction",
            "response": "Test response",
            "context": "Test context",
        }

        # Verify the format of the result
        expected_result = (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request\n"
            "### Instruction: ###Instruction:\n"
            "Test instruction\n"
            "\n"
            "###context:\n"
            "Test context\n"
            "\n"
            "### Response: Test response\n"
            "\n"
            "### End \n"
        )
        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_function(examples, self.tokenizer)
        self.assertEqual(self.tokenizer.decode(result["input_ids"]), expected_result)

    def test_tokenize_function_with_default_chat_template(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        examples = {
            "instruction": "Test instruction",
            "response": "Test response",
            "context": "Test context",
        }

        chat_example = [
            {
                "role": "user",
                "content": "###Instruction:\nTest instruction\n\n###context:\nTest context\n\n",
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

        self.config["gpt_base_model"] = False
        result = self.processer.tokenize_function(examples, self.tokenizer)
        self.assertEqual(self.tokenizer.decode(result["input_ids"]), expected_result)


if __name__ == "__main__":
    unittest.main()
