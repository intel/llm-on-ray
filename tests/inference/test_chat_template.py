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
import pathlib

import pytest
from transformers import AutoTokenizer

from llm_on_ray.inference.utils import parse_jinja_file

# Define the base path for templates
base_path = (
    pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    / "llm_on_ray/inference/models/templates"
)


# Define models, templates, and their corresponding expected outputs
MODEL_TEMPLATE_GENERATON_OUTPUT = [
    (
        "EleutherAI/gpt-j-6b",
        base_path / "default_template.jinja",
        True,
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n"
        "### Instruction: Hello\n"
        "### Response:Hi there!\n"
        "### Response:\n",
    ),
    (
        "EleutherAI/gpt-j-6b",
        base_path / "default_template.jinja",
        False,
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n"
        "### Instruction: Hello\n"
        "### Response:Hi there!\n",
    ),
    ("gpt2", base_path / "template_gpt2.jinja", True, "Hello\nHi there!\nWhat is the capital of\n"),
    (
        "gpt2",
        base_path / "template_gpt2.jinja",
        False,
        "Hello\nHi there!\nWhat is the capital of\n",
    ),
    (
        "mistralai/Mistral-7B-v0.1",
        base_path / "template_mistral.jinja",
        True,
        "<s>\n"
        "[INST] Hello [/INST]\n"
        "Hi there!</s>\n"
        "[INST] What is the capital of [/INST]\n",
    ),
    (
        "mistralai/Mistral-7B-v0.1",
        base_path / "template_mistral.jinja",
        False,
        "<s>\n"
        "[INST] Hello [/INST]\n"
        "Hi there!</s>\n"
        "[INST] What is the capital of [/INST]\n",
    ),
    (
        "Intel/neural-chat-7b-v3-1",
        base_path / "template_neuralchat.jinja",
        True,
        "###System: You are a chatbot developed by Intel. Please answer all "
        "questions to the best of your ability.\n"
        "###User: Hello\n"
        "###Assistant: Hi there!\n"
        "###Assistant: \n",
    ),
    (
        "Intel/neural-chat-7b-v3-1",
        base_path / "template_neuralchat.jinja",
        False,
        "###System: You are a chatbot developed by Intel. Please answer all "
        "questions to the best of your ability.\n"
        "###User: Hello\n"
        "###Assistant: Hi there!\n",
    ),
    (
        "adept/fuyu-8b",
        base_path / "template_llama2.jinja",
        True,
        "|ENDOFTEXT|[INST] Hello [/INST]\n"
        " Hi there! |ENDOFTEXT|\n"
        "|ENDOFTEXT|[INST] What is the capital of [/INST]\n",
    ),
    (
        "adept/fuyu-8b",
        base_path / "template_llama2.jinja",
        False,
        "|ENDOFTEXT|[INST] Hello [/INST]\n"
        " Hi there! |ENDOFTEXT|\n"
        "|ENDOFTEXT|[INST] What is the capital of [/INST]\n",
    ),
    (
        "codellama/CodeLlama-7b-hf",
        base_path / "template_codellama.jinja",
        True,
        "<s>[INST] Hello [/INST]\n"
        " Hi there! </s>\n"
        "<s>[INST] What is the capital of [/INST]\n",
    ),
    (
        "codellama/CodeLlama-7b-hf",
        base_path / "template_codellama.jinja",
        False,
        "<s>[INST] Hello [/INST]\n"
        " Hi there! </s>\n"
        "<s>[INST] What is the capital of [/INST]\n",
    ),
]


TEST_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is the capital of"},
]

TEST_NEURALCHAT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a chatbot developed by Intel. Please answer all questions to the best of your ability.",
    },
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]

TEST_DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    },
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]


@pytest.mark.parametrize(
    "model,template,add_generation_prompt,expected_output", MODEL_TEMPLATE_GENERATON_OUTPUT
)
def test_get_gen_default_prompt(
    model: object, template: object, add_generation_prompt: object, expected_output: object
) -> object:
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.chat_template = parse_jinja_file(template)

    if model == "mistralai/Mistral-7B-v0.1" or model == "codellama/CodeLlama-7b-hf":
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
    elif model == "adept/fuyu-8b":
        tokenizer.bos_token = "|ENDOFTEXT|"
        tokenizer.eos_token = "|ENDOFTEXT|"

    # Call the function and get the result
    if model == "Intel/neural-chat-7b-v3-1":
        result = tokenizer.apply_chat_template(
            conversation=TEST_NEURALCHAT_MESSAGES,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    elif model == "EleutherAI/gpt-j-6b":
        result = tokenizer.apply_chat_template(
            conversation=TEST_DEFAULT_MESSAGES,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        result = tokenizer.apply_chat_template(
            conversation=TEST_MESSAGES, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    # Test assertion
    assert result == expected_output, (
        f"The generated prompt does not match the expected output for "
        f"model {model} and template {template}"
    )
