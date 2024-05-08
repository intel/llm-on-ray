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

default_jinja_path = (
    pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    / "llm_on_ray/common/templates/default_template.jinja"
)
assert default_jinja_path.exists()

gpt2_jinja_path = (
    pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    / "llm_on_ray/common/templates/template_gpt2.jinja"
)
assert gpt2_jinja_path.exists()

gemma_jinja_path = (
    pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    / "llm_on_ray/common/templates/template_gemma.jinja"
)
assert gemma_jinja_path.exists()

mistral_jinja_path = (
    pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    / "llm_on_ray/common/templates/template_mistral.jinja"
)
assert mistral_jinja_path.exists()

neural_chat_jinja_path = (
    pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    / "llm_on_ray/common/templates/template_neuralchat.jinja"
)
assert neural_chat_jinja_path.exists()


llama2_jinja_path = (
    pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
    / "llm_on_ray/common/templates/template_llama2.jinja"
)
assert llama2_jinja_path.exists()


# Define models, templates, and their corresponding expected outputs
MODEL_TEMPLATE_GENERATON_OUTPUT = [
    (
        "EleutherAI/gpt-j-6b",
        default_jinja_path,
        True,
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n"
        "### Instruction: Hello### Response:Hi there!### Instruction: What is the capital of### Response:\n",
    ),
    (
        "EleutherAI/gpt-j-6b",
        default_jinja_path,
        False,
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n"
        "### Instruction: Hello### Response:Hi there!### Instruction: What is the capital of",
    ),
    ("gpt2", gpt2_jinja_path, True, "Hello\nHi there!\nWhat is the capital of\n"),
    ("gpt2", gpt2_jinja_path, False, "Hello\nHi there!\nWhat is the capital of\n"),
    (
        "google/gemma-2b",
        gemma_jinja_path,
        True,
        "<start_of_turn>user\n"
        "Hello<end_of_turn>\n"
        "<start_of_turn>model\n"
        "Hi there!<end_of_turn>\n"
        "<start_of_turn>user\n"
        "What is the capital of<end_of_turn>\n"
        "<start_of_turn>model\n",
    ),
    (
        "google/gemma-2b",
        gemma_jinja_path,
        False,
        "<start_of_turn>user\n"
        "Hello<end_of_turn>\n"
        "<start_of_turn>model\n"
        "Hi there!<end_of_turn>\n"
        "<start_of_turn>user\n"
        "What is the capital of<end_of_turn>\n",
    ),
    (
        "mistralai/Mistral-7B-v0.1",
        mistral_jinja_path,
        True,
        "<s>\n"
        "[INST] Hello [/INST]\n"
        "Hi there!</s>\n"
        "[INST] What is the capital of [/INST]\n",
    ),
    (
        "mistralai/Mistral-7B-v0.1",
        mistral_jinja_path,
        False,
        "<s>\n"
        "[INST] Hello [/INST]\n"
        "Hi there!</s>\n"
        "[INST] What is the capital of [/INST]\n",
    ),
    (
        "Intel/neural-chat-7b-v3-1",
        neural_chat_jinja_path,
        True,
        "'### System:You are a chatbot developed by Intel. Please answer all "
        "questions to the best of your ability.\\n'\n"
        "### User: Hello\n"
        "### Assistant:Hi there!\n"
        "### User: What is the capital of\n"
        "### Assistant:\n"
        "\n",
    ),
    (
        "Intel/neural-chat-7b-v3-1",
        neural_chat_jinja_path,
        False,
        "'### System:You are a chatbot developed by Intel. Please answer all "
        "questions to the best of your ability.\\n'\n"
        "### User: Hello\n"
        "### Assistant:Hi there!\n"
        "### User: What is the capital of\n",
    ),
    (
        "adept/fuyu-8b",
        llama2_jinja_path,
        True,
        "|ENDOFTEXT|[INST] Hello [/INST] Hi there! |ENDOFTEXT||ENDOFTEXT|[INST] What is the capital of [/INST]",
    ),
    (
        "adept/fuyu-8b",
        llama2_jinja_path,
        False,
        "|ENDOFTEXT|[INST] Hello [/INST] Hi there! |ENDOFTEXT||ENDOFTEXT|[INST] What is the capital of [/INST]",
    ),
]


TEST_MESSAGES = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is the capital of"},
]


@pytest.mark.parametrize(
    "model,template,add_generation_prompt,expected_output", MODEL_TEMPLATE_GENERATON_OUTPUT
)
def test_get_gen_default_prompt(
    model: object, template: object, add_generation_prompt: object, expected_output: object
) -> object:
    print(template)
    print(model)
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(tokenizer)
    tokenizer.chat_template = parse_jinja_file(template)

    # Call the function and get the result
    result = tokenizer.apply_chat_template(
        conversation=TEST_MESSAGES, tokenize=False, add_generation_prompt=add_generation_prompt
    )

    # Test assertion
    assert result == expected_output, (
        f"The generated prompt does not match the expected output for "
        f"model {model} and template {template}"
    )
