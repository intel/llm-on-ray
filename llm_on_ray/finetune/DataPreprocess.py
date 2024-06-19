import copy
import re
from itertools import chain

import torch

IGNORE_INDEX = -100


class AlpacaDataPreprocess:
    def __init__(self, eos_token):
        self.end = eos_token
        self.intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        self.instruction = "### Instruction:\n"
        self.input = "### Input:\n"
        self.response = "### Response:\n"

    def prompt(self, examples):
        prompts = {}
        prompts["prompt_sources"] = []
        prompts["prompt_targets"] = []
        for rec in examples:
            instruction = rec["instruction"]
            response = rec["response"]
            context = rec.get("context")
            if not instruction:
                raise ValueError(f"Expected an instruction in: {rec}")
            if not response:
                raise ValueError(f"Expected a response in: {rec}")
            if context:
                prompt = (
                    self.intro
                    + self.end
                    + "\n"
                    + self.instruction
                    + instruction
                    + self.input
                    + context
                    + self.end
                    + "\n"
                    + self.response
                )
                prompts["prompt_sources"].append(prompt)
            else:
                prompt = (
                    self.intro
                    + self.end
                    + "\n"
                    + self.instruction
                    + instruction
                    + self.end
                    + "\n"
                    + self.response
                )
                prompts["prompt_sources"].append(prompt)
            prompt_response = response + self.end
            prompts["prompt_targets"].append(prompt_response)
        return prompts

    def tokenize_func(self, tokenizer, config):
        padding_side = config["Dataset"].get("padding_side", "right")
        truncation_side = config["Dataset"].get("truncation_side", "right")
        max_length = max_source_length = config["Dataset"].get("max_length", 512)
        max_seq_length = config["Dataset"].get("max_seq_length", 1024)
        truncation = config["Dataset"].get("truncation", True)
        padding = config["Dataset"].get("padding", True)
        mask_input = config["Dataset"].get("mask_input", True)
        mask_response = config["Dataset"].get("mask_response", True)

        def truncate_sequences(sequences, max_length):
            """
            Copied from https://github.com/intel/intel-extension-for-transformers/blob/ae54f698b73a66e5729427cb19f69c33e1a5c34d/intel_extension_for_transformers/transformers/llm/finetuning/data_utils.py#L40
            """
            words_to_cut = sum(list(map(len, sequences))) - max_length
            if words_to_cut <= 0:
                return sequences

            while words_to_cut > 0 and len(sequences) > 0:
                words_to_cut -= len(sequences[0])
                sequences = sequences[1:]
            return sequences

        def preprocess_function_with_tokenize(examples):
            """
            Copied from https://github.com/intel/intel-extension-for-transformers/blob/ae54f698b73a66e5729427cb19f69c33e1a5c34d/intel_extension_for_transformers/transformers/llm/finetuning/data_utils.py#L225
            The only differences are:
            - using our own prompt style
            - add left or right padding and truncation
            - add mask_input and mask_response
            """
            keys = list(examples.data.keys())
            if len(keys) != 2:
                raise ValueError("Unsupported dataset format")
            assistant_tokens = tokenizer.tokenize(self.response)
            header = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                + self.end
                + "\n"
            )

            examples["input_ids"] = []
            examples["labels"] = []
            examples["attention_mask"] = []
            for instruction, response in zip(examples[keys[0]], examples[keys[1]]):
                convs = re.findall(
                    r"### Instruction.*?{0}|### Response.*?{0}".format(self.end),
                    instruction,
                    re.DOTALL,
                )
                convs_tokens = [
                    tokenizer.tokenize(conv) + tokenizer.tokenize("\n") for conv in convs
                ]
                header_tokens = tokenizer.tokenize(header) + tokenizer.tokenize("\n")
                max_input = max_source_length - len(header_tokens) - len(assistant_tokens)
                truncated_convs = truncate_sequences(convs_tokens, max_input)
                if len(truncated_convs) == 0:
                    truncated_convs = [convs_tokens[-1][: max_input - 3] + convs_tokens[-1][-3:]]

                prompt_tokens = [header_tokens] + truncated_convs + [assistant_tokens]
                prompt_ids = [
                    tokenizer.convert_tokens_to_ids(prompt_token) for prompt_token in prompt_tokens
                ]
                prompt_ids = list(chain(*prompt_ids))

                resp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response.strip()))
                # keep last and eos_id
                max_resp = max_seq_length - len(prompt_ids) - 1

                # truncating response
                if len(resp_ids) > max_resp:
                    if truncation_side == "right":
                        resp_ids = resp_ids[: max_resp - 1] + resp_ids[-1:]
                    else:
                        resp_ids = resp_ids[-max_resp:]

                # masking
                input_ids = prompt_ids + resp_ids + [tokenizer.eos_token_id]
                if mask_input:
                    labels = [IGNORE_INDEX] * len(prompt_ids) + resp_ids + [tokenizer.eos_token_id]
                elif mask_response:
                    labels = prompt_ids + [IGNORE_INDEX] * len(resp_ids) + [tokenizer.eos_token_id]
                else:
                    labels = input_ids

                # padding
                input_len = len(input_ids)
                pad_len = max_seq_length - input_len
                if padding_side == "right":
                    input_ids = input_ids + [tokenizer.eos_token_id] * pad_len
                    labels = labels + [IGNORE_INDEX] * pad_len
                    attention_mask = [1] * input_len + [0] * pad_len
                else:
                    input_ids = [tokenizer.eos_token_id] * pad_len + input_ids
                    labels = [IGNORE_INDEX] * pad_len + labels
                    attention_mask = [0] * pad_len + [1] * input_len

                assert len(input_ids) == max_seq_length
                assert len(prompt_ids) <= max_source_length
                assert len(labels) == len(input_ids) == len(attention_mask)

                examples["input_ids"].append(torch.tensor(input_ids))
                examples["labels"].append(labels)
                examples["attention_mask"].append(attention_mask)

            return examples

        def preprocess_function_with_tokenizer(examples):
            keys = list(examples.data.keys())
            if len(keys) != 2:
                raise ValueError("Unsupported dataset format")

            examples["input_ids"] = []
            examples["labels"] = []
            examples["attention_mask"] = []
            for s, t in zip(examples[keys[0]], examples[keys[1]]):
                results = tokenizer(
                    s + t,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=None,
                    max_length=max_length,
                )

                input_ids = results["input_ids"]
                input_len = len(input_ids)
                labels = copy.deepcopy(input_ids)
                if mask_input or mask_response:
                    sources_tokenized = tokenizer(
                        s,
                        padding=False,
                        truncation=True,
                        return_tensors=None,
                        max_length=max_length,
                    )
                    input_id_len = len(sources_tokenized["input_ids"])
                    # mask input
                    if mask_input:
                        labels[:input_id_len] = [IGNORE_INDEX] * input_id_len
                    # mask response
                    if mask_response:
                        labels[input_id_len:input_len] = [IGNORE_INDEX] * (input_len - input_id_len)

                examples["input_ids"].append(results["input_ids"])
                examples["labels"].append(labels)
                examples["attention_mask"].append(results["attention_mask"])
            return examples


        return preprocess_function_with_tokenize
