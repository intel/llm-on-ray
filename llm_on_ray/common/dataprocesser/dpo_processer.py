import datasets
import torch
from typing import Dict

from llm_on_ray.common.dataprocesser import DataProcesser

class DPOProcesser(DataProcesser):
    def tokenize_dataset(self, tokenizer, dataset):
        def return_prompt_and_responses(samples) -> Dict[str, str]:
            return {
                "prompt": [system + question for system, question in zip(samples["system"], samples["question"])],
                "chosen": samples["chosen"],
                "rejected": samples["rejected"],
            }

        if isinstance(dataset, datasets.Dataset):
            column_names = dataset.column_names

        if isinstance(dataset, datasets.DatasetDict):
            column_names = dataset["train"].column_names
        column_names = dataset["train"].column_names

        raw_datasets = dataset.map(
            return_prompt_and_responses,
            remove_columns=column_names,
        )

        train_dataset = raw_datasets["train"]
        column_names = train_dataset.column_names

        if train_dataset is not None:
            # Create train feature from dataset
            train_dataset = train_dataset.map(
                lambda examples: self.prepare_features(examples, tokenizer),
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

        eval_examples = raw_datasets["validation"]
        if eval_examples is not None:
            eval_dataset = eval_examples.map(
                lambda examples: self.prepare_features(examples, tokenizer),
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )


    def DPOCollateFn(self, batch, tokenizer):
        input_ids = [torch.tensor(ins["chosen_input_ids"]) for ins in batch] + \
                    [torch.tensor(ins["rejected_input_ids"]) for ins in batch]
        labels = [torch.tensor(ins["chosen_labels"]) for ins in batch] + \
                 [torch.tensor(ins["rejected_labels"]) for ins in batch]
        attention_mask = [torch.tensor(ins["chosen_attention_mask"]) for ins in batch] + \
                         [torch.tensor(ins["rejected_attention_mask"]) for ins in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def prepare_features(self, examples, tokenizer):

        prompts = {p.strip() for p in examples["prompt"]}
        chosens = {c.strip() for c in examples["chosen"]}
        rejects = {r.strip() for r in examples["rejected"]}

        examples = {
                "prompt": [],
                "chosen": [],
                "rejected": [],
                "chosen_response_only": [],
                "rejected_response_only": [],
                "chosen_input_ids": [],
                "chosen_attention_mask": [],
                "chosen_labels": [],
                "rejected_input_ids": [],
                "rejected_attention_mask": [],
                "rejected_labels": [],
                "prompt_input_ids": [],
                "prompt_attention_mask": []}

        for prompt, chosen, reject in zip(prompts, chosens, rejects):

            prompt_tokens = tokenizer.tokenize(prompt)

            if len(prompt_tokens) > data_args.max_prompt_length:
                prompt_tokens = prompt_tokens[:data_args.max_prompt_length]

            prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
            prompt_mask = [1] * len(prompt_ids)

            max_resp = data_args.max_length - len(prompt_ids)
            chosen_tokens = tokenizer.tokenize(chosen)
            chosen_tokens = chosen_tokens[:max_resp - 1]
            chosen_tokens.append(tokenizer.eos_token)
            chosen_ids = tokenizer.convert_tokens_to_ids(chosen_tokens)
            chosen_mask = [1] * len(chosen_ids)

            reject_tokens = tokenizer.tokenize(reject)
            reject_tokens = reject_tokens[:max_resp - 1]
            reject_tokens.append(tokenizer.eos_token)
            reject_ids = tokenizer.convert_tokens_to_ids(reject_tokens)
            reject_mask = [1] * len(reject_ids)

            chosen_input_ids = prompt_ids + chosen_ids
            chosen_attention_mask = prompt_mask + chosen_mask
            chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids

            reject_input_ids = prompt_ids + reject_ids
            reject_attention_mask = prompt_mask + reject_mask
            reject_labels = [IGNORE_INDEX] * len(prompt_ids) + reject_ids

            # padding
            input_len = len(chosen_input_ids)
            if data_args.pad_max:
                pad_len = data_args.max_length - input_len
                chosen_input_ids = chosen_input_ids + [0] * pad_len
                chosen_labels = chosen_labels + [-100] * pad_len
                chosen_attention_mask = chosen_attention_mask + [0] * pad_len
                assert len(chosen_input_ids) == data_args.max_length

            input_len = len(reject_input_ids)
            if data_args.pad_max:
                pad_len = data_args.max_length - input_len
                reject_input_ids = reject_input_ids + [0] * pad_len
                reject_labels = reject_labels + [-100] * pad_len
                reject_attention_mask = reject_attention_mask + [0] * pad_len
                assert len(reject_input_ids) == data_args.max_length

            examples["prompt"].append(prompt)
            examples["chosen"].append(prompt + chosen)
            examples["rejected"].append(prompt + reject)
            examples["chosen_response_only"].append(chosen)
            examples["rejected_response_only"].append(reject)

            examples["chosen_input_ids"].append(chosen_input_ids)
            examples["chosen_attention_mask"].append(chosen_attention_mask)
            examples["chosen_labels"].append(chosen_labels)

            examples["rejected_input_ids"].append(reject_input_ids)
            examples["rejected_attention_mask"].append(reject_attention_mask)
            examples["rejected_labels"].append(reject_labels)

            examples["prompt_input_ids"].append(prompt_ids)
            examples["prompt_attention_mask"].append(prompt_mask)

        return examples
