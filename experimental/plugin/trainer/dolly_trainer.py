from .trainer import Trainer
from itertools import chain
import torch
import transformers
import math
import time
'''

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42
PROMPT_NO_INPUT_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
{response}
{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
PROMPT_WITH_INPUT_FORMAT = """{intro}
{instruction_key}
{instruction}
{input_key}
{input}
{response_key}
{response}
{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    context = rec.get("context")

    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")

    if not response:
        raise ValueError(f"Expected a response in: {rec}")

    # For some instructions there is an input that goes along with the instruction, providing context for the
    # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
    # some piece of information from it.  The response is that information to extract.  In other cases there is
    # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
    # born.
    if context:
        rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
    else:
        rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
    return rec
'''
class DollyTrainer(Trainer):
    def __init__(self, model, tokenizer, dataset, optimizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.optimizer = optimizer
        self.config = config

    def prepare(self):
        self.modify_embedding_size()
        self.train_dataloader, self.eval_dataloader = self.prepare_dataset()
        self.lr_scheduler = self.get_lr_scheduler()
        return self.model, self.optimizer, self.tokenizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler

    def modify_embedding_size(self):
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def get_lr_scheduler(self):
        overrode_max_train_steps = False
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        max_train_steps  = self.config.get("max_train_steps")
        num_train_epochs = self.config.get("num_train_epochs", 1)
        lr_scheduler_type = self.config.get("lr_scheduler_type", "linear")

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True


        num_warmup_steps = self.config.get("num_warmup_steps", 0)
        lr_scheduler = transformers.get_scheduler(
            name=lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        return lr_scheduler


    def prepare_dataset(self):
        column_names = self.dataset["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return self.tokenizer(examples[text_column_name])

        preprocessing_num_workers = self.config.get("preprocessing_num_workers", 1)
        overwrite_cache = self.config.get("overwrite_cache", True)

        tokenized_datasets = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        block_size = self.config.get("block_size")
        if block_size is None:
            block_size = self.tokenizer.model_max_length
            # if block_size > 1024:
            #     logger.warning(
            #         f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            #         "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            #     )
            block_size = 1024
        else:
            if block_size > self.tokenizer.model_max_length:
                # logger.warning(
                #     f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                #     f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                # )
                pass
            block_size = min(block_size, self.tokenizer.model_max_length)

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 2)
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 4)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            collate_fn=transformers.default_data_collator, 
            batch_size=per_device_train_batch_size
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, 
            collate_fn=transformers.default_data_collator, 
            batch_size=per_device_eval_batch_size
        )
        return train_dataloader, eval_dataloader
    def train(self, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, accelerator):
        for epoch in range(self.config.get("epoch", 1)):
            model.train()
            start = time.time()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    print("step:%s/%s loss:%s ppl:%s time:%s"%(step,len(train_dataloader),loss, math.exp(loss), time.time()-start))
                    start = time.time()

            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            print(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
