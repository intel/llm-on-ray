from .trainer import Trainer
from itertools import chain
import torch
import transformers
import math
import time
from ray.air import session
from ray.air.checkpoint import Checkpoint

class DefaultTrainer(Trainer):
    def __init__(self, config):
        self.config = config

    def recovery(self, checkpoint_path, model, optimizer):
        rank_num = session.get_world_rank()
        local_checkpoint_path = os.path.join(checkpoint_path, "rank_"+str(rank_num))
        print("checkpoint_path", checkpoint_path)
        resume_from_checkpoint = False
        try:
            checkpoint_dict = Checkpoint.from_uri(checkpoint_path).to_dict()
            resume_from_checkpoint = True
        except Exception:
            print("Training from scratch.")
            pass
        if resume_from_checkpoint:
            # update model status
            model_state = checkpoint_dict["model"]
            model.load_state_dict(model_state)
            # update optimizer status
            optimizer_state = checkpoint_dict["optimizer_state_dict"]
            optimizer.load_state_dict(optimizer_state)
            # update lr_scheduler status
            scheduler_state = checkpoint_dict["lr_scheduler"]
            lr_scheduler.load_state_dict(scheduler_state)
            # update current epoch
            checkpoint_epoch = checkpoint_dict["epoch"]
            starting_epoch = checkpoint_epoch + 1
            # update the progress_bar if load from checkpoint
            progress_bar.update(starting_epoch * num_update_steps_per_epoch)
            completed_steps = starting_epoch * num_update_steps_per_epoch

    def prepare(self, model, tokenizer, dataset, optimizer, accelerator):
        self.accelerator = accelerator 
        checkpoint_path = self.config.get("checkpoint_path")
        if checkpoint_path is not None:
            model, optimizer = self.recovery(checkpoint_path, model, optimizer)

        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        train_dataloader, eval_dataloader = self.prepare_dataset(dataset, tokenizer)
        lr_scheduler = self.get_lr_scheduler(optimizer, train_dataloader)
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

    def get_lr_scheduler(self, optimizer, train_dataloader):
        overrode_max_train_steps = False
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        max_train_steps  = self.config.get("max_train_steps")
        num_train_epochs = self.config.get("num_train_epochs", 1)
        lr_scheduler_type = self.config.get("lr_scheduler_type", "linear")

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True


        num_warmup_steps = self.config.get("num_warmup_steps", 0)
        lr_scheduler = transformers.get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps * gradient_accumulation_steps,
        )
        return lr_scheduler

    def prepare_dataset(self, dataset, tokenizer):
        column_names = dataset["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        preprocessing_num_workers = self.config.get("preprocessing_num_workers", 1)
        overwrite_cache = self.config.get("overwrite_cache", True)

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        block_size = self.config.get("block_size")
        if block_size is None:
            block_size = tokenizer.model_max_length
            # if block_size > 1024:
            #     logger.warning(
            #         f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            #         "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            #     )
            block_size = 1024
        else:
            if block_size > tokenizer.model_max_length:
                # logger.warning(
                #     f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                #     f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                # )
                pass
            block_size = min(block_size, tokenizer.model_max_length)

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

    def train(self):
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 4)
        for epoch in range(self.config.get("epoch", 1)):
            self.model.train()
            start = time.time()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    print("step:%s/%s loss:%s ppl:%s time:%s"%(step,len(self.train_dataloader),loss, math.exp(loss), time.time()-start))
                    start = time.time()

            self.model.eval()
            losses = []
            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)

                loss = outputs.loss
                losses.append(self.accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            print(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            if checkpoint_hdfs_path is not None:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                checkpoint = Checkpoint.from_dict(
                    {
                        "epoch": epoch,
                        "rank": rank_num,
                        "model": unwrapped_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    }
                )
                Checkpoint.to_uri(checkpoint, checkpoint_path)
