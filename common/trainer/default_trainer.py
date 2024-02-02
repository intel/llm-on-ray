import os
import math
import time
import tempfile
from pathlib import Path

import torch
import transformers
from accelerate.utils import DummyOptim, DummyScheduler

from ray.train import report, Checkpoint
from transformers import get_scheduler

from .. import dataprocesser
from .trainer import Trainer

from ..logging import logger


class DefaultTrainer(Trainer):
    def __init__(self, config):
        self.config = config
        dataprocesser_config = config.get("dataprocesser")
        dataprocesser_type = dataprocesser_config.get("type")
        Factory = dataprocesser.DataProcesser.registory.get(dataprocesser_type)
        if Factory is None:
            raise ValueError(f"there is no {dataprocesser_type} dataprocesser.")
        self.dataprocesser = Factory(dataprocesser_config)
        self.starting_epoch = 0

    def recovery(self, config):
        if config is None or config is {}:
            logger.warning("checkpoint is empty, skip")
            return
        root_path = config.get("root_path")
        model_name = config.get("model_name", "")
        if root_path is None:
            logger.warning("checkpoint root_path is empty, skip")
        local_checkpoint_path = self._get_local_path(root_path, model_name)
        try:
            logger.info(f"start recovery from {local_checkpoint_path}")
            checkpoint = Checkpoint.from_directory(local_checkpoint_path)
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                # update model status
                model_state = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
                self.model.load_state_dict(model_state)

                # update optimizer status
                optimizer_state = torch.load(checkpoint_dir / "optim.pt", map_location="cpu")
                self.optimizer.load_state_dict(optimizer_state)

                # update lr_scheduler status
                if Path.exists(checkpoint_dir / "lr_scheduler.pt") and hasattr(
                    self, "lr_scheduler"
                ):
                    scheduler_state = torch.load(
                        checkpoint_dir / "lr_scheduler.pt", map_location="cpu"
                    )
                    self.lr_scheduler.load_state_dict(scheduler_state)

                # update current epoch
                checkpoint_epoch = torch.load(checkpoint_dir / "epoch.pt", map_location="cpu")
                self.starting_epoch = checkpoint_epoch["epoch"] + 1

            logger.info(f"recovery to epoch {self.starting_epoch}")
        except FileNotFoundError as e:
            logger.info(e)
        except Exception:
            logger.warning("recovery error", exc_info=True)

    def _coordinate(self, accelerator):
        self.accelerator = accelerator
        self.rank = accelerator.process_index
        self.size = accelerator.num_processes
        self.local_rank = accelerator.local_process_index
        accelerator.wait_for_everyone()
        logger.info(
            f"coordinate workers finish, cluster size:{self.size} worker rank:{self.rank} worker local_rank:{self.local_rank}"
        )

    def _get_lr_scheduler(
        self,
        lr_scheduler_config,
        optimizer,
        num_train_epochs,
        num_steps_per_epoch,
        accelerator,
    ):
        enable = lr_scheduler_config.get("enable", False)
        if not enable:
            return None
        max_train_steps = lr_scheduler_config.get("max_train_steps")
        lr_scheduler_type = lr_scheduler_config.get("lr_scheduler_type", "linear")
        num_warmup_steps = lr_scheduler_config.get("num_warmup_steps", 0)

        if max_train_steps is None:
            max_train_steps = num_steps_per_epoch * num_train_epochs

        lr_scheduler = transformers.get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        return lr_scheduler

    def prepare(self, model, tokenizer, dataset, optimizer, accelerator):
        self._coordinate(accelerator)

        embedding_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"model embedding size: {embedding_size}")
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))
            logger.warning(
                f"model embedding size resize to {len(tokenizer)} because of tokenizer size"
            )

        train_dataloader, eval_dataloader = self.dataprocesser.prepare(tokenizer, dataset)

        lr_scheduler_config = self.config.get("lr_scheduler")
        if lr_scheduler_config:
            num_steps_per_epoch = len(train_dataloader)
            num_train_epochs = self.config.get("num_train_epochs", 1)
            lr_scheduler = self._get_lr_scheduler(
                lr_scheduler_config,
                optimizer,
                num_train_epochs,
                num_steps_per_epoch,
                accelerator,
            )
        else:
            lr_scheduler = None

        model.train()

        # self.model, self.optimizer, self.lr_scheduler, ..., are prepared with 2 steps
        # because it is recommended way to prepare model and optimizer while using FSDP.
        # https://huggingface.co/docs/accelerate/usage_guides/fsdp#a-few-caveats-to-be-aware-of
        # self.model = accelerator.prepare(model)

        accelerate_mode = self.config.get("accelerate_mode")
        if accelerate_mode in ["GPU_DEEPSPEED"] :
            dummy_optimizer = DummyOptim(params=model.parameters(), lr=5e-5, weight_decay=0.0)

            def _lr_scheduler_callable(optimizer):
                return get_scheduler(
                    name="linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=1000,
                )
            dummy_lr_scheduler = DummyScheduler(dummy_optimizer, lr_scheduler_callable=_lr_scheduler_callable)
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = accelerator.prepare(
                model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler)
        else:
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

        checkpoint = self.config.get("checkpoint")
        if checkpoint is not None:
            self.recovery(checkpoint)

    def train(self):
        num_train_epochs = self.config.get("num_train_epochs", 1)
        checkpoint = self.config.get("checkpoint")
        logging_steps = self.config.get("logging_steps", 1)
        max_train_step = self.config.get("max_train_step")
        max_eval_step = self.config.get("max_eval_step")
        for idx in range(self.starting_epoch, num_train_epochs, 1):
            logger.info(f"start train epoch {idx}")
            self.model.train()
            start = time.time()
            total_steps = len(self.train_dataloader)
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if step % logging_steps == 0:
                        loss = loss.item()
                        ppl = math.exp(loss)
                        logger.info(
                            f"train epoch:[{idx}/{num_train_epochs}]\tstep:[{step}/{total_steps}]\tloss:{loss:.6f}\tppl:{ppl:.6f}\ttime:{time.time()-start:.6f}"
                        )
                        report(
                            {
                                "loss": loss,
                                "ppl": ppl,
                                "train_epoch": idx,
                                "total_epochs": num_train_epochs,
                                "train_step": step,
                                "total_steps": min(max_train_step, total_steps)
                                if max_train_step
                                else total_steps,
                            }
                        )
                        self.accelerator.log(
                            {"train loss": loss, "train perplexity": ppl},
                            step=idx * total_steps + step,
                        )
                        start = time.time()
                if max_train_step is not None:
                    if step >= max_train_step - 1:
                        break

            if self.eval_dataloader:
                logger.info(f"start eval epoch {idx}")
                self.model.eval()
                start = time.time()
                losses = []
                for step, batch in enumerate(self.eval_dataloader):
                    with torch.no_grad():
                        outputs = self.model(**batch)
                    loss = outputs.loss
                    losses.append(
                        self.accelerator.gather_for_metrics(
                            loss.repeat(batch["input_ids"].shape[0])
                        )
                    )
                    if max_eval_step is not None:
                        if step >= max_eval_step:
                            break

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    eval_loss = float("inf")
                    perplexity = float("inf")
                self.accelerator.log(
                    {"evaluate loss": eval_loss, "evaluate perplexity": perplexity}
                )
                logger.info(
                    f"eval epoch:[{idx}/{num_train_epochs}]\tloss:[{eval_loss:.6f}]\tppl:[{perplexity:.6f}]\ttime:[{time.time()-start:.6f}]"
                )

            if checkpoint is not None:
                self.save(checkpoint, idx)
            self.accelerator.wait_for_everyone()

        output = self.config.get("output", "./output")
        if output is not None:
            logger.info(f"start save model to {output}")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                output,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
            )
            logger.info(f"finish save model to {output}")

        self.accelerator.end_training()

        self.accelerator.wait_for_everyone()

    def _get_local_path(self, root_path, model_name):
        return f"{root_path}/{model_name}_{self.rank}-of-{self.size}"

    def save(self, config, epoch=0):
        if config is None or config is {}:
            logger.warning("checkpoint is empty, skip")
            return
        root_path = config.get("root_path")
        model_name = config.get("model_name", "")
        if root_path is None:
            logger.warning("checkpoint root_path is empty, skip")
            return
        local_checkpoint_path = self._get_local_path(root_path, model_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(self.model.state_dict(), os.path.join(tmpdir, "model.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(tmpdir, "optim.pt"))
            torch.save({"epoch": epoch}, os.path.join(tmpdir, "epoch.pt"))
            if self.lr_scheduler:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(tmpdir, "lr_scheduler.pt"),
                )
            checkpoint = Checkpoint.from_directory(tmpdir)
            checkpoint.to_directory(local_checkpoint_path)
            logger.info(f"save checkpoint to {local_checkpoint_path} finished")
