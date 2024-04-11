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
import math
import time
import tempfile
from pathlib import Path

import torch
import transformers

from ray.train import report, Checkpoint

from llm_on_ray.common import dataprocesser
from llm_on_ray.common.trainer import Trainer
from llm_on_ray.common.logging import logger


class DefaultTrainer(Trainer):
    def __init__(self, config):
        self.model = None
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
        self.accelerate_mode = self.config.get("accelerate_mode")
        if self.accelerate_mode == "DEEPSPEED":
            lr = lr_scheduler_config.get("learning_rate", 0.001)
            weight_decay = lr_scheduler_config.get("weight_decay", 0)
            from accelerate.utils import DummyOptim, DummyScheduler

            dummy_optimizer = DummyOptim(
                params=model.parameters(), lr=lr, weight_decay=weight_decay
            )
            dummy_lr_scheduler = DummyScheduler(dummy_optimizer, lr_scheduler_callable=lr_scheduler)
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.lr_scheduler,
            ) = accelerator.prepare(
                model, dummy_optimizer, train_dataloader, eval_dataloader, dummy_lr_scheduler
            )
        else:
            self.model = accelerator.prepare(model)
            (
                self.optimizer,
                self.train_dataloader,
                self.eval_dataloader,
                self.lr_scheduler,
            ) = accelerator.prepare(optimizer, train_dataloader, eval_dataloader, lr_scheduler)

        self.device = self.config.get("device")
        if self.device == "hpu":
            import habana_frameworks.torch.core as htcore
            from habana_frameworks.torch.utils.internal import is_lazy

            self.htcore = htcore
        else:
            self.htcore = None

        checkpoint = self.config.get("checkpoint")
        if checkpoint is not None:
            self.recovery(checkpoint)

    def train(self):
        num_train_epochs = self.config.get("num_train_epochs", 1)
        checkpoint = self.config.get("checkpoint")
        logging_steps = self.config.get("logging_steps", 1)
        max_train_steps = self.config.get("max_train_steps")
        steps_per_epoch = len(self.train_dataloader)
        completed_steps = self.starting_epoch * steps_per_epoch
        for idx in range(self.starting_epoch, num_train_epochs, 1):
            self.model.train()
            start = time.time()
            logger.info(f"Start training epoch {idx}, steps_per_epoch {steps_per_epoch}")
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.model.train()
                    batch = batch.to(device=self.accelerator.device)
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    if self.htcore is not None:
                        self.htcore.mark_step()
                    self.optimizer.step()
                    if self.htcore is not None:
                        self.htcore.mark_step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                        if self.htcore is not None:
                            self.htcore.mark_step()
                    self.optimizer.zero_grad()

                    if step % logging_steps == 0:
                        loss = loss.item()
                        ppl = math.exp(loss)
                        epochs = idx + step / steps_per_epoch
                        logger.info(
                            f"train epoch:{epochs:.6f}\tloss:{loss:.6f}\tppl:{ppl:.6f}\ttime:{time.time()-start:.6f}"
                        )
                        report(
                            {
                                "train_loss": loss,
                                "train_ppl": ppl,
                                "train_epoch": idx,
                                "total_epochs": num_train_epochs,
                                "train_step": step,
                                "completed_steps": completed_steps,
                                "total_steps": min(
                                    max_train_steps, steps_per_epoch * num_train_epochs
                                )
                                if max_train_steps
                                else steps_per_epoch * num_train_epochs,
                            }
                        )
                        start = time.time()
                completed_steps += 1
                if max_train_steps is not None and completed_steps >= max_train_steps:
                    break

            if self.eval_dataloader:
                logger.info(f"start eval epoch {idx}")
                self.model.eval()
                start = time.time()
                losses = []
                for step, batch in enumerate(self.eval_dataloader):
                    batch = batch.to(device=self.accelerator.device)
                    with torch.no_grad():
                        outputs = self.model(**batch)
                    loss = outputs.loss
                    losses.append(
                        self.accelerator.gather_for_metrics(
                            loss.repeat(batch["input_ids"].shape[0])
                        )
                    )

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    eval_loss = float("inf")
                    perplexity = float("inf")
                logger.info(
                    f"eval epoch:{idx}\tloss:{eval_loss:.6f}\tppl:{perplexity:.6f}\ttime:{time.time()-start:.6f}"
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
