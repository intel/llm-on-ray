import os
import math
import time

import torch
import transformers
from megatron import get_args
from ray.air.checkpoint import Checkpoint

from common import dataprocesser
from .pretrainer import PreTrainer
from common.logging import logger

class MegatronPreTrainer(PreTrainer):
    def __init__(self, config):
        self.config = config
        dataprocesser_config = config.get("dataprocesser")
        dataprocesser_type = dataprocesser_config.get("type")
        Factory = dataprocesser.DataProcesser.registory.get(dataprocesser_type)
        if Factory is None:
            raise ValueError(f"there is no {dataprocesser_type} dataprocesser.")
        self.dataprocesser = Factory(dataprocesser_config)
        self.starting_step = 0
        self.mode = "ddp"

    def _coordinate(self, accelerator):
        self.accelerator = accelerator
        self.rank = accelerator.process_index
        self.size = accelerator.num_processes
        self.local_rank = accelerator.local_process_index
        accelerator.wait_for_everyone()
        logger.info(f"coordinate workers finish, cluster size:{self.size} worker rank:{self.rank} worker local_rank:{self.local_rank}")

    def _get_all_checkpoint_step(self, root_path):
        if not os.path.exists(f"{root_path}"):
            return []
        all_checkpoint_step = os.listdir(root_path)
        list.sort(all_checkpoint_step, key=lambda x: int(x), reverse=True)
        return all_checkpoint_step

    def _check_avaiable(self, root_path, step):
        donefile_path = self._get_donefile_path(root_path, step)
        if not os.path.exists(donefile_path):
            return False
        all_donefile = os.listdir(donefile_path)
        if len(all_donefile) == self.size:
            return True
        else:
            return False

    def _get_latest_checkpoint_step(self, root_path):
        all_checkpoint_step = self._get_all_checkpoint_step(root_path)
        for step in all_checkpoint_step:
            if self._check_avaiable(root_path, step):
                return step
        return -1

    def recovery(self, config):
        if config is None or config is {}:
            logger.warning(f"checkpoint is empty, skip")
            return 0
        root_path = config.get("root_path")

        if root_path is None:
            logger.error(f"checkpoint root_path is None, exit")
            exit(1)
        step = self._get_latest_checkpoint_step(root_path)
        if step is None or step == -1:
            logger.warning(f"step is None, skip")
            return 0
        local_checkpoint_path = self._get_local_path(root_path, step)
        try:
            logger.info(f"start recovery from {local_checkpoint_path}")
            checkpoint_dict = Checkpoint.from_directory(local_checkpoint_path).to_dict()
            model_state = checkpoint_dict["model"]
            self.model.load_state_dict(model_state)
            # update optimizer status
            optimizer_state = checkpoint_dict["optimizer_state_dict"]
            self.optimizer.load_state_dict(optimizer_state)
            # update lr_scheduler status
            if "lr_scheduler" in checkpoint_dict and hasattr(self, "lr_scheduler"):
                scheduler_state = checkpoint_dict["lr_scheduler"]
                self.lr_scheduler.load_state_dict(scheduler_state)
            logger.info(f"recovery to step {int(step)}")
            self.starting_step = int(step) + 1
        except Exception as e:
            logger.warning(f"recovery error", exc_info=True)
            return 0

    def _get_lr_scheduler(self, lr_scheduler_config, optimizer, num_train_epochs, num_steps_per_epoch, accelerator):
        # gradient_accumulation_steps = accelerator.gradient_accumulation_steps
        # num_update_steps_per_epoch = math.ceil(num_steps_per_epoch / gradient_accumulation_steps)
        enable = lr_scheduler_config.get("enable", False)
        if not enable:
            return None
        max_train_steps  = lr_scheduler_config.get("max_train_steps")
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
            logger.warning(f"model embedding size resize to {len(tokenizer)} because of tokenizer size")

        lr_scheduler_config = self.config.get("lr_scheduler")
        if lr_scheduler_config:
            num_steps_per_epoch = len(dataset) // self.config.get("dataprocesser").get("per_device_train_batch_size")
            num_train_epochs = self.config.get("num_train_epochs", 1)
            lr_scheduler = self._get_lr_scheduler(lr_scheduler_config, optimizer, num_train_epochs, num_steps_per_epoch, accelerator)
        else:
            lr_scheduler = None

        if self.mode == "ddp":
            self.model = model
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

            checkpoint = self.config.get("checkpoint")
            if checkpoint is not None:
                self.recovery(checkpoint)

            self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        else:
            # only ddp support
            pass

        self.train_dataloader, self.eval_dataloader, _ = self.dataprocesser.prepare(tokenizer, dataset, step=self.starting_step)

    def train(self):
        checkpoint = self.config.get("checkpoint")
        log_step = self.config.get("log_step", 1)
        max_train_step = self.config.get("max_train_step")
        max_eval_step = self.config.get("max_eval_step")
        checkpoint_step = self.config.get("checkpoint_step", 1)
        # megatron arguments
        args = get_args()

        logger.info(f"start train")
        self.model.train()
        start = time.time()
        for step, batch in enumerate(self.train_dataloader):
            step = step + self.starting_step
            batch["input_ids"] = batch["text"]
            batch["labels"] =  batch["text"]
            del batch["text"]
            #del batch["dummy_sample"]
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                if torch.isnan(loss).any():
                    raise ValueError("loss NaN")
                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if step % log_step == 0:
                    logger.info(f"step:[{step}/{len(self.train_dataloader)}]\tlr:{self.lr_scheduler.get_last_lr() if self.lr_scheduler else None}\tloss:{loss}\tppl:{math.exp(loss)}\ttime:{time.time()-start}")
                    start = time.time()
            if max_train_step is not None:
                if step >= max_train_step:
                    break

            if self.eval_dataloader and args.eval_interval and step and step % args.eval_interval == 0:
                logger.info(f"start eval step {step}")
                self.model.eval()
                start = time.time()
                losses = []
                for step, batch in enumerate(self.eval_dataloader):
                    batch["input_ids"] = batch["text"]
                    batch["labels"] =  batch["text"]
                    del batch["text"]
                    with torch.no_grad():
                        outputs = self.model(**batch)
                    loss = outputs.loss
                    losses.append(self.accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].shape[0])))
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
                logger.info(f"eval step:[{step}/{len(self.train_dataloader)}]\tloss:[{eval_loss}]\tppl:[{perplexity}]\ttime:[{time.time()-start}]")

            if checkpoint is not None and (step+1) % checkpoint_step == 0:
                self.save(checkpoint, step)
            self.accelerator.wait_for_everyone()

        output = self.config.get("output", "./output")
        if output is not None:
            logger.info(f"start save model to {output}")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                output, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            )
            logger.info(f"finish save model to {output}")
        self.accelerator.wait_for_everyone()

    def _get_local_path(self, root_path, step):
        if self.mode == "ddp":
            return f"{root_path}/{step}/{0}-of-{self.size}"
        elif self.mode == "fsdp":
            return f"{root_path}/{step}/{self.rank}-of-{self.size}"
        else:
            pass

    def _save(self, local_checkpoint_path):
        logger.info(f"save checkpoint to {local_checkpoint_path}")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        status = {
            "model": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.lr_scheduler:
            status["lr_scheduler"] = self.lr_scheduler.state_dict()
        checkpoint = Checkpoint.from_dict(status)
        Checkpoint.to_directory(checkpoint, local_checkpoint_path)
        logger.info(f"save checkpoint finish")

    def _get_donefile_path(self, root_path, step):
        return f"{root_path}/{step}/donefile"

    def _get_local_donefile_path(self, root_path, step):
        donefile_path = self._get_donefile_path(root_path, step)
        return f"{donefile_path}/{self.rank}-of-{self.size}"

    def _save_done(self, root_path, step):
        placeholder = Checkpoint.from_dict({0:0})
        donefile = self._get_local_donefile_path(root_path, step)
        Checkpoint.to_directory(placeholder, donefile)

    def save(self, config, step):
        if config is None or config is {}:
            logger.warning(f"checkpoint is empty, skip")
            return
        root_path = config.get("root_path")
        if root_path is None:
            logger.warning(f"checkpoint root_path is empty, skip")
        local_checkpoint_path = self._get_local_path(root_path, step)
        if self.mode == "ddp":
            if int(self.rank) == 0:
                self._save(local_checkpoint_path)
        elif self.mode == "fsdp":
            self._save(local_checkpoint_path)
        else:
            pass
        self._save_done(root_path, step)
