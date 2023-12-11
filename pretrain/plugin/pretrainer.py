import os
import math
import time
import json
import shutil
import tempfile

import torch
import transformers

from ray.train import Checkpoint
from ray.train.torch import TorchCheckpoint
from pathlib import Path

from common import dataprocesser
from common.trainer import Trainer
from common.logging import logger

class PreTrainer(Trainer):
    def __init__(self, config):
        self.config = config
        dataprocesser_config = config.get("dataprocesser")
        dataprocesser_type = dataprocesser_config.get("type")
        Factory = dataprocesser.DataProcesser.registory.get(dataprocesser_type)
        if Factory is None:
            raise ValueError(f"there is no {dataprocesser_type} dataprocesser.")
        self.dataprocesser = Factory(dataprocesser_config)
        self.starting_episode = 0
        self.mode = "ddp"

    def _coordinate(self, accelerator):
        self.accelerator = accelerator
        self.rank = accelerator.process_index
        self.size = accelerator.num_processes
        self.local_rank = accelerator.local_process_index
        accelerator.wait_for_everyone()
        logger.info(f"coordinate workers finish, cluster size:{self.size} worker rank:{self.rank} worker local_rank:{self.local_rank}")

    def _get_all_checkpoint_episode(self, root_path):
        if not os.path.exists(f"{root_path}"):
            return []
        all_checkpoint_episode = os.listdir(root_path)
        list.sort(all_checkpoint_episode, key=lambda x: int(x), reverse=True)
        return all_checkpoint_episode

    def _check_avaiable(self, root_path, episode):
        donefile_path = self._get_donefile_path(root_path, episode)
        if not os.path.exists(donefile_path):
            return False
        all_donefile = os.listdir(donefile_path)
        if len(all_donefile) == self.size:
            return True
        else:
            return False

    def _get_latest_checkpoint_episode(self, root_path):
        all_checkpoint_episode = self._get_all_checkpoint_episode(root_path)
        for episode in all_checkpoint_episode:
            if self._check_avaiable(root_path, episode):
                return episode
        return -1

    def recovery(self, config):
        if config is None or config is {}:
            logger.warning(f"checkpoint is empty, skip")
            return 0
        root_path = config.get("root_path")
        episode = config.get("episode", None)

        if root_path is None:
            logger.error(f"checkpoint root_path is None, exit")
            exit(1)
        if episode is None:
            episode = self._get_latest_checkpoint_episode(root_path)
        if episode is None or episode == -1:
            logger.warning(f"episode is None, skip")
            return 0
        local_checkpoint_path = self._get_local_path(root_path, episode)
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
                if Path.exists(checkpoint_dir / "lr_scheduler.pt") and hasattr(self, "lr_scheduler"):
                    scheduler_state = torch.load(checkpoint_dir / "lr_schduler.pt", map_location="cpu")
                    self.lr_scheduler.load_state_dict(scheduler_state)

            logger.info(f"recovery to episode {int(episode)}")
            self.starting_episode = int(episode) + 1
        except Exception as e:
            logger.warning(f"recovery error", exc_info=True)

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

        train_dataloader, eval_dataloader = self.dataprocesser.prepare(
            tokenizer, dataset
        )

        lr_scheduler_config = self.config.get("lr_scheduler")
        if lr_scheduler_config:
            num_steps_per_epoch = len(train_dataloader)
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
        elif self.mode == "fsdp":
            self.model, self.optimizer, self.lr_scheduler = accelerator.prepare(
                model, optimizer, lr_scheduler
            )
            checkpoint = self.config.get("checkpoint")
            if checkpoint is not None:
                self.recovery(checkpoint)
        else:
            pass

        self.train_dataloader, self.eval_dataloader = accelerator.prepare(
            train_dataloader, eval_dataloader,
        )

    def _check_and_mkdir(self, path):
        path = Path(path) 
        if not path.exists():
            path.mkdir(parents=True)
    
    def _write_json(self, target_dict, save_path):
        json_object = json.dumps(target_dict, indent=4)
        with open(save_path, "w") as outfile:
            outfile.write(json_object)

    def train(self):
        num_train_epochs = self.config.get("num_train_epochs", 1)
        checkpoint = self.config.get("checkpoint")
        log_step = self.config.get("log_step", 1)
        max_train_step_per_episode = self.config.get("max_train_step_per_episode")
        max_eval_step_per_episode = self.config.get("max_eval_step_per_episode")
        save_state_path = self.config.get("save_state_path")

        if save_state_path is not None and int(self.rank) == 0:
            self._check_and_mkdir(save_state_path)
            training_state = {}
        else:
            training_state = None 
        
        for idx in range(self.starting_episode, len(self.train_dataloader), 1):
            logger.info(f"start train episode {idx}")
            if training_state is not None and int(self.rank) == 0:
                training_state[f'episode_{idx}'] = {}
            self.model.train()
            current_train_dataloader = self.train_dataloader[idx]
            start = time.time()
            for step, batch in enumerate(current_train_dataloader):
                if training_state is not None and int(self.rank) == 0:
                    training_state[f'episode_{idx}'][f'step_{step}'] = {}
                    training_state[f'episode_{idx}'][f'step_{step}']['data'] = batch['input_ids'][0].tolist()[:50]
                    training_state[f'episode_{idx}'][f'step_{step}']['learning_rate'] = self.lr_scheduler.state_dict()['_last_lr']
                        
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
                        logger.info(f"train episode:[{idx}/{len(self.train_dataloader)}]\tstep:[{step}]\tloss:{loss}\tppl:{math.exp(loss)}\ttime:{time.time()-start}")
                        start = time.time()
                if training_state is not None and int(self.rank) == 0:
                    training_state[f'episode_{idx}'][f'step_{step}']['loss'] = loss.item()
                    training_state[f'episode_{idx}'][f'step_{step}']['ppl'] = math.exp(loss) 
                    file_name = "stepwise_training_state_recovery" if self.starting_episode > 0 else "stepwise_training_state"
                    self._write_json(training_state, f"{save_state_path}/{file_name}.json")
                
                if max_train_step_per_episode is not None:
                    if step >= max_train_step_per_episode:
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
                    losses.append(self.accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].shape[0])))
                    if max_eval_step_per_episode is not None:
                        if step >= max_eval_step_per_episode:
                            break

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    eval_loss = float("inf")
                    perplexity = float("inf")
                logger.info(f"eval episode:[{idx}/{len(self.train_dataloader)}]\tloss:[{eval_loss}]\tppl:[{perplexity}]\ttime:[{time.time()-start}]")

            if checkpoint is not None:
                self.save(checkpoint, idx)
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

    def _get_local_path(self, root_path, episode):
        if self.mode == "ddp":
            return f"{root_path}/{episode}/{0}-of-{self.size}"
        elif self.mode == "fsdp":
            return f"{root_path}/{episode}/{self.rank}-of-{self.size}"
        else:
            pass

    def _save(self, local_checkpoint_path):
        logger.info(f"save checkpoint to {local_checkpoint_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), os.path.join(tmpdir, "model.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(tmpdir, "optim.pt"))
            if self.lr_scheduler:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(tmpdir, "lr_schduler.pt"))
            checkpoint = Checkpoint.from_directory(tmpdir)
            checkpoint.to_directory(local_checkpoint_path)
        logger.info(f"save checkpoint finish")

    def _get_donefile_path(self, root_path, episode):
        return f"{root_path}/{episode}/donefile"

    def _get_local_donefile_path(self, root_path, episode):
        donefile_path = self._get_donefile_path(root_path, episode)
        return f"{donefile_path}/{self.rank}-of-{self.size}"

    def _save_done(self, root_path, episode):
        placeholder = TorchCheckpoint.from_state_dict({"0": 0})
        donefile = self._get_local_donefile_path(root_path, episode)
        placeholder.to_directory(donefile)

    def _remove_stale_checkpoint(self, root_path, num_to_keep):
        checkpoints = self._get_all_checkpoint_episode(root_path)
        if len(checkpoints) > num_to_keep:
            stale = checkpoints[-1]
            logger.warning("Removing stale checkpoint")
            shutil.rmtree(f"{root_path}/{stale}")

    def save(self, config, episode):
        if config is None or config is {}:
            logger.warning(f"checkpoint is empty, skip")
            return
        root_path = config.get("root_path")
        if root_path is None:
            logger.warning(f"checkpoint root_path is empty, skip")
        num_to_keep = config.get("num_to_keep")
        if num_to_keep <= 0:
            logger.warning(f"checkpoint num_to_keep cannot be zero, ignored")
            num_to_keep = None
        local_checkpoint_path = self._get_local_path(root_path, episode)
        if self.mode == "ddp":
            if int(self.rank) == 0:
                self._save(local_checkpoint_path)
        elif self.mode == "fsdp":
            self._save(local_checkpoint_path)
        else:
            pass
        self._save_done(root_path, episode)
        if num_to_keep > 0 and self.mode == "ddp"and int(self.rank) == 0:
            self._remove_stale_checkpoint(root_path, num_to_keep)
