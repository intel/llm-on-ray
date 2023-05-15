from .trainer import Trainer
from itertools import chain
import torch
import transformers
import math
import time
from ray.air import session
from ray.air.checkpoint import Checkpoint

from .. import dataprocesser

class DefaultTrainer(Trainer):
    def __init__(self, config):
        self.config = config
        dataprocesser_config = config.get("dataprocesser")
        dataprocesser_type = dataprocesser_config.get("type")
        Factory = dataprocesser.DataProcesser.registory.get(dataprocesser_type)
        if Factory is None:
            raise ValueError("there is no %s dataprocesser."%dataprocesser_type)
        self.dataprocesser = Factory(dataprocesser_config)

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
        print("before prepare:", model)
        model, tokenizer, train_dataloader, eval_dataloader, optimizer, lr_scheduler = self.dataprocesser.prepare(
            model, tokenizer, dataset, optimizer
        )
        
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        print("after prepare:", model)
    def train(self):
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 4)
        epoch = self.config.get("epoch", 1)
        for idx in range(epoch):
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
                    print("train epoch:[%s/%s]\tstep:[%s/%s]\tloss:[%s]\tppl:[%s]\ttime:[%s]" % \
                        (idx, epoch, step,len(self.train_dataloader),loss, math.exp(loss), time.time()-start))
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
            print("eval epoch:[%s/%s]\tloss:[%s]\tppl:[%s]\ttime:[%s]" % \
                        (idx, epoch, eval_loss, perplexity, time.time()-start))

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
