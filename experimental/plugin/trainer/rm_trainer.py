from .trainer import Trainer
from itertools import chain
import torch
import transformers
import math
import time
from ray.air import session
from ray.air.checkpoint import Checkpoint

from .. import dataprocesser
from .default_trainer import DefaultTrainer

class RMTrainer(DefaultTrainer):

    def compute_loss(self, batch, return_outputs=False):

        chosen_ids = batch.pop("chosen_input_ids").to(self.model.device)
        chosen_mask = batch.pop("chosen_attention_mask").to(self.model.device)
        rejected_ids = batch.pop("rejected_input_ids").to(self.model.device)
        rejected_mask = batch.pop("rejected_attention_mask").to(self.model.device)

        result = self.model(chosen_ids, chosen_mask, rejected_ids, rejected_mask)
        
        chosen_rewards = result[:, 0, :]
        rejected_rewards = result[:, 1, :]

        batch_size = chosen_rewards.shape[0]

        loss = 0
        for i in range(batch_size):
            
            divergence = (
                chosen_ids[i] * chosen_mask[i] != rejected_ids[i] * rejected_mask[i]
            ).squeeze().nonzero(as_tuple=True)[0]

            if len(divergence) <= 0:
                # Chosen and rejected prompts are identical.
                # Loss will be 0 anyways.
                continue

            start_index = divergence[0].item()
            end_index = divergence[-1].item()

            # Loss is the negative log probability loss between the chosen and rejected prompt.
            selected_chosen_rewards = chosen_rewards[i][start_index:end_index + 1]
            selected_rejected_rewards = rejected_rewards[i][start_index:end_index + 1]

            loss += -torch.log(
                torch.sigmoid(selected_chosen_rewards - selected_rejected_rewards)
            ).mean()

        loss /= batch_size

        return (loss, result) if return_outputs else loss

    def train(self):
        per_device_eval_batch_size = self.config.get("per_device_eval_batch_size", 4)
        epoch = self.config.get("epoch", 1)
        for idx in range(epoch):
            self.model.train()
            start = time.time()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    batch = dict(zip(batch.keys(), map(lambda x: x.unsqueeze(1), batch.values())))
                    loss = self.compute_loss(batch)
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
