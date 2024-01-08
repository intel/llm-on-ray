import os
import torch
from torch.utils.tensorboard import SummaryWriter
import math
import time

from .default_trainer import DefaultTrainer
from ..logging import logger


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
                (chosen_ids[i] * chosen_mask[i] != rejected_ids[i] * rejected_mask[i])
                .squeeze()
                .nonzero(as_tuple=True)[0]
            )

            if len(divergence) <= 0:
                # Chosen and rejected prompts are identical.
                # Loss will be 0 anyways.
                continue

            start_index = divergence[0].item()
            end_index = divergence[-1].item()

            # Loss is the negative log probability loss between the chosen and rejected prompt.
            selected_chosen_rewards = chosen_rewards[i][start_index : end_index + 1]
            selected_rejected_rewards = rejected_rewards[i][start_index : end_index + 1]

            loss += -torch.log(
                torch.sigmoid(selected_chosen_rewards - selected_rejected_rewards)
            ).mean()

        loss /= batch_size

        return (loss, result) if return_outputs else loss

    def train(self):
        num_train_epochs = self.config.get("num_train_epochs", 1)
        log_step = self.config.get("log_step", 1)
        if not os.path.exists(self.config.get("log_path", ".")):
            os.makedirs(self.config.get("log_path", "."), exist_ok=True)
        writer = SummaryWriter(self.config.get("log_path", "."))
        for idx in range(num_train_epochs):
            logger.info(f"start train epoch {idx}")
            self.model.train()
            start = time.time()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    loss = self.compute_loss(batch)
                    writer.add_scalar("training loss", loss, step)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if step % log_step == 0:
                        logger.info(
                            f"train epoch:[{idx}/{num_train_epochs}]\tstep:[{step}/{len(self.train_dataloader)}]\tloss:{loss}\tppl:{math.exp(loss)}\ttime:{time.time()-start}"
                        )
                        start = time.time()

            if self.eval_dataloader:
                logger.info(f"start eval epoch {idx}")
                self.model.eval()
                start = time.time()
                losses = []
                for step, batch in enumerate(self.eval_dataloader):
                    with torch.no_grad():
                        outputs = self.model(**batch)
                    loss = outputs.loss
                    losses.append(self.accelerator.gather_for_metrics(loss.repeat(2)))
                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    eval_loss = float("inf")
                    perplexity = float("inf")
                logger.info(
                    f"eval epoch:[{idx}/{num_train_epochs}]\tloss:[{eval_loss}]\tppl:[{perplexity}]\ttime:[{time.time()-start}]"
                )
                writer.add_scalar("eval loss", eval_loss, idx)
                writer.add_scalar("perplexity", perplexity, idx)
