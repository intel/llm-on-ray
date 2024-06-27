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
import logging
import sys
from torch.utils.data import DataLoader, Dataset
import evaluate
from typing import Optional
from transformers import (
    HfArgumentParser,
    default_data_collator,
)
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers import Trainer, TrainingArguments
from llm_on_ray import common
from llm_on_ray.common import dataprocesser
from llm_on_ray.common.logging import logger
from llm_on_ray.common.trainer import Trainer as RayTrainer

use_habana = True
import importlib

loader = importlib.util.find_spec("habana_frameworks")
if loader is not None:
    from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
    from optimum.habana.utils import set_seed

    try:
        from optimum.habana.utils import check_optimum_habana_min_version
    except ImportError:

        def check_optimum_habana_min_version(*a, **b):
            return ()

    finally:
        # Will error if the minimal version of Transformers and Optimum Habana are not installed. Remove at your own risks.
        check_min_version("4.33.0")
        check_optimum_habana_min_version("1.7.5")
else:
    use_habana = False


class HFCustomerSamplerTrainer(GaudiTrainer if use_habana else Trainer):  # type: ignore
    def set_sampler(self, sampler):
        self.customer_sampler = sampler

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataloader, _, _ = self.customer_sampler.prepare(
            None, (self.train_dataset, None, None)
        )
        return train_dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: training requires a eval_dataset.")

        _, eval_dataloader, _ = self.customer_sampler.prepare(None, (None, eval_dataset, None))
        return eval_dataloader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_dataset = test_dataset if test_dataset is not None else self.test_dataset
        if test_dataset is None:
            raise ValueError("Trainer: training requires a test_dataset.")

        _, _, test_dataloader = self.customer_sampler.prepare(None, (None, None, test_dataset))
        return test_dataloader


class HuggingFacePreTrainer(RayTrainer):
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

    def prepare(self, model, tokenizer, dataset, optimizer, accelerator):
        self.train_dataset, self.eval_dataset, self.test_dataset = dataset

    def train(self):
        if use_habana:
            del os.environ["ACCELERATE_TORCH_DEVICE"]
            parser = HfArgumentParser((GaudiTrainingArguments))
        else:
            parser = HfArgumentParser((TrainingArguments))

        training_args = parser.parse_dict(self.config.get("training_config", None))[0]
        send_example_telemetry("Ray_HF_Trainer", training_args)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        if use_habana:
            gaudi_config = GaudiConfig.from_pretrained(
                training_args.gaudi_config_name,
                cache_dir=self.config.get("cache_dir", None),
                revision=self.config.get("model_revision", None),
                use_auth_token=True if self.config.get("use_auth_token") else None,
            )

            # Log on each process the small summary:
            mixed_precision = (
                training_args.bf16
                or gaudi_config.use_torch_autocast
                or gaudi_config.use_habana_mixed_precision
            )
            logger.warning(
                f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
                + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
                + f"mixed-precision training: {mixed_precision}"
            )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Set seed before initializing model.
        if use_habana:
            set_seed(training_args.seed)

        model_config = self.config.get("model")
        model_config["deepspeed_zero_stage"] = training_args.deepspeed_plugin.zero_stage
        if model_config:
            self.model = common.load.load_model(model_config)
        else:
            common.logger.warn("No internal model plugin provided")
        self.model.train()

        tokenizer_config = self.config.get("tokenizer")
        if tokenizer_config:
            self.tokenizer = common.load.load_tokenizer(tokenizer_config)
        else:
            common.logger.warn("No internal tokenizer plugin provided")

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        if training_args.do_eval:

            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy")

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)

        # Initialize our Trainer
        trainer = None
        if use_habana:
            trainer = HFCustomerSamplerTrainer(
                model=self.model,
                gaudi_config=gaudi_config,
                args=training_args,
                train_dataset=self.train_dataset if training_args.do_train else None,
                eval_dataset=self.eval_dataset if training_args.do_eval else None,
                tokenizer=self.tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=default_data_collator,
                compute_metrics=compute_metrics if training_args.do_eval else None,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
                if training_args.do_eval
                else None,
            )

        else:
            trainer = HFCustomerSamplerTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset if training_args.do_train else None,
                eval_dataset=self.eval_dataset if training_args.do_eval else None,
                tokenizer=self.tokenizer,
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=default_data_collator,
                compute_metrics=compute_metrics if training_args.do_eval else None,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
                if training_args.do_eval
                else None,
            )
            print("use the GPU for training")

        trainer.set_sampler(self.dataprocesser)

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics

            # if data_args.streaming:
            metrics["train_samples"] = (
                training_args.max_steps * training_args.per_device_train_batch_size
            )
            # else:
            #    max_train_samples = (
            #        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            #    )
            #   metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()

            # if not data_args.streaming:
            #     max_eval_samples = (
            #         data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            #     )
            #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            metrics["eval_samples"] = len(self.eval_dataset)

            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        # if data_args.dataset_name is not None:
        #     kwargs["dataset_tags"] = data_args.dataset_name
        #     if data_args.dataset_config_name is not None:
        #         kwargs["dataset_args"] = data_args.dataset_config_name
        #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        #     else:
        #         kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub()
        else:
            trainer.create_model_card()
