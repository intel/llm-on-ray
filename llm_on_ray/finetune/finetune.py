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

#!/usr/bin/env python

import os
import argparse
import re
import sys
import copy

from typing import Any, Dict, Union, Optional

from itertools import chain

import torch
import datasets
import transformers

from peft import get_peft_model, LoraConfig

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig, FailureConfig

from pydantic_yaml import parse_yaml_raw_as

from llm_on_ray import common
from llm_on_ray.finetune.finetune_config import FinetuneConfig


def train_func(config: Dict[str, Any]):
    os.chdir(config["cwd"])
    from .finetuning import Finetuning

    finetuing = Finetuning()
    if config["Training"]["finetuning_model"] is not None:
        use_dpo = config["Training"]["finetuning_model"].get("dpo", False)
        use_ppo = config["Training"]["finetuning_model"].get("ppo", False)
        if use_dpo:
            from .dpo_finetuing import DPOFineTuning

            finetuing = DPOFineTuning()
        elif use_ppo:
            raise ValueError("PPO is not supported yet")
        else:
            raise ValueError("No finetuning model specified")

    if finetuing is None:
        raise ValueError("No finetuning model specified")

    finetuing.adapt_transformers_to_device(config)

    finetuing.set_seed(config)

    tokenizer = finetuing.load_tokenizer(config)

    dataset = finetuing.load_dataset(config)

    tokenized_dataset = finetuing.tokenize_dataset(config, tokenizer, dataset)

    data_collator = finetuing.prepare_data_collator(config, tokenizer)

    model = finetuing.load_model(config)

    training_args, trainer = finetuing.get_trainer(
        config, model, tokenizer, tokenized_dataset, data_collator
    )

    common.logger.info("train start")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    common.logger.info("train finish")


def get_finetune_config():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    # Print help if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file) as f:
        finetune_config = parse_yaml_raw_as(FinetuneConfig, f)
    return finetune_config.dict()


def main(external_config=None):
    if not external_config:
        config = get_finetune_config()
    else:
        config = external_config

    config["cwd"] = os.getcwd()
    num_training_workers = config["Training"].get("num_training_workers")
    resources_per_worker = config["Training"].get("resources_per_worker")

    if config["Training"].get("accelerate_mode", None) is None:
        config["Training"][
            "accelerate_mode"
        ] = "DDP"  # will use DDP to accelerate if no method specified

    ccl_worker_count = 1
    device = config["Training"]["device"]
    if device != "cpu":
        ccl_worker_count = num_training_workers

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "OMP_NUM_THREADS": str(resources_per_worker["CPU"]),
                "CCL_ZE_IPC_EXCHANGE": "sockets",
                "CCL_WORKER_COUNT": str(ccl_worker_count),
                "CCL_LOG_LEVEL": "info",
                "FI_TCP_IFACE": "lo",
                "FI_PROVIDER": "tcp",
            }
        }

        if config["General"]["gpt_base_model"] is True:
            runtime_env["pip"] = ["transformers==4.26.0"]

        if config["Training"]["finetuning_model"] is not None:
            if (
                config["Training"]["finetuning_model"]["dpo"]
                and config["General"]["gpt_base_model"]
            ):
                raise ValueError("DPO is not supported for GPT models")

        if device == "gpu":
            num_cpus = (
                resources_per_worker["CPU"] * num_training_workers + 1
            )  # additional 1 for head worker
            ray.init(num_cpus=num_cpus, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)

    common.logger.info(f"ray available resources = {ray.available_resources()}")
    use_gpu = True if device == "gpu" else False
    scaling_config = ScalingConfig(
        num_workers=num_training_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="SPREAD",
    )

    # if try to use Intel GPU, convert device to 'xpu'
    # due to accelerate internal use 'xpu' represent Intel GPU
    if device == "gpu":
        from accelerate.utils import is_xpu_available

        if is_xpu_available():
            device = "xpu"

    if config.get("torch_config", None) is None:
        backend = None
        if device == "cpu" or device == "xpu" or device == "gpu":
            backend = "ccl"
        elif device == "hpu":
            backend = "hccl"
        torch_config = common.TorchConfig(backend=backend, device=device)
    else:
        customer_torch_config = config.get("torch_config")
        torch_config = common.TorchConfig(**customer_torch_config, device=device)

    if config.get("failure_config", None) is None:
        failure_config = FailureConfig()
    else:
        customer_failure_config = config.get("failure_config")
        failure_config = FailureConfig(**customer_failure_config)

    if config.get("run_config", None) is None:
        run_config = RunConfig(failure_config=failure_config)
    else:
        customer_run_config = config.get("run_config")
        if customer_run_config.get("failure_config", None) is None:
            customer_run_config["failure_config"] = failure_config
        run_config = RunConfig(**customer_run_config)

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )
    results = trainer.fit()
    if external_config is not None:
        return results


if __name__ == "__main__":
    main()
