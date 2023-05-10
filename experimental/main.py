#!/usr/bin/env python

import os
import time
import logging
from typing import Any, Dict

import torch
import accelerate
from accelerate.logging import get_logger

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from raydp.torch.config import TorchConfig
from ray.air import RunConfig, FailureConfig
from ray.air.checkpoint import Checkpoint

import plugin

def train_func(config: Dict[str, Any]):
    torch_thread_num = config.get("torch_thread_num")
    if torch_thread_num is not None:
        torch.set_num_threads(torch_thread_num)
    seed = config.get("seed")
    if seed is not None:
        accelerate.utils.set_seed(seed)

    accelerator = accelerate.Accelerator(**config.get("accelerator"))
    try:
        dataset = plugin.load_dataset(config.get("dataset"))
        if dataset is None:
            raise ValueError("dataset return type error")
    except:
        exit()

    try:
        tokenizer = plugin.load_tokenizer(config.get("tokenizer"))
        if tokenizer is None:
            raise ValueError("tokenizer return type error")
    except:
        print("tokenizer error")
        exit()

    try:
        model = plugin.load_model(config.get("model"))
        if model is None:
            raise ValueError("model return type error")
    except:
        print("model error")
        exit()

    try:
        optimizer = plugin.load_optimizer(model, config.get("optimizer"))
        if optimizer is None:
            raise ValueError("optimizer return type error")
    except:
        print("optimizer error")
        exit()


    try:
        trainer = plugin.get_trainer(config.get("trainer"))
        if trainer is None:
            raise ValueError("trainer return type error")
    except:
        print("trainer error")
        exit()

    trainer.prepare(model, tokenizer, dataset, optimizer, accelerator)
    trainer.train()

def main():
    config = plugin.config.parse_config()
    if config.get("run_mode") == "standalone":
        train_func(config)
    elif config.get("run_mode") == "ray":
        ray_config = config.get("ray_config")
        ray.init(**ray_config.get("init", {}))

        scaling_config = ScalingConfig(**ray_config.get("scaling_config", {}))
        torch_config = TorchConfig(**ray_config.get("torch_config", {}))
        failure_config = FailureConfig(**ray_config.get("failure_config", {}))
        run_config = RunConfig(**ray_config.get("run_config", {}), failure_config=failure_config)

        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
            torch_config = torch_config,
            run_config = run_config
        )
        results = trainer.fit()
    else:
        pass
if __name__ == "__main__":
    main()
