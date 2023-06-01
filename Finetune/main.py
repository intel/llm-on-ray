#!/usr/bin/env python

import os
import time
import traceback
from typing import Any, Dict

import accelerate

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from raydp.torch.config import TorchConfig
from ray.air import RunConfig, FailureConfig

import plugin

def train_func(config: Dict[str, Any]):
    plugin.init(config)
    try :
        accelerator_config = config.get("accelerator")
        plugin.logger.info(f"accelerator_config: {accelerator_config}")
        accelerator = accelerate.Accelerator(**accelerator_config)
    except Exception as e:
        plugin.logger.critical(e, exc_info=True)
        exit(1)
    plugin.logger.info(f"accelerator generate finish")

    datasets = plugin.load_dataset(config.get("datasets"))
    tokenizer = plugin.load_tokenizer(config.get("tokenizer"))
    model = plugin.load_model(config.get("model"))
    optimizer = plugin.load_optimizer(model, config.get("optimizer"))
    trainer = plugin.get_trainer(config.get("trainer"))

    try :
        plugin.logger.info(f"trainer prepare start")
        trainer.prepare(model, tokenizer, datasets, optimizer, accelerator)
    except Exception as e:
        plugin.logger.critical(e, exc_info=True)
        exit(1)
    plugin.logger.info(f"trainer prepare finish")

    try :
        plugin.logger.info(f"train start")
        trainer.train()
    except Exception as e:
        plugin.logger.critical(e, exc_info=True)
        exit(1)
    plugin.logger.info(f"train finish")

def main(external_config = None):
    config = plugin.Config()
    if external_config is not None:
        config.merge(externel_config)
    if config.get("run_mode") == "standalone":
        train_func(config)
    elif config.get("run_mode") == "ray":
        ray_config = config.get("ray_config")
        ray_init_config = ray_config.get("init", {})
        plugin.logger.info(f"ray init config: {ray_init_config}")

        runtime_env = ray_init_config.get("runtime_env")
        ray.init(**ray_init_config)

        scaling_config = ScalingConfig(**ray_config.get("scaling_config", {}))
        plugin.logger.info(f"ray scaling config: {scaling_config}")

        torch_config = TorchConfig(**ray_config.get("torch_config", {}))
        plugin.logger.info(f"ray torch config: {torch_config}")

        failure_config = FailureConfig(**ray_config.get("failure_config", {}))
        plugin.logger.info(f"ray failure config: {failure_config}")

        run_config = RunConfig(**ray_config.get("run_config", {}), failure_config=failure_config)
        plugin.logger.info(f"ray run config: {run_config}")

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
