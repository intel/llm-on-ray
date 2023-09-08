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

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import common

def train_func(config: Dict[str, Any]):
    cwd = config.get("cwd")
    if cwd:
        os.chdir(cwd)
    from common.common import import_all_module
    import_all_module(f"{os.path.dirname(os.path.realpath(__file__))}/plugin","plugin")
    common.init(config)
    
    initializer_config = config.get("initializer")
    if initializer_config:
        try :
            initializer = common.get_initializer(initializer_config)
            initializer.init()
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)   
        common.logger.info(f"Initializer is initialized")

    accelerator = None
    accelerator_config = config.get("accelerator")
    if accelerator_config != None:
        try :
            common.logger.info(f"accelerator_config: {accelerator_config}")
            accelerator = accelerate.Accelerator(**accelerator_config)
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)
        common.logger.info(f"accelerator generate finish")
    
    model = None
    datasets = None
    tokenizer = None
    optimizer = None
    
    datasets_config = config.get("datasets")
    if datasets_config:
        datasets = common.load_dataset(datasets_config)
        common.logger.info(f" ")
    else:
        common.logger.warn(f"No datasets plugin provided, use the built-in datasets of trainer")
    
    tokenizer_config = config.get("tokenizer")
    if tokenizer_config:
        tokenizer = common.load_tokenizer(tokenizer_config)
    else:
        common.logger.warn(f"No tokenizer plugin provided, use the built-in datasets of trainer")
    
    model_config = config.get("model")
    if model_config:
        model = common.load_model(model_config)
    else:
        common.logger.warn(f"No model plugin provided, use the built-in datasets of trainer")
    
    optimizer_config = config.get("optimizer")
    if optimizer_config:
        optimizer = common.load_optimizer(model, config.get("optimizer"))
    else:
        common.logger.warn(f"No optimizer plugin provided, use the built-in datasets of trainer")

    trainer_config = config.get("trainer") 
    if trainer_config:
        trainer = common.get_trainer(config.get("trainer"))

        try :
            trainer.prepare(model, tokenizer, datasets, optimizer, accelerator)
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)
        common.logger.info(f"trainer prepare finish")

        try :
            common.logger.info(f"train start")
            trainer.train()
            common.logger.info(f"train done")
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)
        common.logger.info(f"train finish")
    else:
        common.logger.error(f"Trainer isn't found!")


def main(external_config = None):
    config = common.Config()
    if external_config is not None:
        config.merge(external_config)
    if config.get("run_mode") == "standalone":
        train_func(config)
    elif config.get("run_mode") == "ray":
        config["cwd"] = os.getcwd()
        ray_config = config.get("ray_config")
        ray_init_config = ray_config.get("init", {})
        common.logger.info(f"ray init config: {ray_init_config}")

        runtime_env = ray_init_config.get("runtime_env")
        ray.init(**ray_init_config)

        scaling_config = ScalingConfig(**ray_config.get("scaling_config", {}))
        common.logger.info(f"ray scaling config: {scaling_config}")

        torch_config = TorchConfig(**ray_config.get("torch_config", {}))
        common.logger.info(f"ray torch config: {torch_config}")

        failure_config = FailureConfig(**ray_config.get("failure_config", {}))
        common.logger.info(f"ray failure config: {failure_config}")

        run_config = RunConfig(**ray_config.get("run_config", {}), failure_config=failure_config)
        common.logger.info(f"ray run config: {run_config}")

        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
            torch_config = torch_config,
            run_config = run_config
        )
        results = trainer.fit()
    elif config.get("run_mode") == "initialized":
        ray_config = config.get("ray_config")

        scaling_config = ScalingConfig(**ray_config.get("scaling_config", {}))
        common.logger.info(f"ray scaling config: {scaling_config}")

        torch_config = TorchConfig(**ray_config.get("torch_config", {}))
        common.logger.info(f"ray torch config: {torch_config}")

        failure_config = FailureConfig(**ray_config.get("failure_config", {}))
        common.logger.info(f"ray failure config: {failure_config}")

        run_config = RunConfig(**ray_config.get("run_config", {}), failure_config=failure_config)
        common.logger.info(f"ray run config: {run_config}")
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
