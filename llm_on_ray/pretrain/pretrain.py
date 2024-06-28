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
from typing import Any, Dict

import accelerate

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig, FailureConfig

from llm_on_ray import common

from importlib import util

use_habana = False
loader = util.find_spec("habana_frameworks")
if loader is not None:
    from llm_on_ray.pretrain.backend.habana_backend import TorchConfig

    use_habana = True
else:
    from ray.train.torch import TorchConfig
    from llm_on_ray.pretrain.backend.deepspeed_backend import TorchConfig as DeepSpeedTorchConfig


def train_func(config: Dict[str, Any]):
    cwd = config.get("cwd")
    if cwd:
        os.chdir(cwd)
    from llm_on_ray.common import import_all_modules

    import_all_modules(f"{os.path.dirname(os.path.realpath(__file__))}/plugin", "plugin")
    common.init(config)  # type: ignore
    initializer_config = config.get("initializer")
    if initializer_config:
        try:
            initializer = common.load.get_initializer(initializer_config)
            initializer.init()
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)
        common.logger.info("Initializer is initialized")

    accelerator = None
    accelerator_config = config.get("accelerator")
    if accelerator_config is not None:
        try:
            common.logger.info(f"accelerator_config: {accelerator_config}")
            accelerator = accelerate.Accelerator(**accelerator_config)
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)
        common.logger.info("accelerator generate finish")

    model = None
    datasets = None
    tokenizer = None
    optimizer = None

    datasets_config = config.get("datasets")
    if datasets_config:
        datasets = common.load.load_dataset(datasets_config)
        common.logger.info(" ")
    else:
        common.logger.warn("No datasets plugin provided, use the built-in datasets of trainer")

    tokenizer_config = config.get("tokenizer")
    if tokenizer_config:
        tokenizer = common.load.load_tokenizer(tokenizer_config)
    else:
        common.logger.warn("No tokenizer plugin provided, use the built-in tokenizer of trainer")

    model_config = config.get("model")
    if model_config:
        model = common.load.load_model(model_config)
    else:
        common.logger.warn("No model plugin provided, use the built-in model of trainer")

    optimizer_config = config.get("optimizer")
    if optimizer_config:
        optimizer = common.load.load_optimizer(model, config.get("optimizer"))
    else:
        common.logger.warn("No optimizer plugin provided, use the built-in optimizer of trainer")

    trainer_config = config.get("trainer")
    if trainer_config:
        trainer = common.load.get_trainer(config.get("trainer"))

        try:
            trainer.prepare(model, tokenizer, datasets, optimizer, accelerator)
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)
        common.logger.info("trainer prepare finish")

        try:
            common.logger.info("train start")
            trainer.train()
            common.logger.info("train done")
        except Exception as e:
            common.logger.critical(e, exc_info=True)
            exit(1)
        common.logger.info("train finish")
    else:
        common.logger.error("Trainer isn't found!")


def main(external_config=None):
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

        ray.init(**ray_init_config)

        scaling_config = ScalingConfig(**ray_config.get("scaling_config", {}))
        common.logger.info(f"ray scaling config: {scaling_config}")

        if (
            config["trainer"].get("training_config", None)
            and config["trainer"].get("training_config").get("deepspeed", None)
            and use_habana is False
        ):
            torch_config = DeepSpeedTorchConfig(**ray_config.get("torch_config", {}))
        else:
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
            torch_config=torch_config,
            run_config=run_config,
        )
        trainer.fit()
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
            torch_config=torch_config,
            run_config=run_config,
        )
        trainer.fit()
    else:
        pass


if __name__ == "__main__":
    main()
