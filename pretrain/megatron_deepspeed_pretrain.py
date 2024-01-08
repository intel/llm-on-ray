import os
import sys
from typing import Any, Dict

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig, FailureConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common

import importlib

loader = importlib.util.find_spec("habana_frameworks")
if loader is not None:
    from backend.habana_backend import TorchConfig
else:
    from ray.train.torch import TorchConfig

from megatron.training import pretrain


def train_func(config: Dict[str, Any]):
    cwd = config.get("cwd")
    if cwd:
        os.chdir(cwd)

    try:
        import pretrain_gpt as pretrain_module
    except ImportError:
        megatron_deepspeed_path = config.get("megatron_deepspeed_path", None)
        if megatron_deepspeed_path is not None:
            sys.path.append(megatron_deepspeed_path)
            pretrain_module_name = config.get("pretrain_module", None)
            if pretrain_module_name is not None:
                pretrain_module = importlib.import_module(pretrain_module_name)
            else:
                pretrain_module = importlib.import_module("pretrain_gpt")
        else:
            raise ImportError("Please set megatron_deepspeed_path in config")

    common.init(config)  # type: ignore
    megatron_config = config.get("megatron_config", {})

    if hasattr(pretrain_module, "ModelType"):
        pretrain(
            pretrain_module.train_valid_test_datasets_provider,
            pretrain_module.model_provider,
            pretrain_module.ModelType.encoder_or_decoder,
            pretrain_module.forward_step,
            args_defaults=megatron_config,
            data_post_process=pretrain_module.data_post_process,
        )
    elif hasattr(pretrain_module, "llama_argument_handler"):
        pretrain(
            pretrain_module.train_valid_test_datasets_provider,
            pretrain_module.model_provider,
            pretrain_module.forward_step,
            pretrain_module.llama_argument_handler,
            args_defaults=megatron_config,
        )
    else:
        pretrain(
            pretrain_module.train_valid_test_datasets_provider,
            pretrain_module.model_provider,
            pretrain_module.forward_step,
            args_defaults=megatron_config,
        )


def main(external_config=None):
    config = common.Config()
    if external_config is not None:
        config.merge(external_config)

    config["cwd"] = os.getcwd()
    ray_config = config.get("ray_config")
    ray_init_config = ray_config.get("init", {})
    common.logger.info(f"ray init config: {ray_init_config}")

    ray_init_config.get("runtime_env")
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
        torch_config=torch_config,
        run_config=run_config,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
