#!/usr/bin/env python

import os
import argparse
from typing import Any, Dict, Union

import accelerate
from accelerate.utils import is_xpu_available

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig, FailureConfig

from pydantic_yaml import parse_yaml_raw_as

from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common
from finetune.finetune_config import FinetuneConfig


def get_accelerate_environment_variable(mode: str, config: Union[Dict[str, Any], None]) -> dict:
    mixed_precision = config["Training"]["mixed_precision"] if config else "no"
    mode_env_vars = {
        "CPU_DDP": {
            "ACCELERATE_USE_CPU": "true",
            "ACCELERATE_USE_IPEX": "true",
            "ACCELERATE_MIXED_PRECISION": mixed_precision,
        },
        "GPU_DDP": {
            "ACCELERATE_USE_CPU": "false",
            "ACCELERATE_USE_XPU": "true",
            "ACCELERATE_USE_IPEX": "true",
            "ACCELERATE_MIXED_PRECISION": mixed_precision,
        },
        "GPU_FSDP": {
            "ACCELERATE_USE_CPU": "false",
            "ACCELERATE_USE_XPU": "true",
            "ACCELERATE_USE_IPEX": "true",
            "ACCELERATE_USE_FSDP": "true",
            "FSDP_SHARDING_STRATEGY": "1",
            "FSDP_OFFLOAD_PARAMS": "false",
            "FSDP_AUTO_WRAP_POLICY": "NO_WRAP",
            "FSDP_BACKWARD_PREFETCH": "BACKWARD_PRE",
            "FSDP_STATE_DICT_TYPE": "SHARDED_STATE_DICT",
            "FSDP_FORWARD_PREFETCH": "false",
            "FSDP_USE_ORIG_PARAMS": "false",
            "FSDP_SYNC_MODULE_STATES": "true",
            "ACCELERATE_MIXED_PRECISION": mixed_precision,
        },
    }
    if mode not in mode_env_vars:
        raise ValueError(f"accelerate mode must be one of {list(mode_env_vars.keys())}")
    return mode_env_vars[mode]


def convert_dtype(dtype: str) -> torch.dtype:
    supported_dtypes = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    if dtype in supported_dtypes:
        return supported_dtypes[dtype]
    else:
        raise ValueError(f"only supported torch.dtype list [{supported_dtypes.keys()}]")


def train_func(config: Dict[str, Any]):
    cwd = config.get("cwd")
    if cwd:
        os.chdir(cwd)

    gradient_accumulation_steps = config["Training"].get("gradient_accumulation_steps", 1)

    accelerate_mode = config["Training"]["accelerate_mode"]
    if accelerate_mode in ["GPU_FSDP"]:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=False, rank0_only=False
            ),
        )
    else:
        fsdp_plugin = None
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, fsdp_plugin=fsdp_plugin
    )
    common.logger.info(
        f"accelerator generate finish, accelerator device type = {accelerator.device}"
    )

    seed = config["Training"].get("seed")
    if seed is not None:
        accelerate.utils.set_seed(seed)

    datasets = common.dataset.Dataset.registory.get("HuggingfaceDataset")()(
        config={
            "name": config["Dataset"]["train_file"],
            "validation_file": config["Dataset"]["validation_file"],
            "validation_split_percentage": config["Dataset"]["validation_split_percentage"],
        }
    )

    tokenizer = common.tokenizer.Tokenizer.registory.get("HuggingFaceTokenizer")()(
        config={
            "name": config["General"]["base_model"],
            "config": config["General"]["config"],
        }
    )

    model = common.model.Model.registory.get("HuggingFaceModelForCausalLM")()(
        config={
            "name": config["General"]["base_model"],
            "dtype": convert_dtype(config["Training"]["mixed_precision"]),
            "config": config["General"]["config"],
            "gradient_checkpointing": config["General"]["gradient_checkpointing"],
            "lora_config": config["General"]["lora_config"]
            if config["General"].get("lora_config")
            else None,
        }
    )

    optimizer = common.optimizer.Optimizer.registory.get("DefaultOptimizer")()(
        model,
        config={
            "name": config["Training"]["optimizer"],
            "config": {"lr": config["Training"]["learning_rate"]},
        },
    )

    trainer = common.trainer.Trainer.registory.get("DefaultTrainer")(
        config={
            "num_train_epochs": config["Training"]["epochs"],
            "max_train_step": config["Training"].get("max_train_steps", None),
            "log_step": 1,
            "output": config["General"]["output_dir"],
            "dataprocesser": {
                "type": "GeneralProcesser",
                "per_device_train_batch_size": config["Training"]["batch_size"],
                "per_device_eval_batch_size": config["Training"]["batch_size"],
                "preprocessing_num_workers": config["Dataset"].get("preprocessing_num_workers", 1),
                "shuffle": True,
            },
            "lr_scheduler": {
                "enable": True,
                "max_train_steps": None,
                "lr_scheduler_type": config["Training"]["lr_scheduler"],
                "num_warmup_steps": 0,
            },
            "checkpoint": {
                "root_path": config["General"]["checkpoint_dir"],
            }
            if config["General"].get("checkpoint_dir")
            else None,
        }
    )

    try:
        common.logger.info("trainer prepare start")
        model.training = True
        trainer.prepare(model, tokenizer, datasets, optimizer, accelerator)
    except Exception as e:
        common.logger.critical(e, exc_info=True)
        exit(1)
    common.logger.info("trainer prepare finish")

    try:
        common.logger.info("train start")
        trainer.train()
    except Exception as e:
        common.logger.critical(e, exc_info=True)
        exit(1)
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

    accelerate_mode = config["Training"].get("accelerate_mode")
    if accelerate_mode is None:
        accelerate_mode = "CPU_DDP"

    use_cpu = True if accelerate_mode.startswith("CPU") else False
    use_gpu = True if accelerate_mode.startswith("GPU") else False
    ccl_worker_count = 1 if use_cpu is True else num_training_workers

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "OMP_NUM_THREADS": str(resources_per_worker["CPU"]),
                "CCL_ZE_IPC_EXCHANGE": "sockets",
                "CCL_WORKER_COUNT": str(ccl_worker_count),
                "CCL_LOG_LEVEL": "info",
                "WORLD_SIZE": str(num_training_workers),
                "FI_TCP_IFACE": "lo",
                "FI_PROVIDER": "tcp",
            }
        }

        accelerate_env_vars = get_accelerate_environment_variable(accelerate_mode, config)
        runtime_env["env_vars"].update(accelerate_env_vars)

        if config["General"]["gpt_base_model"] is True:
            runtime_env["pip"] = ["transformers==4.26.0"]

        ray.init(runtime_env=runtime_env)

    common.logger.info(f"ray available resources = {ray.available_resources()}")

    scaling_config = ScalingConfig(
        num_workers=num_training_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="SPREAD",
    )

    device = config["Training"]["device"].lower()
    if device == "gpu" and is_xpu_available():
        device = "xpu"

    if config.get("torch_config", None) is None:
        backend = "ccl" if device == "cpu" or device == "xpu" else None
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

    return results


if __name__ == "__main__":
    main()
