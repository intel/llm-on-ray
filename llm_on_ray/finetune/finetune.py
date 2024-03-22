#!/usr/bin/env python

import os
import argparse
from typing import Any, Dict, Union, Optional

import torch

import ray
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air import RunConfig, FailureConfig

from pydantic_yaml import parse_yaml_raw_as

from accelerate import DeepSpeedPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

from llm_on_ray import common
from llm_on_ray.finetune.finetune_config import FinetuneConfig
from importlib import util

use_habana = False
if util.find_spec("habana_frameworks") is not None:
    from optimum.habana.accelerate import GaudiAccelerator as Accelerator
    from optimum.habana.accelerate.utils import (
        GaudiFullyShardedDataParallelPlugin as FullyShardedDataParallelPlugin,
    )
    from optimum.habana.utils import set_seed

    use_habana = True
else:
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from accelerate.utils import set_seed, is_xpu_available

    use_habana = False


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
        "GPU_DEEPSPEED": {
            "ACCELERATE_USE_CPU": "false",
            "ACCELERATE_USE_XPU": "true",
            "ACCELERATE_USE_IPEX": "true",
            "ACCELERATE_USE_DEEPSPEED": "true",
            "ACCELERATE_MIXED_PRECISION": mixed_precision,
        },
        "HPU_DDP": {
            "ACCELERATE_USE_CPU": "false",
            "ACCELERATE_USE_XPU": "false",
            "ACCELERATE_USE_IPEX": "false",
            "ACCELERATE_MIXED_PRECISION": mixed_precision,
        },
    }
    if mode not in mode_env_vars:
        raise ValueError(f"accelerate mode must be one of {list(mode_env_vars.keys())}")
    return mode_env_vars[mode]


def get_device_environment_variable(device):
    if device == "hpu":
        return {
            "HABANA_VISIBLE_DEVICES": "all",
            "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES": "true",
        }
    return {}


def convert_dtype(dtype: str) -> Optional[torch.dtype]:
    supported_dtypes = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": None,
    }
    return supported_dtypes[dtype]


def train_func(config: Dict[str, Any]):
    cwd = config.get("cwd")
    if cwd:
        os.chdir(cwd)

    gradient_accumulation_steps = config["Training"].get("gradient_accumulation_steps", 1)
    base_model = config["General"]["base_model"]
    dataset_file = config["Dataset"]["train_file"]

    seed = config["Training"].get("seed")
    if seed is not None:
        set_seed(seed)

    datasets = common.dataset.Dataset.registory.get("HuggingfaceDataset")()(
        config={
            "name": dataset_file,
            "validation_file": config["Dataset"]["validation_file"],
            "validation_split_percentage": config["Dataset"]["validation_split_percentage"],
        }
    )

    tokenizer = common.tokenizer.Tokenizer.registory.get("HuggingFaceTokenizer")()(
        config={
            "name": base_model,
            "config": config["General"]["config"],
        }
    )

    model = common.model.Model.registory.get("HuggingFaceModelForCausalLM")()(
        config={
            "name": base_model,
            "dtype": convert_dtype(config["Training"].get("mixed_precision", "no")),
            "config": config["General"]["config"],
            "enable_gradient_checkpointing": config["General"].get(
                "enable_gradient_checkpointing", False
            ),
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

    accelerate_mode = config["Training"]["accelerate_mode"]
    if accelerate_mode in ["GPU_FSDP"]:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=False, rank0_only=False
            ),
        )
        deepspeed_plugin = None
    elif accelerate_mode in ["GPU_DEEPSPEED"]:
        fsdp_plugin = None
        hf_ds_config = config["Training"]["deepspeed_config_file"]
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
    else:
        fsdp_plugin = None
        deepspeed_plugin = None

    output_dir = config["General"]["output_dir"]
    accelerator = Accelerator(
        device_placement=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
    )
    epochs = config["Training"]["epochs"]

    common.logger.info(
        f"accelerator generate finish, accelerator device type = {accelerator.device}"
    )

    trainer = common.trainer.Trainer.registory.get("DefaultTrainer")(
        config={
            "accelerate_mode": config["Training"]["accelerate_mode"],
            "num_train_epochs": epochs,
            "max_train_step": config["Training"].get("max_train_steps", None),
            "logging_steps": config["Training"].get("logging_steps", 1),
            "output": output_dir,
            "dataprocesser": {
                "type": "GeneralProcesser",
                "per_device_train_batch_size": config["Training"]["batch_size"],
                "per_device_eval_batch_size": config["Training"]["batch_size"],
                "preprocessing_num_workers": config["Dataset"].get("preprocessing_num_workers", 1),
                "group": config["Dataset"].get("grouping", False),
                "shuffle": True,
            },
            "lr_scheduler": {
                "enable": True,
                "max_train_steps": None,
                "lr_scheduler_type": config["Training"]["lr_scheduler"],
                "num_warmup_steps": 0,
                "learning_rate": config["Training"]["learning_rate"],
                "weight_decay": config["Training"]["weight_decay"],
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
    device = config["Training"]["device"].lower()

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

        device_env_vars = get_device_environment_variable(device)
        runtime_env["env_vars"].update(device_env_vars)

        if config["General"]["gpt_base_model"] is True:
            runtime_env["pip"] = ["transformers==4.26.0"]

        if use_habana:
            ray.init(runtime_env=runtime_env)
        else:
            import intel_extension_for_pytorch as ipex

            if "xpu" in ipex.__version__:
                num_cpus = (
                    resources_per_worker["CPU"] * num_training_workers + 1
                )  # additional 1 for head worker
                ray.init(num_cpus=num_cpus, runtime_env=runtime_env)
            else:
                ray.init(runtime_env=runtime_env)

    common.logger.info(f"ray available resources = {ray.available_resources()}")

    scaling_config = ScalingConfig(
        num_workers=num_training_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="SPREAD",
    )

    if not use_habana and device == "gpu" and is_xpu_available():
        device = "xpu"

    if config.get("torch_config", None) is None:
        backend = None
        if device == "cpu" or device == "xpu":
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
