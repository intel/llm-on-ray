#!/usr/bin/env python

import plugin
import main
import argparse
import os

TEMPLATE_CONFIG_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/plugin/llm_finetune_configure_template.py"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--finetune_config",
        type=str,
        required=False,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    args, unparsed = parser.parse_known_args()
    return args

CONFIG_MAPPING = {
    "General.base_model": [
       "model.name",
        "tokenizer.name"
    ],
    "General.output_dir": "trainer.output",
    "General.checkpoint_dir": "trainer.checkpoint.root_path", 
    "Dataset.train_file": "datasets.name",
    "Dataset.preprocessing_num_workers": "trainer.dataprocesser.preprocessing_num_workers",
    "Dataset.validation_file": "datasets.validation_file",
    "Dataset.validation_split_percentage": "datasets.validation_split_percentage",
    "Training.optimizer": "optimizer.name",
    "Training.batch_size": "trainer.dataprocesser.per_device_train_batch_size",
    "Training.epochs": "trainer.num_train_epochs",
    "Training.learning_rate": "optimizer.config.lr",
    "Training.lr_scheduler": "trainer.lr_scheduler.lr_scheduler_type",
    "Training.weight_decay": "optimizer.config.weight_decay",
    "Training.num_training_workers": [
        "ray_config.scaling_config.num_workers",
        "ray_config.init.runtime_env.env_vars.CCL_WORKER_COUNT#str"
        "ray_config.init.runtime_env.env_vars.WORLD_SIZE#str"
    ],
    "Training.resources_per_worker.CPU": [
        "ray_config.scaling_config.resources_per_worker.CPU",
        "ray_config.init.runtime_env.env_vars.OMP_NUM_THREADS#str",
        "torch_thread_num"
    ],
    "Training.max_train_steps": "trainer.max_train_step",
    "Training.gradient_accumulation_steps": "accelerator.gradient_accumulation_steps",
    "Training.seed": "seed",
}

if __name__ == "__main__":
    args = parse_args()
    finetune_config_path = args.finetune_config

    template_config = plugin.parse_config(TEMPLATE_CONFIG_PATH)
    user_config = plugin.parse_config(finetune_config_path)

    config = plugin.Config()
    config.merge(template_config)
    config.merge_with_mapping(user_config, CONFIG_MAPPING)

    main.main(config)
