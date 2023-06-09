#!/usr/bin/env python

import os
import time
import traceback
from typing import Any, Dict

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

from rl_algo.ppo.ppo_rlhf import PPORLHF
from rl_algo.ppo.rlhf_ppo_module import RLHFPPOTorchRLModule
from rl_algo.ppo.rlhf_ppo_torch_learner import RLHFPPOTorchLearner

from plugin.agentenv.rlhf_env import RLHFEnv

import plugin

class ValueFunctionInitializerCallback(DefaultCallbacks):

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        learner_group = algorithm.learner_group


def main():

    config = plugin.Config()
    args = plugin.parse_args()

    ray_config = config.get("ray_config")
    ray_init_config = ray_config.get("init", {})
    plugin.logger.info(f"ray init config: {ray_init_config}")

    runtime_env = ray_init_config.get("runtime_env")
    if runtime_env is None:
        ray_init_config["runtime_env"] = {}
    env_vars = ray_init_config["runtime_env"].get("env_vars")
    if env_vars is None:
        ray_init_config["runtime_env"]["env_vars"] = {}
    ray_init_config["runtime_env"]["env_vars"]["CONFIG_PATH"] = os.path.abspath(args.config_path)
    ray.init(**ray_init_config)

    env_creator = lambda config: RLHFEnv(config)
    tune.register_env("RLHFEnv", env_creator)

    ppo_config = (
        PPOConfig(algo_class=PPORLHF)
        .framework("torch")
        .environment(
            "RLHFEnv", 
            env_config=config.get("agentenv"),
            disable_env_checking=True,
        )
        .rl_module(
            _enable_rl_module_api=True,
            rl_module_spec=SingleAgentRLModuleSpec
            (
                RLHFPPOTorchRLModule,
                model_config_dict=config.get("rl_module"),
            ),
        )
        .training(
            learner_class = RLHFPPOTorchLearner,
            **config.get("training")
        )
        .rollouts(
            **config.get("rollouts")
        )
        .evaluation(
            **config.get("evaluation")
        )
        .experimental(
            **config.get("experimental")
        )
        .callbacks(
            callbacks_class=make_multi_callbacks([
                ValueFunctionInitializerCallback,
            ])
        )
    )

    tuner = tune.Tuner(
        PPORLHF,
        param_space=ppo_config,
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                **config.get("run_config").pop('checkpoint')
            ),
            **config.get("run_config")
        ),
        tune_config=tune.TuneConfig(reuse_actors=False)
    )

    results = tuner.fit()

if __name__ == "__main__":
    main()

