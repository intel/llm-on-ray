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

ray.init(local_mode=True)

def main():
    
    config = plugin.Config()
    # agentenv = plugin.get_agentenv(config.get("agentenv"))

    env_creator = lambda config: RLHFEnv(config)
    tune.register_env("RLHFEnv", env_creator)

    # env = RLHFEnv(config.get("agentenv"))

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
                model_config_dict=config.get("agent"),
            ),
        )
        .training(
            num_sgd_iter=1,
            sgd_minibatch_size=2, # minibatch size for each SGD iteration
            train_batch_size=2, # number of on-policy samples to collect
            _enable_learner_api=True,
            learner_class=RLHFPPOTorchLearner,
        )
        .rollouts(
            num_rollout_workers=0,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration_unit="episodes",
        )
        .experimental(
            _disable_preprocessor_api=True,
            # _disable_initialize_loss_from_dummy_batch=True,
        )
        .callbacks(
            callbacks_class=make_multi_callbacks([
                ValueFunctionInitializerCallback,
            ])
        )
    )


    # ppo_algo = ppo_config.build()
    # ppo_algo.train()
    tuner = tune.Tuner(
        PPORLHF,
        param_space=ppo_config,
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=50,
                checkpoint_at_end=True,
            ),
            stop={"training_iteration": 1000},
        ),
        tune_config=tune.TuneConfig(reuse_actors=False)
    )

    results = tuner.fit()



if __name__ == "__main__":
    main()

