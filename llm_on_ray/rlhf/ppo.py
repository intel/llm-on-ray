#!/usr/bin/env python


import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.ppo import PPOConfig

from llm_on_ray.rlhf.rl_algo.ppo.ppo_rlhf import PPORLHF
from llm_on_ray.rlhf.rl_algo.ppo.rlhf_ppo_module import RLHFPPOTorchRLModule
from llm_on_ray.rlhf.rl_algo.ppo.rlhf_ppo_torch_learner import RLHFPPOTorchLearner
import llm_on_ray.common as common
from llm_on_ray.common.agentenv.rlhf_env import RLHFEnv


class ValueFunctionInitializerCallback(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        learner_group = algorithm.learner_group  # noqa: F841 # assigned to but never used


def init_ray(config):
    num_training_workers = config["Training"].get("num_training_workers")
    resources_per_worker = config["Training"].get("resources_per_worker")

    runtime_env = {
        "env_vars": {
            "OMP_NUM_THREADS": str(resources_per_worker["CPU"]),
            "ACCELERATE_USE_CPU": "True",
            "ACCELERATE_MIXED_PRECISION": "no",
            "CCL_WORKER_COUNT": "1",
            "CCL_LOG_LEVEL": "info",
            "WORLD_SIZE": str(num_training_workers),
        }
    }
    ray.init(runtime_env=runtime_env, local_mode=True)


def prepare_ppo(config):
    env_creator = lambda config: RLHFEnv(config)

    tune.register_env("RLHFEnv", env_creator)

    agentenv_config = {
        "type": "RLHFEnv",
        "config": {
            "tokenizer": {
                "type": "HuggingFaceTokenizer",
                "name": config["General"]["model_name"],
                "config": {},
            },
            "reward_model": {
                "type": "HuggingFaceRewardModel",
                "name": config["General"]["rm_name"],
                "config": {},
            },
            "sft_model": {
                "type": "HuggingFaceModelForCausalLM",
                "name": config["General"]["model_name"],
                "config": {},
            },
            "datasets": {
                "type": "HuggingfaceDataset",
                "name": config["Dataset"]["train_file"],
                "load_config": {},
            },
            "kl_coeff": config["Training"]["kl_coeff"],
            "max_generation_length": 50,
            "model_max_length": 1024,
        },
    }

    ppo_config = (
        PPOConfig(algo_class=PPORLHF)
        .framework("torch")
        .environment(
            "RLHFEnv",
            env_config=agentenv_config,
            disable_env_checking=True,
        )
        .rl_module(
            _enable_rl_module_api=True,
            rl_module_spec=SingleAgentRLModuleSpec(
                RLHFPPOTorchRLModule,
                model_config_dict={
                    "actor_base_model": config["General"]["model_name"],
                    "critic_base_model": config["General"]["rm_name"],
                },
            ),
        )
        .training(
            learner_class=RLHFPPOTorchLearner,
            # optimizer=config["Training"]["optimizer"],
            lr=config["Training"]["learning_rate"],
            num_sgd_iter=1,
            sgd_minibatch_size=config["Training"]["experience_batch_size"],
            train_batch_size=config["Training"]["experience_batch_size"],
            _enable_learner_api=True,
        )
        .rollouts(num_rollout_workers=0)
        .evaluation(
            evaluation_interval=1,
            evaluation_duration_unit="episodes",
        )
        .experimental(
            _disable_preprocessor_api=True,
            _disable_initialize_loss_from_dummy_batch=True,
        )
        .callbacks(
            callbacks_class=make_multi_callbacks(
                [
                    ValueFunctionInitializerCallback,
                ]
            )
        )
    )

    return ppo_config


def main(external_config=None):
    config = common.Config()
    if external_config is not None:
        config.merge(external_config)

    init_ray(config)

    ppo_config = prepare_ppo(config)

    tuner = tune.Tuner(
        PPORLHF,
        param_space=ppo_config,
        run_config=air.RunConfig(
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=50,
                checkpoint_at_end=True,
            ),
            stop={"training_iteration": config["Training"]["training_iteration"]},
        ),
        tune_config=tune.TuneConfig(reuse_actors=False),
    )

    tuner.fit()


if __name__ == "__main__":
    main()
