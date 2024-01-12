import torch
import numpy as np

from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    LEARNER_STATS_KEY,
)
from ray.rllib.evaluation.metrics import RolloutMetrics

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from common.agentenv.rlhf_env import generate_response
from .rlhf_buffer import Buffer, BufferItem


class RLHFSampler:
    """This sampler is a local sampler for LLMEnv.

    The underlying env is an LLMEnv which creates a batch of prompts and the agent has
    to generate a response for each prompt. Then the env evaluate those responses and
    returns a reward signal.
    """

    def __init__(self, module, env):
        self._env = env
        self._module = module
        self.max_generation_length = self._env.max_generation_length

    def sample(self, batch_size: int, **kwargs) -> SampleBatch:
        # TODO (Kourosh): Can we use batch inference here?
        batches = Buffer()
        for i in range(batch_size):
            obs, _ = self._env.reset()

            output = generate_response(
                self._module.actor,
                input_ids=torch.tensor(obs["input_ids"])[None],
                max_length=self.max_generation_length,
                eos_token_id=self._env.tokenizer.eos_token_id,
            )

            # construct the action
            n_generated_tokens = output["n_generated_tokens"]
            n_input_tokens = output["n_input_tokens"]
            generated_tokens = output["sequence"][-n_generated_tokens:]

            value = self._module.critic(output["sequence"]).detach().item()

            action = {
                "sequence": generated_tokens.detach().numpy()[0],
                "response_mask": np.array([0] * n_input_tokens + [1] * n_generated_tokens),
                "logits": output["logits"].detach().numpy()[0],  # remove batch dimension
                "attention_mask": np.array([1] * (n_input_tokens + n_generated_tokens)),
            }

            next_obs, reward, terminated, truncated, info = self._env.step(action)

            assert terminated is True, "The env should be terminated after each step."

            # value and reward should be both float scalars here.
            advantages = value - reward

            batches.append(
                BufferItem(
                    **{
                        SampleBatch.OBS: obs,
                        SampleBatch.ACTIONS: action,
                        SampleBatch.REWARDS: reward,
                        SampleBatch.INFOS: info,
                        Postprocessing.VALUE_TARGETS: value,
                        Postprocessing.ADVANTAGES: advantages,
                    }
                )
            )

        return batches.convert_to_sample_batch()


class PPORLHF(PPO):
    def setup(self, config: AlgorithmConfig) -> None:
        super().setup(config)

        self.rlhf_module = self.learner_group._learner.module[DEFAULT_POLICY_ID]

        self.env = self.workers.local_worker().env

        # create a copy of module and env in the algorithm.
        self.sampler = RLHFSampler(self.rlhf_module, self.env)

    def training_step(self):
        train_batch = self.sampler.sample(batch_size=self.config.train_batch_size)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        train_results = self.learner_group.update(
            train_batch,
            minibatch_size=self.config.sgd_minibatch_size,
            num_iters=self.config.num_sgd_iter,
        )

        policies_to_update = {DEFAULT_POLICY_ID}
        kl_dict = {
            pid: train_results[pid][LEARNER_STATS_KEY].get("kl") for pid in policies_to_update
        }
        self.learner_group.additional_update(
            module_ids_to_update=policies_to_update,
            sampled_kl_values=kl_dict,
            timestep=self._counters[NUM_AGENT_STEPS_SAMPLED],
        )

        return train_results

    def evaluate(self):
        # breakpoint()

        train_batch = self.sampler.sample(batch_size=1)
        rewards = train_batch[SampleBatch.INFOS]["r_align"]

        self.evaluation_metrics = {"evaluation": {"reward": rewards.item()}}

        eval_metric = RolloutMetrics(
            episode_length=1,
            episode_reward=rewards.item(),
            agent_rewards={},
            custom_metrics={},
            perf_stats={},
            hist_data={},
            media={},
            episode_faulty={},
            connector_metrics={},
        )

        self.workers.local_worker().sampler.metrics_queue.put(eval_metric)
        # self.evaluation_workers.local_worker().sampler.metrics_queue.put(eval_metric)
        self.workers.local_worker().sampler._env_runner_obj._perf_stats.incr("iters", 1)

        return self.evaluation_metrics
