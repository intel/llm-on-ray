from typing import Any
import gymnasium as gym

import numpy as np
import torch
import tree

from ray.rllib.utils.spaces.repeated import Repeated
import gymnasium.spaces as sp

from .agentenv import AgentEnv
from ..load import load_dataset, load_model, load_tokenizer


def generate_response(
    model: torch.nn.Module, 
    *, 
    input_ids: torch.tensor, 
    max_length:int, 
    eos_token_id: int
):
    """Generate a response using the model."""
    generated_sequence = []
    probs_list = []
    model_in = torch.clone(input_ids)
    with torch.no_grad():
        for i in range(max_length):
            # Get the logits using the forward() method
            logits = model(model_in).logits

            # Get the logits of the last token
            next_token_logits = logits[:, -1, :]

            # Apply a softmax function to the logits to get the probabilities
            probs = next_token_logits.softmax(dim=-1)

            # if not probs_list:
            #     prev_probs = logits[:, :-1, :].softmax(dim=-1)
            #     probs_list.extend(prev_probs.unbind(1))

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the probabilities and the generated token
            generated_sequence.append(next_token)
            # probs_list.append(probs)

            # Update the input_ids with the generated token
            model_in = torch.cat([model_in, next_token], dim=-1)

            if next_token.item() == eos_token_id:
                break

        # Decode and print the generated sequence
        generated_tokens = torch.cat(generated_sequence, dim=-1)

    # Stack the probabilities tensor --> this resulted in N - 1 probs missing the last one, to get that we have to do another forward pass, so we may as well do one round in the end to compute all the probs.
    # probs_tensor = torch.concat(probs_list, dim=0)

    logit_tensor = model(model_in).logits

    return {
        "sequence": torch.cat([input_ids, generated_tokens], dim=-1),
        "logits": logit_tensor,
        "n_input_tokens": input_ids.shape[-1],
        "n_generated_tokens": generated_tokens.shape[-1],
    }

def compute_approx_kl(
    logits: torch.Tensor,
    logits_base: torch.Tensor,
) -> torch.Tensor:
    probs = logits.softmax(dim=-1)
    probs_base = logits_base.softmax(dim=-1)
    log_ratio = (probs / probs_base).log()
    approx_kl = probs * log_ratio
    approx_kl = approx_kl.sum(dim=-1).mean()
    return approx_kl


class RLHFEnv(gym.Env, AgentEnv):

    def __init__(self, config):

        self.config = config
        agentenv_config = config.get("config")

        # Prompt dataset
        self.prompt_dataset  = load_dataset(agentenv_config.get("datasets"))
        self.dsize = len(self.prompt_dataset)
        
        # base tokenizer
        self.tokenizer = load_tokenizer(agentenv_config.get("tokenizer"))
        vocab_size = self.tokenizer.vocab_size
        model_max_length = min(agentenv_config['model_max_length'], self.tokenizer.model_max_length)

        # reward and sft model
        self.reward_model = load_model(agentenv_config.get("reward_model"))
        self.sft_model = load_model(agentenv_config.get("sft_model"))
       
        # the KL coefficient
        self.kl_coeff = agentenv_config["kl_coeff"]
        # The maximum length of the generated text
        self.max_generation_length = agentenv_config["max_generation_length"]

        # action space
        self.action_space = sp.Dict({
            "sequence": Repeated(sp.Discrete(vocab_size), max_len=model_max_length),
            "attention_mask": Repeated(sp.Discrete(2), max_len=model_max_length),
            "response_mask": Repeated(sp.Discrete(2), max_len=model_max_length),
            "logits": Repeated(
                sp.Box(0, 1, shape=(vocab_size,)), max_len=model_max_length
            ),
        })

        # observation space
        self.observation_space = sp.Dict({
            "input_ids": Repeated(sp.Discrete(vocab_size), max_len=model_max_length),
            "attention_mask": Repeated(sp.Discrete(2), max_len=model_max_length),
        })

    def reset(self, *, seed=None, options=None):
    
        if seed:
            np.random.seed(seed)
        index = np.random.randint(self.dsize)
        prompt = self.prompt_dataset[index]["prompt"]
        prompt_tokens = self.tokenizer(prompt, return_tensors="np")
        # remove the batch dimension since we can only do one sentence generation at a time
        prompt_tokens = tree.map_structure(lambda x: x[0], prompt_tokens)

        return prompt_tokens, {}

    def step(self, action):

        sequence = action["sequence"]
        response_mask = action["response_mask"]
        attention_mask = action["attention_mask"]
        logits = action["logits"]

        n_response_tokens = response_mask.sum()

        r_align = self.reward_model.value(sequence, attention_mask)
        # For now, take the reward of the last token of the sentence as the overall reward.
        r_align = r_align[-1].item()

        # Compute the probs from the sft model for the same number of tokens
        sequence = torch.tensor(sequence, dtype=torch.long)[None] # add batch dim
        sft_output = generate_response(
            self.sft_model, 
            input_ids=sequence, 
            max_length=n_response_tokens, 
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logits = torch.tensor(logits, dtype=torch.float32)[None] # add batch dim
        # only compute kl on the response tokens
        r_kl = compute_approx_kl(
            logits[:, -n_response_tokens:], # the inner term
            sft_output["logits"][:, -n_response_tokens:] # the outer term
        ).item()

        reward = r_align - self.kl_coeff * r_kl

        info = {
            "r_align": r_align, 
            "r_kl": r_kl, 
            "n_response_tokens": n_response_tokens
        }

        # Produce a random reward when we reach the goal.
        return self.observation_space.sample(), reward, True, False, info