import transformers
from transformers.models.auto import MODEL_MAPPING

import torch
import torch.nn as nn

from llm_on_ray.common.model import Model


class HuggingFaceRewardModel(Model):
    def __call__(self, config):
        name = config.get("name")
        if name is None:
            raise ValueError("Model config error, config should contain a model name")
        model_config = config.get("config", {})

        auto_config = transformers.AutoConfig.from_pretrained(name, **model_config)
        model_cls = MODEL_MAPPING[type(auto_config)]

        try:
            model = get_reward_model(model_cls, name)
        except Exception:
            print("Load reward model error")
            exit()

        return model


def get_reward_model(model_cls, name):
    class RewardModel(model_cls):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)

            # The additional value head.
            if hasattr(self.config, "hidden_size"):
                self.value_head = nn.Linear(self.config.hidden_size, 1)
            elif hasattr(self.config, "n_embd"):
                self.value_head = nn.Linear(self.config.n_embd, 1)
            else:
                raise ValueError("current model does not support")

            self.post_init()

        def forward(
            self,
            chosen_input_ids,
            chosen_attention_mask,
            rejected_input_ids,
            rejected_attention_mask,
            **kwargs,
        ) -> torch.Tensor:
            chosen_value = self.value(chosen_input_ids, chosen_attention_mask)
            rejected_value = self.value(rejected_input_ids, rejected_attention_mask)
            return torch.stack([chosen_value, rejected_value], dim=1)

        def value(self, input_ids, attention_mask) -> torch.Tensor:
            """Forward function predicts whether chosen response has a higher reward."""
            # Force inputs to be torch tensors.
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids).to(self.device)
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask).to(self.device)

            last_hidden_state = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )["last_hidden_state"]

            values = self.value_head(last_hidden_state)
            # Remove the last dimension, since there is only a single value per token.
            value = values.squeeze(-1)

            return value

        def generate(self, *kwargs):
            raise NotImplementedError("Reward model does not generate token.")

    return RewardModel.from_pretrained(name)
