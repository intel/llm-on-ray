#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import transformers

from llm_on_ray.common.model import Model
from peft import get_peft_model, LoraConfig


class HuggingFaceModelForCausalLM(Model):
    def __call__(self, config):
        name = config.get("name")
        model_dtype = config.get("dtype")
        model_config = config.get("config", {})
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=model_dtype, **model_config
        )

        lora_config = config.get("lora_config", None)
        if lora_config:
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)

        enable_gradient_checkpointing = config.get("enable_gradient_checkpointing")
        if enable_gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        model.to(dtype=model_dtype, device=config.get("device"))

        return model
