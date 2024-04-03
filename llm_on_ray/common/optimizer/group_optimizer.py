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

import torch
from llm_on_ray.common.optimizer import Optimizer


class GroupOptimizer(Optimizer):
    def __call__(self, model, config):
        optimizer_name = config.get("name", "SGD")
        optimizer_config = config.get("config", {})
        optimizer_type = eval("torch.optim.%s" % (optimizer_name))

        optimizer_grouped_parameters = self.get_grouped_parameters(model, config)
        optimizer = optimizer_type(optimizer_grouped_parameters, **optimizer_config)

        return optimizer

    def get_grouped_parameters(self, model, config):
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
