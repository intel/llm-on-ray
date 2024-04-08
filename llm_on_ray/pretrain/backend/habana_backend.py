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

from ray.train.torch.config import _TorchBackend
from ray.train.torch.config import TorchConfig as RayTorchConfig
from ray.train._internal.worker_group import WorkerGroup
from dataclasses import dataclass


@dataclass
class TorchConfig(RayTorchConfig):
    @property
    def backend_cls(self):
        return EnableHabanaBackend


def habana_import():
    try:
        import habana_frameworks.torch
    except ImportError as habana_not_exist:
        raise ImportError("Please install habana_frameworks") from habana_not_exist


class EnableHabanaBackend(_TorchBackend):
    def on_start(self, worker_group: WorkerGroup, backend_config: RayTorchConfig):
        for i in range(len(worker_group)):
            worker_group.execute_single_async(i, habana_import)
        super().on_start(worker_group, backend_config)
