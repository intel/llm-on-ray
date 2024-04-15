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

from megatron.initialize import initialize_megatron
from llm_on_ray.common.initializer import Initializer
from llm_on_ray.common.logging import logger


class MegatronInitializer(Initializer):
    def __init__(self, config):
        self.config = config
        self.args = {}

    def init(self):
        # self._parse_arguments(ARGUMENTS_SCHEMA, config)
        args = None
        if "megatron_config" in self.config:
            args = self.config["megatron_config"]
            initialize_megatron(ignore_unknown_args=True, external_args=args, allow_no_cuda=True)
        else:
            logger.error("cannot initialize the megatron without the megatron_config")
