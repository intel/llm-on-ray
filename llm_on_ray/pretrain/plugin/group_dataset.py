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

import os
import datasets

from llm_on_ray.common.dataset import Dataset


class GroupDataset(Dataset):
    def __call__(self, config):
        path = config.get("path")
        load_from_disk = config.get("load_from_disk", False)
        load_config = config.get("load_config", {})
        names = self.get_all_file(path)
        if load_from_disk:
            return [datasets.load_from_disk(name, **load_config) for name in names]
        else:
            return [datasets.load_dataset(name, **load_config) for name in names]

    def get_all_file(self, path):
        files = os.listdir(path)
        list.sort(files)
        return [os.path.join(path, file) for file in files]
