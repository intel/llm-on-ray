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

from llm_on_ray import common
from llm_on_ray.common.dataset import Dataset


def local_load(name, **load_config):
    if os.path.isfile(name):
        file = os.path.basename(os.path.abspath(name))
        path = os.path.dirname(os.path.abspath(name))
        dataset = datasets.load_dataset(path, data_files=file, **load_config)
    else:
        dataset = datasets.load_dataset(name, **load_config)
    return dataset["train"]


class HuggingfaceDataset(Dataset):
    def __call__(self, config):
        name = config.get("name")
        load_from_disk = config.get("load_from_disk", False)
        validation_file = config.get("validation_file", None)
        validation_split_percentage = config.get("validation_split_percentage", 0)
        if os.path.exists(name):
            train_dataset = local_load(name)
            if validation_file is not None:
                validation_dataset = local_load(validation_file)
                return datasets.DatasetDict(
                    {"train": train_dataset, "validation": validation_dataset}
                )
            if validation_split_percentage / 100 > 0.0 and validation_split_percentage / 100 < 1.0:
                datasets_dict = train_dataset.train_test_split(
                    test_size=validation_split_percentage / 100
                )
                datasets_dict["validation"] = datasets_dict["test"]
                return datasets_dict
            return datasets.DatasetDict({"train": train_dataset})
        else:
            load_config = config.get("load_config", {})
            if load_from_disk:
                raw_datasets = datasets.load_from_disk(name, **load_config)
                if "validation" not in raw_datasets.keys():
                    raw_datasets["validation"] = datasets.load_from_disk(
                        name, split=f"train[:{validation_split_percentage}%]", **load_config
                    )
                    raw_datasets["train"] = datasets.load_from_disk(
                        name, split=f"train[{validation_split_percentage}%:]", **load_config
                    )
            else:
                common.logger.info("load dataset")
                raw_datasets = datasets.load_dataset(name, **load_config)
                common.logger.info(raw_datasets)

                if "validation" not in raw_datasets.keys():
                    common.logger.info("split")
                    raw_datasets["validation"] = datasets.load_dataset(
                        name, split=f"train[:{validation_split_percentage}%]", **load_config
                    )
                    raw_datasets["train"] = datasets.load_dataset(
                        name, split=f"train[{validation_split_percentage}%:]", **load_config
                    )
                common.logger.info(raw_datasets)
            return raw_datasets
