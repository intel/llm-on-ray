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

import yaml
import argparse


def update_finetune_config(config_file, base_model):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # due to compute node can't connect network
        # base models are downloaded as local files in directory ~/models/
        # avaiable base models are:
        #
        # Mistral-7B-v0.1
        # Llama-2-7b
        # pythia-1.4b
        # pythia-2.8b
        # pythia-70m
        # gpt-j-6b
        # pythia-160m
        # pythia-410m
        # pythia-12b
        # pythia-1b
        # pythia-6.9b

        config["General"]["base_model"] = base_model
        config["General"]["output_dir"] = "./output"
        config["General"]["save_strategy"] = "no"
        config["Training"]["device"] = "GPU"
        config["Training"]["resources_per_worker"]["CPU"] = 1
        config["Training"]["resources_per_worker"]["GPU"] = 1
        config["Training"]["accelerate_mode"] = "DDP"
        config["Training"]["logging_steps"] = 1

    with open(config_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Finetuning on Intel GPU")
    parser.add_argument("--config_file", type=str, required=True, default=None)
    parser.add_argument("--base_model", type=str, required=True, default=None)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    update_finetune_config(args.config_file, args.base_model)
