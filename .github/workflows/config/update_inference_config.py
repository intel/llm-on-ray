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


def update_inference_config(
    config_file: str, output_file: str, deepspeed: bool, ipex: bool, vllm: bool
):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["deepspeed"] = deepspeed
        config["ipex"]["enabled"] = ipex
        config["vllm"]["enabled"] = vllm

    with open(output_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Adjust Inference Config File")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--vllm", action="store_true")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    update_inference_config(
        args.config_file, args.output_file, args.deepspeed, args.ipex, args.vllm
    )
