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


def patch_yaml_config():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str)
    parser.add_argument("--models", type=str)
    parser.add_argument("--peft_lora", action="store_true", default=False)
    args = parser.parse_args()

    conf_path = args.conf_path
    with open(conf_path, encoding="utf-8") as reader:
        result = yaml.load(reader, Loader=yaml.FullLoader)
        result["General"]["base_model"] = args.models
        if args.models == "mosaicml/mpt-7b":
            result["General"]["config"]["trust_remote_code"] = True
        else:
            result["General"]["config"]["trust_remote_code"] = False
        if args.models == "EleutherAI/gpt-j-6b" or args.models == "gpt2":
            result["General"]["gpt_base_model"] = True
        else:
            result["General"]["gpt_base_model"] = False
        result["Training"]["epochs"] = 1
        if args.models == "gpt2":
            # to verify oneccl
            result["Training"]["num_training_workers"] = 2
        else:
            result["Training"]["num_training_workers"] = 1

        if args.peft_lora:
            result["General"]["lora_config"] = {
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            }
            if args.models == "mistralai/Mistral-7B-v0.1":
                result["General"]["lora_config"]["target_modules"] = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ]
            elif args.models == "google/gemma-2b":
                result["General"]["lora_config"]["target_modules"] = ["k_proj", "v_proj"]
            else:
                result["General"]["lora_config"]["target_modules"] = None
        else:
            result["General"]["lora_config"] = None

    with open(conf_path, "w") as output:
        yaml.dump(result, output, sort_keys=False)


if __name__ == "__main__":
    patch_yaml_config()
