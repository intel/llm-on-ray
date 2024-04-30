import yaml
import sys


def modify_yaml(argv=None):
    import argparse

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
    modify_yaml(sys.argv[1:])
