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
        config["General"]["checkpoint_dir"] = "./checkpoint"
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
