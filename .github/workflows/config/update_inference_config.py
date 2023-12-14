import yaml
import argparse


def update_inference_config(config_file: str, output_file: str, deepspeed: bool, ipex: bool):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["deepspeed"] = deepspeed
        config["ipex"]["enabled"] = ipex

    with open(output_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Adjust Inference Config File")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--deepspeed", action='store_true')
    parser.add_argument("--ipex", action='store_true')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    update_inference_config(args.config_file, args.output_file, args.deepspeed, args.ipex)
