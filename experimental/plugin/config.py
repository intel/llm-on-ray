import os
import argparse
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    args, unparsed = parser.parse_known_args()
    return args

def parse_config():
    args = parse_args()
    if args.config_path is None:
        args.config_path = os.environ.get("CONFIG_PATH")
    assert args.config_path is not None
    with open(args.config_path) as f:
        config = eval(f.read())
    return config

def _singleton(cls):
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner
    
@_singleton
class Config(Dict):
    def __init__(self):
        dict.__init__(self)
        _config = parse_config()
        for key, value in _config.items():
            self.__setitem__(key, value)
