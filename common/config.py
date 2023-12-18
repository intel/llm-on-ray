import os
import yaml
import argparse
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    args, unparsed = parser.parse_known_args()
    return args

def parse_config(config_file=None):
    if config_file is None:
        args = parse_args()
        config_file = args.config_file
    if config_file is None:
        return {}
    if config_file.endswith("yaml"):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open(config_file) as f:
            config = eval(f.read())
    assert isinstance(config, dict)
    return config

def _singleton(cls):
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

def flat(x, separator="."):
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in flat(value):
                k = f'{key}{separator}{k}'
                yield (k, v)
        else:
            yield (key, value)

def pack(x, separator="."):
    return {k:v for k,v in flat(x, separator)}

def rank(key, value):
    if len(key) == 1:
        return {key[0]: value}
    else:
        prefix = key.pop(0)
        return {prefix: rank(key, value)}

def deflat(x, separator="."):
    for key, value in x.items():
        yield rank(key.split(separator), value)

def recursive_merge(dst, src):
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            recursive_merge(dst[key], value)
        else:
            dst[key] = value

def unpack(x, separator="."):
    result = {}
    for i in deflat(x, separator):
        recursive_merge(result, i)
    return result

def mapping(x, table, only_in_table = True):
    new_x = {}
    for k,v in x.items():
        if k in table:
            new_keys = table[k]
            if isinstance(new_keys, list):
                for new_key in new_keys:
                    new_key = new_key.split("#")
                    if len(new_key) == 1:
                        new_x[new_key[0]] = v
                    elif len(new_key) == 2:
                        new_x[new_key[0]] = eval(f"{new_key[1]}(v)")
                    else:
                        pass
            elif isinstance(new_keys, str):
                new_key = new_keys.split("#")
                if len(new_key) == 1:
                    new_x[new_key[0]] = v
                elif len(new_key) == 2:
                    new_x[new_key[0]] = eval(f"{new_key[1]}(v)")
                else:
                    pass
            else:
                pass
        else:
            if not only_in_table:
                new_x[k] = v
    return new_x


def merge_with_mapping(dict1, dict2, table, only_in_table = True):
    dict1_pack = pack(dict1)
    dict2_pack = pack(dict2)
    dict2_pack = mapping(dict2_pack, table, only_in_table)
    recursive_merge(dict1_pack, dict2_pack)
    dict1.clear()
    for k,v in unpack(dict1_pack).items():
        dict1[k] = v
    return dict1

@_singleton
class Config(Dict):
    def __init__(self):
        dict.__init__(self)
        config = parse_config()
        if config is not None:
            self.merge(config)

    def merge(self, config: dict):
        recursive_merge(self, config)

    def merge_with_mapping(self, config: dict, table: dict, only_in_table: bool = True):
        merge_with_mapping(self, config, table, only_in_table)
