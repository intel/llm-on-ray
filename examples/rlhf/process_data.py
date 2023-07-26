import os
import numpy as np
import pandas as pd 
from datasets import load_dataset
import argparse



def prep_data(df, colume):
    df = df[
        colume
    ]
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default='Dahoas/rm-static',
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default='data',
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default='reward',
        choices=['reward', 'rlhf']
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset)

    train = ds['train'].to_pandas()
    test  = ds['test'].to_pandas()

    if args.mode == 'reward':
        df_train = prep_data(train, colume=["prompt", "chosen", "rejected"])
        df_test  = prep_data(test, colume=["prompt", "chosen", "rejected"])
    elif args.mode == 'rlhf':
        df_train = prep_data(train, colume=["prompt"])
        df_test  = prep_data(test, colume=["prompt"])
    else:
        raise ValueError('unsupport mode')

    save_dir = os.path.join(args.save_dir, args.mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_train.to_json(os.path.join(save_dir, 'train.jsonl'), lines=True, orient="records")
    df_test.to_json(os.path.join(save_dir, 'test.jsonl'), lines=True, orient="records")
