import os
import numpy as np
import pandas as pd 
from datasets import load_dataset
ds = load_dataset("tatsu-lab/alpaca")
train = ds['train'].to_pandas()

def prep_data(df):
    df["context"] = df["input"]
    df["response"] = df["output"]
    df = df[df.response != ""]
    df = df[
        ["instruction", "context", "response"]
    ]

    return df
df_train = prep_data(train)
if not os.path.exists("data/train"):
    os.makedirs("data/train")
df_train.to_json("data/train/train.jsonl", lines=True, orient="records")

