import os
import numpy as np
import pandas as pd 
from datasets import load_dataset
ds = load_dataset("OpenAssistant/oasst1")
train = ds['train'].to_pandas()
val = ds['validation'].to_pandas()

def prep_data(df):
    df_assistant = df[(df.role == "assistant") & (df["rank"] == 0.0)].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["response"] = df_assistant["text"].values

    inputs = []
    contexts = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)
        contexts.append("")
        parent_ids.append(input.parent_id)

    df_assistant["instruction"] = inputs
    df_assistant["context"] = contexts

    df_assistant = df_assistant[df_assistant.lang == "en"]

    df_assistant = df_assistant[
        ["instruction", "context", "response"]
    ]

    return df_assistant
df_train = prep_data(train)
df_val = prep_data(val)
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/validation"):
    os.makedirs("data/validation")
df_train.to_json("data/train/train.jsonl", lines=True, orient="records")
df_val.to_json("data/validation/validation.jsonl", lines=True, orient="records")

