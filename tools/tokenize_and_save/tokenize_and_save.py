"""
this script is for tokenizing various data input in json format and saving to megatron-format.
"""

import os
import time 
import argparse
from pprint import pprint
from typing import Dict
import glob 

import ray
import ray.data
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
    
from indexed_dataset import MMapIndexedDatasetBuilder


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="directory contains the input files to be processed"
    )
    group.add_argument(
        "--file-type",
        type=str,
        default="json",
        help="the file type of the input data, only json, jsonl and parquet are supported, default type is json"
    )
    group.add_argument(
        "--ray-load",
        action='store_true',
        default=False,
        help="whether or not to use native ray data API to load data; if set true, you need to ensure that the data is accessible by each worker, e.g. it NFS"
    )
    group.add_argument(
        "--tokenizer",
        type=str,
        default='togethercomputer/LLaMA-2-7B-32K',
        help="tokenizer name"
    )
    group.add_argument(
        "--use-slow",
        action='store_true',
        default=False,
        help="whether or not to use slow tokenizer"
    )
    group.add_argument(
        "--model-max-length", type=int, default=100000000000, help="batch size"
    )
    group.add_argument(
        "--data-field",
        type=str,
        required=True,
        help="the column where the text is stored"
    )
    group.add_argument(
        "--load-batch-size", type=int, default=1000, help="batch size"
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="directory to save output files",
    )
    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--cpu-per-node", type=int, default=1, help="Number of CPUs to use per cluster node"
    )
    
    args = parser.parse_args()   
    args.output_path = '/home/user/local'
    return args


def tokenize_batch(tokenizer_name, batch, data_field, model_max_length, use_fast):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast= use_fast)
    tokenizer.model_max_length = model_max_length

    eos_token = tokenizer.eos_token
    eos_id = tokenizer(eos_token)['input_ids']

    samples = batch[data_field].tolist()

    ids = []

    for sample in samples:
        encoded = tokenizer(sample, 
                            truncation=False,
                            padding=False)  
        sample_id = encoded['input_ids'] + eos_id
        ids.append(sample_id)

    tokenized_batch = pd.DataFrame({"tokens": ids})

    return tokenized_batch


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def save_megatron(output_dir, task_id, tokenized_batch, vocab_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_file = os.path.join(output_dir, f"{task_id}.bin")
    idx_file = os.path.join(output_dir, f"{task_id}.idx")
    
    data_builder = MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))

    for doc in tokenized_batch['tokens']:    
        data_builder.add_item(np.array(doc, dtype=data_builder.dtype))
    
    data_builder.end_document()                        
    data_builder.finalize(idx_file)


def main():
    args = get_args()

    output_dir = args.output_dir
    tokenizer_name = args.tokenizer
    model_max_length = args.model_max_length
    data_field = args.data_field
    input_dir = args.input_dir
    file_type = args.file_type
    use_fast = not args.use_slow
    ray_load = args.ray_load
        
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True

    if file_type == 'json':
        input_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    elif file_type == 'jsonl':
        input_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    elif file_type == 'parquet':
        input_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    else:
        raise ValueError("Please specify the correct file type. Choose one from json, jsonl and parquet.")

    vocab_size = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast).vocab_size 
    
    # init ray
    ray.init(address='auto')
    pprint(ray.cluster_resources())
    num_nodes = len(ray.nodes())
    parallelism = num_nodes * args.cpu_per_node
    
    ray_job_id = ray.runtime_context.get_runtime_context().get_job_id()
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_parent = os.path.dirname(output_dir)
    name_dir = os.path.join(output_parent, f"{timestr}_{ray_job_id}.csv")
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

    def preprocess_megatron(batch: Dict[str, np.ndarray]) -> pd.DataFrame:
        
        task_id = ray.get_runtime_context().get_task_id()
        tokenized_batch = tokenize_batch(tokenizer_name, batch, data_field, model_max_length, use_fast)
        save_megatron(output_dir, task_id, tokenized_batch, vocab_size)

        return pd.DataFrame({'task_id': [task_id]})

    if ray_load:
        if file_type == 'parquet':
            ray_dataset = ray.data.read_parquet(input_files, columns=[args.data_field])
        else:
            ray_dataset = ray.data.read_json(input_files)
        tokenized_data = ray_dataset.map_batches(preprocess_megatron, batch_format="numpy", batch_size=None)
        tokenized_data.write_csv(name_dir)
    else:
        if file_type == 'parquet':
            dataset = load_dataset("parquet", data_files=input_files, streaming=True)['train']
        else:
            dataset = load_dataset("json", data_files=input_files, streaming=True)['train']
        dataset = dataset.select_columns(args.data_field)
        
        idx = 1
        for rows in dataset.iter(batch_size=args.load_batch_size):
            df = pd.DataFrame(rows)
            ray_dataset = ray.data.from_pandas(df)
            ray_dataset = ray_dataset.repartition(parallelism)

            tokenized_data = ray_dataset.map_batches(preprocess_megatron, batch_format="numpy", batch_size=None)
            tokenized_data = tokenized_data.repartition(1)
            tokenized_data.write_csv(name_dir)        
            idx += 1

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nthis script took {end-start}s.")