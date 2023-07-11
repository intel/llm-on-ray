"""this script is for processing redpajama data for pre-training."""

import os
import time 
import argparse
from pprint import pprint
from typing import Dict, List

import ray
import ray.data
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from indexed_dataset import MMapIndexedDatasetBuilder


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Name of the dataset repository,e.g. togethercomputer/RedPajama-Data-1T-Sample"
    )
    group.add_argument(
        '--stream', 
        default=False, 
        action='store_true', 
        help="whether to load data from hugging face using streaming mode"
    )
    group.add_argument(
        "--load-batch-size", type=int, default=1000, help="only needed if you use streaming mode to read data from hugging face"
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="One sample's maximal token length, for sample with less tokens, we concatenate multiple tokens up to this many tokens ",
    )
    group.add_argument(
        '--drop_tokens', 
        default=False, 
        action='store_true', 
        help="whether to drop tokens while truncating after concatenating"
    )
    group.add_argument(
        "--eos-text",
        type=str,
        default='<|endoftext|>',
        help="Path to binary output file without suffix",
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--output-format",
        type=str,
        required=True,
        choices=[
            "megatron",
            "json"
        ],
        help="use megatron to the tokenized text to bin and idx files",
    )
    group.add_argument(
        '--save-on-host', 
        default=False, 
        action='store_true', 
        help="whether to write data only on the host machine. Only applicable for non-streaming work"
    )
    group.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="number of rows in one JSON file. This argument is only applicable for json output format.",
    )
    group.add_argument(
        "--save-on-source",
        default=False,
        action='store_true', 
        help="whether to write the megatron data based on data sources",
    )
    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--cpu-per-worker", type=int, default=1, help="Number of CPUs to use per worker"
    )
    
    args = parser.parse_args()   
    args.output_path = '/home/user/local'
    return args


def save_megatron(out_file, idx_file, docs):
    data_builder = MMapIndexedDatasetBuilder(out_file, dtype=np.uint16)

    for doc in docs['tokens']:    
        data_builder.add_item(np.array(doc, dtype=data_builder.dtype))
    
    data_builder.end_document()                        
    data_builder.finalize(idx_file)
    

def build_megatron_central(tokenized_data, save_on_source, output_dir):
    
    if save_on_source:
        all_sources = tokenized_data.unique('source')

        for src in all_sources:
            def filter_source(batch: pd.DataFrame) -> pd.DataFrame:
                return batch[batch['source'] == src]

            tmp_data = tokenized_data.map_batches(filter_source, batch_format='pandas', batch_size=None)
            tmp_rows = tmp_data.count()  

            out_file = f'{output_dir}/{src}.bin'
            idx_file = f'{output_dir}/{src}.idx'
            
            for docs in tmp_data.iterator().iter_batches(batch_size=tmp_rows):
                save_megatron(out_file, idx_file, docs)
    else:
        out_file = f'{output_dir}/all_redpajama.bin'
        idx_file = f'{output_dir}/all_redpajama.idx'
        
        num_rows = tokenized_data.count()
        for docs in tokenized_data.iterator().iter_batches(batch_size=num_rows):
            save_megatron(out_file, idx_file, docs)


def build_megatron_distributed(tokenized_data, save_on_source, output_dir):
                            
    if save_on_source:
        def write_megatron(batch: pd.DataFrame) -> pd.DataFrame:
            task_id = ray.get_runtime_context().get_task_id()
            all_sources = batch['source'].unique()
            for src in all_sources:
                batch = batch[batch['source'] == src]

                if not os.path.exists(f"{output_dir}/{src}"):
                    os.makedirs(f"{output_dir}/{src}")

                out_file = f'{output_dir}/{src}/{task_id[:20]}.bin'
                idx_file = f'{output_dir}/{src}/{task_id[:20]}.idx'
            
                save_megatron(out_file, idx_file, batch)

            return pd.DataFrame({'task_id': [task_id]})
    else:
        def write_megatron(batch: pd.DataFrame) -> pd.DataFrame:

            task_id = ray.get_runtime_context().get_task_id()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            out_file = f'{output_dir}/{task_id[:20]}.bin'
            idx_file = f'{output_dir}/{task_id[:20]}.idx'
        
            save_megatron(out_file, idx_file, batch)

            return pd.DataFrame({'task_id': [task_id]})

    task_ids = tokenized_data.map_batches(write_megatron, batch_format="pandas", batch_size=None)
    task_ids.materialize()


def make_megatron_dataset(tokenized_data, save_on_host, save_on_source, output_dir):

    if save_on_host:
        tokenized_data = tokenized_data.repartition(1)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        build_megatron_central(tokenized_data, save_on_source, output_dir)
        
        if save_on_source:
            for src in ['arxiv', 'book', 'common_crawl', 'c4', 'wikipedia', 'stackexchange', 'github']:
                build_megatron_central(tokenized_data, src, output_dir)
                print(f"samples from {src} data source were written to disk!") 
        else:
            build_megatron_central(tokenized_data, 'all', output_dir)
            print(f"all samples from data source were written to disk") 

    else:
        build_megatron_distributed(tokenized_data, save_on_source, output_dir)


def main():
    args = get_args()

    output_dir = f'{args.output_path}/{args.output_prefix}'
    max_length = args.max_length
    drop_tokens = args.drop_tokens 
    
    use_sample = 'sample' in args.input.lower()
    text_field = 'text'
    meta_field = 'meta' if use_sample else 'red_pajama_subset'

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', use_fast=True)
    eos_tokens = tokenizer(args.eos_text)['input_ids']
    #pad_tokens = tokenizer('<|padding|>')['input_ids']

    ray.init(address='auto')
    pprint(ray.cluster_resources())
    num_nodes = len(ray.nodes())
    parallelism = num_nodes * args.cpu_per_worker

    def preprocess_json(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    
        # load tokenizer took 0.15s 
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', use_fast=True)
        samples = batch[text_field].tolist()
        buffer = []
        token_chunked = []
        
        for sample in samples:
            encoded = tokenizer(sample, 
                                truncation=False,
                                padding=False)                    
            ids = encoded['input_ids']
            buffer = buffer + ids + eos_tokens
            
            while len(buffer) >= max_length:
                concat_sample = buffer[:max_length]
                token_chunked.append(concat_sample)
                buffer = buffer[max_length:]
        
        if not drop_tokens:
            #add padding to sequence shorter than max_length
            buffer = buffer + [1]*(max_length - len(buffer))
            token_chunked[-1] = buffer    
        
        return {"tokens": np.asarray(token_chunked)}
    
    if use_sample:
        def preprocess_megatron(batch: Dict[str, np.ndarray]) -> pd.DataFrame:

            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', use_fast=True)
            
            metas = batch[meta_field].tolist()
            samples = batch[text_field].tolist()
            
            ids = []
            lens = []
            
            for sample in samples:
                encoded = tokenizer(sample, 
                                    truncation=False,
                                    padding=False)  
                sample_id = encoded['input_ids'] + eos_tokens
                ids.append(sample_id)
                lens.append(len(sample))
            
            sources = []
            for meta in metas:
                meta_dict = eval(meta)
                meta_keys = meta_dict.keys()
                if 'arxiv_id' in meta_keys:
                    sources.append('arxiv')
                elif 'pred_label_prob' in meta_keys:
                    sources.append('common_crawl')
                elif 'short_book_title' in meta_keys:
                    sources.append('book')
                elif 'title' in meta_keys:
                    if 'url' in meta_keys:
                        if 'wikipedia' in meta_dict['url']:
                            sources.append('wikipedia')
                        else:
                            sources.append('book')
                    else:
                        sources.append('book')    
                else:
                    sources.append(meta_dict['source'])

            return pd.DataFrame({"tokens": ids, "length": lens, "source": sources})
    else:
        def preprocess_megatron(batch: Dict[str, np.ndarray]) -> pd.DataFrame:

            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', use_fast=True)

            sources = batch[meta_field].tolist()
            samples = batch[text_field].tolist()
            
            ids = []
            lens = []
            
            for sample in samples:
                encoded = tokenizer(sample, 
                                    truncation=False,
                                    padding=False)  
                sample_id = encoded['input_ids'] + eos_tokens
                ids.append(sample_id)
                lens.append(len(sample))
        
            return pd.DataFrame({"tokens": ids, "length": lens, "source": sources})
    
    if args.stream:
        if use_sample:
            dataset = load_dataset(args.input, streaming=True)
        else:
            dataset = load_dataset(args.input, 'default', streaming=True)

        idx = 1
        for rows in dataset['train'].iter(batch_size=args.load_batch_size):
            df = pd.DataFrame(rows)
            ray_dataset = ray.data.from_pandas(df)
            ray_dataset = ray_dataset.repartition(parallelism)
            if args.output_format == 'json':
                tokenized_data = ray_dataset.map_batches(preprocess_json, batch_format="numpy", batch_size=None)

                total_rows = tokenized_data.count()
                num_partition = total_rows//args.num_samples if not args.save_on_host else 1 
                tokenized_data = tokenized_data.repartition(num_partition)

                if args.save_on_host: 
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    index = 0
                    for batch in tokenized_data.iterator().iter_batches(batch_size=args.num_samples):
                        batch.to_json(f'{output_dir}/partition_{index}.json', orient="records", lines=True)
                        index += 1
                else:
                    tokenized_data.write_json(output_dir)
            elif args.output_format == 'megatron':
                tokenized_data = ray_dataset.map_batches(preprocess_megatron, batch_format="numpy", batch_size=None)
                make_megatron_dataset(tokenized_data, args.save_on_host, args.save_on_source, output_dir)

            idx += 1 
            if idx % 100 == 0:
                print(f"{idx} * {args.load_batch_size} samples are written to disk.")

    else:
        raw_dataset = load_dataset(args.input)['train']
        ray_dataset = ray.data.from_huggingface(raw_dataset)
        # create multiple data blocks  
        ray_dataset = ray_dataset.repartition(parallelism, shuffle=True)

        if args.output_format == 'json':
            fn_name = "preprocess_json"    
        elif args.output_format == 'megatron':
            fn_name = "preprocess_megatron"

        tokenized_data = ray_dataset.map_batches(eval(fn_name), batch_format="numpy", batch_size=None)
        total_rows = tokenized_data.count()
        print(f"Total number of rows after processing: {total_rows}")

        if args.output_format == 'json':
            num_partition = total_rows//args.num_samples if not args.save_on_host else 1 
            tokenized_data = tokenized_data.repartition(num_partition)
            if args.save_on_host: 
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                index = 0
                for batch in tokenized_data.iterator().iter_batches(batch_size=args.num_samples):
                    batch.to_json(f'{output_dir}/partition_{index}.json', orient="records", lines=True)
                    index += 1
            else:
                tokenized_data.write_json(output_dir)
        elif args.output_format == 'megatron':
            make_megatron_dataset(tokenized_data, args.save_on_host, args.save_on_source, output_dir)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nthis script took {end-start}s.")

