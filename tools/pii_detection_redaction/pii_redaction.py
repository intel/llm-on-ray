"""
this script is for processing pii-detection-dedaction 
"""

from presidio_analyzer import AnalyzerEngine
from utils import summarize_pii_entities
import time
import json
from redact_pii import redact_pii_with_random_values, redact_pii_with_tags
from detect_pii import detect_other_piis, detect_phone_numbers, merge_outputs

import os, sys
import time 
import argparse
from pprint import pprint
from typing import Dict, List

import ray
import ray.data
import pandas as pd
import numpy as np
from datasets import load_dataset


def detect_redact_pii_for_one_text(text, meta, analyzer):

    detected_phone_numbers = detect_phone_numbers(text, analyzer)

    # get output from bigscience-pii
    detected_other_piis = detect_other_piis(text)
    # merge the two outputs
    piis = merge_outputs(detected_phone_numbers, detected_other_piis)
    #print('Merged PIIs: ', piis)

    if len(piis)>0:
        # save result
        #redact
        redacted_text = redact_pii_with_random_values(text, piis)
        #redacted_text = redact_pii_with_tags(text, piis)

        output = {
            'text': text,
            'redacted': redacted_text,
            'pii': piis,
            "meta": meta
        }

        return output
    else: 
        return None


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        default="tiiuae/falcon-refinedweb",
        required=False,
        help="Name of the dataset repository,e.g. togethercomputer/RedPajama-Data-1T"
    )
    group.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="for local mode, you need to provide local RedPajama dataset repository, e.g. /home/user/local"
    )
    group.add_argument(
        "--cache-dir",
        type=str,
        default='/root/.cache',
        help="Hugging Face cache dir, where the hugging face dataset it stored"
    )
    group.add_argument(
        '--local', 
        default=False, 
        action='store_true', 
        help="whether to use local mode to preprocess data"
    )
    group.add_argument(
        "--load-batch-size", type=int, default=1000, help="only needed if you use streaming mode to read data from hugging face"
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=False,
        default="processed",
        help="Path to binary output file without suffix",
    )
    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--cpu-per-worker", type=int, default=1, help="Number of CPUs to use per worker"
    )
    
    args = parser.parse_args()   
    args.output_path = '/home/user/local'
    return args


def main():
    args = get_args()

    output_dir = os.path.join(args.output_path, args.output_prefix)
    cache_dir = args.cache_dir 

    # init ray
    ray.init(address='auto')
    pprint(ray.cluster_resources())
    num_nodes = len(ray.nodes())
    parallelism = num_nodes * args.cpu_per_worker

    def preprocess_fn(batch: Dict[str, List]) -> pd.DataFrame:
        
        #task_id = ray.get_runtime_context().get_task_id()

        analyzer = AnalyzerEngine()
        outputs = []

        contents = batch['content'].tolist()
        urls = batch['url'].tolist()

        print(contents[0])
        print(urls[0])

        for text, meta in zip(contents, urls):

            output = detect_redact_pii_for_one_text(text, meta, analyzer)

            if output != None:
                outputs.append(output)

        return pd.DataFrame({"results": outputs})

    if args.load_batch_size and not args.local:
        dataset = load_dataset(args.input, streaming=True)['train']
        idx = 1
        for rows in dataset.iter(batch_size=args.load_batch_size):
            print("-----------------------------")
            df = pd.DataFrame(rows)
            ray_dataset = ray.data.from_pandas(df)
            ray_dataset = ray_dataset.repartition(parallelism)

            tokenized_data = ray_dataset.map_batches(preprocess_fn, batch_format="numpy", batch_size=None)
            tokenized_data.select_columns(cols=["results"]).write_json(output_dir)

            print(f"{idx} * {args.load_batch_size} samples were written to disk.")
            idx += 1
            print("============================")
            if idx == 2:
                sys.exit()
    else:
        pass 
        #os.environ["RED_PAJAMA_DATA_DIR"] = args.data_dir 
        #dataset = load_dataset(args.input, cache_dir=cache_dir, streaming=True)['train']
    

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nthis script took {end-start}s.")

