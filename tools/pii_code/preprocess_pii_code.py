"""
This script is doing PII detection and redaction for code. Asumes data is available as a folder of jsonl files.
These can be downloaded by the user from huggingface or be the result of a previous data preprocessing pipeline
(filtering, dedup,..). The processed data is written to an output folder in json format as well
"""

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

from pii_detection import scan_pii_batch
from pii_redaction import redact_pii_batch, random_replacements
import logging
from datetime import datetime
    
def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--data_dir",
        type=str,
        required=False,
        help="provide local dataset folder with json files, e.g. /home/user/local/github"
    )
    group.add_argument(
        "--load-batch-size", type=int, default=1000, help="only needed if you use streaming mode to read data with hugging face load_dataset (also works with pre-downloaded datasets)"
    )
    group.add_argument(
        "--skip", type=int, default=None, help="how many samples to skip"
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory path to store processed output e,g '/home/user/local/output_pii_github'",
    )
    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--cpu-per-worker", type=int, default=1, help="Number of CPUs to use per worker"
    )
    
    args = parser.parse_args()   
    return args

def generate_list_jsonl(dir):
    files = []
    for file in os.listdir(dir):
        if file.endswith('.jsonl'):
            files.append(os.path.join(dir, file))
    print(files)
    logger.info('list of jsons: {}'.format(files))
    return files   

def pii_detection_redaction_batch(batch):
    pii_batch = scan_pii_batch(batch)
    replacements = random_replacements()
    redacted_pii_batch = redact_pii_batch(pii_batch, replacements)
    redacted_batch_df = pd.DataFrame(redacted_pii_batch)
    return redacted_batch_df

def main():
    args = get_args()
    print(args)
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    # exception_dir = args.output_path+'/exceptions/'
    # if not os.path.exists(exception_dir):
    #     os.mkdir(exception_dir)
    log_dir = args.output_path+'/logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    dataset_name = os.path.basename(os.path.normpath(args.data_dir))
    log_filename = f"{dataset_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logging.basicConfig(filename=log_dir+log_filename+"_log.txt",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(args)
    logger.info('processing {} data.....'.format(args.data_dir))
    # init ray
    ray.init(address='auto') #for multi node
    #ray.init() #for testing on single node
    pprint(ray.cluster_resources())
    cluster_resources = ray.cluster_resources()
    num_nodes = len(ray.nodes())
    parallelism = num_nodes * args.cpu_per_worker
    
    logger.info('cluster resources: {}'.format(cluster_resources))
    logger.info('num of ray nodes: {}'.format(num_nodes))
    logger.info('parallelism: {}'.format(parallelism))

    
    def preprocess_pii(batch: Dict[str, np.ndarray]) -> pd.DataFrame:
        
        task_id = ray.get_runtime_context().get_task_id()
        nopii_batch_df = pii_detection_redaction_batch(batch)
        return nopii_batch_df
        #return pd.DataFrame({'task_id': [task_id]})
    
    dataset = load_dataset('json', data_files=generate_list_jsonl(args.data_dir), streaming=True)['train']
    
    if args.skip != None:
        dataset_to_process = dataset.skip(args.skip)
    else:
        dataset_to_process = dataset

    idx = 1
    t0 = time.time()
    for rows in dataset_to_process.iter(batch_size=args.load_batch_size):
        logger.info('Start processing batch # {}'.format(idx))
        print("-----------------------------")
        start = time.time()
        df = pd.DataFrame(rows)
        ray_dataset = ray.data.from_pandas(df)
        ray_dataset = ray_dataset.repartition(parallelism)

        nopii_data = ray_dataset.map_batches(preprocess_pii, batch_format="numpy", batch_size=None)
        #we repartition back to 1 to save one file per batch (rather than one per worker)
        nopii_data=nopii_data.repartition(num_nodes) #set this to the number of nodes in the cluster
        nopii_data.write_json(os.path.join(args.output_path,"pii_redacted_batch_ray.json"))

        #print(f"{idx} * {args.load_batch_size} samples were written to disk.")
        logger.info('Finished processing batch # {}'.format(idx))
        logger.info(f"{idx} * {args.load_batch_size} samples were written to disk.")
        idx += 1
        end = time.time()
        print(f"\nthis batch took {end-start}s.")
        logger.info('This batch took {}s'.format(end-start))
        logger.info('Output filename {}s'.format(nopii_data._uuid))
        #_uuid: 8fc196ad2e6498e94c62b337a90b0fd 
        #output file created {self._uuid}_{block_idx}.json: 38fc196ad2e6498e94c62b337a90b0fd_000001.json 
        # #matches except first character
        print("============================")
    t1 = time.time()
    logger.info('Processing {} samples took {:.3f} sec'.format((idx-1)*args.load_batch_size, t1-t0))
  

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nthis script took {end-start}s.")

