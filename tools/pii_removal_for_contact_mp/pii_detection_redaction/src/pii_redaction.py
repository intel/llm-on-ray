"""
this script is for processing pii-detection-dedaction 
"""

# from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import PhoneRecognizer

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

import logging
import glob


def detect_redact_pii_for_one_text(text, analyzer):

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
            'redacted': redacted_text,
            'pii': piis,
            "modified": True
        }

        
    else: 
        output = {
            'redacted': None,
            'pii': None,
            "modified": False

        }

    return output


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

    # group.add_argument(
    #     "--format",
    #     type=str,
    #     default="parquet",
    #     required=False,
    #     help="input data format, parquet or json"
    # )

    group.add_argument(
        "--dataset-family",
        type=str,
        default="refinedweb",
        required=False,
        help="choose from: refinedweb, slimpajama, pile"
    )

    group.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="for local mode, you need to provide local dataset repository, e.g. /home/user/local"
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
    group.add_argument(
        "--skip", type=int, default=None, help="how many samples to skip"
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

    # if args.format not in ['parquet', 'json']:
    #     raise ValueError('data file format must be parquet or json')

    output_dir = os.path.join(args.output_path, args.output_prefix)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    exception_dir = output_dir+'/exceptions/'
    cache_dir = args.cache_dir 
    dataset_family = args.dataset_family
    log_dir = output_dir+'/logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(filename=log_dir+"newlog.txt",
                    format='%(asctime)s %(message)s',
                    filemode='w')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(args)
    logger.info('processing {} data.....'.format(dataset_family))

    # init ray
    ray.init(address='auto')
    pprint(ray.cluster_resources())
    num_nodes = len(ray.nodes())
    parallelism = num_nodes * args.cpu_per_worker

    logger.info('num of ray nodes: {}'.format(num_nodes))
    logger.info('parallelism: {}'.format(parallelism))

    def preprocess_fn(contents, metas, analyzer) -> pd.DataFrame:
        # inputs are in batches
        redacted_content = []
        modified = []
        piis = []
        meta_output = []
        original_content = []

        exceptions = []

        for i, text in enumerate(contents):
            try:
                # # for testing exception
                # if i%5 == 0:
                #     raise ValueError
                output = detect_redact_pii_for_one_text(text, analyzer)
                modified.append(output['modified'])
                piis.append(output['pii'])
                if output['pii'] != None:
                    redacted_content.append(output['redacted'])
                else:
                    redacted_content.append(text)
                meta_output.append(metas[i])
            except:
                logger.debug('exception occurred!') # seems cannot log from ray actor using this method
                exceptions.append({
                    'text':text,
                    'meta': metas[i]
                })
        if len(exceptions)>0:
            if not os.path.exists(exception_dir):
                os.mkdir(exception_dir)
            task_id = ray.get_runtime_context().get_task_id()
            with open(exception_dir + task_id+'.json', 'w') as f:
                json.dump(exceptions, f)

        return pd.DataFrame({#"original": original_content, 
                             'redacted': redacted_content,
                             'piis': piis,
                             'meta': meta_output,
                             'modified': modified})



    def pii_removal_refinedweb(batch: Dict[str, List]) -> pd.DataFrame:
        # analyzer = AnalyzerEngine()
        analyzer = PhoneRecognizer()

        contents = batch['content'].tolist()

        try:
            urls = batch['url'].tolist()
            timestamps = batch['timestamp'].tolist()
            dump = batch['dump'].tolist()
            segment = batch['segment'].tolist()
            metas = []

            for i in range(len(urls)):
                metas.append({
                    'url': urls[i],
                    'timestamp': timestamps[i],
                    'dump': dump[i],
                    'segment': segment[i]
                })
        except:
            metas = [None]*len(contents)

        return preprocess_fn(contents, metas, analyzer)
    
    def pii_removal_slimpajama_pile(batch: Dict[str, List]) -> pd.DataFrame:
        # analyzer = AnalyzerEngine()
        analyzer = PhoneRecognizer()
        # try:
        contents = batch['text'].tolist()
        try:
            metas = batch['meta'].tolist()
            # print(metas)
        except:
            metas = [None]*len(contents)
        return preprocess_fn(contents, metas, analyzer)
        # except:
        #     if not os.path.exists(exception_dir):
        #         os.mkdir(exception_dir)
        #     task_id = ray.get_runtime_context().get_task_id()
        #     with open(exception_dir + task_id+'.json', 'w') as f:
        #         json.dump(batch, f)

    

    if not args.local:
        dataset = load_dataset(args.input, streaming=True)['train']
    else:
        data_dir = args.data_dir
        if dataset_family == 'refinedweb':
            datafiles = glob.glob(data_dir + '*.parquet')
            dataset = load_dataset('parquet', data_files = datafiles, streaming=True)['train']
        elif dataset_family == 'slimpajama' or dataset_family == 'pile':
            datafiles = glob.glob(data_dir+'*.jsonl')
            dataset = load_dataset('json', data_files = datafiles, streaming=True)['train']
        else:
            raise ValueError('{} not supported'.format(dataset_family))
        
    if args.skip != None:
        dataset_to_process = dataset.skip(args.skip)
    else:
        dataset_to_process = dataset

    idx = 1
    
    t0 = time.time()
    for rows in dataset_to_process.iter(batch_size=args.load_batch_size):
        logger.info('Start processing batch # {}'.format(idx))
        print("-----------------------------")
        df = pd.DataFrame(rows)
        # logger.info(df['meta'])
        ray_dataset = ray.data.from_pandas(df)
        ray_dataset = ray_dataset.repartition(parallelism) #, shuffle = True)

        if dataset_family == 'refinedweb':
            print('process refinedweb')
            process_fn = pii_removal_refinedweb
        elif dataset_family == 'slimpajama' or dataset_family == 'pile':
            print('process slimpj or pile')
            process_fn = pii_removal_slimpajama_pile
        else:
            raise ValueError('{} not supported'.format(dataset_family))

        tokenized_data = ray_dataset.map_batches(process_fn, batch_format="numpy", batch_size=None)

        if dataset_family == 'refinedweb':
            tokenized_data.write_parquet(output_dir)
        elif dataset_family == 'slimpajama' or dataset_family == 'pile':
            tokenized_data.write_json(output_dir)
        else:
            raise ValueError('{} not supported'.format(dataset_family))

        logger.info('Finished processing batch # {}'.format(idx))
        logger.info(f"{idx} * {args.load_batch_size} samples were written to disk.")
        idx += 1
        print("============================")
        if idx == 2:
            #sys.exit()
            break
    t1 = time.time()
    logger.info('Processing {} samples took {:.3f} sec'.format((idx-1)*args.load_batch_size, t1-t0))


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nthis script took {end-start}s.")

