import argparse
import os
from multiprocessing import Pool, cpu_count
from pyrecdp.core.utils import Timer
from math import ceil
from tqdm import tqdm
import json
from pyrecdp.primitives.llmutils.text_to_jsonl import *

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("-n", dest="n_part", type=int, default = 10)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    n_part = args.n_part
    
    with Timer(f"apply duplicates.pickle to create new data"):
        text_to_jsonl_MP(data_dir, out_dir, n_part)
    