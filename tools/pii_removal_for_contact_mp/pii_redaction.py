import argparse
import os
from multiprocessing import Pool, cpu_count
from pyrecdp.core.utils import Timer
from math import ceil
from tqdm import tqdm
import json
from pii_redaction_impl import *

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("-mp", dest="mp", type=int, default=-1)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    n_parallel = args.mp
    
    with Timer(f"generate hash to {data_dir}"):
        pii_remove_MP(data_dir, out_dir, n_parallel)