import argparse
import os
import sys
import pickle
import queue
from multiprocessing import Pool, cpu_count
from pyrecdp.core.utils import Timer
from math import ceil
from tqdm import tqdm
from pyrecdp.primitives.llmutils.shrink_jsonl import *
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data_files, dup_dir, ngram_size, num_perm, bands, ranges
    #pipeline = minHashLSH_prepare(df, num_perm = 256, ngram_size = 6, bands = 9, ranges = 13)
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-f", dest="dup_dict", type=str, default=None)
    parser.add_argument("-o", dest="out_dir", type=str, default=None)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    dup_dir = os.path.join(data_dir, "deduplicate")
    if args.dup_dict is None:
        dup_dict = os.path.join(dup_dir, "duplicates.pickle")
    else:
        dup_dict = args.dup_dict
        
    if args.out_dir is None:
        out_dir = os.path.join(dup_dir, "output")
    else:
        out_dir = args.out_dir
    
    with Timer(f"apply duplicates.pickle to create new data"):
        shrink_document_MP(data_dir, dup_dict, out_dir)
    