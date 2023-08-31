import argparse
import os
import sys
import ftfy
import re
import numpy as np
import pickle
from pyrecdp.core.utils import Timer
from pyrecdp.core import SparkDataProcessor
from pyrecdp.core.utils import Timer
from pyrecdp.primitives.llmutils.near_dedup import *
from pyrecdp.primitives.llmutils.utils import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window 
import shutil
from nltk import ngrams
import string

def run(data_files, dup_dir, ngram_size, num_perm, bands, ranges, enable_ray):
    if enable_ray:
        rdp = SparkDataProcessor(spark_mode='ray')
    else:
        rdp = SparkDataProcessor()
    spark=rdp.spark  
    try:
        with Timer("Load data with RowID"):
            df = read_json(data_files, spark).cache()
            total_length = df.count()

        pipeline = minHashLSH_prepare(df, num_perm, ngram_size, bands, ranges)
        with Timer("generate minHashLsh"):
            if os.path.exists(dup_dir):
                shutil.rmtree(dup_dir, ignore_errors=True)
            results = pipeline.saveAsTextFile(dup_dir)
            
        
        with Timer(f"generate_connected_components all"):
            dup_connected_args = argparse.Namespace()
            dup_connected_args.input_dir = dup_dir
            dup_connected_args.out_file = os.path.join(
                dup_dir, "connected_components.pickle"
            )
            generate_connected_components.generate_connected_components_mp(
                dup_connected_args
            )
            
        with Timer(f"generate_duplicates_dict all"):
            dup_docs = os.path.join(dup_dir, "duplicates.pickle")
            dup_dict_args = argparse.Namespace()
            dup_dict_args.input_file = os.path.join(
                dup_dir, "connected_components.pickle"
            )
            dup_dict_args.out_file = dup_docs
            generate_duplicates_dict.generate_duplicates(dup_dict_args)

        dup_dict = pickle.load(open(os.path.join(dup_dir, "duplicates.pickle"), 'rb'))
        dup_sum = 0
        for _, v in dup_dict.items():
            dup_sum += len(list(v))

        print(f"Completed!!")
        print(f"    total processed {total_length} documents")
        print(f"    total detected {dup_sum} duplicated documents")
        print(f"    duplicate ratio is {dup_sum/total_length}")
    except Exception as e:
        spark.stop()
        print("Failed", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data_files, dup_dir, ngram_size, num_perm, bands, ranges
    #pipeline = minHashLSH_prepare(df, num_perm = 256, ngram_size = 6, bands = 9, ranges = 13)
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("--nperm", dest="num_perm", type=int, default=256)
    parser.add_argument("--ngram", dest="ngram_size", type=int, default=6)
    parser.add_argument("--bands", dest="bands", type=int, default=9)
    parser.add_argument("--ranges", dest="ranges", type=int, default=13)
    parser.add_argument("--enable_ray", dest="enable_ray", action='store_true', default=False)
    args = parser.parse_args()
    data_dir = args.data_dir
    
    data_files = get_data_files(data_dir)
    dup_dir = os.path.join(data_dir, "deduplicate")
    
    num_perm = args.num_perm
    ngram_size = args.ngram_size
    bands = args.bands
    ranges = args.ranges
    enable_ray = args.enable_ray
    with Timer(f"Generate duplicate dict for {data_dir}"):
        run(data_files, dup_dir, ngram_size, num_perm, bands, ranges, enable_ray)
