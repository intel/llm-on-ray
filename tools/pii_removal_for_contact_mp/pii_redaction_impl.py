import argparse
import os, sys
from pyrecdp.core.utils import Timer
import json
from pyrecdp.primitives.llmutils.utils import get_nchunks_and_nproc, clean_str
import hashlib
import pandas as pd
from tqdm import tqdm
import subprocess #nosec
import io
import re
import time
import hashlib

from presidio_analyzer.predefined_recognizers import PhoneRecognizer

import sys, pathlib
cur_path = str(pathlib.Path(__file__).parent.resolve())
import_path = os.path.join(cur_path, "pii_detection_redaction", "src")
print(f"add new import_path: {import_path}")
sys.path.append(import_path)

from pii_redaction_v2 import *
            
def pii_removal_impl_parquet_to_parquet(in_file_name, out_file_name, base_file_name):
    analyzer = PhoneRecognizer()
    batch = pd.read_parquet(in_file_name).reset_index(drop=True)
    text = batch['text'].tolist()
    redacted_content = []
    modified = []
    piis = []
    
    #for txt in tqdm(text, total=len(text), desc = f"process {in_file_name}"):
    for txt in text:
        # # for testing exception
        # if i%5 == 0:
        #     raise ValueError
        output = detect_redact_pii_for_one_text(txt, analyzer)
        modified.append(output['modified'])
        piis.append(output['pii'])
        if output['pii'] != None: # have PII so output redacted text
            redacted_content.append(output['redacted'])
        else: # did not have PII so output original text
            redacted_content.append(txt)
    
    batch['text'] = pd.Series(redacted_content)
    batch['secrets'] = pd.Series(piis)
    batch['modified'] = pd.Series(modified)

    batch.to_parquet(out_file_name)

# define actual work
def pii_remove(proc_id, x_list, out_type):
    #for x in tqdm(x_list, total=len(x_list), desc=f"proc-{proc_id}", position=proc_id+1):
    for x in x_list:
        try:
            in_file_name, out_file_name, base_file_name = x
            base_file_name = os.path.basename(base_file_name)
            out_dir = os.path.dirname(out_file_name)
            os.makedirs(out_dir, exist_ok=True)
            pii_removal_impl_parquet_to_parquet(in_file_name, out_file_name, base_file_name)

        except Exception as e:
            with open(f"{out_file_name}.error.log", 'w') as f:
                f.write(f"Failed to process {base_file_name}, error is {e}")
    return True
    
def wait_and_check(pool):
    for proc_id, (process, cmd) in pool.items():
        std_out, std_err = process.communicate()
        rc = process.wait()
        if rc != 0:
            file_name = f"pii-redaction-proc-{proc_id}.error.log"
            print(f"Task failed, please check {file_name} for detail information")
            with open(file_name, "a") as f:
                f.write(f"=== {time.ctime()} {' '.join(cmd)} failed. ===\n")
                f.write(std_err.decode(sys.getfilesystemencoding()))
                f.write("\n")
                                
def launch_cmdline_mp(args, data_dir, out_dir, mp):
    pool = {}
    for arg in tqdm(args, total=len(args), desc="pii redaction"):
        proc_id, x_list = arg
        cmd = ["python", "pii_redaction_impl.py", "--proc_id", f"{proc_id}", "--in_dir", f"{data_dir}", "--out_dir", f"{out_dir}", "--file_list", f"{x_list}"]
        #f.write(' '.join(cmd) + "\n")
        pool[proc_id] = (subprocess.Popen(cmd , stdout=subprocess.PIPE, stderr=subprocess.PIPE), cmd)
        
        if len(pool) >= mp:
            wait_and_check(pool)
            pool = {}
        
    wait_and_check(pool)

def get_target_file_list(data_dir, file_type):
    cmd = ["find", data_dir, "-name", f"*.{file_type}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    exitcode = proc.returncode
    if exitcode != 0:
        return []
    else:
        ret = stdout.decode("utf-8").split('\n')[:-1]
        ret = [i.replace(data_dir, "") for i in ret]
        ret = [i[1:] if i[0] == '/' else i for i in ret]
        return ret
    
def pii_remove_MP(data_dir, out_dir, n_part = -1):
    files = get_target_file_list(data_dir, 'parquet')
    #print(files)

    if len(files) == 0:
        print("Detect 0 files, exit here")
        return

    if n_part != -1:
        n_proc = n_part
    else:
        _, n_proc = get_nchunks_and_nproc(len(files), n_part = n_part)
    print(f"resetting to {n_proc} for number of processes")
    
    args = [(idx, [i]) for idx, i in enumerate(files)]
    launch_cmdline_mp(args, data_dir, out_dir, n_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_id", dest="proc_id", type=int)
    parser.add_argument("--in_dir", dest="in_dir", type=str)
    parser.add_argument("--out_dir", dest="out_dir", type=str)
    parser.add_argument("--file_list", dest="file_list", type=str)
    args = parser.parse_args()

    proc_id = args.proc_id
    in_dir = args.in_dir
    out_dir = args.out_dir
    in_file_list = eval(args.file_list)
    out_type = 'parquet'
    
    file_args = [(os.path.join(in_dir, f_name), os.path.join(out_dir, f"{f_name}.pii_remove.{out_type}"), f_name) for f_name in in_file_list]

    with Timer(f"generate hash index with proc-id {proc_id}"):
        pii_remove(proc_id, file_args, out_type)
