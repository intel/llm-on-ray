import pyarrow.parquet as pq
import argparse
import glob 
import os 
import time 

parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="the directory containing the parquet files")
args = parser.parse_args()

row_nums = []
parquet_files = sorted(glob.glob(os.path.join(args.file_path, "*.parquet")))

start = time.time()
for file in parquet_files:
    table = pq.read_table(file, columns=[])
    print(table.num_rows)
    row_nums.append(table.num_rows)
end = time.time()

print(f"the total number of rows in all parquet files = {sum(row_nums)}.")
print(f"the total time: {end-start}s.")
