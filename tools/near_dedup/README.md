# Near Dedup

## Intro

Near Dedup is to Detect duplicated documents and output as a duplicates list.

Step 1. We use [DataSketch minHash](https://ekzhu.com/datasketch/minhash.html) as the base algorithm to calculate (hash, band_id) pair for each documents.

Step 2. We use Spark Groupby to find the local lists for documents sharing the same (hash, band_id) pair.

Step 3. We use SlimPajama [connected component graph](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/slimpajama/dedup/generate_connected_components.py) to detect global lists and [generate duplication list](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/transformers/data_processing/slimpajama/dedup/generate_duplicates_dict.py) for documents sharing the same (hash, band_id) pair.

Step 4(Optional). We apply the duplication list to original file to elimate duplicated documents.

Now, only support to run with single node. Distributed Run will support by Ray in near Future.

## Expected input and Output

Input format: a folder of *.jsonl.

near_dedup.py Output: a folder with duplicates.pickle(index of duplicated documents), connected_components.pickle(graph indicates the connections between duplications)

dedup_convert.py Output: a folder of *.jsonl files (deduplicated)

## How to RUN

0. setup
```
cd llm-ray/tools/near_dedup
cd docker
docker compose build && docker compose run autofe-notebook-dev
```

### option 1. using cmdline

```
docker exec -it `docker ps | grep 'autofe-notebook-dev' | awk '{print $NF}'` /bin/bash
```
Now you are inside container
0. (Optional) convert text files to jsonl
```
python convert_jsonl.py -d ${raw_data_dir} -o ${output_dir} -n {num_partition} | tee -a ${log}
```

1. generate dedup pickle
```
python near_dedup.py -d ${to_dedup_data_folder} | tee -a ${log}
```

2. using dedup pickle to shink data
```
python dedup_convert.py -d ${to_dedup_data_folder} -f ${dedup_dict_file} -o ${output_folder} | tee -a ${log}
```

### option 2. using notebook

There will be notebook url shown as below
```
[I 2023-08-28 16:24:48.132 ServerApp] Serving notebooks from local directory: /home/vmagent/app/workspace
[I 2023-08-28 16:24:48.132 ServerApp] Jupyter Server 2.7.1 is running at:
[I 2023-08-28 16:24:48.132 ServerApp] http://${hostname}:8890/lab
[I 2023-08-28 16:24:48.132 ServerApp]     http://127.0.0.1:8890/lab
[I 2023-08-28 16:24:48.132 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

```
template notebooks are at [template_notebooks](template_notebooks)
PILE processed notebooks are at [PILE_notebooks](PILE_notebooks)