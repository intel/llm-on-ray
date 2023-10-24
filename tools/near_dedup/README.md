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
conda create --name pyrecdp
conda activate pyrecdp
DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-8-jre graphviz
pip install pyrecdp --pre
```

### option 1. using cmdline
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

template notebooks are at [template_notebooks](template_notebooks)
PILE processed notebooks are at [PILE_notebooks](PILE_notebooks)

## Learn More

We support all general used LLM data utils in [RecDP LLM](https://github.com/intel/e2eAIOK/blob/main/RecDP/pyrecdp/LLM/README.md)