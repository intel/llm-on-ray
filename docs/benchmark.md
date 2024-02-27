# Benchmarking LLM-on-Ray Serving

## Overview

`benchmark_serving.py` is a Python script used for benchmarking LLM-on-Ray serving system.

## Features

* Send requests to the serving system in one of the following fasions:
    * Samples from prompt dataset
    * Random samples from model vocaburary with specific distribution (TODO)
* Generate a load on the server by specifying requests per second to test its performance under stress
* Track the throughput (requests per second) and latency (seconds) of the server
* Report and record key statistics of the serving performance
    * throughput requests per second
    * throughput tokens per second
    * average latency per request
    * average latency per token
    * first and next token latency per request
* Record individual request and response for further analysis

## Dataset

`benchmark_serving.py` currently support two formats of the prompt dataset: `ShareGPT` and `IPEX`

You can download the datasets by running:

```bash

# ShareGPT dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# IPEX sample dataset
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

```

## Benchmark

For benchmarking, user needs to run `llm-on-ray` serving first and then run `benchmark_serving.py`.

Currently only serving models from `inference/models/*.yaml` are supported. It can be specified with `model_id` from the serving script.

On the server side, run the following command:

```bash
python inference/serve.py --models <model_id> --simple --keep_serve_terminal
```

For example:

```bash
python inference/serve.py --models gpt-j-6b --simple --keep_serve_terminal
```

On the client side, run the following command:

```bash
python benchmarks/benchmark_serving.py \
    --model_endpoint_base <model_endpoint_base_url> \
    --model_name <model_id> \
    --dataset <target_dataset> \
    --dataset-format <ShareGPT / IPEX> \
    --num-prompts <num_prompts> \
    --request-rate <request_rate>
```

Example:

* Send 5 prompts from ShareGPT dataset in total at 1 request per second for `gpt-j-6b` model:

```bash
python benchmarks/benchmark_serving.py \
    --model_endpoint_base http://127.0.0.1:8000 \
    --model_name gpt-j-6b \
    --dataset benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-format ShareGPT \
    --num-prompts 5 \
    --request-rate 1
```

* Send 10 identical prompts from IPEX dataset for 32 input_tokens and 32 max_new_tokens at once for `gpt-j-6b` model:

```bash
python benchmarks/benchmark_serving.py \
    --model_endpoint_base http://127.0.0.1:8000 \
    --model_name gpt-j-6b \
    --dataset benchmarks/prompt.json \
    --dataset-format IPEX \
    --input-tokens 32 \
    --max-new-tokens 32 \
    --num-prompts 10
```

## Arguments

Run the following commands to get argument details of the the script:

```bash
python benchmarks/benchmark_serving.py --help
```

## Save Benchmark Results

`benchmark_serving.py` supports saving benchmark statistics and individual requests and responses into json file. Please specify `--results-dir` argument when running the script, then the benchmark results will be saved to the specified directory.