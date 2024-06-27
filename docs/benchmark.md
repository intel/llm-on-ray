# Benchmarking LLM-on-Ray Serving

## Overview

The `benchmark_serving.py` script is designed for benchmarking the LLM-on-Ray serving system. This document outlines its key features, testing approach, and dataset information.

## Key Features

* **Request Types**:
   - We will send requests to the serving system using two different methods:
     - **Samples from Prompt Dataset**: These requests will be based on actual prompts from the dataset.
     - **Random Samples from Model Vocabulary**: We'll generate random samples using the model's vocabulary, following a specific distribution (TODO).
* **Load Generation**:
   - To stress-test the server, we'll generate load by specifying the desired queries per second (QPS).
* **Performance Metrics**:
   - We'll track the following metrics at both the request and token levels::
     - **Throughput**: The number of queries or tokens processed by the server per second.
     - **Latency**: The time taken by the server to respond to a request or generate a token.
* **Key Statistics**:
   - We'll record the following key statistics, including:
     - **Throughput (QPS)**: The average number of queries handled per second.
     - **Throughput (Tokens per Second)**: The rate at which tokens (words or units) are processed.
     - **Average Latency per Request**: The average time taken to process a single request.
     - **Average Latency per Token**: The average time per token processed.
     - **First and Next Token Latency per Request**: The time taken for the first token and subsequent tokens in a request.
* **Individual Request and Response Logging**:
   - We'll maintain detailed logs of each request and its corresponding response for further analysis.

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

### Performance
You can run these command to reproduce the results of deploying 2 replicas on Intel(R) Xeon(R) Platinum 8480L 1 socket.
1. Start ray cluster
```cmd
numactl -N 0 -m 0 -C 49-55 ray start --head --num-cpus 0
numactl -N 0 -m 0 -C 0-47 ray start --address=$HEAD_NODE_IP:PORT --num-cpus 48
```
2. Modify parameters in model config file
```
num_replicas: 2
cpus_per_worker: 24
```
3. Deploy server
```cmd
OMP_NUM_THREAD=24 numactl -N 0 -m 0 -C 0-47 python -u inference/serve.py --config_file inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --keep_serve_terminal --max_ongoing_requests 1000 --max_num_seqs 256 --simple
```
4. Send requests
```cmd
numactl -N 1 -m 1 python benchmarks/benchmark_serving.py --model-endpoint-base http://127.0.0.1:8000 --model-name llama-2-7b-chat-hf --dataset ./dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --dataset-format ShareGPT --vllm-engine --simple
```
5. Results
```cmd
Total time: 1132.641 s
Prompt Length (Min/Med/Max): 4 / 95 / 1024
Request Throughput (QPS): 0.883 requests/s
Input Token Throughput: 205.797 tokens/s
output Token Throughput: 216.418 tokens/s
Average latency per Request: 524.764 s
Average latency per Token: 8.211 s
```

## Arguments

Run the following commands to get argument details of the the script:

```bash
python benchmarks/benchmark_serving.py --help
```

## Save Benchmark Results

`benchmark_serving.py` supports saving benchmark statistics and individual requests and responses into json file. Please specify `--results-dir` argument when running the script, then the benchmark results will be saved to the specified directory.

## Recommendations
- Ensure that the serving system can handle the specified load without compromising performance.
- Monitor the system during testing to identify bottlenecks or issues.
- Use the recorded statistics to fine-tune the system for optimal performance.
