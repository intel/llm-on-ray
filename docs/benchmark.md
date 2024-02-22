# Benchmarking LLM-on-Ray Serving

## Download the datasets

You can download the datasets by running:

```bash

# ShareGPT dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# IPEX dataset
wget https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json

```

## Benchmark

Currently only models from `inference/models/*.yaml` are supported. On the server side, run the following command:

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

## Help on parameters

```bash
python benchmarks/benchmark_serving.py --help
```