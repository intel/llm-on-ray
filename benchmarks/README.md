# Benchmarking LLM-on-Ray Serving

## Download the ShareGPT dataset

You can download the dataset by running:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Benchmark

On the server side, run the following command:

```bash
python inference/serve.py --config_file <model_config_file.yaml> --simple --keep_serve_terminal
```

For example:

```bash
python inference/serve.py --config_file inference/models/gpt-j-6b.yaml --simple --keep_serve_terminal
```

On the client side, run the following command:

```bash
python benchmarks/benchmark_serving.py \
    --model_endpoint_base <model_endpoint_base_url> \
    --model_name <model_id>
    --dataset <target_dataset> \
    --request-rate <request_rate>
```

For example:

```bash
python benchmarks/benchmark_serving.py \
    --model_endpoint_base http://127.0.0.1:8000 \
    --model_name gpt-j-6b
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 5
```

## Help on parameters

```bash
python benchmarks/benchmark_serving.py --help
```