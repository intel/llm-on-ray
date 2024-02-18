#!/usr/bin/env bash

LLM_ON_RAY_HOME=$(cd $(dirname $0)/.. && pwd -P)
echo LLM_ON_RAY_HOME is $LLM_ON_RAY_HOME

cd $LLM_ON_RAY_HOME

export PYTHONPATH=~/Works/llm-on-ray-xwu99_add-benchmark-serving
export http_proxy=10.24.221.149:911
export https_proxy=10.24.221.149:911

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export NUMEXPR_MAX_THREADS=$(nproc)

serve shutdown
ray stop

RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external

python ./inference/serve.py --simple --config_file ./inference/models/gpt-j-6b.yaml --keep_serve_terminal
