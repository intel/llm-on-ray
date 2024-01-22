#!/usr/bin/env bash

LLM_ON_RAY_HOME=$(cd $(dirname $0)/.. && pwd -P)
echo LLM_ON_RAY_HOME is $LLM_ON_RAY_HOME

cd $LLM_ON_RAY_HOME

# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

export NUMEXPR_MAX_THREADS=$(nproc)

serve shutdown
ray stop

RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external

KEEP_SERVE_TERMINAL='false' python ./inference/run_model_serve.py --config_file ./inference/models/llama-2-7b-chat-hf.yaml
