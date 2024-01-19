#!/usr/bin/env bash

LLM_RAY_HOME=$(cd $(dirname $0)/.. && pwd -P)
echo LLM_RAY_HOME is $LLM_RAY_HOME

cd $LLM_RAY_HOME

MODEL_ID=llama2_7b
# MODEL_ID=gpt-j-6B

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export NUMEXPR_MAX_THREADS=$(nproc)

serve shutdown
ray stop

RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external

KEEP_SERVE_TERMINAL='false' python ./inference/run_model_serve.py --config_file ./inference/models/llama-2-7b-chat-hf.yaml
