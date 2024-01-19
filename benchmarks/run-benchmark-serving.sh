#!/usr/bin/env bash

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LLM_RAY_HOME=$(cd $(dirname $0)/.. && pwd -P)
echo LLM_RAY_HOME is $LLM_RAY_HOME

export PYTHONPATH=$LLM_RAY_HOME:$PYTHONPATH

python benchmark_serving.py --model_name llama-2-7b-chat-hf --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 5 --request-rate 1