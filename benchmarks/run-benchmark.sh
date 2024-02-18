#!/usr/bin/env bash

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

LLM_ON_RAY_HOME=$(cd $(dirname $0)/.. && pwd -P)
echo LLM_ON_RAY_HOME is $LLM_ON_RAY_HOME

export PYTHONPATH=$LLM_RAY_HOME:$PYTHONPATH

python benchmark_serving.py --model_name gpt-j-6b --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 5 --max_new_tokens 10