#!/bin/bash
set -e

# Check if an environment variable exists and print its value
if [ -n "$hf_token" ]; then
  echo "The hf_token environment variable is: $hf_token"
  # Execute Hugging Face CLI login command
  huggingface-cli login --token "${hf_token}"
else
  echo "Environment variable 'hf_token' is not set."
fi

# Default serve cmd
if ! pgrep -f 'ray'; then
    echo "Ray is not running. Starting Ray..."
    # 启动 Ray
    ray start --head
    echo "Ray started."
else
    echo "Ray is already running."
fi

if [ -n "$model_name" ]; then
    echo "Using User Model: $model_name"
    llm_on_ray-serve --models $model_name
else
    echo "Using Default Model: gpt2"
    llm_on_ray-serve --config_file llm_on_ray/inference/models/gpt2.yaml 
fi

