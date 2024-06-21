#!/bin/bash
set -e

# Default serve cmd
if ! pgrep -f 'ray'; then
    echo "Ray is not running. Starting Ray..."
    # start Ray
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

#Keep the service not be exited
tail -f /dev/null