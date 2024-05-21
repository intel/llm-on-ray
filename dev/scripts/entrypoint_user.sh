#!/bin/bash

su 
# Default model path
DEFAULT_MODEL_PATH="/app/models/default_model"

# Set proxies if needed

export http_proxy=${http_proxy:-http://10.24.221.169:911}
export https_proxy=${https_proxy:-http://10.24.221.169:911}

# # Default volume mounts if needed
# VOLUME_SOURCE1=${VOLUME_SOURCE1:-/home/yutianchen/Project/pr_lib/llm-on-ray}
# VOLUME_TARGET1=${VOLUME_TARGET1:-/root/llm-on-ray}

VOLUME_SOURCE2=${VOLUME_SOURCE2:-/home/yutianchen/.cache/huggingface/hub}
VOLUME_TARGET2=${VOLUME_TARGET2:-/root/.cache/huggingface/hub}

# Mount volumes
mkdir -p /root/llm-on-ray
mkdir -p /root/.cache/huggingface/hub
mount --bind "$VOLUME_SOURCE1" "$VOLUME_TARGET1"
mount --bind "$VOLUME_SOURCE2" "$VOLUME_TARGET2"

# Check if any arguments are provided
if [ "$#" -gt 0 ]; then
    # If arguments are provided, use them
    exec "$@"
else
    # If no arguments are provided, start the default model service
    echo "No arguments provided, starting default model service..."
    # exec python serve_model.py --model-path $DEFAULT_MODEL_PATH
fi
