#!/bin/bash
set -e
#!/bin/bash

# Check if an environment variable exists and print its value
if [ -n "$hf_token" ]; then
  echo "The hf_token environment variable is: $hf_token"
  # Execute Hugging Face CLI login command
  huggingface-cli login --token "${hf_token}"
else
  echo "Environment variable 'hf_token' is not set."
fi

# Default serve cmd
ray start --head

if [ -n "$MODEL_NAME"]; then
    echo "Using User Model: $MODEL_NAME"
    llm_on_ray-serve --models $MODEL_NAME
else
    echo "Using Default Model: gpt2"
    llm_on_ray-serve --config_file llm_on_ray/inference/models/gpt2.yaml 
fi

