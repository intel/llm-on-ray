#!/bin/bash
set -eo pipefail

# Step 1: Python environment
# Check Python version is or later than 3.9
if ! python -c 'import sys; assert sys.version_info >= (3,9)' > /dev/null; then
    exit "Python should be 3.9 or later!"
fi

# Clone repo
git clone https://github.com/intel/llm-on-ray.git && cd llm-on-ray

# Install dependencies
pip install .[cpu] -f https://developer.intel.com/ipex-whl-stable-cpu -f https://download.pytorch.org/whl/torch_stable.html

# Dynamic link oneCCL and Intel MPI libraries
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh

# Start Ray (already started in CI)
#ray start --head

# Step 2: Serving
# take gpt2 for example
llm_on_ray-serve --config_file llm_on_ray/inference/models/gpt2.yaml

# Three ways to access OpenAI API
# 1.Using curl
export ENDPOINT_URL=http://localhost:8000/v1
curl $ENDPOINT_URL/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "gpt2",
    "messages": [{"role": "assistant", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    "temperature": 0.7
    }'

# 2.Using requests library
python examples/inference/api_server_openai/query_http_requests.py

# 3.Using serve.py and query_single.py
# Deploy models by serve.py
llm_on_ray-serve --config_file llm_on_ray/inference/models/gpt2.yaml --simple

# Access models by query_single.py
python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/gpt2
