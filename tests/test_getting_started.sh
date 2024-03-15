#!/bin/bash
set -eo pipefail

# Step 1: Python environment
# Check Python version is or later than 3.9
echo "Step 1: Python environment"
echo "Checking Python version which should be equal or later than 3.9"
if ! python -c 'import sys; assert sys.version_info >= (3,9)' > /dev/null; then
    exit "Python should be 3.9 or later!"
fi

# Clone repo
echo "Cloning repo"
git clone https://github.com/intel/llm-on-ray.git && cd llm-on-ray

# Install dependencies
echo "Installing dependencies"
pip install .[cpu] -f https://developer.intel.com/ipex-whl-stable-cpu -f https://download.pytorch.org/whl/torch_stable.html

# Dynamic link oneCCL and Intel MPI libraries
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh

# Step 2: Serving
# take gpt2 for example
echo "Step 2: Serving"
echo "Starting ray server for gpt2 with 3 cpu per worker"
llm_on_ray-serve --config_file .github/workflows/config/gpt2-ci.yaml --simple --cpus_per_worker 3 

# Step 3: access OpenAI API
# 1.Using curl
echo "Step 3: access OpenAI API"
echo "Way 1: Using curl to access model"
export ENDPOINT_URL=http://0.0.0.0:8000/v1
curl $ENDPOINT_URL/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "gpt2",
    "messages": [{"role": "assistant", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    "temperature": 0.7
    }'

# 2.Using requests library
echo "Way 2: Using requests library to access model"
python examples/inference/api_server_openai/query_http_requests.py

# 3.Using OpenAI SDK
echo "Way 3: Using OpenAI SDK to access model"
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY="not_a_real_key"
python examples/inference/api_server_openai/query_openai_sdk.py --model_name gpt2 

# 4.Using serve.py and query_single.py
# Access models by query_single.py
# TODO: some bugs in query_single.py, enable it after fixed by follow-up PRs.
# echo "Way 4: Using query_single.py to access model"
# python examples/inference/api_server_simple/query_single.py --model_endpoint http://0.0.0.0:8000/gpt2
