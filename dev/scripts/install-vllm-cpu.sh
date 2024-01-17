#!/usr/bin/env bash

# g++ version should be >=12.3
g++ --version

# Install from source
MAX_JOBS=8 pip install -v git+https://github.com/bigPYJ1151/vllm@PR_Branch \
    -f https://download.pytorch.org/whl/torch_stable.html
