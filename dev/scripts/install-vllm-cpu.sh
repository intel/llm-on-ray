#!/usr/bin/env bash

conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3
MAX_JOBS=8 pip install -v git+https://github.com/bigPYJ1151/vllm@PR_Branch
