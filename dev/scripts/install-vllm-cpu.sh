#!/usr/bin/env bash

conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
MAX_JOBS=8 pip install -v git+https://github.com/bigPYJ1151/vllm@PR_Branch
