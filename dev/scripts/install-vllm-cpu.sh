#!/usr/bin/env bash

# The script will install vllm-cpu into current conda environment
# Use the following command to create a new conda env if necessary
# $ conda create -n vllm-cpu python=3.10
# $ conda activate vllm-cpu

# Install g++ 12.3 for building
conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3

# Install from source
# TODO: need to verify if conda env needed to reactivate to setup g++ envs
MAX_JOBS=8 pip install -v git+https://github.com/bigPYJ1151/vllm@PR_Branch
