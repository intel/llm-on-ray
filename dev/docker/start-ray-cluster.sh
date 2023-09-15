#!/usr/bin/env bash

# Setup oneapi envs before starting Ray
source /opt/intel/oneapi/setvars.sh

export CCL_ZE_IPC_EXCHANGE=sockets

# Setup Ray cluster
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --address='127.0.0.1:6379' --ray-debugger-external
