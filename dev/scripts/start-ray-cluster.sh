#!/usr/bin/env bash

set -eo pipefail

# Setup oneapi envs before starting Ray
if [[ -e "/opt/intel/oneapi/setvars.sh" ]]; then
  source /opt/intel/oneapi/setvars.sh
  export CCL_ZE_IPC_EXCHANGE=sockets
else
  echo "/opt/intel/oneapi/setvars.sh doesn't exist, not loading."
fi

# Setup Ray cluster
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --address='127.0.0.1:6379' --ray-debugger-external
ray status
