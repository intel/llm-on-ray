#!/usr/bin/env bash

# Check tools
[[ -n $(which g++) ]] || { echo "GNU C++ Compiler (g++) is not found!";  exit 1; }
[[ -n $(which pip) ]] || { echo "pip command is not found!";  exit 1; }

# g++ version should be >=12.3. On Ubuntu 22.4, you can run:
# sudo apt-get update  -y
# sudo apt-get install -y gcc-12 g++-12
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
version_greater_equal()
{
    printf '%s\n%s\n' "$2" "$1" | sort --check=quiet --version-sort
}
gcc_version=$(g++ --version | grep -o -E '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
echo
echo Current GNU C++ Compiler version: $gcc_version
echo
version_greater_equal "${gcc_version}" 12.3.0 || { echo "GNU C++ Compiler 12.3.0 or above is required!"; exit 1; }

# Refer to https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html to install from source
# We use this one-liner to install latest vllm-cpu
MAX_JOBS=8 VLLM_TARGET_DEVICE=cpu pip install -v git+https://github.com/vllm-project/vllm.git \
    --extra-index-url https://download.pytorch.org/whl/cpu
