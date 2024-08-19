#!/usr/bin/env bash

# Check tools
[[ -n $(which g++) ]] || { echo "GNU C++ Compiler (g++) is not found!";  exit 1; }
[[ -n $(which pip) ]] || { echo "pip command is not found!";  exit 1; }

# conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3 libxcrypt
version_greater_equal()
{
    printf '%s\n%s\n' "$2" "$1" | sort --check=quiet --version-sort
}
gcc_version=$(g++ --version | grep -o -E '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
echo
echo Current GNU C++ Compiler version: $gcc_version
echo
VLLM_VERSION=0.5.2

echo Installing vLLM v$VLLM_VERSION ...
# Install VLLM from source, refer to https://docs.vllm.ai/en/latest/getting_started/cpu-installation.html for details
is_avx512_available=$(cat /proc/cpuinfo | grep avx512)
if [ -z "$is_avx512_available" ]; then
    echo "AVX512 is not available, vLLM CPU backend using other ISA types."
    MAX_JOBS=8 VLLM_TARGET_DEVICE=cpu VLLM_CPU_DISABLE_AVX512="true" pip install -v git+https://github.com/vllm-project/vllm.git@v$VLLM_VERSION \
    --extra-index-url https://download.pytorch.org/whl/cpu
else
    # g++ version should be >=12.3. You can run the following to install GCC 12.3 and dependencies on conda:
    version_greater_equal "${gcc_version}" 12.3.0 || { echo "GNU C++ Compiler 12.3.0 or above is required!"; exit 1; }
    echo "Install vllm-cpu with AVX512 ISA support"
    MAX_JOBS=8 VLLM_TARGET_DEVICE=cpu pip install -v git+https://github.com/vllm-project/vllm.git@v$VLLM_VERSION \
        --extra-index-url https://download.pytorch.org/whl/cpu
fi
echo Done!