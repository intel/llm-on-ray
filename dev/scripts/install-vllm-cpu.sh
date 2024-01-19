#!/usr/bin/env bash

# Check tools
[[ -n $(which g++) ]] || { echo "GNU C++ Compiler (g++) is not found!";  exit 1; }
[[ -n $(which pip) ]] || { echo "pip command is not found!";  exit 1; }

# g++ version should be >=12.3
version_greater_equal()
{
    printf '%s\n%s\n' "$2" "$1" | sort --check=quiet --version-sort
}
gcc_version=$(g++ -dumpversion)
echo
echo Current GNU C++ Compiler version: $gcc_version
echo
version_greater_equal "${gcc_version}" 12.3.0 || { echo "GNU C++ Compiler 12.3.0 or above is required!"; exit 1; }

# Install from source
MAX_JOBS=8 pip install -v git+https://github.com/bigPYJ1151/vllm@PR_Branch \
    -f https://download.pytorch.org/whl/torch_stable.html
