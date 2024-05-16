#!/usr/bin/env bash

# Check tools
[[ -n $(which g++) ]] || { echo "GNU C++ Compiler (g++) is not found!";  exit 1; }
[[ -n $(which pip) ]] || { echo "pip command is not found!";  exit 1; }

# g++ version should be >=12.3. You can run the following to install GCC 12.3 and dependencies on conda:
# conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3 libxcrypt
version_greater_equal()
{
    printf '%s\n%s\n' "$2" "$1" | sort --check=quiet --version-sort
}
gcc_version=$(g++ --version | grep -o -E '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
echo
echo Current GNU C++ Compiler version: $gcc_version
echo
version_greater_equal "${gcc_version}" 12.3.0 || { echo "GNU C++ Compiler 12.3.0 or above is required!"; exit 1; }
