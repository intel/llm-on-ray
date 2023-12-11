#!/usr/bin/env bash

set -eo pipefail

echo "Installing oneAPI components ..."

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

apt update
apt-get install -y intel-oneapi-ccl-devel intel-oneapi-compiler-dpcpp-cpp-runtime

echo "oneAPI components installed!"
