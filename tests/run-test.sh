#!/bin/bash
export https_proxy=10.24.221.149:911
export http_proxy=10.24.221.149:911
work_path=$(dirname $0)
cd $(dirname $0)

# This script runs pytest to execute all tests in the test_utils.py file.

# Exit immediately if a command exits with a non-zero status.
set -e

# Run pytest with the test file
pytest  -vs ./inference

echo "Pytest finished running tests."
