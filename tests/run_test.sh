#!/bin/bash
export https_proxy=10.24.221.149:911
export http_proxy=10.24.221.149:911
work_path=$(dirname $0)
cd $(dirname $0)

# This script runs pytest to execute all tests in the test_utils.py file.

# Exit immediately if a command exits with a non-zero status.
set -e

# Optional: activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Run pytest with the test file
# pip install -r ./requirements.txt
pytest  -vs ./inference/test_utils.py
# Optional: deactivate your virtual environment if you used one
# deactivate
# pr draft
echo "Pytest finished running tests."
