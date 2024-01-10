#!/bin/bash
cd $(dirname $0)

# This script runs pytest to execute all tests in the test_utils.py file.

# Exit immediately if a command exits with a non-zero status.
set -e

# Run pytest with the test file
pytest -vs ./inference

echo "Pytest finished running tests."