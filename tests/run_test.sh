#!/bin/bash

# This script runs pytest to execute all tests in the test_utils.py file.

# Exit immediately if a command exits with a non-zero status.
set -e

# Optional: activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Run pytest with the test file
pytest -vs ./inference/test_utils.py

# Optional: deactivate your virtual environment if you used one
# deactivate

echo "Pytest finished running tests."
