#!/bin/bash
cd $(dirname $0)

# Run pytest with the test file
pytest -vs ./inference

echo "Pytest finished running tests."
