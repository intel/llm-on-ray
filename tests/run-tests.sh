#!/bin/bash
cd $(dirname $0)

# Run pytest with the test file
pytest -vs --capture=tee-sys --show-capture=all ./inference

echo "Pytest finished running tests."
