#!/bin/bash
set -eo pipefail
cd $(dirname $0)

pip install -r ./requirements.txt

# Run pytest with the test file
pytest -vv --capture=tee-sys --show-capture=all ./inference

echo "Pytest finished running tests."
