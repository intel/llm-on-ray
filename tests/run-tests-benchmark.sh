#!/bin/bash
set -eo pipefail
cd $(dirname $0)


# Run pytest with the test file
pytest -vv --capture=tee-sys --show-capture=all ./benchmarks

echo "Pytest finished running tests."
