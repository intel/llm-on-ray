#!/bin/bash
set -eo pipefail
cd $(dirname $0)

export OPENAI_BASE_URL=http://localhost:8000/v
export OPENAI_API_BASE=http://localhost:8000/v
export OPENAI_API_KEY=not_a_real_key
export no_proxy=localhost,127.0.0.1

# Run pytest with the test file
pytest -vv --capture=tee-sys --show-capture=all ./inference

echo "Pytest finished running tests."
