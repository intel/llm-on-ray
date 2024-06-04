#!/usr/bin/env bash
sed -e 's\setuptools.*\\g' requirements.txt > .requirements.txt.unis
pip uninstall -y -r .requirements.txt.unis
pip uninstall -y vllm-ext
rm .requirements.txt.unis
