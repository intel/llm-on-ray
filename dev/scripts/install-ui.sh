#!/usr/bin/env bash

# install dependency
pip install "gradio<=3.36.1" "langchain<=0.0.329" "langchain_community<=0.0.13" "paramiko<=3.4.0" "sentence-transformers" "faiss-cpu"

# install pyrecdp from source
pip install 'git+https://github.com/intel/e2eAIOK.git#egg=pyrecdp&subdirectory=RecDP'
