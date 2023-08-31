#!/bin/bash
if [[ $1 = "dp" ]]; then
    dockerfile=Dockerfile-dp
fi 

docker build \
    -f ${dockerfile} . \
    -t llm-ray:$1 \
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy} \