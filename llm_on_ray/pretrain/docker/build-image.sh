#!/bin/bash
dockerfile=Dockerfile
if [[ $1 = "nvidia" ]]; then
    dockerfile=Dockerfile.nvidia
elif [[ $1 = "megatron-habana" ]]; then
    dockerfile=Dockerfile.megatron.habana
elif [[ $1 = "optimum-habana" ]]; then
    dockerfile=Dockerfile.optimum.habana
fi
docker build \
    -f ${dockerfile} ../../ \
    -t llm-ray:$1 \
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy} \
