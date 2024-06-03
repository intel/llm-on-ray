#!/usr/bin/env bash
set -eo pipefail

##Set Your proxy and cache path here
HTTP_PROXY='http://10.24.221.169:911'
HTTPS_PROXY='http://10.24.221.169:911'
HF_TOKEN='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'
code_checkout_path='/home/yutianchen/Project/pr_lib/llm-on-ray'
model_cache_path='/home/yutianchen/.cache/huggingface/hub'
MODEL_CACHE_PATH_LOACL='/root/.cache/huggingface/hub'
CODE_CHECKOUT_PATH_LOCAL='/root/llm-on-ray'


build_docker() {

    docker_args=()
    docker_args+=("--build-arg=CACHEBUST=1")
    docker_args+=("--build-arg=DOCKER_NAME=".cpu_and_deepspeed"")
    docker_args+=("--build-arg=PYPJ="cpu,deepspeed"")

    # # If you need to use proxy,activate the following two lines
    docker_args+=("--build-arg=http_proxy=${HTTP_PROXY}")
    docker_args+=("--build-arg=https_proxy=${HTTPS_PROXY}")


    echo "Build Docker image and perform cleaning operation"
    echo "docker build ./ ${docker_args[@]} -f dev/docker/Dockerfile.user -t serving:latest"

    # Build Docker image and perform cleaning operation
    docker build ./ "${docker_args[@]}" -f dev/docker/Dockerfile.user -t serving:latest 

}

start_docker() {

    docker_args=()
    docker_args+=("--name=serving" )
    docker_args+=("--hostname=serving-container")

    docker_args+=("-e=hf_token=${HF_TOKEN}")

    # # If you need to use proxy,activate the following two lines
    docker_args+=("-e=http_proxy=${HTTP_PROXY}")
    docker_args+=("-e=https_proxy=${HTTPS_PROXY}")

    # # If you need to use the modified llm-on-ray repository or huggingface model cache, activate the corresponding row
    docker_args+=("-v=${code_checkout_path}:${CODE_CHECKOUT_PATH_LOCAL}")
    docker_args+=("-v=${model_cache_path}:${MODEL_CACHE_PATH_LOACL}")

    echo "docker run -tid  "${docker_args[@]}" "serving:latest""
    # docker run -tid  "${docker_args[@]}" "serving:latest"

}