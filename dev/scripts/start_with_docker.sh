#!/usr/bin/env bash
set -eo pipefail

##Set Your proxy and cache path here
HTTP_PROXY='Your proxy'
HTTPS_PROXY='Your proxy'
HF_TOKEN='Your hf_token'
code_checkout_path='If you need to use the modified llm-on-ray repository, define your path here'
model_cache_path='If you need to use huggingface model cache, define your path here'
MODEL_CACHE_PATH_LOACL='/root/.cache/huggingface/hub'
CODE_CHECKOUT_PATH_LOCAL='/root/llm-on-ray'


build_docker() {
    local DOCKER_NAME=$1

    docker_args=()
    docker_args+=("--build-arg=CACHEBUST=1")
    if [ "$DOCKER_NAME" == "vllm" ]; then
        docker_args+=("--build-arg=DOCKER_NAME=".vllm"")
        docker_args+=("--build-arg=PYPJ="vllm"")
    elif [ "$DOCKER_NAME" == "ipex-llm" ]; then
        docker_args+=("--build-arg=DOCKER_NAME=".ipex-llm"")
        docker_args+=("--build-arg=PYPJ="ipex-llm"")
    else 
        docker_args+=("--build-arg=DOCKER_NAME=".cpu_and_deepspeed"")
        docker_args+=("--build-arg=PYPJ="cpu,deepspeed"")
    fi

    # # If you need to use proxy,activate the following two lines
    # docker_args+=("--build-arg=http_proxy=${HTTP_PROXY}")
    # docker_args+=("--build-arg=https_proxy=${HTTPS_PROXY}")


    echo "Build Docker image and perform cleaning operation"
    echo "docker build ./ ${docker_args[@]} -f dev/docker/Dockerfile.user -t serving:latest"

    # Build Docker image and perform cleaning operation
    docker build ./ "${docker_args[@]}" -f dev/docker/Dockerfile.user -t serving:latest 

}

start_docker() {
    local MODEL_NAME=$1

    docker_args=()
    docker_args+=("--name=serving" )
    docker_args+=("-e=hf_token=${HF_TOKEN}")
    if [ -z "$MODEL_NAME" ];  then
        echo "use default model"
    else
        docker_args+=("-e=model_name=${MODEL_NAME}")
    fi

    # # If you need to use proxy,activate the following two lines
    # docker_args+=("-e=http_proxy=${HTTP_PROXY}")
    # docker_args+=("-e=https_proxy=${HTTPS_PROXY}")

    # # If you need to use the modified llm-on-ray repository or huggingface model cache, activate the corresponding row
    # docker_args+=("-v=${code_checkout_path}:${CODE_CHECKOUT_PATH_LOCAL}")
    # docker_args+=("-v=${model_cache_path}:${MODEL_CACHE_PATH_LOACL}")

    echo "docker run -tid  "${docker_args[@]}" "serving:latest""
    docker run -tid  "${docker_args[@]}" "serving:latest"

}
