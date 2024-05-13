#!/usr/bin/env bash
set -eo pipefail

##Set Your proxy and cache path here
HTTP_PROXY='http://10.24.221.169:911'
HTTPS_PROXY='http://10.24.221.169:911'
MODEL_CACHE_PATH_LOACL='/root/.cache/huggingface/hub'
CODE_CHECKOUT_PATH_LOCAL='/root/llm-on-ray'
HF_TOKEN='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'

build_and_prune() {
    # Set TARGET and DF-SUFFIX using the passed in parameters
    local TARGET="$1"
    local DF_SUFFIX="$2"

    docker_args=()
    docker_args+=("--build-arg=CACHEBUST=1")
    docker_args+=("--build-arg=http_proxy=${HTTP_PROXY}")
    docker_args+=("--build-arg=https_proxy=${HTTPS_PROXY}")


    echo "Build Docker image and perform cleaning operation"
    echo "docker build ./ ${docker_args[@]} -f dev/docker/Dockerfile${DF_SUFFIX} -t ${TARGET}:latest && yes | docker container prune && yes | docker image prune -f"

    # Build Docker image and perform cleaning operation
    docker build ./ "${docker_args[@]}" -f dev/docker/Dockerfile${DF_SUFFIX} -t ${TARGET}:latest && yes | docker container prune && yes
    docker image prune -f
}

start_docker() {
    local TARGET="$1"
    local code_checkout_path="$2"
    local model_cache_path="$3"

    cid=$(docker ps -q --filter "name=${TARGET}")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid; fi
    # check and remove exited container
    cid=$(docker ps -a -q --filter "name=${TARGET}")
    if [[ ! -z "$cid" ]]; then docker rm $cid; fi
    docker ps -a

    docker_args=()
    docker_args+=("-v=${code_checkout_path}:${CODE_CHECKOUT_PATH_LOCAL}")

    docker_args+=("-v=${model_cache_path}:${MODEL_CACHE_PATH_LOACL}")


    docker_args+=("--name=${TARGET}" )
    docker_args+=("--hostname=${TARGET}-container")

    docker_args+=("-e=http_proxy=${HTTP_PROXY}")
    docker_args+=("-e=https_proxy=${HTTPS_PROXY}")


    echo "docker run -tid  "${docker_args[@]}" "${TARGET}:latest""
    docker run -tid  "${docker_args[@]}" "${TARGET}:latest"

    docker exec "${TARGET}" bash -c "huggingface-cli login --token ${HF_TOKEN}"

}
