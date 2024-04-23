#!/usr/bin/env bash
set -eo pipefail

HTTP_PROXY='http://10.24.221.149:911'
HTTPS_PROXY='http://10.24.221.149:911'
MODEL_CACHE_PATH_LOACL='/root/.cache/huggingface/hub'
CODE_CHECKOUT_PATH_LOCAL='/root/llm-on-ray'

build_and_prune() {
    # Set TARGET and DF-SUFFIX using the passed in parameters
    local TARGET=$1
    local DF_SUFFIX=$2
    local PYTHON_V=$3  ## same name
    local USE_PROXY=$4

    docker_args=()
    docker_args+=("--build-arg=CACHEBUST=1")

    if [ -n "$PYTHON_V" ]; then
        docker_args+=("--build-arg=python_v=${PYTHON_V}")
    fi

    if [ -n "$USE_PROXY" ]; then
        docker_args+=("--build-arg=http_proxy=${HTTP_PROXY}")
        docker_args+=("--build-arg=https_proxy=${HTTPS_PROXY}")
    fi
    
    echo "docker build ./ ${docker_args[@]} -f dev/docker/Dockerfile${DF_SUFFIX} -t ${TARGET}:latest && yes | docker container prune && yes | docker image prune -f"

    # Build Docker image and perform cleaning operation
    docker build ./ "${docker_args[@]}" -f dev/docker/Dockerfile${DF_SUFFIX} -t ${TARGET}:latest && yes | docker container prune && yes 
    docker image prune -f

    
}


start_docker() {
    local TARGET=$1
    local code_checkout_path=$2
    local model_cache_path=$3
    local USE_PROXY=$4
    
    cid=$(docker ps -q --filter "name=${TARGET}")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid; fi
    # check and remove exited container
    cid=$(docker ps -a -q --filter "name=${TARGET}")
    if [[ ! -z "$cid" ]]; then docker rm $cid; fi
    docker ps -a

    docker_args=()
    docker_args+=("-v=${code_checkout_path}:${CODE_CHECKOUT_PATH_LOCAL}")
    docker_args+=("--name=${TARGET}" )
    docker_args+=("--hostname=${TARGET}-container")

    if [ -n "$model_cache_path" ]; then
        docker_args+=("-v="${model_cache_path }:${MODEL_CACHE_PATH_LOACL}"")
    fi
     if [ -n "$USE_PROXY" ]; then
        docker_args+=("-e=http_proxy=${HTTP_PROXY}")
        docker_args+=("-e=https_proxy=${HTTPS_PROXY}")
    fi
    echo "docker run -tid  "${docker_args[@]}" "${TARGET}:latest""
    docker run -tid  "${docker_args[@]}" "${TARGET}:latest"
}

install_dependencies(){
    local TARGET=$1

    # Install Dependencies for Tests
    docker exec "${TARGET}" bash -c "pip install -r ./tests/requirements.txt"
}

strat_ray(){
    local TARGET=$1

    # Start Ray Cluster
    docker exec "${TARGET}" bash -c "./dev/scripts/start-ray-cluster.sh"
}

run_tests(){
    local TARGET=$1
    
    # Run Tests
    docker exec "${TARGET}" bash -c "./tests/run-tests.sh"
}

stop_container(){
    local TARGET=$1

    # Stop Container
    cid=$(docker ps -q --filter "name=${TARGET}")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid; fi
}