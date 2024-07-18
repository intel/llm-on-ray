#!/bin/bash
set -eo pipefail

# If your model needs HF_TOKEN. Please modify the "model_description.config.use_auth_token" in the config file such as "llm_on_ray/inference/models/llama-2-7b-chat-hf.yaml" 
# Mount your own llm-on-ray directory here
code_checkout_path=$PWD
# Mount your own huggingface cache path here
model_cache_path=$HOME'/.cache/huggingface/hub'
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

    if [ -n "$http_proxy" ]; then
        docker_args+=("--build-arg=http_proxy=$http_proxy")
    fi

    if [ -n "$https_proxy" ]; then
        docker_args+=("--build-arg=https_proxy=$http_proxy")
    fi


    echo "Build Docker image and perform cleaning operation"
    echo "docker build ./ ${docker_args[@]} -f dev/docker/Dockerfile.user -t serving${DOCKER_NAME}:latest"

    # Build Docker image and perform cleaning operation
    docker build ./ "${docker_args[@]}" -f dev/docker/Dockerfile.user -t serving${DOCKER_NAME}:latest

}

start_docker() {
    local DOCKER_NAME=$1
    local MODEL_NAME=$2

    docker_args=()
    docker_args+=("--name=serving${DOCKER_NAME}" )
    if [ -z "$MODEL_NAME" ];  then
        echo "use default model"
    else
        docker_args+=("-e=model_name=${MODEL_NAME}")
    fi

    if [ -n "$http_proxy" ]; then
        docker_args+=("-e=http_proxy=$http_proxy")
    fi
    
    if [ -n "$https_proxy" ]; then
        docker_args+=("-e=https_proxy=$http_proxy")
    fi

    docker_args+=("-e=OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://localhost:8000/v1}")
    docker_args+=("-e=OPENAI_API_KEY=${OPENAI_API_KEY:-not_a_real_key}")

    # # If you need to use the modified llm-on-ray repository or huggingface model cache, activate the corresponding row
    docker_args+=("-v=$code_checkout_path:${CODE_CHECKOUT_PATH_LOCAL}")
    docker_args+=("-v=${model_cache_path}:${MODEL_CACHE_PATH_LOACL}")

    echo "docker run -ti  ${docker_args[@]} serving${DOCKER_NAME}:latest"
    docker run -ti ${docker_args[@]} serving${DOCKER_NAME}:latest
}
