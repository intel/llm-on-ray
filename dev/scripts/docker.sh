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
build_and_prune_inference() {
    # Set TARGET and DF-SUFFIX using the passed in parameters
    local TARGET=$1
    local DF_SUFFIX=$2

    docker_args=()
    docker_args+=("--build-arg=CACHEBUST=1")

    docker_args+=("--build-arg=http_proxy=${HTTP_PROXY}")
    docker_args+=("--build-arg=https_proxy=${HTTPS_PROXY}")
    
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

    if [-n "$model_cache_path"]; then
        docker_args+=("-v="${model_cache_path }:${MODEL_CACHE_PATH_LOACL}"")
    fi

    docker_args+=("-e=http_proxy=${HTTP_PROXY}")
    docker_args+=("-e=https_proxy=${HTTPS_PROXY}")

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

# declare map
declare -A DF_SUFFIX_MAPPER
DF_SUFFIX_MAPPER=(
    ["mpt-7b-ipex-llm"]=".ipex-llm"
    ["llama-2-7b-chat-hf-vllm"]=".vllm"
    ["gpt-j-6b"]=".cpu_and_deepspeed.pip_non_editable"
)


get_DF_SUFFIX_MAPPER() {
    local key="$1"
    if [[ ${DF_SUFFIX_MAPPER[$key]+_} ]]; then
        echo "${DF_SUFFIX_MAPPER[$key]}"
    else
        echo ".cpu_and_deepspeed"
    fi
}

declare -A TARGET_MAPPER
TARGET_MAPPER=(
    ["mpt-7b-ipex-llm"]="_ipex-llm"
    ["llama-2-7b-chat-hf-vllm"]="_vllm"
)

get_TARGET_MAPPER() {
    local key="$1"
    if [[ ${TARGET_MAPPER[$key]+_} ]]; then
        echo "${TARGET_MAPPER[$key]}"
    else
        echo ""
    fi
}


declare -A INFERENCE_MAPPER
INFERENCE_MAPPER=(
    ["mpt-7b-ipex-llm"]="llm_on_ray-serve --config_file llm_on_ray/inference/models/ipex-llm/mpt-7b-ipex-llm.yaml --simple"
    ["llama-2-7b-chat-hf-vllm"]="llm_on_ray-serve --config_file .github/workflows/config/llama-2-7b-chat-hf-vllm-fp32.yaml --simple"
    ["default"]="llm_on_ray-serve --simple --models ${model}"
)

inference_test(){
local TARGET=$1
local model=$2
if [[ ${INFERENCE_MAPPER[$model]+_} ]]; then
    echo "${INFERENCE_MAPPER[$model]}"
    docker exec "${TARGET}" bash -c "${INFERENCE_MAPPER["${model}"]}"
else
    echo "${INFERENCE_MAPPER["default"]}"
    docker exec "${TARGET}" bash -c "llm_on_ray-serve --simple --models ${model}"
fi

echo Non-streaming query:
docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model}"
echo Streaming query:
docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model} --streaming_response"
}

inference_test_deltatuner(){
    local TARGET=$1
    local dtuner_model=$2
    local model=$3
    echo "${dtuner_model}"
    if [ "${dtuner_model}" ];then
        docker exec "${TARGET}" bash -c "llm_on_ray-serve --config_file .github/workflows/config/mpt_deltatuner.yaml --simple"
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model }"
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model } --streaming_response"
    fi
}


inference_test_deepspeed(){
    local TARGET=$1
    local model=$2
    if [[ ${model} =~ ^(gemma-2b|gpt2|falcon-7b|starcoder|mpt-7b.*)$ ]]; then
        echo ${model} is not supported!
    elif [[ ! ${model} == "llama-2-7b-chat-hf-vllm" ]]; then
        docker exec "${TARGET}" bash -c "python .github/workflows/config/update_inference_config.py --config_file llm_on_ray/inference/models/\"${model}\".yaml --output_file \"${model}\".yaml.deepspeed --deepspeed"
        docker exec "${TARGET}" bash -c "llm_on_ray-serve --config_file \"${model}\".yaml.deepspeed --simple"
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model}"
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model} --streaming_response"
    fi
}

inference_test_deepspeed_deltatuner(){
    local TARGET=$1
    local dtuner_model=$2
    local model=$3
    if [[ ${model} =~ ^(gemma-2b|gpt2|falcon-7b|starcoder|mpt-7b.*)$ ]]; then
        echo ${model} is not supported!
    elif [[ ! ${model} == "llama-2-7b-chat-hf-vllm" ]]; then
        docker exec "${TARGET}" bash -c "llm_on_ray-serve --config_file .github/workflows/config/mpt_deltatuner_deepspeed.yaml --simple"
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${{ matrix.model }}"
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${{ matrix.model }} --streaming_response"
    fi
}

# model="mpt-7b-ipex-llm"
# target=${TARGET_MAPPER[$model]}