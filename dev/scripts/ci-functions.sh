#!/usr/bin/env bash
set -eo pipefail

HTTP_PROXY='http://proxy-prc.intel.com:912'
HTTPS_PROXY='http://proxy-prc.intel.com:912'
MODEL_CACHE_PATH_LOACL='/root/.cache/huggingface/hub'
CODE_CHECKOUT_PATH_LOCAL='/root/llm-on-ray'

build_and_prune() {
    # Set TARGET and DF-SUFFIX using the passed in parameters
    local TARGET=$1
    local DF_SUFFIX=$2
    local USE_PROXY=$3
    local PYTHON_V=$4

    docker_args=()
    docker_args+=("--build-arg=CACHEBUST=1")

    if [ -n "$PYTHON_V" ]; then
        docker_args+=("--build-arg=python_v=${PYTHON_V}")
    fi

    if [  "$USE_PROXY" == "1" ]; then
        docker_args+=("--build-arg=http_proxy=${HTTP_PROXY}")
        docker_args+=("--build-arg=https_proxy=${HTTPS_PROXY}")
    fi

    echo "Build Docker image and perform cleaning operation"
    echo "docker build ./ ${docker_args[@]} -f dev/docker/ci/Dockerfile${DF_SUFFIX} -t ${TARGET}:latest && yes | docker container prune && yes | docker image prune -f"

    # Build Docker image and perform cleaning operation
    docker build ./ "${docker_args[@]}" -f dev/docker/ci/Dockerfile${DF_SUFFIX} -t ${TARGET}:latest && yes | docker container prune && yes
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

    if [ -z "$model_cache_path" ];  then
        echo "no cache path"
    else
        docker_args+=("-v=${model_cache_path}:${MODEL_CACHE_PATH_LOACL}")
    fi

    docker_args+=("--name=${TARGET}" )
    docker_args+=("--hostname=${TARGET}-container")

    if [ "$USE_PROXY" == "1" ]; then
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

start_ray(){
    local TARGET=$1

    # Start Ray Cluster
    docker exec "${TARGET}" bash -c "./dev/scripts/start-ray-cluster.sh"
}

stop_ray(){
    local TARGET=$1

    cid=$(docker ps -q --filter "name=${TARGET}")
    if [[ ! -z "$cid" ]]; then
        docker exec "${TARGET}" bash -c "ray stop"
    fi
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
    ["llama-2-7b-chat-hf-no-vllm"]=".cpu_and_deepspeed"
    ["gpt-j-6b"]=".cpu_vllm_and_deepspeed.pip_non_editable"
)


get_DF_SUFFIX() {
    local key="$1"
    if [[ ${DF_SUFFIX_MAPPER[$key]+_} ]]; then
        echo "${DF_SUFFIX_MAPPER[$key]}"
    else
        echo ".cpu_vllm_and_deepspeed"
    fi
}

declare -A TARGET_SUFFIX_MAPPER
TARGET_SUFFIX_MAPPER=(
    ["mpt-7b-ipex-llm"]="_ipex-llm"
    ["llama-2-7b-chat-hf-no-vllm"]="_wo_vllm"
)

get_TARGET_SUFFIX() {
    local key="$1"
    if [[ ${TARGET_SUFFIX_MAPPER[$key]+_} ]]; then
        echo "${TARGET_SUFFIX_MAPPER[$key]}"
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

    echo Basic inference test:
    echo Non-streaming query:
    docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model}"
    echo Streaming query:
    docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model} --streaming_response"
}

inference_deepspeed_test(){
    local TARGET=$1
    local model=$2
    if [[ ${model} =~ ^(gemma-2b|gpt2|falcon-7b|starcoder|mpt-7b.*)$ ]]; then
        echo ${model} is not supported!
    elif [[ ! ${model} == "llama-2-7b-chat-hf-no-vllm" ]]; then
        echo update_inference_config with deepspeed:
        docker exec "${TARGET}" bash -c "python .github/workflows/config/update_inference_config.py --config_file llm_on_ray/inference/models/\"${model}\".yaml --output_file \"${model}\".yaml.deepspeed --deepspeed"
        echo Start deepspeed simple serve :
        docker exec "${TARGET}" bash -c "llm_on_ray-serve --config_file \"${model}\".yaml.deepspeed --simple"
        echo Non-streaming query:
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model}"
        echo Streaming query:
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/${model} --streaming_response"
    fi
}

inference_restapi_test(){
    local TARGET=$1
    local model=$2
    if [[ ${model} == "mpt-7b-ipex-llm" ]]; then
        echo Start mpt-7b-ipex-llm simple serve :
        docker exec "${TARGET}" bash -c "llm_on_ray-serve --config_file llm_on_ray/inference/models/ipex-llm/mpt-7b-ipex-llm.yaml"
    else
        echo Start "${TARGET}"  serve :
        docker exec "${TARGET}" bash -c "llm_on_ray-serve --models ${model}"
        echo Http query:
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_openai/query_http_requests.py --model_name ${model}"
    fi
}

inference_agent_restapi_test(){
    local TARGET=$1
    local model=$2
    if [[ ${model} == "llama-2-7b-chat-hf" ]]; then
        echo Start "${TARGET}"  serve :
        docker exec "${TARGET}" bash -c "llm_on_ray-serve --models ${model}"
        echo Http tool query:
        docker exec "${TARGET}" bash -c "python examples/inference/api_server_openai/query_http_requests_tool.py --model_name ${model}"
    fi
}

finetune_test(){
    local model=$1
    echo Set finetune source config :
    docker exec "finetune" bash -c "source \$(python -c 'import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)')/env/setvars.sh; RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external; RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1  ray start --address='127.0.0.1:6379' --ray-debugger-external"
    echo Set "${model}" patch_yaml_config :
    docker exec "finetune" bash -c "python dev/scripts/patch_yaml_config.py --conf_path "llm_on_ray/finetune/finetune.yaml" --models ${model} "
    echo Stert "${model}" finetune :
    docker exec "finetune" bash -c "llm_on_ray-finetune --config_file llm_on_ray/finetune/finetune.yaml"
}

peft_lora_test(){
    local model=$1
    docker exec "finetune" bash -c "rm -rf /tmp/llm-ray/*"
    echo Set "${model}" patch_yaml_config :
    docker exec "finetune" bash -c "python dev/scripts/patch_yaml_config.py --conf_path "llm_on_ray/finetune/finetune.yaml" --models ${model} --peft_lora"
    echo Stert "${model}" peft lora finetune :
    docker exec "finetune" bash -c "llm_on_ray-finetune --config_file llm_on_ray/finetune/finetune.yaml"
}

