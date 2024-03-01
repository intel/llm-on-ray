set -eo pipefail
## $1 TARGET

build_and_prune() {
    # Set TARGET and DF-SUFFIX using the passed in parameters
    local TARGET="$1"
    local DF_SUFFIX="$2"
    local http_proxy="$3:-"
    local https_proxy="$4:-"

    # Build Docker image and perform cleaning operation
    docker build ./ \
        --build-arg CACHEBUST=1 \
        # ${http_proxy:+--build-arg http_proxy="$http_proxy"} \
        # ${https_proxy:+--build-arg https_proxy="$https_proxy"} \
        -f dev/docker/Dockerfile${DF_SUFFIX} \
        -t ${TARGET}:latest && yes | docker container prune && yes | docker image prune -f
}

run_docker() {
    local TARGET="$1" 
    local DF_SUFFIX="$2"  
    local http_proxy="$3"
    local https_proxy="$4"
    local model_cache_path="$5"
    local code_checkout_path="$6"

    cid=$(docker ps -q --filter "name=${TARGET}${DF_SUFFIX}")
    if [[ ! -z "$cid" ]]; then
        docker stop "$cid" && docker rm "$cid"
    fi

    # check and remove exited container
    cid=$(docker ps -a -q --filter "name=${TARGET}${DF_SUFFIX}")
    if [[ ! -z "$cid" ]]; then
        docker rm "$cid"
    fi

    docker run -tid \
        -v "${{model_cache_path }}:/root/.cache/huggingface/hub" \
        -v "${{code_checkout_path }}:/root/llm-on-ray" \
        -e http_proxy="${{http_proxy }}" \
        -e https_proxy="${{https_proxy }}" \
        --name="${TARGET}${DF_SUFFIX}" \
        --hostname="${TARGET}-container" \
        "${TARGET}:latest"
}

docker_bash(){
    local TARGET="$1" 
    local bash_command="$2"

    docker exec "${TARGET}" bash -c "${bash_command}"
}