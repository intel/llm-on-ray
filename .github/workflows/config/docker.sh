
## $1 TARGET

fuction build_and_prune() {
    # Set TARGET and DF-SUFFIX using the passed in parameters
    local TARGET="$1"
    local DF_SUFFIX="$2"
    local http_proxy="$3"
    local https_proxy="$4"

    # Build Docker image and perform cleaning operation
    docker build ./ \
        --build-arg CACHEBUST=1 \
        --build-arg http_proxy="${http_proxy}" \
        --build-arg https_proxy="${https_proxy}" \
        -f "dev/docker/Dockerfile${DF_SUFFIX}" \
        -t "${TARGET}:latest" && \
    yes | docker container prune && \
    yes | docker image prune -f
}