#!/bin/bash
set -eo pipefail

# Usage: ./test_setup [CPU, GPU, Gaudi] [True for Deepspeed and false for disable]

if [ "$#" != 2 ]; then
    echo "Error, there should be 2 arguments! See Usage: ./test_setup [CPU, GPU, Gaudi] [True for Deepspeed and false for disable]"
    exit 1
fi

# Step 1: Clone the repository, install llm-on-ray and its dependencies.
echo "Checkout already done in the workflow."

# Step 2: Check CPU, GPU or Gaudi and install dependencies
hardware=0
deepspeed=false
case $(echo $1 | tr 'a-z' 'A-Z') in
    "CPU")
        hardware=1
        pip install .[cpu] --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
        bash ./dev/scripts/install-vllm-cpu.sh
        ;;
    "GPU")
        pip install .[gpu] --extra-index-url https://developer.intel.com/ipex-whl-stable-xpu
        hardware=2
        ;;
    "GAUDI")
        hardware=3
        ;;
    *)
        exit "Error: CPU, GPU, Gaudi should be setted as the first variable of this script."
        ;;
esac

# Check if it needs deepspeed
if [ $(echo $2 | tr 'A-Z' 'a-z') == "true" ]
then
    deepspeed=true
    source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl; print(torch_ccl.cwd)")/env/setvars.sh
fi

# If Gaudi, build docker
if [ "$hardware" == 3 ]; then
    export dockerfile=inference/habana/Dockerfile
    export no_proxy="localhost,127.0.0.1"
    docker build \
        -f ${dockerfile} ../../ \
        -t llm-ray-habana:latest \
        --network=host \
        --build-arg no_proxy=${no_proxy}
fi


# Step 3: Launch ray cluster
# Start head node
ray start --head --node-ip-address 127.0.0.1 --dashboard-host='0.0.0.0' --dashboard-port=8265
# Start worker node
ray start --address='127.0.0.1:6379'

# Step 4: Check if it is installed correctly
llm_on_ray-serve --help
llm_on_ray-finetune --help
