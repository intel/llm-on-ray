# Setting up For Intel CPU/GPU/Gaudi

This guide provides detailed steps for settting up for Intel CPU, Intel Data Center GPU or Intel Gaudi2.

## Hardware and Software requirements

### Hardware Requirements

Ensure your setup includes one of the following Intel hardware:

* CPU:
Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processor

* GPU:

    |Product Name|Launch Date|Memory Size|Xe-cores|
    |---|---|---|---|
    |[Intel® Data Center GPU Max 1550](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html)|Q1'23|128 GB|128|
    |[Intel® Data Center GPU Max 1100](https://www.intel.com/content/www/us/en/products/sku/232876/intel-data-center-gpu-max-1100/specifications.html)|Q2'23|48 GB|56|

* Gaudi: Gaudi2

### Software Requirements
- Python 3.9

## Setup

#### 1. Prerequisites
For Intel GPU, ensure the [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) is installed.

For Gaudi, ensure the [SynapseAI SW stack and container runtime](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) is installed.

#### 2. Clone the repository and install dependencies.
```bash
git clone https://github.com/intel/llm-on-ray.git
cd llm-on-ray
```
For CPU:
```bash
pip install .[cpu] -f https://developer.intel.com/ipex-whl-stable-cpu -f https://download.pytorch.org/whl/torch_stable.html
```
For GPU:
```bash
pip install .[gpu] --extra-index-url https://developer.intel.com/ipex-whl-stable-xpu
```
If DeepSpeed is enabled or doing distributed finetuing, oneCCL and Intel MPI libraries should be dynamically linked in every node before Ray starts:
```bash
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
```

For Gaudi:

Please use the [Dockerfile](../inference/habana/Dockerfile) to build the image. Alternatively, you can install the dependecies on a bare metal machine. In this case, please refer to [here](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#build-docker-bare).

```bash
docker build \
    -f ${dockerfile} ../../ \
    -t llm-ray-habana:latest \
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy} \
```

After the image is built successfully, start a container:

```bash
docker run -it --runtime=habana -v ./llm-on-ray:/root/llm-ray --name="llm-ray-habana-demo" llm-ray-habana:latest 
```

#### 3. Launch Ray cluster
Start the Ray head node using the following command.
```bash
ray start --head --node-ip-address 127.0.0.1 --dashboard-host='0.0.0.0' --dashboard-port=8265
```
Optionally, for a multi-node cluster, start Ray worker nodes:
```bash
ray start --address='127.0.0.1:6379'
```
