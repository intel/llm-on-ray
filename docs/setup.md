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
- Git
- Conda
- Docker

## Setup

#### 1. Prerequisites
For Intel GPU, ensure the [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) is installed.

For Gaudi, ensure the [SynapseAI SW stack and container runtime](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) is installed.

#### 2. Clone the repository and install dependencies.
```bash
git clone https://github.com/intel/llm-on-ray.git
cd llm-on-ray
conda create -n llm-on-ray python=3.9
conda activate llm-on-ray
```
##### For CPU:
```bash
pip install .[cpu] --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```
##### For GPU:
```bash
pip install .[gpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
If DeepSpeed is enabled or doing distributed finetuing, oneCCL and Intel MPI libraries should be dynamically linked in every node before Ray starts:
```bash
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl; print(torch_ccl.cwd)")/env/setvars.sh
```

##### For Gaudi:

Please use the [Dockerfile](../dev/docker/Dockerfile.habana) to build the image. Alternatively, you can install the dependecies on a bare metal machine. In this case, please refer to [here](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#build-docker-bare).

```bash
# Under dev/docker/
cd ./dev/docker
docker build \
    -f Dockerfile.habana ../../ \
    -t llm-on-ray:habana \
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy}
```

After the image is built successfully, start a container:

```bash
# llm-on-ray mounting is necessary.
# Please replace /path/to/llm-on-ray with your actual path to llm-on-ray.
# Add -p HOST_PORT:8080 or --net host if using UI.
# Add --cap-add sys_ptrace to enable py-spy in container if you need to debug.
# Set HABANA_VISIBLE_DEVICES if multi-tenancy is needed, such as "-e HABANA_VISIBLE_DEVICES=0,1,2,3"
# For multi-tenancy, refer to https://docs.habana.ai/en/latest/PyTorch/Reference/PT_Multiple_Tenants_on_HPU/Multiple_Dockers_each_with_Single_Workload.html
docker run -it --runtime=habana --name="llm-ray-habana-demo" -v /path/to/llm-on-ray:/root/llm-on-ray -v /path/to/models:/models/in/container -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host llm-on-ray:habana 
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
