# Setting up vLLM For Intel CPU

__NOTICE: The support for vLLM is experimental and subject to change.__

## Install vLLM for Intel CPU

vLLM for CPU currently supports Intel® 4th Gen Xeon® Scalable Performance processor (formerly codenamed Sapphire Rapids) for best performance and is runnable with FP32 precision for other Xeon processors.

Currently a GNU C++ compiler with >=12.3 version is required to build and install. In Ubuntu 22.4, you can use the following commands to install GCC 12.3:

```bash
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

Then please run the following script to install vLLM for CPU into your current environment.

```bash
dev/scripts/install-vllm-cpu.sh
```

## Setup

Please follow [Deploying and Serving LLMs on Intel CPU/GPU/Gaudi](serve.md) document to setup other environments.

## Run

#### Serving

To serve model with vLLM, run the following:

```bash
llm_on_ray-serve --config_file llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --simple --keep_serve_terminal
```

In the above example, `vllm` property is set to `true` in the config file for enabling vLLM.

#### Querying

To start a non-streaming query, run the following:

```bash
python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/llama-2-7b-chat-hf
```

To start a streaming query, run the following:

```bash
python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/llama-2-7b-chat-hf --streaming_response
```