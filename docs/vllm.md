# Setting up vLLM For Intel CPU

__NOTICE: The support for vLLM is experimental and subject to change.__

## Install vLLM for Intel CPU

vLLM for CPU currently supports Intel® 4th Gen Xeon® Scalable Performance processor (formerly codenamed Sapphire Rapids) for best performance and is runnable with FP32 precision for other Xeon processors.

Please run the following script to install vLLM for CPU into your current environment. Currently a GNU C++ compiler with >=12.3 version is required to build and install.

```bash
$ dev/scripts/install-vllm-cpu.sh
```

## Setup

Please follow [Deploying and Serving LLMs on Intel CPU/GPU/Gaudi](serve.md) document to setup other environments.

## Run

#### Serving

To serve model with vLLM, run the following:

```bash
$ python serve.py --config_file inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --simple --keep_serve_terminal
```

In the above example, `vllm` property is set to `true` in the config file for enabling vLLM.

#### Querying

To start a non-streaming query, run the following:

```bash
$ python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/llama-2-7b-chat-hf
```

To start a streaming query, run the following:

```bash
$ python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/llama-2-7b-chat-hf --streaming_response
```