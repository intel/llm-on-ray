# Deploying and Serving LLMs on Intel CPU/GPU/Gaudi

This guide provides detailed steps for deploying and serving LLMs on Intel CPU/GPU/Gaudi.

## Setup
Please follow [setup.md](setup.md) to setup the environment first.


## Configure Serving Parameters
We provide preconfigured yaml files in [inference/models](../inference/models) for popular open source models. You can customize a few configurations such as the resource used for serving. 

To deploy on CPU, please make sure `device` is set to CPU and `cpus_per_worker` is set to a correct number.
```
cpus_per_worker: 24
device: "cpu"
```
To deploy on GPU, please make sure `device` is set to GPU and `gpus_per_worker` is set to 1.
```
gpus_per_worker: 1
device: "gpu"
```
To deploy on Gaudi, please make sure `device` is set to hpu and `hpus_per_worker` is set to 1.
```
hpus_per_worker: 1
device: "hpu"
```
LLM-on-Ray also supports serving with [Deepspeed](serve_deepspeed.md) for AutoTP and [BigDL-LLM](serve_bigdl.md) for INT4/FP4/INT8/FP8 to reduce latency. You can follow the corresponding documents to enable them.

## Deploy the Model
To deploy your model, execute the following command with the model's configuration file. This will create an OpenAI-compatible API for serving.
```bash
python inference/run_model_serve.py --config_file <path to the conf file>
```
To deploy and serve multiple models concurrently, place all models' configuration files under `inference/models` and directly run `python inference/run_model_serve.py` without passing any conf file.

## Test the Model Endpoint
After deploying the model, you can access and test it  in many ways:
```bash
# using curl
export ENDPOINT_URL=http://localhost:8000/v1
curl $ENDPOINT_URL/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "gpt2",
    "messages": [{"role": "assistant", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    "temperature": 0.7
    }'

# using requests library
python examples/inference/api_server_openai/query_chat_streaming.py

# using OpenAI SDK
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=$your_openai_api_key
python examples/inference/api_server_openai/query_openai_sdk.py
```

