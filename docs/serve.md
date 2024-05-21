# Deploying and Serving LLMs on Intel CPU/GPU/Gaudi

This guide provides detailed steps for deploying and serving LLMs on Intel CPU/GPU/Gaudi.

## Setup
Please follow [setup.md](setup.md) to setup the environment first.


## Configure Serving Parameters
We provide preconfigured yaml files in [inference/models](../llm_on_ray/inference/models) for popular open source models. You can customize a few configurations for serving.

### Resource
To deploy on CPU, please make sure `device` is set to CPU and `cpus_per_worker` is set to a correct number.
```
cpus_per_worker: 24
device: cpu
```
To deploy on GPU, please make sure `device` is set to GPU and `gpus_per_worker` is set to 1.
```
gpus_per_worker: 1
device: gpu
```
To deploy on Gaudi, please make sure `device` is set to hpu and `hpus_per_worker` is set to 1.
```
hpus_per_worker: 1
device: hpu
```
LLM-on-Ray also supports serving with [Deepspeed](serve_deepspeed.md) for AutoTP and [IPEX-LLM](serve_ipex-llm.md) for INT4/FP4/INT8/FP8 to reduce latency. You can follow the corresponding documents to enable them.

### Autoscaling
LLM-on-Ray supports automatically scales up and down the number of serving replicas based on the resources of ray cluster and the requests traffic. You can adjust the autoscaling strategy through the following parameters in configuration file. You can follow the [guides-autoscaling-config-parameters](https://docs.ray.io/en/master/serve/advanced-guides/advanced-autoscaling.html#autoscaling-config-parameters) for more detailed explanation of these parameters.

```
max_ongoing_requests: 64
autoscaling_config:
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 2
    target_ongoing_requests: 24
    downscale_delay_s: 30
    upscale_delay_s: 10
```

## Serving
We support two methods to specify the models to be served, and they have the following priorities.
1. Use inference configuration file if config_file is set.
```
llm_on_ray-serve --config_file llm_on_ray/inference/models/gpt2.yaml
```
```
2. If --config_file is None, it will serve GPT2 by default, or the models specified by --models.
```
llm_on_ray-serve --models gpt2 gpt-j-6b
```
### OpenAI-compatible API
To deploy your model, execute the following command with the model's configuration file. This will create an OpenAI-compatible API ([OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)) for serving.
```bash
llm_on_ray-serve --config_file <path to the conf file>
```
To deploy and serve multiple models concurrently, place all models' configuration files under `llm_on_ray/inference/models` and directly run `llm_on_ray-serve` without passing any conf file.

After deploying the model, you can access and test it in many ways:
```bash
# using curl
export ENDPOINT_URL=http://localhost:8000/v1
curl $ENDPOINT_URL/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": $MODEL_NAME,
    "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
    "temperature": 0.7
    }'

# using requests library
python examples/inference/api_server_openai/query_http_requests.py

# using OpenAI SDK
# please install openai in current env by running: pip install openai>=1.0
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY="not_a_real_key"
python examples/inference/api_server_openai/query_openai_sdk.py
```
### Serving Model to a Simple Endpoint
This will create a simple endpoint for serving according to the `port` and `route_prefix` parameters in conf file, for example: http://127.0.0.1:8000/gpt2.
```bash
llm_on_ray-serve --config_file <path to the conf file> --simple
```
After deploying the model endpoint, you can access and test it by using the script below:
```bash
python examples/inference/api_server_simple/query_single.py --model_endpoint <the model endpoint URL>
```

