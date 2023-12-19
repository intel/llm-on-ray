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
To deploy your model, execute the following command with the model's configuration file. This will create an endpoint for serving, for example: http://127.0.0.1:8000/gpt2.
```bash
python inference/run_model_serve.py --config_file <path to the conf file>
```
To deploy and serve multiple models concurrently, place all models' configuration files under `inference/models` and directly run `python inference/run_model_serve.py` without passing any conf file.

## Test the Model Endpoint
After deploying the model endpoint, you can access and test it through curl, OpenAI SDK, or by using the script below:
```bash
python examples/inference/api_server/run_model_infer.py --model_endpoint <the model endpoint URL>
```

