## Deploying and Serving LLMs with IPEX-LLM
[IPEX-LLM](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/index.html) is a library for running LLM (large language model) on Intel XPU (from Laptop to GPU to Cloud) using INT4 with very low latency (for any PyTorch model).

The integration with IPEX-LLM currently only supports running on Intel CPU.

## Setup
Please follow [setup.md](setup.md) to setup the environment first. Additional, you will need to install IPEX-LLM dependencies as below.
```bash
pip install .[ipex-llm] --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

## Configure Serving Parameters
Please follow the serving [document](serve.md#configure-deploying-parameters) for configuring the parameters. In the configuration file, you need to set `ipex-llm` and `load_in_4bit` to true. Example configuration files for enalbing ipex-llm are availabe [here].(../inference/models/ipex-llm)

```bash
  ipexllm: true
  config:
    load_in_4bit: true
```

## Deploy and Test
Please follow the serving [document](serve.md#deploy-the-model) for deploying and testing.