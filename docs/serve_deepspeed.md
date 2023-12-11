## Deploying and Serving LLMs with Deepspeed
This guide provides steps for deploying and serving LLMs with Deepspeed, to legerage features such as automatic tensor parallelism (AutoTP).

## Setup
Please follow [setup.md](setup.md) to setup the environment first. Additional, you will need to install deepspeed dependencies as below.
```bash
pip install .[deepspeed]
```

## Configure Serving Parameters
Please follow the serving [document](serve.md#configure-deploying-parameters) for configuring the parameters. In the configuration file, you need to set `deepspeed` to true to enable Deepspeed AutoTP feature.

```
deepspeed: true
```

## Deploy and Test
Please follow the serving [document](serve.md#deploy-the-model) for deploying and testing.
