## Deploying and Serving LLMs with Intel Habana Gaudi(HPU)
[Habana Gaudi AI Processors (HPUs)](https://habana.ai) are AI hardware accelerators designed by Habana Labs. For more information, see [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/index.html) and [Gaudi Developer Docs](https://developer.habana.ai/).

This guide provides steps for deploying and serving LLMs with HPU, including these topics:

* Deploy Llama2-7b model using single HPU
* Deploy Llama2-70b model using 8 HPUs
* How to deploy them on our Web UI

## Setup
Please follow [setup.md](setup.md) to setup the environment for HPU first.

## How to serve model on HPU
Please follow the serving [document](serve.md#configure-deploying-parameters) for configuring the parameters. For HPU, we need to set:

```
device: "hpu"
hpus_per_worker: 1
```

As an example, you can try [this configuration file](../llm_on_ray/inference/models/hpu/llama-2-7b-chat-hf-hpu.yaml).

### DeepSpeed on HPU

To serve Llama2-70b on HPU, we need to enable DeepSpeed. In addition to the config introduced above, the following config are required:

```
deepspeed: true
workers_per_group: 8
```

As an example, you can try [this configuration file](../llm_on_ray/inference/models/hpu/llama-2-70b-chat-hf-hpu.yaml).

## Deploy and Test
Please follow the serving [document](serve.md#deploy-the-model) for deploying and testing.

## Web UI

If you want to use our web ui, you need to first setup SSH password-less login:

```bash
# In the container
# Hit enter for all prompts
ssh-keygen
# Add the generated public key 
cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys
# If using multiple nodes, run ssh-keygen in all containers
# then copy all public keys to the container where ray head is
```

You can now start the UI: `python llm_on_ray/ui/start-ui.py --node_port 3022 --master_ip_port RAY_HEAD_ADDRESS`

Under the **inference** tab, choose the models to deploy. Then, expand **Parameters**, drag the slider for CPU to 1, and the one for HPU to 1. You can now hit the **Deploy** button and wait for the service is up. After you see model endpoint gets printed, you can now chat with the deployed model.
