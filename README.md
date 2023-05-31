# LLM on Ray Workflow

## Introduction
There are many reasons that you may want to build and serve your own Large Language Models(LLMs) such as cost, latency, data security, etc. With more high quality open source models being released, it becomes possible to finetune a LLM to meet your specific needs. However, finetuning a LLM is not a simple task as it involves many different technologies such as PyTorch, Huggingface, Deepspeed and more. It becomes more complex when scaling to a cluster as you also need to take care of infrastructure management, fault tolerance, etc. Serving a small LLM model on a single node might be simple. But deploying a production level online inference service is also challenging.
In this workflow, we show how you can finetune your own LLM with your proprietary data and deploy an online inference service easily on an Intel CPU cluster.



## Solution Technical Overview
Ray is a leading solution for scaling AI workloads, allowing people to train and deploy models faster and more efficiently. Ray is used by leading AI organizations to train LLMs at scale (e.g., by OpenAI to train ChatGPT, Cohere to train their models, EleutherAI to train GPT-J). Ray provides the ability to recover from training failure and auto scale the cluster resource. Ray is developer friendly with the ability to debug remote tasks easily. In this workload, we run LLM finetuning and serving on Ray to leverage all the benefits provided by Ray. Meanwhile we also integrate LLM optimizations from Intel such as IPEX.


## Solution Technical Details
To finetune a LLM, usually you start with selecting an open source base model. In rare case, you can also train the base model yourself and we plan to provide a pretraining workflow as well in the future.  Next, you can prepare your proprietary data and training configurations and feed them to the finetuning workflow. After the training is completed, a finetuned model will be generated. The serving workflow can take the new model and deploy it on Ray as an online inference service. You will get a model endpoint that can be used to integrate with your own application such as a chatbot.

![image](https://github.com/intel-sandbox/llm-ray/assets/9278199/11f07b52-d42d-4aeb-9693-ccb54fd20fc0)


## Hardware and Software requirements


### Hardware Requirements


### Software Requirements
- Docker 
- NFS 
- Python3


## Run this reference

### Finetuning Workflow
This `Finetune` folder contains code for fine-tuning large language model on top of Ray cluster. 

#### 1. Prepare Env
```bash
# on head node 
git clone https://github.com/intel-sandbox/llm-ray.git
cd llm-ray/Finetune
./build-image.sh 
# save docker image
docker save -o ray-image.tar ray-llm:latest
# copy over to worker nodes, this is an optional step if all your cluster nodes are NFS-shared
scp ray-image.tar <worker_node_ip>:<destination_path_on_worker_node>
# on worker nodes   
docker load -i ray-image.tar 
```

#### 2. Define workflow yaml file 
modify the `workflow.yaml` based on your needs


#### 3. Start the workflow
```python
python launch_workflow.sh -w workflow.yaml
```


### Inference Workflow

## How to customize this reference





