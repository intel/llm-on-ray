# LLM on Ray Workflow

## Introduction
There are many reasons that you may want to build and serve your own Large Language Models(LLMs) such as cost, latency, data security, etc. With more high quality open source models being released, it becomes possible to finetune a LLM to meet your specific needs. However, finetuning a LLM is not a simple task as it involves many different technologies such as PyTorch, Huggingface, Deepspeed and more. It becomes more complex when scaling to a cluster as you also need to take care of infrastructure management, fault tolerance, etc. Serving a small LLM model on a single node might be simple. But deploying a production level online inference service is also challenging.
In this workflow, we show how you can finetune your own LLM with your proprietary data and deploy an online inference service easily on an Intel CPU cluster.



## Solution Technical Overview
Ray is a leading solution for scaling AI workloads, allowing people to train and deploy models faster and more efficiently. Ray is used by leading AI organizations to train LLMs at scale (e.g., by OpenAI to train ChatGPT, Cohere to train their models, EleutherAI to train GPT-J). Ray provides the ability to recover from training failure and auto scale the cluster resource. Ray is developer friendly with the ability to debug remote tasks easily. In this workload, we run LLM finetuning and serving on Ray to leverage all the benefits provided by Ray. Meanwhile we also integrate LLM optimizations from Intel such as IPEX.


## Solution Technical Details
To finetune a LLM, usually you start with selecting an open source base model. In rare case, you can also train the base model yourself and we plan to provide a pretraining workflow as well in the future.  Next, you can prepare your proprietary data and training configurations and feed them to the finetuning workflow. After the training is completed, a finetuned model will be generated. The serving workflow can take the new model and deploy it on Ray as an online inference service. You will get a model endpoint that can be used to integrate with your own application such as a chatbot.

![image](https://github.com/intel-sandbox/llm-ray/assets/9278199/addd7a7f-83ef-43ae-b3ac-dd81cc2570e4)



## Hardware and Software requirements

There are workflow-specific hardware and software setup requirements depending on how the workflow is run.
### Hardware Requirements

|Recommended Hardware|Precision|
|-|-|
|Intel® 1st, 2nd, 3rd, and 4th Gen Xeon® Scalable Performance processo | FP32|

### Software Requirements
Workflow has been tested on Linux-4.18.0-408.el8.x86_64 and Ubuntu 22.04
- Docker
- NFS
- Python3 > 3.7.16

## Run This Workflow
### Prepare Environment
#### 1. Download the Workflow Repository
Create a working directory for the workflow and clone the repository into your working directory.
```bash
mkdir ~/workspace && cd ~/workspace
git clone https://github.com/intel-sandbox/llm-ray.git
cd llm-ray
git checkout main
```
#### 2. build Docker images for pretrain
```bash
cd llm-ray/tools/workload_in_containers
./build-image.sh megatron-habana # for Gaudi2 platform
```
or
```bash
./build-image.sh megatron-nvidia # for Nvidia GPU platform
```
#### 3. Install Dependencies

Skip this step, if using docker container.

```bash
pip install -r requirements.txt
pip install -r requirements.intel.txt -f https://developer.intel.com/ipex-whl-stable-cpu
```
(Optional) If you want to use DeepSpeed, please install the following requirements:
```bash
pip install -r requirements.deepspeed.txt
```

#### 4. Run the docker containers on head node and worker nodes for pretrain.
make the logs directory for saving the ray logs.
```bash
mkdir ~/workspace/logs
```
Gaudi2:
```bash
docker run -it --name megatron-habana --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v ~/workspace:/home/user/workspace -v ~/workspace/logs:/tmp --cap-add=sys_nice --net=host --ipc=host llm-ray:megatron-habana
```
Nvidia GPU:
```bash
docker run --gpus all -it --ulimit memlock=-1 --ulimit stack=67108864 --network host --name megatron-nvidia --shm-size=64g -v ~/workspace/logs:/tmp -v ~/workspace:/home/user/workspace llm-ray:megatron-gpu  /bin/bash
```

#### 5. Launch ray cluster
if using docker container, run the following commands in docker containers.
#### head node
```bash
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external --dashboard-host='0.0.0.0' --dashboard-port=8265
```
#### worker node
```bash
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1  ray start --address='127.0.0.1:6379' --ray-debugger-external
```

If deploying a ray cluster on multiple nodes, please download the workflow repository on each node. More information about ray cluster, please refer to https://www.ray.io/

### Pretrain Workflow
This workflow integrates the Megatron-DeepSpeed and Ray for pretrain.
For GPU version, we use the [Microsoft Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). For Gaudi2 version, we use the [HabanaAI Megatron-DeepSpeed](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed)


#### 1. Generate megatron datasets
Please refer to [this tutorial](../tools/redpajama_data_processing/README.md). Copy the datasets bin and idx files into ~/workspace/data

#### 2. Download the vocab file and merge table.
Download the GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) into ~/workspace/data.
```bash
cd ~/workspace/data/
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt 
```



#### 3. Pretrain Command

Please ensure that you check and modify the configuration files located in ~/workspace/llm-ray/pretrain/config/ before proceeding.

After your environment configuration are properly set up, you can use the following instructions to pretrain the language model:

##### Gaudi2:

Set up `megatron_deepspeed_path` in the configuration.

```bash
cd /home/user/workspace/llm-ray
#Bloom-7B
python pretrain/pretrain.py --config_path pretrain/config/bloom_7b_megatron_deepspeed_zs0_8Gaudi_pretrain.conf
#llama-7B
python pretrain/pretrain.py --config_path pretrain/config/llama_7b_megatron_deepspeed_zs0_8Gaudi_pretrain.conf
```
##### Nvidia GPU:
```bash
cd /home/user/workspace/llm-ray
#llama2-7B
python pretrain/pretrain.py --config_path pretrain/config/llama2_3b_megatron_deepspeed_zs0_8gpus_pretrain.conf
```
### Finetune Workflow


#### 1. Prepare Dataset

Now, the workflow only supports datasets in the specified format

The format of dataset similar to [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k). This type of data is used for finetuning in prompt mode and this type of data is characterized by containing `instruction` `context` and `response` fields where `instruction` and `response` are required fields and `context` is an optional field. In the data preprocessing stage, the three fields will be concatenated to the corresponding format according to [dolly](https://github.com/databrickslabs/dolly/blob/master/training/trainer.py#LL93).


The meaning of the above three columns:
+ Instruction Column: The column in the dataset is the user input, such as a question or a command.
+ Context Column: This column is other information used by instruction, such as the options used in the question and so on. It can be empty.
+ Response: The column in the dataset containing the expected output.


Therefore, if the your data meets the above two formats, you can use the data by configuring the local data path or huggingface dataset. If not, please refer to the following **Adopt to Your Dataset**.

#### 2. Finetune

The workflow is designed for configure driven. All argument about the workflow are record to just one configure. So user can launch a task with a simple common command. Once the prerequisits have been met, use the following commands to run the workflow:
```bash
python finetune/finetune.py --config_path finetune/finetune.conf
```

#### 3. Expected Output
The successful execution of this stage will create the below contents under `output` and `checkpoint` directory. Model for deploying is in `output`. `checkpoint` help us to recovery from fault. `TorchTrainer_xxx` is generated by ray cluster automatically.
```
/tmp/llm-ray/output/
|-- config.json
|-- generation_config.json
|-- pytorch_model-00001-of-00003.bin
|-- pytorch_model-00002-of-00003.bin
|-- pytorch_model-00003-of-00003.bin
`-- pytorch_model.bin.index.json
/tmp/llm-ray/output/
|-- test_0-of-2
|   `-- dict_checkpoint.pkl
`-- test_1-of-2
    `-- dict_checkpoint.pkl
/tmp/llm-ray/TorchTrainer_2023-06-05_08-50-46/
|-- TorchTrainer_10273_00000_0_2023-06-05_08-50-47
|   |-- events.out.tfevents.1685955047.localhost
|   |-- params.json
|   |-- params.pkl
|   |-- rank_0
|   |-- rank_1
|   `-- result.json
|-- basic-variant-state-2023-06-05_08-50-46.json
|-- experiment_state-2023-06-05_08-50-46.json
|-- trainable.pkl
|-- trainer.pkl
`-- tuner.pkl
```


### Inference Workflow
The inference workflow provides two execution methods, deploying it by UI or terminal execution.
#### Deploy by UI
![image](https://github.com/intel-sandbox/llm-ray/assets/97155466/c676e2f1-9e17-4bea-815d-8f7e21d68582)


This method will launch a UI interface and deploy an online inference service.
- (Optional) If customed models need to be added, please update `inference/config.py`.
```bash
python inference/start_ui.py
# Running on local URL:  http://0.0.0.0:8080
# Running on public URL: https://180cd5f7c31a1cfd3c.gradio.live
```
Access url and deploy service in a few simple clicks.
#### Terminal Execution
Ray serve is used to deploy models. First the model is exposed over HTTP by using Deployment, then test it over HTTP request.

A specific model can be deployed by specifying the model path and tokenizer path. If you want to enabled DeepSpeed, please add `--deepspeed` to the command line.

```bash
# If you dont' want to view serve logs, you can set env var, "KEEP_SERVE_TERMINAL" to false

# Run model serve with specified model and tokenizer
python inference/run_model_serve.py --model $model --tokenizer $tokenizer

# Run model serve with DeepSpeed
python inference/run_model_serve.py --model $model --tokenizer $tokenizer --deepspeed

# INFO - Deployment 'custom-model_PredictDeployment' is ready at `http://127.0.0.1:8000/custom-model`. component=serve deployment=custom-model_PredictDeployment
# Service is deployed successfully

python inference/run_model_infer.py --model_endpoint http://127.0.0.1:8000/custom-model
```
Otherwise, all the models configured in `inference/config.py` will be deployed by default. If you want to choose a specific model to deploy, you can set env var, "MODEL_TO_SERVE", to your choice.  You can add customized models in it as needed.


## Customize
### Adopt to your dataset
If your data do not meets the supported formats, you may need to preprocess the data into the standard format. Here we provide several examples include `example/finetune/open_assistant` and `example/finetune/dolly1`. After running `cd examples/finetune/open_assistant; python process_data.py`, a directory named `data` will output to the `examples/finetune/open_assistant` directory, just modify the configuration items `train_file` and `validation_file` to the corresponding file in `data` to start your own task. So does `example/finetune/dolly1` and other user-defined dataset.

