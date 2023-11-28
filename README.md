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


## Workflows

* [Setup](#setup)
* [Finetuning](#finetune)
* [Serving](#serving)
* [Pretrain](#pretrain)

### <a name="setup"></a>Setup

#### <a name="installdep"></a> 1. Clone the repository and install dependencies.
```bash
git clone https://github.com/intel-sandbox/llm-ray.git
cd llm-ray
pip install -r ./requirements.txt -f https://developer.intel.com/ipex-whl-stable-cpu -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. Launch ray cluster
Start ray head
```bash
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address 127.0.0.1 --ray-debugger-external --dashboard-host='0.0.0.0' --dashboard-port=8265
```
Start ray worker (optional)
```bash
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1  ray start --address='127.0.0.1:6379' --ray-debugger-external
```

If deploying a ray cluster on multiple nodes, please download the workflow repository on each node. More information about ray cluster, please refer to https://www.ray.io/

### <a name="finetune"></a>Finetuning

#### 1. Prepare Dataset

The workflow only supports datasets with JSONL (JSON Lines) format, where each line is a separate JSON object. Here’s the structure each line should follow:

``` json
{"instruction":"<User Input>", "context":"<Additional Information>", "response":"<Expected Output>"}
```

- Instruction: This is the user's input, such as a question, command, or prompt for content generation.
- Context: Supplementary information that aids the instruction. This can include previous conversation parts, background details, or specificities influencing the response. It's optional and can be left empty.
- Response: The model's expected output in response to the 'instruction', considering the 'context' if provided.

##### Examples:
``` json
{"instruction":"Which is a species of fish? Tope or Rope", "context":"", "response":"Tope"}
{"instruction":"What is the average lifespan of a Golden Retriever?","context":"Golden Retrievers are a generally healthy breed; they have an average lifespan of 12 to 13 years. Irresponsible breeding to meet high demand has led to the prevalence of inherited health problems in some breed lines, including allergic skin conditions, eye problems and sometimes snappiness. These problems are rarely encountered in dogs bred from responsible breeders.","response":"The average lifespan of a Golden Retriever is 12 to 13 years."}
```

An example dataset can be accessed at `examples/data/sample_finetune_data.jsonl`. Ensure each line in your dataset follows the above format.

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


### <a name="serving"></a>Serving

Llm-ray provides two ways to serve models, GUI or terminal.

#### GUI
![image](https://github.com/intel-sandbox/llm-ray/assets/97155466/c676e2f1-9e17-4bea-815d-8f7e21d68582)


This method will launch a UI interface and deploy an online inference service.
- (Optional) If customed models need to be added, please update `inference/config.py`.
```bash
# Start ray head
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head  --node-ip-address $node_ip --dashboard-host 0.0.0.0 --resources='{"queue_hardware": 5}'
# Start ray worker (optional)
RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --address='$node_ip:6379'
# Start ui (Please make sure passwordless SSH login is set up for $user on node $node_ip) 
python -u inference/start_ui.py --node_user_name $user --conda_env_name $conda_env --master_ip_port "$node_ip:6379"
# Get urls from the log
# Running on local URL:  http://0.0.0.0:8080
# Running on public URL: https://180cd5f7c31a1cfd3c.gradio.live
```
Then you can access url and deploy service in a few simple clicks.

#### Terminal
Ray serve is used to deploy models. First the model is exposed over HTTP by using Deployment, then test it over HTTP request.

A specific model can be deployed by specifying the model path and tokenizer path.

```bash
# If you dont' want to view serve logs, you can set env var, "KEEP_SERVE_TERMINAL" to false

# Run model serve with specified model and tokenizer
python inference/run_model_serve.py --model $model --tokenizer $tokenizer

# INFO - Deployment 'custom-model_PredictDeployment' is ready at `http://127.0.0.1:8000/custom-model`. component=serve deployment=custom-model_PredictDeployment
# Service is deployed successfully

# Verfiy the inference on deployed model
python inference/run_model_infer.py --model_endpoint http://127.0.0.1:8000/custom-model --streaming_response
```
Otherwise, all the models placed under `inference/models` folder will be deployed by default. If you want to choose a specific model to deploy, you can set env var, "MODEL_TO_SERVE", to your choice. You can also specify your model by either `--model` or `--config_file`.
For `--config_file`, you can copy one of them from `inference/models` and make necessary changes.

Llm-ray also supports serving with deepspeed. Please follow the [guide](inference/deepspeed/README.md) under inference/deepspeed folder.

### <a name="pretrain"></a>Pretrain

This workflow integrates the Megatron-DeepSpeed and Ray for pretrain. It's docker-based solution. You can follow the [guide](pretrain/README.md) under pretrain folder for details.


## Customize
### Adopt to your dataset
If your data do not meets the supported formats, you may need to preprocess the data into the standard format. Here we provide several examples include `example/finetune/open_assistant` and `example/finetune/dolly1`. After running `cd examples/finetune/open_assistant; python process_data.py`, a directory named `data` will output to the `examples/finetune/open_assistant` directory, just modify the configuration items `train_file` and `validation_file` to the corresponding file in `data` to start your own task. So does `example/finetune/dolly1` and other user-defined dataset.

## Disclaimer 
To the extent that any public datasets are referenced by Intel or accessed using tools or code on this site those datasets are provided by the third party indicated as the data source. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s), or using a model trained on those datasets, you agree to the terms associated with those datasets and that your use complies with the applicable license. 
 
Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data.  Intel is not liable for any liability or damages relating to your use of public datasets.
