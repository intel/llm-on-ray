## Pretrain LLMs on Intel Gaudi
### Prepare Environment
#### 1. Download the Workflow Repository
Create a working directory for the workflow and clone the repository into your working directory.
```bash
mkdir ~/workspace && cd ~/workspace
git clone https://github.com/intel/llm-on-ray.git
cd llm-on-ray
git checkout main
```
#### 2. build Docker images for pretrain
```bash
cd llm-on-ray/pretrain/docker
```
Build the habana docker image for Megatron-DeepSpeed.
```bash
./build-image.sh megatron-habana
```
Build the habana docker image for Huggingface trainer
```bash
./build-image.sh optimum-habana
```
Build the Nvidia docker image for both Megatron-DeepSpeed and Huggingface trainer
```bash
./build-image.sh nvidia
```

#### 3. Run the docker containers on head node and worker nodes for pretrain.
make the logs directory for saving the ray logs.
```bash
mkdir ~/workspace/logs
```
Gaudi2:

##### Megatron-DeepSpeed
```bash
docker run -it --name megatron-habana --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v ~/workspace:/home/user/workspace -v ~/workspace/logs:/tmp --cap-add=sys_nice --net=host --ipc=host llm-ray:megatron-habana
```

##### Huggingface trainer
```bash
docker run -it --name megatron-habana --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -v ~/workspace:/home/user/workspace -v ~/workspace/logs:/tmp --cap-add=sys_nice --net=host --ipc=host llm-ray:optimum-habana
```
Nvidia GPU:
```bash
docker run --gpus all -it --ulimit memlock=-1 --ulimit stack=67108864 --network host --name megatron-nvidia --shm-size=64g -v ~/workspace/logs:/tmp -v ~/workspace:/home/user/workspace llm-ray:nvidia  /bin/bash
```

#### 4. Launch ray cluster
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
This workflow integrates two different pretrain solutions.
#### Megatron-DeepSpeed
For GPU version, we use the [Microsoft Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed). For Gaudi2 version, we use the [HabanaAI Megatron-DeepSpeed](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed)

#### Huggingface Trainer
It integrates the megatron dataloader for pretrain. For habana support, it uses the [optimum-habana](https://github.com/huggingface/optimum-habana). It can use deepspeed ZeRO stage3 to train medium and large language models

#### 1. Generate megatron datasets
Please refer to [this tutorial](https://github.com/intel/e2eAIOK/tree/main/RecDP/pyrecdp/primitives/llmutils/tokenize_and_save). Copy the datasets bin and idx files into ~/workspace/data

#### 2. Tokenizer
##### Download the vocab file and merge table.
If using the tokenizer files for Megatron_DeepSpeed pretrain, Download the GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) into ~/workspace/data.
```bash
cd ~/workspace/data/
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt 
```
Modify the vocab_file and merge_file of megatron_config in config files
```bash
#llama_7b_megatron_deepspeed_zs0_8Gaudi_pretrain.conf
"megatron_config": {
        "vocab_file": "megatron-data/gpt2-vocab.json",
        "merge_file": "megatron-data/gpt2-merges.txt",
}
```
##### Huggingface Tokenizer
For Huggingface trainer, the Huggingface tokenizer is preferred.
Modify the tokenizer_type and tokenizer_model of megatron_config for megatron dataset.
```bash
#llama_7b_8Guadi_pretrain.conf
"megatron_config": {
        "tokenizer_type": "HFTokenizer",
        "tokenizer_model": "huggyllama/llama-7b",
}
```
Modify the tokenizer parameters of trainer. The tokenizer of trainer and megatron dataset should be consistent
```bash
#llama_7b_8Guadi_pretrain.conf
"tokenizer": {
        # The type of dataset, now only HuggingfaceTokenizer is supported.
        "type": "HuggingFaceTokenizer",
        # The name/path of tokenizer in huggingface.
        "name": "huggyllama/llama-7b",
        # Config of tokenizer, all items will be transfered to transformers.AutoTokenizer.from_pretrained().
        "config": {}
}
```

#### 3. Pretrain Command

Please ensure that you check and modify the configuration files located in ~/workspace/llm-on-ray/pretrain/config/ before proceeding.

After your environment configuration are properly set up, you can use the following instructions to pretrain the language model:

##### Gaudi2:
###### Megatron-DeepSpeed

Set up `megatron_deepspeed_path` in the configuration.

```bash
cd /home/user/workspace/llm-on-ray
#Bloom-7B
python -m llm_on_ray.pretrain.megatron_deepspeed_pretrain --config_file llm_on_ray/pretrain/config/bloom_7b_megatron_deepspeed_zs0_8Gaudi_pretrain.conf
#llama-7B
python -m llm_on_ray.pretrain.megatron_deepspeed_pretrain --config_file llm_on_ray/pretrain/config/llama_7b_megatron_deepspeed_zs0_8Gaudi_pretrain.conf
```

##### Huggingface Trainer
```bash
cd /home/user/workspace/llm-on-ray
#llama-7B
python -m llm_on_ray.pretrain.pretrain --config_file llm_on_ray/pretrain/config/llama_7b_8Guadi_pretrain.conf
```
##### Nvidia GPU:
###### Megatron-DeepSpeed
```bash
cd /home/user/workspace/llm-on-ray
#llama2-7B
python -m llm_on_ray.pretrain.megatron_deepspeed_pretrain --config_file llm_on_ray/pretrain/config/llama2_3b_megatron_deepspeed_zs0_8gpus_pretrain.conf
```
##### Huggingface Trainer
```bash
cd /home/user/workspace/llm-on-ray
#llama-7B
python -m llm_on_ray.pretrain.pretrain --config_file llm_on_ray/pretrain/config/llama_7b_8gpu_pretrain.conf
```