## Pretrain
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

#### 3. Run the docker containers on head node and worker nodes for pretrain.
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
python pretrain/megatron_deepspeed_pretrain.py --config_path pretrain/config/bloom_7b_megatron_deepspeed_zs0_8Gaudi_pretrain.conf
#llama-7B
python pretrain/megatron_deepspeed_pretrain.py --config_path pretrain/config/llama_7b_megatron_deepspeed_zs0_8Gaudi_pretrain.conf
```
##### Nvidia GPU:
```bash
cd /home/user/workspace/llm-ray
#llama2-7B
python pretrain/megatron_deepspeed_pretrain.py --config_path pretrain/config/llama2_3b_megatron_deepspeed_zs0_8gpus_pretrain.conf
```