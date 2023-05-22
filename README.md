## Finetune LLM models on Ray
This `Finetune` folder contains code for fine-tuning large language model on top of Ray cluster. 

### Getting Started 
#### 0. Prerequisites
Please make sure that the following softwares are installed on all your cluster nodes
- Docker 
- NFS 
- Python3

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


