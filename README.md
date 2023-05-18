# Finetune LLM models on Ray
This repo contains code for fine-tuning large language model on top of Ray cluster. 

## Getting Started 
### 0. Prerequisites
Please make sure that the following softwares are installed on all your cluster nodes
- Docker 
- NFS 
- Python3

### 1. Prepare Env
```bash
# on head node 
git clone https://github.com/intel-sandbox/llm-ray.git
cd llm-ray 
./build-image.sh 
docker save -o ray-image.tar ray-llm:latest
# on all worker nodes   
docker load -i ray-image.tar 
```

### 2. Define workflow yaml file 
modify the `workflow.yaml` based on your needs


### 3. Start the workflow
```python
python launch_workflow.sh -w workflow.yaml
```


