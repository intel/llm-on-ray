# How to run this workload inside Docker containers?

## Step 1: Build Docker Images  
```bash
# on head node 
./build-image.sh 
# save docker image
docker save -o ray-image.tar llm-ray:latest
# copy over to worker nodes, this is an optional step if all your cluster nodes are NFS-shared
scp ray-image.tar <worker_node_ip>:<destination_path_on_worker_node>
# on worker nodes   
docker load -i ray-image.tar 
```

## Step 2: Specify Workload Config File
Before launch the workload, you need to configure the `workflow.yaml`. The parameters in the config file are self-explained and you need to set your own values, e.g. the cluster node name, user name and the password. Please note that setting `run_training_job` to `True` will start the training job automatically when the containers are up. Therefore, if you only need the dev environment of the workload, remember to set the `run_training_job` to `False`. 


## Step 3: Launch Workload 
```bash 
python launch_workload.py -w workload.yaml
```