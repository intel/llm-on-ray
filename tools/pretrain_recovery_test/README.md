# How to run End2End Validation of the Recovery Test?

## Step 1: Set up Env
Please follow [this guide](../workload_in_containers/README.md) on how to set-up the container environment of this workload. When the containers are running, you can enter the container on head node using following command:
```bash  
docker exec -it ray-leader bash 
```

## Step 2: Start the script
You can use the `test_end2end.sh` to run the end-to-end validation for ray recovery mechanism.
```bash
cd tools/pretrain_recovery_test 
./test_end2end.sh
```