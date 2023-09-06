# How to run PII Detection and Redaction pipeline on Code Datasets on top of Ray

This pii_code pipeline is doing PII detection and redaction for code. Asumes data is available as a folder of jsonl files.
These can be downloaded by the user from huggingface or be the result of a previous data preprocessing pipeline
(filtering, dedup,..). The processed data is written to an output folder in json format as well

## Step 1: Set up Env
Please follow [this guide](../workload_in_containers/README.md) on how to set-up the container environment of this workload. When the containers are running, you can enter the container on head node using following command:
```bash  
docker exec -it ray-leader bash 
```
## Example command
```python
python ./tools/pii_code/preprocess_pii_code.py --data_dir "/home/user/local/github/" --load-batch-size 100000 --output_path '/home/user/local/output_PIIcode' --cpu-per-worker 20
```
## Observations
on opa-clx-4024 node, BS=100000, pipeline produces 1 file pr batch ~1.5GB each (will vary based on the text length of the rows) and takes <3 minutes per large batch.
