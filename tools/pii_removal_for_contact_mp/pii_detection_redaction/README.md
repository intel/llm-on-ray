# How to run PII-for-text pipeline

## Overview
The pipeline detects 5 different types of PIIs: 'PHONE_NUMBER, 'IP_ADDRESS', 'EMAIL', 'USER', 'KEY'. The detection is based on regular expressions using open-source packages including presidio and bigscience-pii. The detection precision/recall has been tuned for web scrapes based datasets, for example, Falcon-RefinedWeb, SlimPajama-StackExchange, PILE-Hackernews. But the detection precion/recall is not good for code data like Github. </p>

Two redaction methods have been implemented:
1. Replacement with random values
2. Replacement with tags such as [PHONE_NUMBER], [EMAIL], etc. </br>
Currently, the option 1) is used.


## How to run
### Step 1: Set up Env
Please follow [this guide](../workload_in_containers/README.md) on how to set-up the container environment of this workload. When the containers are running, you can enter the container on head node using following command:
```bash  
docker exec -it ray-leader bash 
```

### Step 2: Run PII removal
Once you are inside the ray-leader container, go to the scripts folder. You can change the `BATCH_SIZE` and `CPUCORES` depending on the memory and number of cores on your systems. Then you can run the pii script, for example:
```
bash pii-refinedweb.sh
```

### Step 3: Validate outputs
We implemented 3 checks:
1. Check schema and sample rows in output parquets by loading parquet with pandas
2. Count numbers of PIIs per category by sampling from the outputs. You can further get an estimate of the total number of PIIs per category by multiplying total_num_samples/sample_used_for_this_check
3. Visual check of a small sample by producing a html with yellow highlights of the PIIs and annotating with corresponding category (note that sometimes the highlights are not at the exact location, but should be quite close).

```
# First change the path to the data files in the python script
python src/validate_ray_outputs.py
```