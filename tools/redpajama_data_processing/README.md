# How to preprocess RedPajama Dataset on top of Ray?

## Step 1: Set up Env
Please follow [this guide](../workload_in_containers/README.md) on how to set-up the container environment of this workload. When the containers are running, you can enter the container on head node using following command:
```bash  
docker exec -it ray-leader bash 
```

## Step 2: Run Preprocessing job 
```python 
cd tools/redpajama_data_processing
./run-dp.sh 
```
The `run-dp.sh` will run the `preprocess_data.py` under the hood. By default, it will preprocess the RedPajama Sample Data named `togethercomputer/RedPajama-Data-1T-Sample` from HuggingFace and save them under the temporary path of the container, namely the `/home/user/tmp`, with the folder name `processed_megatron`. But you can modify this script based on your need. The `preprocess_data.py` accept various flags. Run the following command to find out the possible params:
```bash
python preprocess_data.py -h
```
If the data preprocessing gets finished, you will see the total execution time of this script in the command-line output. 

## Step 3: Merge Multiple Megatron Data Files [Optional]
For Megatron-format data, you may need to do an extra step to merge multiple data files in to one. You can use the `merge_datasets.py` as follows:

```python
python merged_datasets.py --input <data_path_of_the_megatron_files> --output-prefix <data_path_without_file_extensions>
``` 

# Notes of running on full RedPajama dataset 
Due to the large size and long processing time for the full RedPajama dataset, we provided an additional script optimized for full RedPajama dataset. Checkout the args to see the possible preprocessing options. 
```python
python preprocess_full.py -h
```
We recommend users to first download the RedPajama dataset to local disk and preprocess the data per data source. The following command will download the full dataset to your local disk:
```bash
wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
while read line; do
    dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
    mkdir -p $(dirname $dload_loc)
    wget "$line" -O "$dload_loc"
done < urls.txt
```
Here is an example for starting full dataset preprocessing on the full dataset of source book:
```python
python preprocess_full.py \
         --input togethercomputer/RedPajama-Data-1T \
         --data-dir /home/user/local \
         --cache-dir /home/user/local \
         --source book \
         --output-prefix full_megatron \
         --cpu-per-worker 90
```

## NLTK
If you are using the `split-sentences` flag, please make sure to run the following lines of code on each worker first before getting started with the data preprocessing for better performance. 
```python
import nltk
nltk.download('punkt')
nltk.data.load("tokenizers/punkt/english.pickle")
```
## Common_Crawl
Please note that the current `togethercomputer/RedPajama-Data-1T` data loading script is not compatible with HuggingFace streaming mode implementation for common_crawl datset. For common_crawl, you have to first download the dataset to local disk and  preprocess the data in local mode.

## Disk Space
For the full dataset mode, we need to have both 



# Troubleshooting
## Connection Error
When running data preprocessing on the full redpajama dataset using HuggingFace streaming API, you may encounter following errors:
```bash 
aiohttp.client_exceptions.ClientPayloadError: Response payload is not completed
```
or 
```bash
MemoryError
[2m[36m(MapBatches(preprocess_megatron) pid=299104, ip=10.165.9.23)[0m [2023-07-01 11:40:34,924 E 299104 300227] gcs_rpc_client.h:542: Failed to connect to GCS within 60 seconds. GCS may have been killed. It's either GCS is terminated by `ray stop` or is killed unexpectedly. If it is killed unexpectedly, see the log file gcs_server.out. https://docs.ray.io/en/master/ray-observability/ray-logging.html#logging-directory-structure. The program will terminate.
```
or 
```bash
aiohttp failed response.json() with status 500
```
or 
```
aiohttp.client_exceptions.ClientResponseError: 520, message=''
``
All these errors are related to the network issue. The solution is to set a lower batch-size, e.g. 10000 or even lower.





