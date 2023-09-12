# How to preprocess RedPajama Dataset on top of Ray?

## Step 1: Set up Env
Please follow [this guide](../workload_in_containers/README.md) on how to set-up the container environment of this workload. When the containers are running, you can enter the container on head node using following command:
```bash  
docker exec -it ray-leader bash 
```

## Step 2: Run Preprocessing job 
```python 
# cd to this folder
cd tools/tokenize_and_save
# run the preprocessing job, below is an example
python tokenize_and_save.py \
        --input-dir /home/user/shared/PILE_dedup/EuroParl \
        --data-field text \
        --tokenizer togethercomputer/LLaMA-2-7B-32K \
        --output-dir /home/user/shared/processed \
        --load-batch-size 1000 \
        --cpu-per-node 90
```
If the data preprocessing gets finished, you will see the total execution time of this script in the command-line output. 

To find out the possible params, run the following command:
```bash
python tokenize_and_save.py -h
```

## Step 3: Merge Multiple Megatron Data Files [Optional]
For Megatron-format data, you may need to do an extra step to merge multiple data files in to one. You can use the `merge_datasets.py` as follows:

```python
python merge_datasets.py --input <directory_containing_megatron_files> --output-prefix <output_file_name_without_file_extensions>
``` 

## Validation
When the data preprocessing gets finished, you will see the total execution time at the end of the command line output. Now, it is your responsibility to gather all data partition files on each worker to the head node. When all the data partition files are under one folder on the head node, you can run the `merge_datasets.py` script to merge multiple megatron `bin` and `idx` files into one `bin` and `idx` files on each worker node. To count the token numbers in the dataset, you can use the `count_tokens.py` script, e.g.
```python
python count_tokens.py <megatron_file_without_file_extension> <output_file_containing_the_token_number_per_row> <tokenizer_name>
```






