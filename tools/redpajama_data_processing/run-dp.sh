echo -e "\n distributed save with data source"
python preprocess_data.py \
        --input togethercomputer/RedPajama-Data-1T-Sample \
        --load-batch-size 100000 \
        --max-length 2048 \
        --output-prefix processed_megatron \
        --output-format megatron \
        --num-samples 1024 \
        --parallelism 180 \
        --save-on-source
