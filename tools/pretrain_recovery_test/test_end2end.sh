
echo -e "\npreprocessing RedPajama-Data-1T-Sample..."
python ../redpajama_data_preprocessing/preprocess_data.py \
        --input togethercomputer/RedPajama-Data-1T-Sample \
        --load-batch-size 100000 \
        --max-length 2048 \
        --output-prefix processed_json \
        --output-format json \
        --num-samples 1024 \
        --parallelism 180 

echo -e "\ncreating sample training data for recovery validation..."
python ../redpajama_data_preprocessing/group_files.py \
        --src-data-path /home/user/tmp/processed_json \
        --des-data-path /home/user/tmp/pretrain_data \
        --test 

echo -e "\nstart pre-training in the background..."
python ../../Finetune/main.py --config_path ../../Finetune/llm_pretrain_template.conf &> training_log.txt &

echo -e "\nlet the training run for 8 mins..."
sleep 400

echo -e "\nmanually stop the training..."
pkill -f main.py

echo -e "\nrestart the training..."
python ../../Finetune/main.py --config_path ../../Finetune/llm_pretrain_template.conf &> training_log2.txt &

echo -e "\nlet the training run for 8 mins..."
sleep 400

echo -e "\nmanually stop the training..."
pkill -f main.py

echo -e "\ncompare the results..."
python compare_logs.py --file_path /home/user/tmp/state















