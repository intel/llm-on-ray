echo -e "\n distributed tokenization with ray"
python tokenize_and_save.py \
        --input-dir /home/user/shared/PILE_dedup/EuroParl \
        --data-field text \
        --tokenizer togethercomputer/LLaMA-2-7B-32K \
        --output-dir /home/user/shared/EuroParl_tokenized \
        --load-batch-size 1000 \
        --cpu-per-node 90

sleep 30
echo -e "\n merging multiple megatron data files.."
python merge_datasets.py --input /home/user/shared/EuroParl_tokenized --output-prefix /home/user/shared/EuroParl_tokenized

sleep 15
echo -e "\n removing multiple megatron files.."
rm -fr /home/user/shared/EuroParl_tokenized

sleep 5
echo -e "\n counting token numbers.."
python count_tokens.py /home/user/shared/EuroParl_tokenized /home/user/shared/EuroParl_tokenized.stat



