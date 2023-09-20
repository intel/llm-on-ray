#!/bin/bash
BATCHSIZE=50000
CPUCORES=48
INPUT=cerebras/SlimPajama-627B
DATA=slimpajama
OUTPUT_PREFIX=pii_slimpajama_se
DATA_DIR=/home/user/local/

python ../src/pii_redaction_v2.py \
--load-batch-size $BATCHSIZE \
--cpu-per-worker $CPUCORES \
--input $INPUT \
--dataset-family $DATA \
--output-prefix $OUTPUT_PREFIX \
--data-dir $DATA_DIR \
--local