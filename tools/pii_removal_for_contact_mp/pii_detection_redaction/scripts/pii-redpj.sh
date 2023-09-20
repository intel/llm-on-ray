#!/bin/bash
BATCHSIZE=50000
CPUCORES=48
INPUT=togethercomputer/RedPajama-Data-1T-Sample
DATA=slimpajama
OUTPUT_PREFIX=redpajama
DATA_DIR=/home/user/local/dataset/RedPajama-Data-1T-Sample/

python ../src/pii_redaction.py \
--load-batch-size $BATCHSIZE \
--cpu-per-worker $CPUCORES \
--input $INPUT \
--dataset-family $DATA \
--output-prefix $OUTPUT_PREFIX \
--data-dir $DATA_DIR \
--local \
#--skip 500000