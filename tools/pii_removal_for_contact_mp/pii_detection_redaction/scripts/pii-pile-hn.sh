#!/bin/bash
BATCHSIZE=1000
CPUCORES=48
DATA=pile_hn
OUTPUT_PREFIX=pile_hn
DATA_DIR=/home/user/local/PILE/hn

python ../src/pii_redaction_v2.py \
--load-batch-size $BATCHSIZE \
--cpu-per-worker $CPUCORES \
--dataset-family $DATA \
--output-prefix $OUTPUT_PREFIX \
--data-dir $DATA_DIR \
--local