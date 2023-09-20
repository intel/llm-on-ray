#!/bin/bash
BATCHSIZE=1000
CPUCORES=48
DATA=refinedweb
OUTPUT_PREFIX=pii_test_output
DATA_DIR=/home/user/local/refinedweb_samples

python ../src/pii_redaction_v2.py \
--load-batch-size $BATCHSIZE \
--cpu-per-worker $CPUCORES \
--dataset-family $DATA \
--output-prefix $OUTPUT_PREFIX \
--data-dir $DATA_DIR \
--local