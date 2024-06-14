#! /bin/bash
set -eo pipefail

CHOICE=${1}
RUN_MODE=${2}   # "test" or "benchmark", where "test" will only use a small part of the dataset
if [ -z "$CHOICE" ]
then
    echo "Please pass in the value of parameter CHOICE, which can be any subset of 1,2,3,4."
fi
if [ -z "$RUN_MODE" ]
then
    echo "Please pass in the value of parameter RUN_MODE, which can be 'test' or 'benchmark'."
fi
VALUE_INF=2000
MODEL_ENDPOINT="http://localhost:8000/llama-2-7b-chat-hf"
MODEL_NAME="llama-2-7b-chat-hf"
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
BENCHMARK_SCRIPT=$SHELL_FOLDER"/benchmark_serving.py"
WITH_VLLM_CONFIG_FILE=$SHELL_FOLDER"/../llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml"
WO_VLLM_CONFIG_FILE=$SHELL_FOLDER"/../llm_on_ray/inference/models/llama-2-7b-chat-hf.yaml"
DATASET_PATH=$SHELL_FOLDER"/../dataset"
DATASET_SHAREGPT_PATH=$SHELL_FOLDER"/../dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_IPEX_PATH=$SHELL_FOLDER"/../dataset/prompt.json"
DATASET_BENCHMARK_NUM=1000
DATASET_COMPARE_NUM=128
NUMA_SERVER_COMMAND=""
NUM_REPLICA=4
if [ ! -f $DATASET_SHAREGPT_PATH ]
then
    echo "Dataset $DATASET_SHAREGPT_PATH not found, download ShareGPT dataset first."
    wget -q -P $DATASET_PATH https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi
if [ ! -f $DATASET_IPEX_PATH ]
then
    echo "Dataset $DATASET_IPEX_PATH not found, download IPEX dataset first."
    wget -q -P $DATASET_PATH https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt.json
fi
if [ $RUN_MODE = "test" ]
then
    SAVE_DIR=$SHELL_FOLDER"/results_test"
    NUMA_CLIENT_COMMAND=""
elif [ $RUN_MODE = "benchmark" ]
then
    SAVE_DIR=$SHELL_FOLDER"/results"
    NUMA_CLIENT_COMMAND="numactl -N 1 -m 1"
else
    echo "Invalid RUN_MODE, expected value 'test' or 'benchmark', but got '$RUN_MODE'."
    exit 1

fi

get_peak_throughpt(){
    echo "get performance results of llm-on-ray with vllm based on different bs"
    bs=${1}
    echo "batch_size: $bs"
    num_prompts=${2}
    choice_dir=${3}
    for vllm_bs in ${bs}
    do
        bs_dir_vllm=$choice_dir"/bs_"$vllm_bs
        echo "RUN llm-on-ray with vllm"
        echo "RUN bs ${vllm_bs}"
        # server:
        $NUMA_SERVER_COMMAND llm_on_ray-serve --config_file $WITH_VLLM_CONFIG_FILE --simple --max_ongoing_requests $VALUE_INF --max_num_seqs $vllm_bs
        # client:
        $NUMA_CLIENT_COMMAND python $BENCHMARK_SCRIPT --model-endpoint-base $MODEL_ENDPOINT --model-name $MODEL_NAME --dataset $DATASET_SHAREGPT_PATH --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple --results-dir $bs_dir_vllm
    done
    echo "CHOICE 1 generation completed"
}

metric_bs(){
    echo "get performance results of llm-on-ray with vllm and llm-on-ray based on different bs"
    bs=${1}
    num_prompts=${2}
    choice_dir_vllm=${3}
    choice_dir_wo_vllm=${4}
    for vllm_bs in ${bs}
    do
        bs_dir_vllm=$choice_dir_vllm"/bs_"$vllm_bs
        echo "RUN llm-on-ray with vllm"
        echo "RUN bs ${vllm_bs}"
        # server:
        $NUMA_SERVER_COMMAND llm_on_ray-serve --config_file $WITH_VLLM_CONFIG_FILE --simple --max_ongoing_requests $VALUE_INF --max_num_seqs $vllm_bs
        # client:
        $NUMA_CLIENT_COMMAND python $BENCHMARK_SCRIPT --model-endpoint-base $MODEL_ENDPOINT --model-name $MODEL_NAME --dataset $DATASET_SHAREGPT_PATH --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple --results-dir $bs_dir_vllm
    done
    for wo_vllm_bs in ${bs}
    do
        echo "RUN llm-on-ray"
        echo "RUN bs ${wo_vllm_bs}"
        bs_dir_wo_vllm=$choice_dir_wo_vllm"/bs_"$wo_vllm_bs
        # server:
        $NUMA_SERVER_COMMAND llm_on_ray-serve --config_file $WO_VLLM_CONFIG_FILE --simple --max_ongoing_requests $wo_vllm_bs
        # client:
        $NUMA_CLIENT_COMMAND python $BENCHMARK_SCRIPT --model-endpoint-base $MODEL_ENDPOINT --model-name $MODEL_NAME --dataset $DATASET_SHAREGPT_PATH --num-prompts $num_prompts --dataset-format ShareGPT --simple  --results-dir $bs_dir_wo_vllm
    done
    echo "CHOICE 2 generation completed"
}

latency_throughput(){
    echo "get performance results of llm-on-ray with vllm when responding different sizes of requests"
    num_iter=${1}
    query_num=${2}
    input_tokens_length=${3}
    output_tokens_length=${4}
    choice_dir=${5}
    tokens_dir=$choice_dir"/tokens_"$input_tokens_length"_"$output_tokens_length

    # server
    $NUMA_SERVER_COMMAND llm_on_ray-serve --config_file $WITH_VLLM_CONFIG_FILE --simple --max_ongoing_requests $VALUE_INF --max_num_seqs $VALUE_INF

    # client
    for i in $(seq 1 $num_iter)
    do
        echo "Run iter $i"
        iter_dir=$tokens_dir"/iter_"$i
        for num_prompts in ${query_num}
        do
            results_dir=$iter_dir"/num_prompts_"$num_prompts
            echo "Run num_prompts ${num_prompts}"
            echo "results_dir: ${results_dir}"
            $NUMA_CLIENT_COMMAND python $BENCHMARK_SCRIPT --model-endpoint-base $MODEL_ENDPOINT --model-name $MODEL_NAME --dataset $DATASET_IPEX_PATH --num-prompts $num_prompts  --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --vllm-engine --simple --results-dir $results_dir
        done
    done
    echo "CHOICE 3 generation completed"
}

get_best_latency(){
    echo "get performance results of llm-on-ray with vllm when responding to input tokens of different lengths"
    num_iter=${1}
    input_tokens_length_li=${2}
    output_tokens_length=${3}
    choice_dir=${4}

    # server
    $NUMA_SERVER_COMMAND llm_on_ray-serve --config_file $WITH_VLLM_CONFIG_FILE --simple --max_ongoing_requests $VALUE_INF --max_num_seqs $VALUE_INF

    # client
    for i in $(seq 1 $num_iter)
    do
        echo "Run iter $i"
        iter_dir=$choice_dir"/iter_"$i
        for input_tokens_length in ${input_tokens_length_li}
        do
            echo "Run input_tokens_length ${input_tokens_length}"
            token_dir=$iter_dir"/tokens_"$input_tokens_length"_"$output_tokens_length
            $NUMA_CLIENT_COMMAND python $BENCHMARK_SCRIPT --model-endpoint-base $MODEL_ENDPOINT --model-name $MODEL_NAME --dataset $DATASET_IPEX_PATH --num-prompts 1 --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --vllm-engine --simple --results-dir $token_dir
        done
    done
    echo "CHOICE 4 generation completed"
}

if [[ "$CHOICE" == *"1"* ]]
then
    benchmark_dir=$SAVE_DIR"/choice_1"
    echo "results will be saved in $benchmark_dir"
    # get the results of choice1(the peak output throughput of llm-on-ray with vllm)
    if [ "$RUN_MODE" == "benchmark" ]
    then
        bs=(1 2 4 8 16 32 64 128 256 300 400 512)
        prompt_num=$DATASET_BENCHMARK_NUM
    elif [ "$RUN_MODE" == "test" ]
    then
        bs=(1 2 4)
        prompt_num=8
    fi
    get_peak_throughpt "${bs[*]}" $prompt_num $benchmark_dir
fi
if [[ "$CHOICE" == *"2"* ]]
then
    benchmark_dir=$SAVE_DIR"/choice_2"
    echo "results will be saved in $benchmark_dir"
    benchmark_dir_vllm=$benchmark_dir"/vllm"
    benchmark_dir_wo_vllm=$benchmark_dir"/wo_vllm"
    # get the results of choice2(compare output token throughput(average latency per token) between llm-on-ray with vllm and llm-on-ray)
    if [ "$RUN_MODE" == "benchmark" ]
    then
        bs=(1 2 4 8 16 32 64)
        prompt_num=$DATASET_COMPARE_NUM
    elif [ "$RUN_MODE" == "test" ]
    then
        bs=(1 2 4)
        prompt_num=1
    fi
    metric_bs "${bs[*]}" $prompt_num $benchmark_dir_vllm $benchmark_dir_wo_vllm
fi
if [[ "$CHOICE" == *"3"* ]]
then
    benchmark_dir=$SAVE_DIR"/choice_3"
    echo "results will be saved in $benchmark_dir"
    # get the results of choice3(latency vs throughput tradeoff for various number of requests)
    if [ "$RUN_MODE" == "benchmark" ]
    then
        iter=10
        concurrent_query_num=(1 2 4 8 16 32 64)
        for i in "${!concurrent_query_num[@]}"; do
            concurrent_query_num[$i]=$[${concurrent_query_num[$i]}*$NUM_REPLICA]
        done
        # 32/64
        input_tokens_length=32
        output_tokens_length=64
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $benchmark_dir
        # 1024/128
        input_tokens_length=1024
        output_tokens_length=128
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $benchmark_dir
    elif [ "$RUN_MODE" == "test" ]
    then
        iter=2
        concurrent_query_num=(1 2 4)
        input_tokens_length=32
        output_tokens_length=20
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $benchmark_dir
    fi
fi
if [[ "$CHOICE" == *"4"* ]]
then
    benchmark_dir=$SAVE_DIR"/choice_4"
    echo "results will be saved in $benchmark_dir"
    # get the results of choice4(get the latency of llm-on-Ray with vllm)
    if [ "$RUN_MODE" == "benchmark" ]
    then
        iter=10
        input_tokens_length=(32 128 1024 2016)
    elif [ "$RUN_MODE" == "test" ]
    then
        iter=2
        input_tokens_length=(32 128)
    fi
    output_tokens_length=32
    get_best_latency $iter "${input_tokens_length[*]}" $output_tokens_length $benchmark_dir
fi