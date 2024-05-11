#! /bin/bash
set -eo pipefail

choice=${1}
run_mode=${2}   # "test" or "benchmark", where "test" will only use a small part of the dataset
VALUE_INF=2000
model_endpoint="http://localhost:8000/llama-2-7b-chat-hf"
model_name="llama-2-7b-chat-hf"
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
benchmark_script=$SHELL_FOLDER"/benchmark_serving.py"
with_vllm_config_file=$SHELL_FOLDER"/../llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml"
wo_vllm_config_file=$SHELL_FOLDER"/../llm_on_ray/inference/models/llama-2-7b-chat-hf.yaml"
dataset_ShareGPT_path=$SHELL_FOLDER"/../dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
dataset_IPEX_path=$SHELL_FOLDER"/../dataset/prompt.json"

if [ ! -f $dataset_ShareGPT_path ]
then
    echo "Dataset $dataset_ShareGPT_path not found, Please download ShareGPT dataset."
fi
if [ ! -f $dataset_IPEX_path ]
then
    echo "Dataset $dataset_IPEX_path not found, Please download IPEX dataset."
fi

dataset_benchmark_num=1000
dataset_compare_num=128
numa_server_command=""
numa_client_command="numactl -N 1 -m 1"
num_replica=4
if [ $run_mode = "test" ]
then
    save_dir=$SHELL_FOLDER"/results_test"
elif [ $run_mode = "benchmark" ]
then
    save_dir=$SHELL_FOLDER"/results"
else
    echo "Invalid run_mode, expected value 'test' or 'benchmark', but got '$run_mode'."
    exit 1

fi

get_peak_throughpt(){
    echo "get performance results of llm-on-ray with vllm based on different bs"
    echo "results will be saved in $log_path"
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
        $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $vllm_bs
        # client:
        $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_ShareGPT_path --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple --results-dir $bs_dir_vllm
    done
    echo "choice 1 generation completed"
}

metric_bs(){
    echo "get performance results of llm-on-ray with vllm and llm-on-ray based on different bs"
    echo "results will be saved in $log_path_vllm and $log_path_llmonray"
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
        $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $vllm_bs
        # client:
        $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_ShareGPT_path --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple --results-dir $bs_dir_vllm
    done
    for wo_vllm_bs in ${bs}
    do
        echo "RUN llm-on-ray"
        echo "RUN bs ${wo_vllm_bs}"
        bs_dir_wo_vllm=$choice_dir_wo_vllm"/bs_"$wo_vllm_bs
        # server:
        $numa_server_command llm_on_ray-serve --config_file $wo_vllm_config_file --simple --max_concurrent_queries $wo_vllm_bs
        # client:
        $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_ShareGPT_path --num-prompts $num_prompts --dataset-format ShareGPT --simple  --results-dir $bs_dir_wo_vllm
    done
    echo "choice 2 generation completed"
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
    $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $VALUE_INF

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
            $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_IPEX_path --num-prompts $num_prompts  --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --vllm-engine --simple --results-dir $results_dir
        done
    done
    echo "choice 3 generation completed"
}

get_best_latency(){
    echo "get performance results of llm-on-ray with vllm when responding to input tokens of different lengths"
    num_iter=${1}
    input_tokens_length_li=${2}
    output_tokens_length=${3}
    choice_dir=${4}

    # server
    $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $VALUE_INF

    # client
    for i in $(seq 1 $num_iter)
    do
        echo "Run iter $i"
        iter_dir=$choice_dir"/iter_"$i
        for input_tokens_length in ${input_tokens_length_li}
        do
            echo "Run input_tokens_length ${input_tokens_length}"
            token_dir=$iter_dir"/tokens_"$input_tokens_length"_"$output_tokens_length
            $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_IPEX_path --num-prompts 1 --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --vllm-engine --simple --results-dir $token_dir
        done
    done
    echo "choice 4 generation completed"
}

if [[ "$choice" == *"1"* ]]
then
    benchmark_dir=$save_dir"/choice_1"
    echo "results will be saved in $benchmark_dir"
    # get the results of choice1(the peak output throughput of llm-on-ray with vllm)
    if [ "$run_mode" == "benchmark" ]
    then
        bs=(1 2 4 8 16 32 64 128 256 300 400 512)
        prompt_num=$dataset_benchmark_num
    elif [ "$run_mode" == "test" ]
    then
        bs=(1 2 4)
        prompt_num=8
    fi
    get_peak_throughpt "${bs[*]}" $prompt_num $benchmark_dir
fi
if [[ "$choice" == *"2"* ]]
then
    benchmark_dir=$save_dir"/choice_2"
    echo "results will be saved in $benchmark_dir"
    benchmark_dir_vllm=$benchmark_dir"/vllm"
    benchmark_dir_wo_vllm=$benchmark_dir"/wo_vllm"
    # get the results of choice2(compare output token throughput(average latency per token) between llm-on-ray with vllm and llm-on-ray)
    if [ "$run_mode" == "benchmark" ]
    then
        bs=(1 2 4 8 16 32 64)
        prompt_num=$dataset_compare_num
    elif [ "$run_mode" == "test" ]
    then
        bs=(1 2 4)
        prompt_num=1
    fi
    metric_bs "${bs[*]}" $prompt_num $benchmark_dir_vllm $benchmark_dir_wo_vllm
fi
if [[ "$choice" == *"3"* ]]
then
    benchmark_dir=$save_dir"/choice_3"
    echo "results will be saved in $benchmark_dir"
    # get the results of choice3(latency vs throughput tradeoff for various number of requests)
    if [ "$run_mode" == "benchmark" ]
    then
        iter=10
        concurrent_query_num=(1 2 4 8 16 32 64)
        for i in "${!concurrent_query_num[@]}"; do
            concurrent_query_num[$i]=$[${concurrent_query_num[$i]}*$num_replica]
        done
        # 32/64
        input_tokens_length=32
        output_tokens_length=64
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $benchmark_dir
        # 1024/128
        input_tokens_length=1024
        output_tokens_length=128
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $benchmark_dir
    elif [ "$run_mode" == "test" ]
    then
        iter=2
        concurrent_query_num=(1 2 4)
        input_tokens_length=32
        output_tokens_length=20
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $benchmark_dir
    fi
fi
if [[ "$choice" == *"4"* ]]
then
    benchmark_dir=$save_dir"/choice_4"
    echo "results will be saved in $benchmark_dir"
    # get the results of choice4(get the latency of llm-on-Ray with vllm)
    if [ "$run_mode" == "benchmark" ]
    then
        iter=10
        input_tokens_length=(32 128 1024 2016)
    elif [ "$run_mode" == "test" ]
    then
        iter=2
        input_tokens_length=(32 128)
    fi
    output_tokens_length=32
    get_best_latency $iter "${input_tokens_length[*]}" $output_tokens_length $benchmark_dir
fi
