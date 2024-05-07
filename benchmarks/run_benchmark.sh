#! /bin/bash
set -eo pipefail

choice=${1}
run_mode=${2}   # "test" or "benchmark", where "test" will only use a small part of the dataset
VALUE_INF=2000
benchmark_script="benchmarks/benchmark_serving.py"
model_endpoint="http://localhost:8000/llama-2-7b-chat-hf"
model_name="llama-2-7b-chat-hf"
with_vllm_config_file="llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf.yaml"
wo_vllm_config_file="llm_on_ray/inference/models/llama-2-7b-chat-hf.yaml"
dataset_ShareGPT_path="./dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
dataset_IPEX_path="./dataset/prompt.json"
dataset_benchmark_num=1000
dataset_compare_num=128
numa_server_command=""
numa_client_command="numactl -N 1 -m 1"
num_replica=4

get_peak_throughpt(){
    echo "get performance results of llm-on-ray with vllm based on different bs"
    echo "results will be saved in $log_path"
    bs=${1}
    echo "batch_size: $bs"
    num_prompts=${2}
    log_path=${3}
    if [ -f $log_path ]; then
        rm $log_path
    fi
    for vllm_bs in ${bs}
    do
        echo "RUN llm-on-ray with vllm"
        echo "RUN bs ${vllm_bs}"
        echo "bs: ${vllm_bs}" >> $log_path
        # server:
        $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $vllm_bs
        # client:
        $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_ShareGPT_path --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple --results-dir "benchmarks/logs/1/"
    done
    echo "choice 1 generation completed"
}

metric_bs(){
    echo "get performance results of llm-on-ray with vllm and llm-on-ray based on different bs"
    echo "results will be saved in $log_path_vllm and $log_path_llmonray"
    bs=${1}
    num_prompts=${2}
    log_path_vllm=${3}
    log_path_llmonray=${4}
    if [ -f $log_path_vllm ]; then
        rm $log_path_vllm
    fi
    if [ -f $log_path_llmonray ]; then
        rm $log_path_llmonray
    fi
    for vllm_bs in ${bs}
    do
        echo "RUN llm-on-ray with vllm"
        echo "RUN bs ${vllm_bs}"
        echo "bs: ${vllm_bs}" >> $log_path_vllm
        # server:
        $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $vllm_bs
        # client:
        $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_ShareGPT_path --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple >> $log_path_vllm
    done
    for llmonray_bs in ${bs}
    do
        echo "RUN llm-on-ray"
        echo "RUN bs ${llmonray_bs}"
        echo "bs: ${llmonray_bs}" >> $log_path_llmonray
        # server:
        $numa_server_command llm_on_ray-serve --config_file $wo_vllm_config_file --simple --max_concurrent_queries $llmonray_bs
        # client:
        $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_ShareGPT_path --num-prompts $num_prompts --dataset-format ShareGPT --simple >> $log_path_llmonray
    done
    echo "choice 2 generation completed"
}

latency_throughput(){
    echo "get performance results of llm-on-ray with vllm when responding different sizes of requests"
    echo "results will be saved in $log_path"
    num_iter=${1}
    query_num=${2}
    input_tokens_length=${3}
    output_tokens_length=${4}
    log_path=${5}
    if [ -f $log_path ]; then
        rm $log_path
    fi

    # server
    $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $VALUE_INF

    # client
    for i in $(seq 1 $num_iter)
    do
        echo "Run iter $i"
        echo "iter: $i" >> $log_path
        for num_prompts in ${query_num}
        do
            echo "Run num_prompts ${num_prompts}"
            echo "num_prompts: ${num_prompts}" >> $log_path
            $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_IPEX_path --num-prompts $num_prompts  --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --vllm-engine --simple >> $log_path
        done
    done
    echo "choice 3 generation completed"
}

get_best_latency(){
    echo "get performance results of llm-on-ray with vllm when responding to input tokens of different lengths"
    echo "results will be saved in $log_path"
    num_iter=${1}
    input_tokens_length_li=${2}
    output_tokens_length=${3}
    log_path=${4}
    if [ -f $log_path ]; then
        rm $log_path
    fi

    # server
    $numa_server_command llm_on_ray-serve --config_file $with_vllm_config_file --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $VALUE_INF

    # client
    for i in $(seq 1 $num_iter)
    do
        echo "Run iter $i"
        echo "iter: $i" >> $log_path
        for input_tokens_length in ${input_tokens_length_li}
        do
            echo "Run input_tokens_length ${input_tokens_length}"
            echo "input_tokens_length: ${input_tokens_length}" >> $log_path
            $numa_client_command python $benchmark_script --model-endpoint-base $model_endpoint --model-name $model_name --dataset $dataset_IPEX_path --num-prompts 1 --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --vllm-engine --simple >> $log_path
        done
    done
    echo "choice 4 generation completed"
}

if [[ "$choice" == *"1"* ]]
then
    # get the results of choice1(the peak output throughput of llm-on-ray with vllm)
    log_path="benchmarks/logs/1_result.txt"
    if [ "$run_mode" == "benchmark" ]
    then
        bs=(1 2 4 8 16 32 64 128 256 300 400 512)
        prompt_num=$dataset_benchmark_num
    elif [ "$run_mode" == "test" ]
    then
        bs=(1 2 4)
        prompt_num=8
    else
        echo "Invalid run_mode, expected value 'test' or 'benchmark', but got $run_mode."
        exit 1
    fi
    get_peak_throughpt "${bs[*]}" $prompt_num $log_path
fi
if [[ "$choice" == *"2"* ]]
then
    # get the results of choice2(compare output token throughput(average latency per token) between llm-on-ray with vllm and llm-on-ray)
    log_path_vllm="benchmarks/logs/2_result_vllm.txt"
    log_path_llmonray="benchmarks/logs/2_result_llmonray.txt"
    if [ "$run_mode" == "benchmark" ]
    then
        bs=(1 2 4 8 16 32 64)
        prompt_num=$dataset_compare_num
    elif [ "$run_mode" == "test" ]
    then
        bs=(1 2 4)
        prompt_num=1
    else
        echo "Invalid run_mode, expected value 'test' or 'benchmark', but got $run_mode."
        exit 1
    fi
    metric_bs "${bs[*]}" $prompt_num $log_path_vllm $log_path_llmonray
fi
if [[ "$choice" == *"3"* ]]
then
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
        log_path="benchmarks/logs/3_result_32_64.txt"
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $log_path
        # 1024/128
        input_tokens_length=1024
        output_tokens_length=128
        log_path="benchmarks/logs/3_result_1024_128.txt"
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $log_path
    elif [ "$run_mode" == "test" ]
    then
        iter=2
        concurrent_query_num=(1 2 4)
        input_tokens_length=32
        output_tokens_length=20
        latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $log_path
    else
        echo "Invalid run_mode, expected value 'test' or 'benchmark', but got $run_mode."
        exit 1
    fi
fi
if [[ "$choice" == *"4"* ]]
then
    # get the results of choice4(get the latency of llm-on-Ray with vllm)
    log_path="benchmarks/logs/4_result.txt"
    if [ "$run_mode" == "benchmark" ]
    then
        iter=10
        input_tokens_length=(32 128 1024 2016)
    elif [ "$run_mode" == "test" ]
    then
        iter=2
        input_tokens_length=(32 128)
    else
        echo "Invalid run_mode, expected value 'test' or 'benchmark', but got $run_mode."
        exit 1
    fi
    output_tokens_length=32
    get_best_latency $iter "${input_tokens_length[*]}" $output_tokens_length $log_path
fi
