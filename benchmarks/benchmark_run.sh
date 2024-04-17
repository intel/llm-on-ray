#! /bin/bash
set -e

OMP_NUM_THREADS=24 # need to be modified based on cpus_per_worker
VALUE_INF=2000
choice=${1}

get_peak_throughpt(){
    bs=${1}
    num_prompts=${2}
    log_path=${3}
    if [ -f $log_path ]; then
        rm $log_path
    fi
    for vllm_bs in ${bs[*]};
    do
        echo "RUN VLLM"
        echo "RUN bs ${vllm_bs}"
        echo "bs: ${vllm_bs}" >> $log_path
        # server:
        OMP_NUM_THREADS=$OMP_NUM_THREADS numactl -N 0 -m 0 -C 0-$(($OMP_NUM_THREADS-1)) llm_on_ray-serve --config_file llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $vllm_bs
        # client:
        numactl -N 1 -m 1 python benchmarks/benchmark_serving.py --model-endpoint-base http://localhost:8000/llama-2-7b-chat-hf --model-name llama-2-7b-chat-hf --dataset ./dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple >> $log_path
    done
}

metric_bs(){
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
    for vllm_bs in ${bs[*]};
    do
        echo "RUN VLLM"
        echo "RUN bs ${vllm_bs}"
        echo "bs: ${vllm_bs}" >> $log_path_vllm
        # server:
        OMP_NUM_THREADS=$OMP_NUM_THREADS numactl -N 0 -m 0 -C 0-$(($OMP_NUM_THREADS-1)) llm_on_ray-serve --config_file llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $vllm_bs
        # client:
        numactl -N 1 -m 1 python benchmarks/benchmark_serving.py --model-endpoint-base http://localhost:8000/llama-2-7b-chat-hf --model-name llama-2-7b-chat-hf --dataset ./dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts $num_prompts --dataset-format ShareGPT --vllm-engine --simple >> $log_path_vllm
    done
    for llmonray_bs in ${bs}
    do
        echo "RUN LLMonRay"
        echo "RUN bs ${llmonray_bs}"
        echo "bs: ${llmonray_bs}" >> $log_path_llmonray
        # server:
        OMP_NUM_THREADS=$OMP_NUM_THREADS numactl -N 0 -m 0 -C 0-$(($OMP_NUM_THREADS-1)) llm_on_ray-serve --config_file llm_on_ray/inference/models/llama-2-7b-chat-hf.yaml --simple --max_concurrent_queries $llmonray_bs
        # client:
        numactl -N 1 -m 1 python benchmarks/benchmark_serving.py --model-endpoint-base http://localhost:8000/llama-2-7b-chat-hf --model-name llama-2-7b-chat-hf --dataset ./dataset/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts $num_prompts --dataset-format ShareGPT --simple >> $log_path_llmonray
    done
}

latency_throughput(){
    num_iter=${1}
    concurrent_query_num=${2}
    input_tokens_length=${3}
    output_tokens_length=${4}
    log_path=${5}
    if [ -f $log_path ]; then
        rm $log_path
    fi

    # # server
    OMP_NUM_THREADS=$OMP_NUM_THREADS numactl -N 0 -m 0 -C 0-$(($OMP_NUM_THREADS-1)) llm_on_ray-serve --config_file llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $VALUE_INF

    # client
    for i in $(seq 1 $num_iter)
    do
        echo "Run iter $i"
        echo "iter: $i" >> $log_path
        for num_prompts in ${concurrent_query_num}
        do
            echo "Run num_prompts ${num_prompts}"
            echo "num_prompts: ${num_prompts}" >> $log_path
            numactl -N 1 -m 1 python benchmarks/benchmark_serving.py --model-endpoint-base http://localhost:8000/llama-2-7b-chat-hf --model-name llama-2-7b-chat-hf --dataset ./dataset/prompt.json --num-prompts $num_prompts  --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --simple >> $log_path
        done
    done
}

get_best_latency(){
    num_iter=${1}
    input_tokens_length_li=${2}
    output_tokens_length=${3}
    log_path=${4}
    if [ -f $log_path ]; then
        rm $log_path
    fi

    # server
    OMP_NUM_THREADS=$OMP_NUM_THREADS numactl -N 0 -m 0 -C 0-$(($OMP_NUM_THREADS-1)) llm_on_ray-serve --config_file llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --simple --max_concurrent_queries $VALUE_INF --vllm_max_num_seqs $VALUE_INF

    # client
    for i in $(seq 1 $num_iter)
    do
        echo "Run iter $i"
        echo "iter: $i" >> $log_path
        for input_tokens_length in ${input_tokens_length_li}
        do
            echo "Run input_tokens_length ${input_tokens_length}"
            echo "input_tokens_length: ${input_tokens_length}" >> $log_path
            numactl -N 1 -m 1 python benchmarks/benchmark_serving.py --model-endpoint-base http://localhost:8000/llama-2-7b-chat-hf --model-name llama-2-7b-chat-hf --dataset ./dataset/prompt.json --num-prompts 1 --dataset-format IPEX --input-tokens $input_tokens_length --max-new-tokens $output_tokens_length --track-token-latency --simple >> $log_path
        done
    done
}

if [ $choice -eq 1 ]
then
    # figure_1: vllm peak throughput
    bs=(1 2 4 8 16 32 64 128 256 300 400 512)
    log_path="benchmarks/logs/1_result.txt"
    get_peak_throughpt "${bs[*]}" 1000 $log_path
elif [ $choice -eq 2 ]
then
    # figure_2: output token throughput_bs(average latency per token_vs) between vllm & llmonray
    # bs=(1 2 4 8 16 32 64)
    # log_path_vllm="benchmarks/logs/2_result_vllm.txt"
    # log_path_llmonray="benchmarks/logs/2_result_llmonray.txt"
    # metric_bs "${bs[*]}" 128 $log_path_vllm $log_path_llmonray

    # test
    bs=(1 2 4)
    log_path_vllm="benchmarks/logs/2_result_vllm.txt"
    log_path_llmonray="benchmarks/logs/2_result_llmonray.txt"
    metric_bs "${bs[*]}" 1 $log_path_vllm $log_path_llmonray
elif [ $choice -eq 3 ]
then
    # figure_3: average_latency_for_next_token vs output tokens throughput
    # iter=10
    # concurrent_query_num=(1 2 4 8 16 32 64)
    # log_path="benchmarks/logs/3_result.txt"
    # # 32/64
    # input_tokens_length=32
    # output_tokens_length=64
    # latency_throughput iter "${concurrent_query_num[*]}" input_tokens_length output_tokens_length
    # # 1024/128
    # input_tokens_length=1024
    # output_tokens_length=128
    # latency_throughput iter "${concurrent_query_num[*]}" input_tokens_length output_tokens_length

    # test
    iter=2
    concurrent_query_num=(1 2 4)
    input_tokens_length=32
    output_tokens_length=20
    log_path="benchmarks/logs/3_result.txt"
    latency_throughput $iter "${concurrent_query_num[*]}" $input_tokens_length $output_tokens_length $log_path
elif [ $choice -eq 4 ]
then
    # get llm on Ray with vllm latency
    iter=10
    input_tokens_length=(32 128 1024 2016)
    output_tokens_length=32
    log_path="benchmarks/logs/4_result.txt"

    # test
    iter=2
    input_tokens_length=(32 128)
    get_best_latency $iter "${input_tokens_length[*]}" $output_tokens_length $log_path
else
    echo "Invalid choice"
fi
