NS_MODEL_PERF_STEPS=50 OMP_PROC_BIND=true NS_NUM_THREADS=24 OMP_NUM_THREADS=1  numactl -N 0 -m 0 -C 0-26 nohup python api_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization ns --device cpu --max-num-seqs 64 --block-size 4096 --max-model-len 4096 --host 10.0.11.5 --port 8070 > s8070.log 2>&1 &
sleep 2

NS_MODEL_PERF_STEPS=50 OMP_PROC_BIND=true NS_NUM_THREADS=24 OMP_NUM_THREADS=1  numactl -N 0 -m 0 -C 28-54 nohup python api_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization ns --device cpu --max-num-seqs 64 --block-size 4096 --max-model-len 4096 --host 10.0.11.5 --port 8071 > s8071.log 2>&1 &
sleep 2

NS_MODEL_PERF_STEPS=50 OMP_PROC_BIND=true NS_NUM_THREADS=24 OMP_NUM_THREADS=1  numactl -N 1 -m 1 -C 56-82 nohup python api_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization ns --device cpu --max-num-seqs 64 --block-size 4096 --max-model-len 4096 --host 10.0.11.5 --port 8072 > s8072.log 2>&1 &
sleep 2

NS_MODEL_PERF_STEPS=50 OMP_PROC_BIND=true NS_NUM_THREADS=24 OMP_NUM_THREADS=1  numactl -N 1 -m 1 -C 84-110 nohup python api_server.py --model meta-llama/Llama-2-7b-chat-hf --quantization ns --device cpu --max-num-seqs 64 --block-size 4096 --max-model-len 4096 --host 10.0.11.5 --port 8073 > s8073.log 2>&1 &
sleep 2

