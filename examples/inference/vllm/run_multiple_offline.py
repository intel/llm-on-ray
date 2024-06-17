NS_NUM_THREADS=25 OMP_NUM_THREADS=1 OMP_PROC_BIND=true NS_MODEL_PERF_STEPS=50 nohup numactl -N 1 -m 1 -C 56-83   python vllm_offline_inference.py > 1.log 2>&1 &
NS_NUM_THREADS=25 OMP_NUM_THREADS=1 OMP_PROC_BIND=true NS_MODEL_PERF_STEPS=50 nohup numactl -N 0 -m 0 -C 0-45   python vllm_offline_inference.py > 2.log 2>&1 &
