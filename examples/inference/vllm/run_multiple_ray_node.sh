# NS_MODEL_PERF_STEPS=50 
OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 1 -C 83-109 -m 1 ray start --address='10.0.11.8:6379' --num-cpus 27 --resources='{"inference_engine": 1}'
sleep 1
OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 0 -C 0-26 -m 0 ray start --address='10.0.11.8:6379' --num-cpus 27 --resources='{"inference_engine": 1}'
sleep 1
OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 1 -C 56-82 -m 1 ray start --address='10.0.11.8:6379' --num-cpus 27 --resources='{"inference_engine": 1}'
sleep 1
OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 0 -C 27-53 -m 0 ray start --address='10.0.11.8:6379' --num-cpus 27 --resources='{"inference_engine": 1}'
sleep 1
numactl -N 1 -C 110-111 -m 1 ray start --address='10.0.11.8:6379' --num-cpus 1 --resources='{"app_router": 1}'

sleep 1
echo 'done'
