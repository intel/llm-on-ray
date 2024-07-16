# NS_MODEL_PERF_STEPS=50 
#OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 1 -C 83-109 -m 1 ray start --address='10.0.11.8:6379' --num-cpus 1
#sleep 1
#OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 0 -C 0-26 -m 0 ray start --address='10.0.11.8:6379' --num-cpus 1
#sleep 1
#OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 1 -C 56-82 -m 1 ray start --address='10.0.11.8:6379' --num-cpus 1
#sleep 1
#OMP_PROC_BIND=true NS_NUM_THREADS=25 OMP_NUM_THREADS=1 numactl -N 0 -C 27-53 -m 0 ray start --address='10.0.11.8:6379' --num-cpus 1

# two instances two sockets
#OMP_PROC_BIND=true NS_NUM_THREADS=44 OMP_NUM_THREADS=1  numactl -N 0 -C 0-45 -m 0 ray start --address='10.0.11.8:6379' --num-cpus 1
#sleep 1
#OMP_PROC_BIND=true NS_NUM_THREADS=44 OMP_NUM_THREADS=1  numactl -N 1 -C 56-101 -m 1 ray start --address='10.0.11.8:6379' --num-cpus 1

#LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" numactl -N 0 -m 0 -C 0-55 ray start --address='10.0.11.8:6379' --num-cpus 56
#sleep 1
#LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" numactl -N 1 -m 1 -C 56-111 ray start --address='10.0.11.8:6379' --num-cpus 56

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" numactl -N 0 -m 0 -C 0-27 ray start --address='10.0.11.8:6379' --num-cpus 27
sleep 1
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" numactl -N 0 -m 0 -C 28-55 ray start --address='10.0.11.8:6379' --num-cpus 27
sleep 1
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" numactl -N 1 -m 1 -C 56-83 ray start --address='10.0.11.8:6379' --num-cpus 27
sleep 1
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" numactl -N 1 -m 1 -C 84-111 ray start --address='10.0.11.8:6379' --num-cpus 27


sleep 1
echo 'done'
