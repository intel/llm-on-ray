numactl -C 60-70 -m 1 ray start --head --include-dashboard false --num-cpus 0
numactl -C 0-55 -m 0 ray start --address='127.0.0.1:6379'

