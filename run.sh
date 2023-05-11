run_type="$1"


if [[ $run_type = "head" ]]; then

        docker rm -f ray-llm-head

        docker run -itd --privileged  --network host \
                -e http_proxy=${http_proxy} \
                -e https_proxy=${https_proxy} \
                -e no_proxy=${no_proxy} \
                -v $(pwd)/Finetune:/home/user/workspace \
                -w /home/user/workspace \
                --shm-size=64gb --name ray-llm-head ray-llm:latest
        
        docker exec ray-llm-head /bin/bash -c "ray start --head --node-ip-address=10.165.9.166 --dashboard-port=9999 --ray-debugger-external"
        
        docker exec -it ray-llm-head /bin/bash

elif [[ $run_type = "worker" ]]; then

        docker rm -f ray-llm-worker 

        docker run -itd --privileged  --network host \
                -e http_proxy=${http_proxy} \
                -e https_proxy=${https_proxy} \
                -e no_proxy=${no_proxy} \
                -v $(pwd)/Finetune:/home/user/workspace \
                -w /home/user/workspace \
                --shm-size=64gb --name ray-llm-worker ray-llm:latest
        
        docker exec ray-llm-worker /bin/bash -c "ray start --address=10.165.9.166:6379 --ray-debugger-external"

        docker exec -it ray-llm-worker /bin/bash

fi 
