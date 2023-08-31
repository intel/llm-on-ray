#!/bin/bash

OPTIND=1
WORKSPACE_DIR=/home/user/workspace
MODEL_DIR=/root/.cache/huggingface/
TMP_DIR=/home/user/shared
LOCAL_DIR=/home/user/local

RAM=$(awk '/^Mem/ {print $2}' <(free -mh))
RAM=${RAM/Gi/}
tri_RAM=$((RAM / 3))
docker_ram=$((RAM - tri_RAM))
shm_size=$docker_ram'g'


usage() {
  echo "Usage: $0 -r [run_type] [optional parameters]"
  echo "  options:"
  echo "    -h Display usage"
  echo "    -r run_type"
  echo "         Run type = [startup_head, startup_worker, exec_pipeline, stop_ray, clean_ray, clean_all]"
  echo "         The recommendation is a single instance using no more than a single socket."
  echo "    -a head_address"
  echo "         Ray head address (127.0.0.1)"
  echo "    -c cores_range"
  echo "         Cores range for ray containers"
  echo "    -f mkldnn_verbose"
  echo "         MKLDNN_VERBOSE value"
  echo "    -w workspace"
  echo "         folder path of workspace which includes you source code"
  echo "    -m model_dir"
  echo "         hugging face model directory"
  echo "    -t tmp_dir"
  echo "         temporary directory"
  echo "    -l local_dir"
  echo "         non-nfs directoy which is only accessible by each worker"
  echo "    -u user"
  echo "         user name for access worker server"
  echo "    -p password"
  echo "         password for access worker server"
  echo "    -i image"
  echo "         docker image for head, worker or database"
  echo "    -s worker_ip"
  echo "         worker ip or hostname for access remote worker"
  echo ""
  echo "  examples:"
  echo "    Startup the ray head"
  echo "      $0 -r startup_head -c 10"
  echo ""
}

while getopts "h?r:a:c:f:w:m:t:l:u:p:i:s:" opt; do
    case "$opt" in
    h|\?)
        usage
        exit 1
        ;;
    r)  run_type=$OPTARG
        ;;
    a)  head_address=$OPTARG
        ;;
    c)  cores_range=$OPTARG
        ;;
    f)  verbose=$OPTARG
        ;;
    w)  workspace=$OPTARG
        ;;
    m)  model_dir=$OPTARG
        ;;
    t)  tmp_dir=$OPTARG
        ;;
    l)  local_dir=$OPTARG
        ;;
    u)  user=$OPTARG
        ;;
    p)  password=$OPTARG
        ;;
    u)  user=$OPTARG
        ;;
    p)  password=$OPTARG
        ;;
    i)  image=$OPTARG
        ;;
    s)  worker_ip=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

post_fix=`date +%Y%m%d`'-'`date +%s`

if [[ $run_type = "startup_head" ]]; then

        echo -e "${GREEN} Startup the Ray head with ${cores_range} cores on ${head_address} !${NC}" 
        
        docker run -itd --network host \
                -e http_proxy=${http_proxy} \
                -e https_proxy=${https_proxy} \
                -e no_proxy=${no_proxy} \
                -v ${workspace}:${WORKSPACE_DIR} \
                -v ${model_dir}:${MODEL_DIR} \
                -v ${tmp_dir}:${TMP_DIR} \
                -v ${local_dir}:${LOCAL_DIR} \
                -w /home/user/workspace \
                --shm-size ${shm_size} \
                --cpuset-cpus=${cores_range} \
                --name ray-leader ${image}

        docker exec ray-leader /bin/bash -c "ray start --head --node-ip-address=${head_address} --dashboard-port=9999 --ray-debugger-external --temp-dir=/home/user/local/ray"
        
elif [[ $run_type = "startup_worker" ]]; then

        echo "cores_range = ${cores_range}"

        worker_name='ray-worker'

        head_address=${head_address}':6379'
        
        echo -e "${GREEN} Access ${worker_ip} and startup the Ray on ${cores_range} cores!${NC}"

        sshpass -p $password ssh -o StrictHostKeychecking=no $user@$worker_ip bash << EOF
        docker run -itd \
                -v ${workspace}:${WORKSPACE_DIR} \
                -v ${model_dir}:${MODEL_DIR} \
                -v ${tmp_dir}:${TMP_DIR} \
                -v ${local_dir}:${LOCAL_DIR} \
                --cpuset-cpus=${cores_range} \
                --network host \
                -w /home/user/workspace \
                --shm-size ${shm_size} \
                -e http_proxy=${http_proxy} \
                -e https_proxy=${https_proxy} \
                --name $worker_name ${image} 
EOF

        sshpass -p $password ssh -o StrictHostKeychecking=no $user@$worker_ip bash << EOF
        docker exec $worker_name /bin/bash -c "ray start --address=${head_address} --ray-debugger-external"
EOF

elif [[ $run_type = "stop_ray" ]]; then
    echo "Stop ray containers"
    docker stop $(docker ps -a |grep -E 'ray-head|ray-worker'|awk '{print $1 }')

elif [[ $run_type = "clean_ray" ]]; then
    echo "Clean ray containers"
    docker rm $(docker ps -a |grep -E 'ray-head|ray-worker'|awk '{print $1 }')

elif [[ $run_type = "clean_all" ]]; then
    echo "Stop and clean ray and elasticsearch containers"
    docker stop $(docker ps -a |grep -E 'ray-head|ray-worker|ray-elasticsearch|ray-postgres'|awk '{print $1 }')
    docker rm $(docker ps -a |grep -E 'ray-head|ray-worker|ray-elasticsearch|ray-postgres'|awk '{print $1 }')
fi

