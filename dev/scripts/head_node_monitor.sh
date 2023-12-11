#!/bin/bash

# eedq[040,186,196,203] eedq206
head_nodes=("eedq186" "eedq203")    # backup head node
head_node="eedq040"
worker_node=("eedq206" "eedq196")
conda_env="LLM_finetune"

CURRENT_DIR=$(cd $(dirname $0); cd ..; pwd)
if [ $CONDA_PREFIX_1 ]; then
    PREFIX=$CONDA_PREFIX_1
else
    PREFIX=$CONDA_PREFIX
fi

function ray_cluster() {
    OPTION=$1
    if [ "$OPTION" == "up" ]; then
        ssh -o "StrictHostKeyChecking no" $head_node ". ${PREFIX}//etc/profile.d/conda.sh; conda activate $conda_env; ray status > /dev/null 2>&1;"
        if [ $? -ne 0 ]; then
            ssh -o "StrictHostKeyChecking no" $head_node ". ${PREFIX}//etc/profile.d/conda.sh; conda activate $conda_env; ray start --head --node-ip-address $head_node --dashboard-host 0.0.0.0 --temp-dir=$HOME/ray --num-cpus=1 > /dev/null 2>&1;"
        fi
        if [ -z $PORT ]; then
            PORT="6379"
        fi
        _MASTER=$head_node:$PORT
        for NODE in ${worker_node[*]}; do
            ssh -o "StrictHostKeyChecking no" $NODE ". ${PREFIX}//etc/profile.d/conda.sh; conda activate $conda_env; ray status > /dev/null 2>&1;"
            if [ $? -ne 0 ]; then
                ssh -o "StrictHostKeyChecking no" $NODE ". ${PREFIX}//etc/profile.d/conda.sh; conda activate $conda_env; $NUMAPREFIX ray start --address=$_MASTER > /dev/null 2>&1;"
            fi
        done
    fi

    if [ "$OPTION" == "stop" ]; then
        ssh -o "StrictHostKeyChecking no" $head_node ". ${PREFIX}//etc/profile.d/conda.sh; conda activate $conda_env; ray stop > /dev/null 2>&1;"
        for NODE in ${worker_node[*]}; do
            ssh -o "StrictHostKeyChecking no" $NODE ". ${PREFIX}//etc/profile.d/conda.sh; conda activate $conda_env; ray stop > /dev/null 2>&1;"
        done
    fi
}

ray_cluster up
echo "Checked ray cluster status."
while true
do
    ssh -o "StrictHostKeyChecking no" $head_node "ps -fe|grep gcs_server |grep -v grep > /dev/null 2>&1"
    if [ $? -ne 0 ]; then
        # stop cluster first
        echo "Head node ${head_node} failure, stop worker nodes first."
        ray_cluster stop

        echo "Choose a new head node:"
        if [ ${#head_nodes[@]} -eq 0 ]; then
            echo "No head node available."
            exit
        fi
        tmp_nodes=( ${head_nodes[*]} )
        for NODE in ${tmp_nodes[@]}; do
            echo "Check node $NODE"
            ping -w 3 $NODE > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                head_node=$NODE
                echo "Node $head_node works well, become the new head node."
                head_nodes=( ${head_nodes[@]/${head_nodes[0]}} )
                break
            fi
            echo "Node $NODE is not working."
            head_nodes=( ${head_nodes[@]/${head_nodes[0]}} )
        done
        # check worker nodes status
        echo "Check worker nodes status: "
        tmp_nodes=( ${worker_node[*]} )
        for(( i=0;i<${#tmp_nodes[@]};i++)) 
        do
            NODE=${tmp_nodes[${i}]}
            ping -w 3 $NODE > /dev/null 2>&1
            if [ $? -ne 0 ]; then
                echo "Worker node $NODE failure. Remove it."
                worker_node=( ${worker_node[@]/$NODE} )
            else
                echo "Worker node $NODE works well."
            fi
        done;
        echo "Restart ray cluster on head node: ${head_node}, worker nodes: ${worker_node[*]}"
        ray_cluster up
        echo "Resubmit job."
        RAY_ADDRESS="http://${head_node}:8265" ray job submit --no-wait --working-dir $CURRENT_DIR/Finetune/ -- python $CURRENT_DIR/Finetune/main.py --config_file $CURRENT_DIR/Finetune/llm_finetune_template.conf

    else
        echo "Report: head node ${head_node} works well"
    fi
    sleep 10
done
