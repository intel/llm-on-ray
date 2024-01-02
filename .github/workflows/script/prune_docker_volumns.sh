#!/bin/bash

while [ 1 ]
do
    disk_info=$(df | grep /mnt/DP_disk1)
    mount_point=$(echo $disk_info | awk '{ print $6 }')
    ratio=$(echo $disk_info | awk '{ print $5 }')
    ratio_nbr=$(echo $ratio | tr -c -d 0-9)
    if [[ "$mount_point" == "/mnt/DP_disk1" ]] && [[ $ratio_nbr -gt 70 ]]; then
        echo "pruning docker volumns..."
        echo y | docker system prune --volumes
    else
        echo "$mount_point usage less than 70%"
    fi

    echo "sleeping 1h"
    sleep 3600
done
