#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -m master_ip -w worker_ips"
   echo -e "\t-m IP address of master node"
   echo -e "\t-w IP address of worker nodes, use @ to combine multiple worker ip addresses"
   exit 1 # Exit script after printing help
}

while getopts "m:w:" opt
do
   case "$opt" in
      m ) master_ip="$OPTARG" ;;
      w ) worker_ips="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$master_ip" ] || [ -z "$worker_ips" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

echo -e "\ncopy hadoop configuration files..."
cp /home/user/workspace/tools/workload_in_containers/configs/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh
cp /home/user/workspace/tools/workload_in_containers/configs/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
cp /home/user/workspace/tools/workload_in_containers/configs/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml && \
cp /home/user/workspace/tools/workload_in_containers/configs/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
cp /home/user/workspace/tools/workload_in_containers/configs/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
cp /home/user/workspace/tools/workload_in_containers/configs/workers $HADOOP_HOME/etc/hadoop/workers


sed -i 's@hadoop-leader@'"$master_ip"'@'  $HADOOP_HOME/etc/hadoop/core-site.xml && \
sed -i 's@hadoop-leader@'"$master_ip"'@'  $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
sed -i 's@hadoop-leader@'"$master_ip"'@'  $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
sed -i 's@hadoop-leader@'"$master_ip"'@'  $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
sed -i 's@hadoop-leader@'"$master_ip"'@'  $HADOOP_HOME/etc/hadoop/workers  


if [ $worker_ips != "" ]; then 
    worker_ips=$(echo $worker_ips | tr "@" "\n")

    for ip in $worker_ips
    do
        echo "" >> $HADOOP_HOME/etc/hadoop/workers 
        echo "${ip}" >> $HADOOP_HOME/etc/hadoop/workers 
    done

    echo -e "\ncopy files to worker nodes..."
    for ip in $worker_ips
    do  
        scp -o StrictHostKeyChecking=no $HADOOP_HOME/etc/hadoop/workers ${ip}:$HADOOP_HOME/etc/hadoop/workers
        scp -o StrictHostKeyChecking=no $HADOOP_HOME/etc/hadoop/core-site.xml ${ip}:$HADOOP_HOME/etc/hadoop/core-site.xml
        scp -o StrictHostKeyChecking=no $HADOOP_HOME/etc/hadoop/hdfs-site.xml ${ip}:$HADOOP_HOME/etc/hadoop/hdfs-site.xml
        scp -o StrictHostKeyChecking=no $HADOOP_HOME/etc/hadoop/yarn-site.xml ${ip}:$HADOOP_HOME/etc/hadoop/yarn-site.xml
        scp -o StrictHostKeyChecking=no $HADOOP_HOME/etc/hadoop/mapred-site.xml ${ip}:$HADOOP_HOME/etc/hadoop/mapred-site.xml
        scp -o StrictHostKeyChecking=no $HADOOP_HOME/etc/hadoop/hadoop-env.sh ${ip}:$HADOOP_HOME/etc/hadoop/hadoop-env.sh
    done 

    for ip in $worker_ips
    do 
        ssh ${ip} "
            echo -e "\nformat datanode on ${ip}..."
            (sleep 5; echo y) | $HADOOP_HOME/bin/hdfs datanode -format
        "
    done 

fi 

echo -e "\nformat namenode..."
(sleep 5; echo y) | $HADOOP_HOME/bin/hdfs namenode -format

echo "starting HDFS..."
$HADOOP_HOME/sbin/start-dfs.sh
echo -e "\n"

hdfs dfsadmin -report 