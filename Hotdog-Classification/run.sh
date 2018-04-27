#!/bin/bash
git pull origin master

export IP=`ifconfig -a| grep inet | grep 192.168 | awk {'print $2'}| cut -d':' -f2`

if [ "$IP" = "192.168.1.171" ]; then
  echo "171"
  nvidia-docker run -it -v /root/test/scripts/Hotdog-Classification:/root/code -v /root/test/result:/root/result tensorflow/tensorflow:1.8.0-rc1-gpu-fix python /root/code/retrain.py   --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="worker"   --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=0 --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="ps"
elif [ "$IP" = "192.168.1.172" ]
then
  echo "172"
nvidia-docker run -it -v /root/test/scripts/Hotdog-Classification:/root/code -v /root/test/result:/root/result tensorflow/tensorflow:1.8.0-rc1-gpu-fix python /root/code/retrain.py   --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="worker"   --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=1 --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="worker"
elif [ "$IP" = "192.168.1.173" ]
then
  echo "173"
  nvidia-docker run -it -v /root/test/scripts/Hotdog-Classification:/root/code -v /root/test/result:/root/result tensorflow/tensorflow:1.8.0-rc1-gpu-fix python /root/code/retrain.py   --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="ps"   --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=0 --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="worker" 
fi
