# Hotdog-Classification

A simple TensorFlow app to classify whether a supplied image is a hotdog or not.
Source for post on medium :

https://medium.com/@faze.php/using-tensorflow-to-classify-hotdogs-8494fb85d875

### Buy me a cup of coffee <3

1LHT8uYsmQW8rCQjx58CJVZLgGxKL5pL89

^ Bitcoin RULZ ALL


### Command to run Cluster

PS node:
nvidia-docker run -it -p 2222:2222 -v /root/test/scripts/Hotdog-Classification:/root/code -v /root/test/result:/root/result tensorflow/tensorflow:1.6.0-gpu python /root/code/retrain.py   --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="ps"   --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=0

worker node 0:
nvidia-docker run -it -p 2222:2222 -v /root/test/scripts/Hotdog-Classification:/root/code -v /root/test/result:/root/result tensorflow/tensorflow:1.6.0-gpu python /root/code/retrain.py   --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="worker"   --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=0

worker node 1:
nvidia-docker run -it -p 2222:2222 -v /root/test/scripts/Hotdog-Classification:/root/code -v /root/test/result:/root/result tensorflow/tensorflow:1.6.0-gpu python /root/code/retrain.py   --bottleneck_dir=/root/result/bottlenecks   --model_dir=/root/result/inception   --summaries_dir=/root/result/training_summaries/long   --output_graph=/root/result/retrained_graph.pb   --output_labels=/root/result/retrained_labels.txt   --image_dir=/root/code/images   --job_name="worker"   --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=1 
