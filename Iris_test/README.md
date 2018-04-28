# This test case must use tf1.6

# Command to use to execute job
Mounting the /root/code/Iris_test path into docker

nvidia-docker run -itd -p 2222:2222 -v /root/code/Iris_test:/root/code -v /root/result/Iris_result:/root/result tensorflow/tensorflow:latest  python /root/code/distributed.py --job_name="ps" --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222" --task_index=0

nvidia-docker run -itd -p 2222:2222 -v /root/code:/root/code -v /root/result:/root/result tensorflow/tensorflow:latest  python /root/code/distributed.py --job_name="worker" --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222"  --task_index=0

nvidia-docker run -itd -p 2222:2222 -v /root/code:/root/code -v /root/result:/root/result tensorflow/tensorflow:latest  python /root/code/distributed.py --job_name="worker" --ps_hosts="192.168.1.173:2222" --worker_hosts="192.168.1.171:2222,192.168.1.172:2222"  --task_index=1
