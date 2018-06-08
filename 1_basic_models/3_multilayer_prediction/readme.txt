#173
nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/3_multilayer_prediction:/root/code  tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --data_dir=/root/code/occupancy_data/datatest.txt  --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="ps"   --task_index=0

#172
nvidia-docker run -it -p 2222:2222 -v /root/tensorflow/model/3_multilayer_prediction:/root/code  tensorflow/tensorflow:1.7.0-gpu  python /root/code/train_mul.py --data_dir=/root/code/occupancy_data/datatest.txt  --ps_hosts="192.168.1.173:2222"   --worker_hosts="192.168.1.172:2222"   --job_name="worker"   --task_index=0
