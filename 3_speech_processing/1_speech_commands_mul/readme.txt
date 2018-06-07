#train tensorflow:1.7.0-gpu
#101
python ./code/train_mul.py --data_dir=./speech_dataset/  --summaries_dir=./result/retrain_logs  --train_dir=./result/speech_commands_train  --how_many_training_steps='1000,2000,3000'  --learning_rate='0.01,0.001,0.0001' --batch_size=1000 --ps_hosts="192.168.0.101:2222"   --worker_hosts="192.168.0.100:2222"   --job_name="ps"   --task_index=0

#100
python ./code/train_mul.py --data_dir=./speech_dataset/  --summaries_dir=./result/retrain_logs  --train_dir=./result/speech_commands_train  --how_many_training_steps='1000,2000,3000'  --learning_rate='0.01,0.001,0.0001' --batch_size=1000 --ps_hosts="192.168.0.101:2222"   --worker_hosts="192.168.0.100:2222"   --job_name="worker"   --task_index=0



#Create pb file from ckpt file:
python freeze.py --start_checkpoint=172/speech_commands_train/conv.ckpt-6000 --output_file=172/my_frozen_graph.pb

#loan model to predict:
python label_wav.py --graph=172/my_frozen_graph.pb --labels=172/speech_commands_train/conv_labels.txt --wav=tmp/speech_dataset/left/a5d485dc_nohash_0.wav