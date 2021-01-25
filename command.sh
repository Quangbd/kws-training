# DNN
python train.py --data_dir /home/ubuntu/kws-data/speech_commands_v0.02 \
  --model_architecture dnn \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 40 \
  --model_size_info 144 144 144 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/dnn/dnn1/logs \
  --train_dir work/dnn/dnn1/training \
  --wanted_words yes,no,up,down,left,right,on,off,stop,go

python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture dnn \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 40 \
  --model_size_info 256 256 256 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/dnn/dnn2/logs \
  --train_dir work/dnn/dnn2/training

python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture dnn \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 40 \
  --model_size_info 436 436 436 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/dnn/dnn3/logs \
  --train_dir work/dnn/dnn3/training

python train.py --data_dir /home/ubuntu/kws-vinai/clean \
  --model_architecture dnn \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 40 \
  --model_size_info 436 436 436 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/dnn/dnn3/logs \
  --train_dir work/dnn/dnn3/training \
  --wanted_words quang

# CNN
python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture cnn \
  --model_size_info 28 10 4 1 1 30 10 4 2 1 16 128 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/cnn/cnn1/logs \
  --train_dir work/cnn/cnn1/training

python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture cnn \
  --model_size_info 64 10 4 1 1 48 10 4 2 1 16 128 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/cnn/cnn2/logs \
  --train_dir work/cnn/cnn2/training

python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture cnn \
  --model_size_info 60 10 4 1 1 76 10 4 2 1 58 128 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/cnn/cnn3/logs \
  --train_dir work/cnn/cnn3/training

# LSTM
python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture lstm \
  --model_size_info 98 144 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 40 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/lstm/lstm1/logs \
  --train_dir work/lstm/lstm1/training

python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture lstm \
  --model_size_info 130 280 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/lstm/lstm2/logs \
  --train_dir work/lstm/lstm2/training

python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture lstm \
  --model_size_info 188 500 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/lstm/lstm3/logs \
  --train_dir work/lstm/lstm3/training

# DS_CNN
python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture ds_cnn \
  --model_size_info 5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/ds_cnn/ds_cnn1/logs \
  --train_dir work/ds_cnn/ds_cnn1/training

python train.py --data_dir /home/ubuntu/speech_commands_v0.02 \
  --model_architecture ds_cnn \
  --model_size_info 5 172 10 4 2 1 172 3 3 2 2 172 3 3 1 1 172 3 3 1 1 172 3 3 1 1 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/ds_cnn/ds_cnn2/logs \
  --train_dir work/ds_cnn/ds_cnn2/training

python train.py --data_dir /home/ubuntu/kws-data/speech_commands_v0.02 \
  --model_architecture ds_cnn \
  --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0005,0.0001,0.00002 \
  --training_steps 10000,10000,10000 \
  --summaries_dir work/ds_cnn/ds_cnn3/logs \
  --train_dir work/ds_cnn/ds_cnn3/training \
  --wanted_words yes,no,up,down,left,right,on,off,stop,go \
  --batch_size 1024

env CUDA_VISIBLE_DEVICES=4, python train.py --data_dir /home/ubuntu/viet_nam_20201113 \
  --model_architecture ds_cnn \
  --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.0001,0.00005,0.00001 \
  --training_steps 15000,25000,20000 \
  --summaries_dir work/ds_cnn/ds_cnn3/logs \
  --train_dir work/ds_cnn/ds_cnn3/training \
  --wanted_words viet_nam \
  --batch_size 128 \
  --eval_step_interval 1000

# DS_CNN
env CUDA_VISIBLE_DEVICES=7, python train.py --data_dir /home/ubuntu/viet_nam_20201116 \
  --model_architecture crnn \
  --model_size_info 48 10 4 2 2 2 60 84 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20 \
  --learning_rate 0.00005,0.00003,0.00001 \
  --training_steps 15000,25000,20000 \
  --summaries_dir work/crnn/crnn1/logs \
  --train_dir work/crnn/crnn1/training \
  --wanted_words viet_nam \
  --batch_size 128 \
  --eval_step_interval 1000

# Test
python test.py --data_dir /Users/quangbd/Documents/data/kws-data/speech_commands_v0.02 \
  --checkpoint /Users/quangbd/Documents/data/model/kws/speech_commands_v0.02/ds_cnn/ds_cnn3/training/best/ds_cnn_9457.ckpt-23200 \
  --model_architecture ds_cnn \
  --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20

# freeze
python convert.py --data_dir /Users/quangbd/Documents/data/kws-data/speech_commands_v0.02 \
  --checkpoint /Users/quangbd/Documents/data/model/kws/speech_commands_v0.02/ds_cnn/ds_cnn3/training/best/ds_cnn_9457.ckpt-23200_bnfused \
  --output_file /Users/quangbd/Documents/data/model/kws/speech_commands_v0.02/ds_cnn/ds_cnn3.pb \
  --model_architecture ds_cnn \
  --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20

python fold_batchnorm.py --data_dir /Users/quangbd/Documents/data/kws-data/speech_commands_v0.02 \
  --checkpoint /Users/quangbd/Documents/data/model/kws/speech_commands_v0.02/ds_cnn/ds_cnn3/training/best/ds_cnn_9457.ckpt-23200 \
  --model_architecture ds_cnn \
  --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 \
  --dct_coefficient_count 10 \
  --window_size_ms 40 \
  --window_stride_ms 20

# Test
env CUDA_VISIBLE_DEVICES=1, python test.py --test_dir /home/ubuntu/new_kws/kws_data/test \
  --checkpoint /home/ubuntu/new_kws/kws_data/models/ds_cnn20210113/ds_cnn1/training/best/ds_cnn_9734.ckpt-31000 \
  --pb /home/ubuntu/new_kws/kws_data/models/ds_cnn20210113/ds_cnn1.pb \
  --tflite /home/ubuntu/new_kws/kws_data/models/ds_cnn20210113/ds_cnn1.tflite \
  --test_model_type checkpoint \
  --name kws-training-0

# DSCNN
env CUDA_VISIBLE_DEVICES=0, python train.py --data_dir /raid/data/quangbd/data/kws/heyvf/train \
  --init_checkpoint /raid/data/quangbd/data/kws/heyvf/models/init/ds_cnn2/ds_cnn2 \
  --model_architecture ds_cnn \
  --model_size_info 5 172 10 4 2 1 172 3 3 2 2 172 3 3 1 1 172 3 3 1 1 172 3 3 1 1 \
  --summaries_dir work/dscnn2-0/ds_cnn/ds_cnn2/logs \
  --train_dir work/dscnn2-0/ds_cnn/ds_cnn2/training \
  --batch_size 128 \
  --augment_dir augment_dir/dscnn2-0 \
  --name dscnn2-0

python convert.py --checkpoint work/dscnn1-4/ds_cnn/ds_cnn1/training/best/ds_cnn_9957.ckpt-56000 \
  --pb work/dscnn1-4/ds_cnn/ds_cnn1/ds_cnn1.pb \
  --tflite work/dscnn1-4/ds_cnn/ds_cnn1/ds_cnn1.tflite

python test.py --test_dir /home/ubuntu/new_kws/kws_data/test \
  --tflite work/dscnn1-4/ds_cnn/ds_cnn1/ds_cnn1.tflite \
  --test_model_type tflite \
  --name dscnn1-4

# CRNN
env CUDA_VISIBLE_DEVICES=1, python train.py --data_dir /Users/quangbd/Documents/data/kws/train \
  --init_checkpoint /Users/quangbd/Documents/data/model/kws/heyvf/init/crnn1/crnn1 \
  --model_architecture crnn \
  --model_size_info 48 10 4 2 2 2 60 84 \
  --summaries_dir work/dscnn1-0/crnn/crnn1/logs \
  --train_dir work/dscnn1-0/crnn/crnn1/training \
  --batch_size 128 \
  --augment_dir augment_dir/crnn1-0 \
  --name crnn1-0 \
  --loss_method ce

env CUDA_VISIBLE_DEVICES=1, python train.py --data_dir /home/ubuntu/new_kws/kws_data/train \
  --init_checkpoint /home/ubuntu/new_kws/kws_data/models/init/crnn1/crnn1 \
  --model_architecture crnn \
  --model_size_info 48 10 4 2 2 2 60 84 \
  --summaries_dir work/crnn1-0/crnn/crnn1/logs \
  --train_dir work/crnn1-0/crnn/crnn1/training \
  --batch_size 128 \
  --augment_dir augment_dir/crnn1-0 \
  --name crnn1-0 \
  --loss_method ce

# LSTM
env CUDA_VISIBLE_DEVICES=1, python train.py --data_dir /Users/quangbd/Documents/data/kws/train \
  --init_checkpoint /Users/quangbd/Documents/data/model/kws/heyvf/init/lstm1/lstm1 \
  --model_architecture lstm \
  --model_size_info 98 144 \
  --summaries_dir work/lstm1-0/lstm/lstm1/logs \
  --train_dir work/lstm1-0/lstm/lstm1/training \
  --batch_size 128 \
  --augment_dir augment_dir/lstm1-0 \
  --name lstm1-0 \
  --loss_method ce

env CUDA_VISIBLE_DEVICES=1, python train.py --data_dir /home/ubuntu/new_kws/kws_data/train \
  --init_checkpoint /home/ubuntu/new_kws/kws_data/models/init/lstm1/lstm1 \
  --model_architecture lstm \
  --model_size_info 98 144 \
  --summaries_dir work/lstm1-0/lstm/lstm1/logs \
  --train_dir work/lstm1-0/lstm/lstm1/training \
  --batch_size 128 \
  --augment_dir augment_dir/lstm1-0 \
  --name lstm1-0 \
  --loss_method ce

# CNN
env CUDA_VISIBLE_DEVICES=1, python train.py --data_dir /Users/quangbd/Documents/data/kws/train \
  --init_checkpoint /Users/quangbd/Documents/data/model/kws/heyvf/init/cnn1/cnn1 \
  --model_architecture cnn \
  --model_size_info 28 10 4 1 1 30 10 4 2 1 16 128 \
  --summaries_dir work/cnn1-0/cnn/cnn1/logs \
  --train_dir work/cnn1-0/cnn/cnn1/training \
  --batch_size 128 \
  --augment_dir augment_dir/cnn1-0 \
  --name cnn1-0 \
  --loss_method ce

env CUDA_VISIBLE_DEVICES=1, python train.py --data_dir /home/ubuntu/new_kws/kws_data/train \
  --init_checkpoint /home/ubuntu/new_kws/kws_data/models/init/cnn1/cnn1 \
  --model_architecture cnn \
  --model_size_info 28 10 4 1 1 30 10 4 2 1 16 128 \
  --summaries_dir work/cnn1-0/cnn/cnn1/logs \
  --train_dir work/cnn1-0/cnn/cnn1/training \
  --batch_size 128 \
  --augment_dir augment_dir/cnn1-0 \
  --name cnn1-0 \
  --loss_method ce