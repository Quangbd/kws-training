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

python train.py --data_dir /home/ubuntu/kws-vinai/clean \
                --model_architecture ds_cnn \
                --model_size_info 6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 \
                --dct_coefficient_count 10 \
                --window_size_ms 40 \
                --window_stride_ms 20 \
                --learning_rate 0.0005,0.0001,0.00002 \
                --training_steps 10000,10000,10000 \
                --summaries_dir work/ds_cnn/ds_cnn3/logs \
                --train_dir work/ds_cnn/ds_cnn3/training \
                --wanted_words quang \
                --batch_size 100

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