#/usr/bin/bash

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--weight_clipping false --dropout_at_beginning_steps true --step_start_dropout 3 --diffusion_steps 200 --noise_schedule exponential --diffusion_dropout 0.002 --num_sample 100"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --save_interval 10000"
export OPENAI_LOGDIR="log"
export OPENAI_LOG_FORMAT="stdout,csv"

python scripts/image_train.py --data_dir cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
