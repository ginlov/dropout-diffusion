#/usr/bin/bash
# change diffusion steps before training

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--weight_clipping false --diffusion_steps 200 --noise_schedule exponential"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --save_interval 10000 --data_dir cifar_train --suffix_prefix baseline --api_key bekqVkrLIJvPbnh9U5LPMdeW0 --project_name exponential-noise --workspace ginlov"
export OPENAI_LOGDIR="log"
export OPENAI_LOG_FORMAT="csv"

python scripts/image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
