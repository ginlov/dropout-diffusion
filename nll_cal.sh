#/usr/bin/bash

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--weight_clipping false --diffusion_steps 1000 --noise_schedule linear --diffusion_dropout 0.0"
# export TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --save_interval 10000"
export NLL_FLAGS="--data_dir cifar_train --num_samples 100 --clip_denoised true --batch_size 64 --model_path baseline.T1000.510000/ema_0.9999_500000.pt --num_to_correct_variance 0 --suffix_prefix nll_baseline"
export OPENAI_LOGDIR="log"
export OPENAI_LOG_FORMAT="stdout"

python scripts/image_nll.py --data_dir cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $NLL_FLAGS
