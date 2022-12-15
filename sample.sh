#/usr/bin/bash

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--weight_clipping false --diffusion_steps 200 --noise_schedule linear --diffusion_dropout 0.0 --num_sample 1 --correct_sigma true"
export SAMPLE_FLAGS="--data_dir cifar_train --num_samples 10 --batch_size 4 --num_to_correct_variance 10 --num_step_save 0 3 5 7 10 20"
export OPENAI_LOGDIR="sample_log"
export OPENAI_LOG_FORMAT="stdout"
export CHECKPOINT_PATH="--model_path baseline.T200.210000/ema_0.9999_210000.pt"

python scripts/image_sample.py $CHECKPOINT_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
