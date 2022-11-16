#/usr/bin/bash

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule exponential --diffusion_dropout 0.1 --num_sample 10"
export SAMPLE_FLAGS="--num_loops 1 --batch_size 2"
export OPENAI_LOGDIR="sample_log"
export OPENAI_LOG_FORMAT="stdout"
export CHECKPOINT_PATH="--model_path /Users/leo/Downloads/model050000.pt"

python scripts/lipschitz_calculation.py $CHECKPOINT_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
