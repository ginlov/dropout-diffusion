#/usr/bin/bash
# Change diffusion steps and model path

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--weight_clipping false--diffusion_steps 200 --noise_schedule exponential --correct_sigma false"
export LIPSCHITZ_FLAGS="--clip_denoised true --num_loops 10 --batch_size 64"
export OPENAI_LOGDIR="sample_log"
export OPENAI_LOG_FORMAT="stdout"
export CHECKPOINT_PATH="--model_path path_to_model"

python scripts/lipschitz_calculation.py $CHECKPOINT_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $LIPSCHITZ_FLAGS
