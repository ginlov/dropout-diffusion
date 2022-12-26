#/usr/bin/bash
# Change diffusion steps and model path before sampling

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--weight_clipping false --diffusion_steps 200 --noise_schedule linear --correct_sigma false"
export SAMPLE_FLAGS="--data_dir cifar_train --num_samples 50000 --batch_size 64 --num_to_correct_variance 10000 --num_step_save 0 3 5 7 10 20 --suffix_prefix baseline"
export OPENAI_LOGDIR="sample_log"
export OPENAI_LOG_FORMAT="stdout"
export CHECKPOINT_PATH="--model_path path_to_model"

python scripts/image_sample.py $CHECKPOINT_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
