#/usr/bin/bash
# change diffusion steps before calculating

export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export DIFFUSION_FLAGS="--weight_clipping false --diffusion_steps 100 --noise_schedule exponential --correct_sigma false"
export NLL_FLAGS="--batch_size 64 --data_dir cifar_train --suffix_prefix baseline --num_samples 10000 --num_to_correct_variance 10000 --model_path path/to/model"
export OPENAI_LOGDIR="log"
export OPENAI_LOG_FORMAT="stdout"

python scripts/image_nll.py $MODEL_FLAGS $DIFFUSION_FLAGS $NLL_FLAGS
