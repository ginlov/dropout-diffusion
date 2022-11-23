# Dropout Diffusion

This codebase is forked from [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

This repo implements code for Dropout Diffusion, an variant of [DDPM](https://arxiv.org/abs/2006.11239) focusing on boosting up sampling speed.

# Quick start

## Install requirements
```bash
pip install -e .
```
```bash
pip install mpi4py
```

## Training
The are several configuration arguments for training, sampling and lipschitz calculation. The easiest way to manage this is export is as environment variables.

Example:
```bash
export DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule exponential --diffusion_dropout 0.002 --num_sample 100"
export MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.1"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --save_interval 10000"

python scripts/image_train.py --data_dir {path_to_data_dir} $DIFFUSION_FLAGS $MODEL_FLAGS $TRAIN_FLAGS
```

For more detail, read CONFIGURATION.md.

## Sampling
Sampling is similar to training, the only diffirence is replacement of SAMPLE_FLAGS for TRAIN_FLAGS.

## Lipschitz calculation
This repos supports calculating lipschitz constant to understand the generalization capacity of models. Use scripts/lipschitz_calcuation.py for rekon the lipschitz constant.
