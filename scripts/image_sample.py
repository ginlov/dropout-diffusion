"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from dropout_diffusion import dist_util, logger
from dropout_diffusion.image_datasets import load_data
from dropout_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(log_suffix=args.suffix_prefix)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    diffusion.dropout_layer.to(dist_util.dev())
    diffusion.dropout_layer.eval()

    if args.correct_sigma is True:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
        )
        if int(args.num_to_correct_variance) % args.batch_size != 0:
            num_batch = int(int(args.num_to_correct_variance)/ args.batch_size) + 1
        else:
            num_batch = int(int(args.num_to_correct_variance)/ args.batch_size)
        data_shape = args.image_size * args.image_size * 3
        diffusion.calculate_corrected_reverse_variance(model, data, data_shape, num_batch=num_batch, batch_size=args.batch_size, device=dist_util.dev())

    logger.log("sampling...")
    all_images = []
    all_labels = []
    for i in range(len(args.num_step_save)):
        all_images.append([])
        all_labels.append([])
    while len(all_images[0]) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_list = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            num_step_save=args.num_step_save,
            save_sample=args.save_sample,
            remember_x0=args.remember_x0
        )
        # sample *= 1 / (1 - args.diffusion_dropout)
        for i, sample in enumerate(sample_list):
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [
                th.zeros_like(sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images[i].extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels[i].extend(
                    [labels.cpu().numpy() for labels in gathered_labels]
                )
        logger.log(f"created {len(all_images[0]) * args.batch_size} samples")

    for i, result_image in enumerate(all_images):
        arr = np.concatenate(result_image, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels[i], axis=0)
            label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(
                logger.get_dir(),
                f"{args.suffix_prefix}_samples_{shape_str}_{args.num_step_save[len(args.num_step_save) - i - 1]}.npz",
            )
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=False,
        num_to_correct_variance=0,
        num_samples=10000,
        remember_x0=False,
        num_step_save=[0],
        batch_size=16,
        save_sample=False,
        use_ddim=False,
        model_path="",
        suffix_prefix=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
