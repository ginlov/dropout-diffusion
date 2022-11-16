"""
Evaluation lipschitz constant of models.
Does not support distributed setting.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from dropout_diffusion import dist_util, logger
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
    logger.configure()

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

    logger.log("sampling...")

    lipschitz_constant = {
        "norm2_normmax": 0.0,
        "norm1_normmax": 0.0,
        "normmax_normmax": 0.0,
        "norm2_norm2": 0.0,
        "norm1_norm2": 0.0,
        "normmax_norm2": 0.0,
    }
    for _ in range(args.num_loops):
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, noise = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            save_noise=True,
        )
        noise = th.cat(
            noise[:-1], dim=1
        )  # Remove final noise because no noise at timestep 0
        noise = noise.view(args.batch_size, -1)
        sample = sample.contiguous().view(args.batch_size, -1)

        norm_max_dis_output = th.nn.functional.pdist(sample, p=float("inf"))
        norm_2_dis_output = th.nn.functional.pdist(sample, p=2)

        norm_1_dis_input = th.nn.functional.pdist(noise, p=1)
        norm_2_dis_input = th.nn.functional.pdist(noise, p=2)
        norm_max_dis_input = th.nn.functional.pdist(noise, p=float("inf"))

        lipschitz_constant["norm1_normmax"] = max(
            lipschitz_constant["norm1_normmax"],
            th.max(norm_max_dis_output / norm_1_dis_input).item(),
        )
        lipschitz_constant["norm2_normmax"] = max(
            lipschitz_constant["norm2_normmax"],
            th.max(norm_max_dis_output / norm_2_dis_input).item(),
        )
        lipschitz_constant["normmax_normmax"] = max(
            lipschitz_constant["normmax_normmax"],
            th.max(norm_max_dis_output / norm_max_dis_input).item(),
        )
        lipschitz_constant["norm1_norm2"] = max(
            lipschitz_constant["norm1_norm2"],
            th.max(norm_2_dis_output / norm_1_dis_input).item(),
        )
        lipschitz_constant["norm2_norm2"] = max(
            lipschitz_constant["norm2_norm2"],
            th.max(norm_2_dis_output / norm_2_dis_input).item(),
        )
        lipschitz_constant["normmax_norm2"] = max(
            lipschitz_constant["normmax_norm2"],
            th.max(norm_2_dis_output / norm_max_dis_input).item(),
        )
        logger.log(f"Running {_+1} loops")

    logger.log(f"Lipschitz constant {lipschitz_constant}")

    dist.barrier()
    logger.log("calculating complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_loops=10,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
