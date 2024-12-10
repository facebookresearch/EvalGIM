# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s

import argparse
import os
import subprocess

DATASET_SIZE = {
    "cc12m_validation_dataset": 14749,
    "coco_txt_dataset": 40504,
    "imagenet_validation_txt_dataset": 50000,
    "geode_dataset": 29160,
    "tifa160_dataset": 160,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="coco_txt_dataset")
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--output_path", type=str, default="./projects/generated")
    parser.add_argument("--marginal_metrics", type=str, default="fid_torchmetrics,prdc")
    parser.add_argument("--conditional_metrics", type=str, default="clipscore")
    parser.add_argument("--groups", type=str, default="")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--local", action="store_true", help="run on local GPU")
    parser.add_argument(
        "--generated_images_path",
        type=str,
        default=None,
        help="path of generated images; if None it's inferred based on the model name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep:
        guidance_scales = [2.0, 5.0, 7.5]
        if args.cfg_scale not in guidance_scales:
            guidance_scales.append(args.cfg_scale)
    else:
        guidance_scales = [args.cfg_scale]

    if args.local:
        ngpus = 1
        partition = None
        local = True
        cluster_type = "local"
    else:
        ngpus = 8
        partition = "learnlab"
        local = True
        cluster_type = "slurm"

    for cfg_scale in guidance_scales:
        if not args.generated_images_path:
            generated_images_path = (
                f"{args.output_path}/{args.model_id.replace('/', '--')}__{args.dataset_name}__cfg{cfg_scale}"  # noqa
            )
        else:
            generated_images_path = args.generated_images_path
        if (
            len([name for name in os.listdir(f"{generated_images_path}/images")])
            != DATASET_SIZE[args.dataset_name]
        ):
            print("Note: Not all images are generated for the full dataset.")

        groups = f"--groups {args.groups} "
        command = (
            f"python3 -m evaluation_library.launcher_with_accelerate "
            f"--ngpus {ngpus} "
            f"--nodes 1 "
            f"--partition {partition} "  ## TODO: Customize for your set-up
            f"--name evaluate "
            f"--seed 1 "
            f"--logdir {args.output_path}/evals "
            f"--generated_images_path {generated_images_path} "
            f"--model_id {args.model_id} "
            f"--cfg_scale {cfg_scale} "
            f"--eval_dataset_name {args.dataset_name} "
            f"{groups if len(args.groups)>0 else ''}"
            f"--marginal_metrics {args.marginal_metrics} "
            f"--cluster_type {cluster_type} "
            f"--conditional_metrics {args.conditional_metrics}"
        )

        subprocess.call(command.split(" "))
