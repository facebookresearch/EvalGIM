# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    set_seed,
)

from evaluation_library.data.dataset_evaluation import (
    get_gen_image_dataset_evaluation,
    get_image_datasets_evaluation,
)
from evaluation_library.evaluator import Evaluator
from evaluation_library.metrics.customCLIPScore import CustomCLIPScore
from evaluation_library.metrics.customFID import CustomFID
from evaluation_library.metrics.customFIDTorchEval import CustomFIDTorchEval
from evaluation_library.metrics.DSG import DSG
from evaluation_library.metrics.groupedMarginalMetric import GroupedMarginalMetric
from evaluation_library.metrics.PRDC import PRDC

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


_METRICS = {
    "clipscore": CustomCLIPScore,
    "fid_torchmetrics": CustomFID,
    "fid_torcheval": CustomFIDTorchEval,
    "prdc": PRDC,
    "dsg": DSG,
}


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="output",
        nargs="?",
        help="postfix for logdir",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="seed for seed_everything",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size",
    )

    parser.add_argument(
        "-mp",
        "--mixed_precision",
        type=str,
        default="no",
        choices=["fp16", "bf16", "no"],
        help="Mixed precision",
    )

    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="data logging directory",
    )
    parser.add_argument("--generated_images_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--eval_dataset_name", type=str, default=None)
    parser.add_argument(
        "--marginal_metrics",
        type=str,
        default="fid_torchmetrics",
        help=f"List of comma-separated metrics from {_METRICS.keys()}.",
    )
    parser.add_argument(
        "--conditional_metrics",
        type=str,
        default="fid_torchmetrics",
        help=f"List of comma-separated metrics from {_METRICS.keys()}.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help="List of comma-separated group names.",
    )

    return parser


def eval(args):
    logger = get_logger("Eval")
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%m/%d/%Y-%H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    # Load accelerator
    # increase timeout to not raise an error during long evals.
    kwargs = [
        InitProcessGroupKwargs(timeout=timedelta(seconds=500000)),
        DistributedDataParallelKwargs(),
    ]

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=kwargs,
        rng_types=[],
    )

    accelerator.init_trackers(project_name="Eval", init_kwargs=None)

    accelerator.print("\n")
    accelerator.print("Command line args:")
    accelerator.print(args)

    # Set seed
    logger.info(f"Using random seed : {args.seed}")
    set_seed(seed=args.seed, device_specific=True)

    logger.info("Loading dataset.")

    real_attribute_dataloader = get_gen_image_dataset_evaluation(
        args.generated_images_path,
        batch_size=args.batch_size,
    )

    if len(args.groups) > 0:
        conditional_metrics = {
            metric: _METRICS[metric](groups=args.groups.split(",")).to(accelerator.device)
            for metric in args.conditional_metrics.split(",")
        }
    else:
        conditional_metrics = {
            metric: _METRICS[metric]().to(accelerator.device)
            for metric in args.conditional_metrics.split(",")
        }

    if len(args.marginal_metrics) == 0:
        real_attribute_dataloader = accelerator.prepare(real_attribute_dataloader)
        real_image_dataloader = None
        marginal_metrics = {}
    else:
        real_image_dataloader = get_image_datasets_evaluation(
            dataset_name=args.eval_dataset_name,
            batch_size=args.batch_size,
            img_size=args.image_size,
        )
        real_attribute_dataloader, real_image_dataloader = accelerator.prepare(
            real_attribute_dataloader, real_image_dataloader
        )
        marginal_metrics = {}
        if len(args.marginal_metrics) > 0:
            if len(args.groups) > 0:
                for metric in args.marginal_metrics.split(","):
                    marginal_metrics[metric] = GroupedMarginalMetric(
                        groups=args.groups.split(","),
                        metric=_METRICS[metric],
                        device=accelerator.device,
                    )
            else:
                for metric in args.marginal_metrics.split(","):
                    marginal_metrics[metric] = _METRICS[metric]().to(accelerator.device)

    evaluator = Evaluator(
        marginal_metrics=marginal_metrics,
        conditional_metrics=conditional_metrics,
        accelerator=accelerator,
        grouped_eval=len(args.groups) > 0,
    )

    scores = evaluator.unpaired_eval(
        real_attribute_dataloader=real_attribute_dataloader,
        real_image_dataloader=real_image_dataloader,
    )

    for score_key, score_value in scores.items():
        scores[score_key] = torch.tensor(score_value).to("cpu").float().item()
    output_dir = Path(args.logdir) / args.name
    with open(f"{output_dir}/scores.yaml", "w") as f:
        yaml.dump(scores, f)

    if os.path.exists(Path(args.logdir) / "results.csv"):
        results_df = pd.read_csv(f"{args.logdir}/results.csv", index_col=0)
    else:
        results_df = pd.DataFrame(columns=["model_id", "dataset", "cfg_scale", "output_dir"])
    if results_df[results_df["output_dir"] == output_dir].shape[0] == 0:
        for score_key, _ in scores.items():
            if score_key not in results_df.columns:
                results_df[score_key] = [None] * results_df.shape[0]
        vals = []
        for col in results_df.columns:
            if col == "model_id":
                vals.append(args.model_id)
            elif col == "dataset":
                vals.append(args.eval_dataset_name)
            elif col == "cfg_scale":
                vals.append(args.cfg_scale)
            elif col == "output_dir":
                vals.append(output_dir)
            elif col in scores:
                vals.append(scores[col])
            else:
                vals.append(None)
        results_df = pd.concat([results_df, pd.DataFrame([vals], columns=results_df.columns)])
        results_df.to_csv(f"{args.logdir}/results.csv")

    return output_dir, scores


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.marginal_metrics == "none":
        args.marginal_metrics = ""
    if "vqascore" in args.conditional_metrics:
        try:
            from metrics.vqascore import VQAScore

            _METRICS["vqascore"] = VQAScore
        except Exception as e:
            print(f"Not able to import VQAScore. Error: {e}")

    output_dir, scores = eval(args)
