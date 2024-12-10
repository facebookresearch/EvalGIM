# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Mapping

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torcheval.metrics.toolkit import sync_and_compute
from torchmetrics import Metric
from tqdm import tqdm

from evaluation_library.metrics.groupedMarginalMetric import GroupedMarginalMetric

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator to compute matrics over the validation set."""

    def __init__(
        self,
        marginal_metrics: Mapping[str, Metric | GroupedMarginalMetric],
        conditional_metrics: Mapping[str, Metric],
        accelerator: Accelerator,
        num_samples: int = 50000,
        grouped_eval: bool = False,
    ) -> None:
        self.marginal_metrics = marginal_metrics
        self.conditional_metrics = conditional_metrics
        self.accelerator = accelerator
        self.num_samples = num_samples
        self.grouped_eval = grouped_eval

    @property
    def all_metrics(self):
        return list(self.marginal_metrics.items()) + list(self.conditional_metrics.items())

    def reset(self) -> None:
        self.accelerator.wait_for_everyone()
        for _, metric in self.all_metrics:
            if self.grouped_eval:
                if getattr(metric, "grouped_metric", None) is not None:
                    for _, grouped_metric in metric.grouped_metric.items():
                        grouped_metric.reset()
                else:
                    metric.reset()
            else:
                metric.reset()

    def renormalize(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images from [-1, 1] to [0, 1]"""
        return ((images + 1.0) / 2.0).clip(0, 1).to(torch.float32, non_blocking=True)

    @torch.inference_mode()
    def update_marginal_metrics_real_images(
        self, reference_images: torch.Tensor, real_image_datapoint_batch: dict
    ) -> None:
        """
        Performs standardized update() call to marginal metrics, i.e. metrics that
        reference real images in their calculations, of real images only.

        Parameters:
            reference_images: Real image tensors scaled to [0, 1]
            real_attribute_datapoint_batch: Contains prompt / conditioning info for image generation
                and metric metadata (e.g. real images, question graphs, group info, etc.) in 1:1
                correspondance to generated_images.
        """
        for _, metric in self.marginal_metrics.items():
            with self.accelerator.autocast():
                metric.update_real_images(
                    reference_images=reference_images, real_image_datapoint_batch=real_image_datapoint_batch
                )

    @torch.inference_mode()
    def update_marginal_metrics_generated_images(
        self, generated_images: torch.Tensor, real_attribute_datapoint_batch: dict
    ) -> None:
        """
        Performs standardized update() call to marginal metrics, i.e. metrics that
        reference real images in their calculations, of generated images only.

        Parameters:
            generated_images: Generated image tensors scaled to [0, 1]
            real_attribute_datapoint_batch: Contains prompt / conditioning info for image generation
                and metric metadata (e.g. real images, question graphs, group info, etc.) in 1:1
                correspondance to generated_images.
        """
        for _, metric in self.marginal_metrics.items():
            with self.accelerator.autocast():
                metric.update_generated_images(
                    generated_images=generated_images,
                    real_attribute_datapoint_batch=real_attribute_datapoint_batch,
                )

    @torch.inference_mode()
    def update_conditional_metrics(
        self, generated_images: torch.Tensor, real_attribute_datapoint_batch: dict
    ) -> None:
        """
        Performs standardized update() call to conditional metrics, i.e. metrics that do
        not require reference real images in their calculations.

        Parameters:
            generated_images: Generated image tensors scaled to [0, 1]
            real_attribute_datapoint_batch: Contains prompt / conditioning info for image generation
                and metric metadata (e.g. real images, question graphs, group info, etc.) in 1:1
                correspondance to generated_images.
        """
        assert generated_images.size(0) == len(real_attribute_datapoint_batch["condition"]["class_id"])
        for _, metric in self.conditional_metrics.items():
            with self.accelerator.autocast():
                metric.update(
                    generated_images_batch=generated_images,
                    real_attribute_datapoint_batch=real_attribute_datapoint_batch,
                )

    def sync_and_compute(self, name, metric) -> dict:
        return {name: sync_and_compute(metric)}

    def compute_metrics(self) -> None:
        """
        Computes metrics, and assumes the requisite `update()` steps have been performed.
        """
        scores = {}
        for name, metric in self.all_metrics:
            if self.grouped_eval:
                metric_value = metric.compute()
            else:
                metric_value = (
                    self.sync_and_compute(name, metric) if "torcheval" in name else metric.compute()
                )
            for name, value in metric_value.items():
                print({f"eval/{name}": value})
                self.accelerator.log({f"eval/{name}": value})
                scores[name] = value
        return scores

    def unpaired_eval(
        self,
        real_attribute_dataloader: DataLoader,
        real_image_dataloader: DataLoader = None,
    ) -> None:
        """
        Performs evaluation with de-coupled image generation and metric calculation
        metadata (e.g. real images, question graphs, group info, etc.).

        Parameters:
            `real_attribute_dataloader`: Contains prompt / conditioning info for image generation
                and metric metadata (e.g. real images, question graphs, group info, etc.)
            `real_image_dataloader`: Contains real images needed for marginal metrics
            `size`: Size requirements for images.
        """
        if real_image_dataloader is None:
            real_image_dataloader = []
        self.reset()

        for real_image_datapoint_batch in tqdm(real_image_dataloader):
            image_tensors = torch.stack([i for i in real_image_datapoint_batch["image"]])
            reference_images = self.renormalize(image_tensors).to(self.accelerator.device)
            self.update_marginal_metrics_real_images(
                reference_images=reference_images, real_image_datapoint_batch=real_image_datapoint_batch
            )

        num_generated_images = 0
        for _, gen_batch in enumerate(tqdm(real_attribute_dataloader)):
            if num_generated_images >= self.num_samples // self.accelerator.num_processes:
                break

            generated_images = gen_batch["image"]
            num_generated_images += generated_images.shape[0]

            self.update_marginal_metrics_generated_images(
                generated_images=generated_images, real_attribute_datapoint_batch=gen_batch
            )
            self.update_conditional_metrics(
                generated_images=generated_images, real_attribute_datapoint_batch=gen_batch
            )

        return self.compute_metrics()
