# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from typing import Any

import t2v_metrics
import torch
from torchmetrics import Metric
from torchvision.utils import save_image


class VQAScore(Metric):
    def __init__(
        self,
        model_name_or_path: str = "instructblip-flant5-xxl",  # to see all options run: `t2v_metrics.list_all_vqascore_models()`
        temporary_image_dir: str = None,  # if not provided, this will default to system's temporary directory
        **kwargs: Any,
    ):
        """
        Args:
            model_name_or_path: VQA model name
            temporary_image_dir: path to directory of temporarily saved generated images
        """

        super().__init__(**kwargs)

        self.vqascore_model = t2v_metrics.VQAScore(model=model_name_or_path)
        self.temporary_image_dir = temporary_image_dir
        if self.temporary_image_dir:
            os.makedirs(self.temporary_image_dir, exist_ok=True)

        self.add_state("scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_images_batch: torch.Tensor, real_attribute_datapoint_batch: dict[Any]) -> None:
        # temporarily save images to disk because VQAScore uses filepaths
        # then loads the images instead taking the images themselves

        for idx in range(generated_images_batch.shape[0]):
            generated_image = generated_images_batch[idx].unsqueeze(0)
            with tempfile.NamedTemporaryFile(dir=self.temporary_image_dir, suffix=".png") as image_file:
                save_image(generated_image, image_file.name)
                score = self.vqascore_model(
                    images=image_file.name, texts=real_attribute_datapoint_batch["prompt"][idx]
                ).item()
                self.scores += score
                self.n_samples += 1

    def compute(self) -> torch.Tensor:
        return {"vqascore": self.scores / self.n_samples}
