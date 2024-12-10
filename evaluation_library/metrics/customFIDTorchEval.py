# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torcheval.metrics import FrechetInceptionDistance


class CustomFIDTorchEval(FrechetInceptionDistance):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def update_real_images(
        self,
        reference_images: torch.Tensor,
        real_image_datapoint_batch: dict,
    ) -> None:
        assert reference_images.shape[0] == len(real_image_datapoint_batch["image"])
        assert (
            torch.max(reference_images) <= 1.0 and torch.min(reference_images) >= 0
        ), "images should be between [0, 1]"
        self.update(reference_images, is_real=True)

    def update_generated_images(
        self, generated_images: torch.Tensor, real_attribute_datapoint_batch: dict
    ) -> None:
        assert generated_images.shape[0] == len(real_attribute_datapoint_batch["prompt"])
        assert (
            torch.max(generated_images) <= 1.0 and torch.min(generated_images) >= 0
        ), "images should be between [0, 1]"
        self.update(generated_images, is_real=False)

    def compute(self) -> dict:
        return {"fid_torcheval": super().compute()}
