# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Primarily copied from https://github.com/Lightning-AI/torchmetrics/blob/v1.4.0/src/torchmetrics/functional/multimodal/clip_score.py
# with adjustments to enable grouped measurements

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.multimodal.clip_score import _clip_score_update, _get_clip_model_and_processor
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_10
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CLIPScore.plot"]

if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_10:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

    def _download_clip_for_clip_score() -> None:
        _CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", resume_download=True)
        _CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", resume_download=True)

    if not _try_proceed_with_timeout(_download_clip_for_clip_score):
        __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]
else:
    __doctest_skip__ = ["CLIPScore", "CLIPScore.plot"]


class CustomCLIPScore(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound = 100.0

    score: Tensor
    n_samples: Tensor
    feature_network: str = "model"

    def __init__(
        self,
        groups=None,
        model_name_or_path: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        ] = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model, self.processor = _get_clip_model_and_processor(model_name_or_path)
        self.groups = groups
        if self.groups is None:
            self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        else:
            for group in self.groups:
                self.add_state(f"{group}_score", torch.tensor(0.0), dist_reduce_fx="sum")
                self.add_state(f"{group}_n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_images_batch: torch.Tensor, real_attribute_datapoint_batch: dict) -> None:
        assert generated_images_batch.shape[0] == len(real_attribute_datapoint_batch["prompt"])
        assert (
            torch.max(generated_images_batch) <= 1.0 and torch.min(generated_images_batch) >= 0
        ), "generated images should be between [0, 1]"

        if self.groups is None:
            images = (generated_images_batch * 255).to(torch.uint8)
            text = real_attribute_datapoint_batch["prompt"]

            score, n_samples = _clip_score_update(images, text, self.model, self.processor)
            self.score += score.sum(0)
            self.n_samples += n_samples
        else:
            for group in self.groups:
                grouped_idx = [i for i, g in enumerate(real_attribute_datapoint_batch["group"]) if group in g]

                if len(grouped_idx) > 0:
                    grouped_idx_tensor = torch.Tensor(grouped_idx).int().to(self.device)
                    grouped_images = torch.index_select(generated_images_batch, 0, grouped_idx_tensor)
                    grouped_images = (grouped_images * 255).to(torch.uint8)
                    grouped_text = [real_attribute_datapoint_batch["prompt"][i] for i in grouped_idx]

                    score, n_samples = _clip_score_update(
                        grouped_images, grouped_text, self.model, self.processor
                    )
                    setattr(self, f"{group}_score", getattr(self, f"{group}_score") + score.sum(0))
                    setattr(self, f"{group}_n_samples", getattr(self, f"{group}_n_samples") + n_samples)

    def compute(self) -> dict:
        if self.groups is None:
            results = {"clipscore": torch.max(self.score / self.n_samples, torch.zeros_like(self.score))}
        else:
            results = {}
            for group in self.groups:
                results[f"clipscore_{group}"] = torch.max(
                    getattr(self, f"{group}_score") / getattr(self, f"{group}_n_samples"),
                    torch.zeros_like(getattr(self, f"{group}_score")),
                )
        return results

    def plot(
        self, val: Tensor | Sequence[Tensor] | None = None, ax: _AX_TYPE | None = None
    ) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)
