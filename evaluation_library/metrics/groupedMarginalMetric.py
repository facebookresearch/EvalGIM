# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class GroupedMarginalMetric:
    def __init__(self, groups=None, metric=None, device=None) -> None:
        """
        Instatiate a new metric for each group.
        TODO: In the future we may need to pass the same feature extractor around to avoid
        instantiate many repeated feature extractors.
        """
        assert groups is not None
        assert len(list(set(groups))) == len(groups), "groups must be unique"
        self.device = device
        self.grouped_metric = {g: metric().to(self.device) for g in groups} if metric is not None else {}

    def update_real_images(self, reference_images: torch.Tensor, real_image_datapoint_batch: dict) -> None:
        """
        Partition the inputs by group and update the group-specific metric.
        """
        for group, metric in self.grouped_metric.items():
            grouped_idx = [i for i, g in enumerate(real_image_datapoint_batch["group"]) if group in g]

            if len(grouped_idx) > 0:
                grouped_idx_tensor = torch.Tensor(grouped_idx).int().to(self.device)
                grouped_reference_images = torch.index_select(reference_images, 0, grouped_idx_tensor)

                grouped_real_image_datapoint_batch = {}
                for k, v in real_image_datapoint_batch.items():
                    grouped_real_image_datapoint_batch[k] = [v[i] for i in grouped_idx]

                metric.update_real_images(grouped_reference_images, grouped_real_image_datapoint_batch)

    def update_generated_images(
        self, generated_images: torch.Tensor, real_attribute_datapoint_batch: dict
    ) -> None:
        """
        Partition the inputs by group and update the group-specific metric.
        """
        for group, metric in self.grouped_metric.items():
            grouped_idx = [i for i, g in enumerate(real_attribute_datapoint_batch["group"]) if group in g]

            if len(grouped_idx) > 0:
                grouped_idx_tensor = torch.Tensor(grouped_idx).int().to(self.device)
                grouped_generated_images = torch.index_select(generated_images, 0, grouped_idx_tensor)

                grouped_real_attribute_datapoint_batch = {}
                for k, v in real_attribute_datapoint_batch.items():
                    if k == "condition":
                        grouped_real_attribute_datapoint_batch["condition"] = {}
                        for condition_k, condition_v in v.items():
                            grouped_real_attribute_datapoint_batch["condition"][condition_k] = [
                                condition_v[i] for i in grouped_idx
                            ]
                    else:
                        grouped_real_attribute_datapoint_batch[k] = [v[i] for i in grouped_idx]

                metric.update_generated_images(
                    grouped_generated_images, grouped_real_attribute_datapoint_batch
                )

    def compute(self) -> dict:
        """
        Compute each group-specific metric and coalesce into a single metric dictionary
        """
        ret = {}
        for group, metric in self.grouped_metric.items():
            grouped_compute = metric.compute()  # TODO: sync_and_compute if torcheval
            for m, v in grouped_compute.items():
                ret[f"{m}_{group}"] = v

        return ret
