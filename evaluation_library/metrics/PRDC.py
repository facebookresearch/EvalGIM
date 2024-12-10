# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from torchmetrics import Metric
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.utilities import dim_zero_cat


def compute_pairwise_distance(data_x, data_y=None):
    """
    From: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = pairwise_distances(data_x, data_y, metric="euclidean", n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    From: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    From: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k, compute_dc=False):
    """
    From: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    print(f"Num real: {real_features.shape[0]} Num fake: {fake_features.shape[0]}")
    print("Computing manifold...")
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)
    print("Computing precision...")
    precision = (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
    )
    print("Computing recall...")
    recall = (
        (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()
    )
    if compute_dc:
        print("Computing density...")
        density = (1.0 / float(nearest_k)) * (
            distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()
        print("Computing coverage...")
        coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()
        return dict(precision=precision, recall=recall, density=density, coverage=coverage)
    else:
        return dict(precision=precision, recall=recall)


def reduce_concat(input):
    k = [i for i in input]
    return torch.concat(k)


class PRDC(Metric):
    def __init__(self, k=3, **kwargs):
        super().__init__(**kwargs)
        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")
        self.k = k
        self.feature_length = 2048
        self.feature_extractor = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=[str(self.feature_length)]
        )

    def update(self, imgs: torch.Tensor, real: bool) -> None:
        assert torch.max(imgs) <= 1.0 and torch.min(imgs) >= 0, "images should be between [0, 1]"
        features = self.feature_extractor((imgs * 255).to(torch.uint8))
        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def update_real_images(self, reference_images: torch.Tensor, real_image_datapoint_batch: dict) -> None:
        assert reference_images.shape[0] == len(real_image_datapoint_batch["image"])
        return self.update(reference_images, real=True)

    def update_generated_images(
        self, generated_images: torch.Tensor, real_attribute_datapoint_batch: dict
    ) -> None:
        assert generated_images.shape[0] == len(real_attribute_datapoint_batch["prompt"])
        return self.update(generated_images, real=False)

    def compute(self) -> dict:
        real_features = dim_zero_cat(
            self.real_features
        )  # Recommended per https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
        fake_features = dim_zero_cat(self.fake_features)
        assert (real_features.shape[1] == self.feature_length) and (
            len(real_features.shape) == 2
        ), "Dimensions are incorrect, potential synchronization problem"

        real_features = np.asarray(real_features.cpu().detach().float().numpy())
        fake_features = np.asarray(fake_features.cpu().detach().float().numpy())

        return compute_prdc(real_features, fake_features, self.k, compute_dc=True)
