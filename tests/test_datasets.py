# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from data import real_datasets


class TestRealDatasets:
    @pytest.mark.parametrize(
        "dataset_class,image_size",
        [
            (real_datasets.COCORealImageDataset, 256),
            (real_datasets.ImageNetValidationRealImageDataset, 256),
            (real_datasets.CC12MValidationRealImageDataset, 256),
            (real_datasets.GeoDERealImageDataset, 256),
        ],
    )
    def test_mini_coco_real_image_dataset(self, dataset_class, image_size: int):
        dataset = dataset_class()
        assert len(dataset) > 1
        x, y, meta_data = dataset[3]
        assert x.shape == (3, image_size, image_size)

    @pytest.mark.parametrize(
        "dataset_class",
        [
            real_datasets.COCORealAttributeDataset,
            real_datasets.ImageNetValidationRealAttributeDataset,
            real_datasets.CC12MValidationRealAttributeDataset,
            real_datasets.GeoDERealAttributeDataset,
        ],
    )
    def test_attribute_datasets(self, dataset_class):
        dataset = dataset_class()
        assert len(dataset) > 1
        datapoint = dataset[3]
        assert hasattr(datapoint, "prompt")
        assert hasattr(datapoint, "condition")
