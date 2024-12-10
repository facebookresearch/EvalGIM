# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from torch.utils.data import DataLoader

from .real_datasets import (
    CC12MValidationRealAttributeDataset,
    CC12MValidationRealImageDataset,
    COCORealAttributeDataset,
    COCORealImageDataset,
    GenImageDataset,
    GeoDERealAttributeDataset,
    GeoDERealImageDataset,
    ImageNetValidationRealAttributeDataset,
    ImageNetValidationRealImageDataset,
    TIFA160RealAttributeDataset,
    gen_image_dataset_collate,
    real_attribute_dataset_collate,
    real_image_dataset_collate,
)


def get_attribute_datasets_evaluation(
    dataset_name,
    batch_size=1,
):
    if dataset_name == "coco_txt_dataset":
        dataset = COCORealAttributeDataset()
    elif dataset_name == "imagenet_validation_dataset":
        dataset = ImageNetValidationRealAttributeDataset()
    elif dataset_name == "cc12m_validation_dataset":
        dataset = CC12MValidationRealAttributeDataset()
    elif dataset_name == "geode_dataset":
        dataset = GeoDERealAttributeDataset()
    elif dataset_name == "tifa160_dataset":
        dataset = TIFA160RealAttributeDataset()
    else:
        i = __import__(".".join(dataset_name.split(".")[:-1]), fromlist=[dataset_name.split(".")[:-1]])
        dataset = getattr(i, f"{dataset_name.split('.')[-1]}RealAttributeDataset")()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        prefetch_factor=None,
        drop_last=False,
        collate_fn=real_attribute_dataset_collate,
    )


def get_image_datasets_evaluation(dataset_name, batch_size=1, img_size=256):
    if dataset_name == "coco_txt_dataset":
        dataset = COCORealImageDataset(img_size=img_size)
    elif dataset_name == "imagenet_validation_dataset":
        dataset = ImageNetValidationRealImageDataset(img_size=img_size)
    elif dataset_name == "cc12m_validation_dataset":
        dataset = CC12MValidationRealImageDataset(img_size=img_size)
    elif dataset_name == "geode_dataset":
        dataset = GeoDERealImageDataset(img_size=img_size)
    else:
        i = __import__(".".join(dataset_name.split(".")[:-1]), fromlist=[dataset_name.split(".")[:-1]])
        dataset = getattr(i, f"{dataset_name.split('.')[-1]}RealImageDataset")(img_size=img_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        prefetch_factor=None,
        drop_last=False,
        collate_fn=real_image_dataset_collate,
    )


def get_gen_image_dataset_evaluation(generated_images_path: str | Path, batch_size=1) -> DataLoader:
    data_file = Path(generated_images_path) / "index.json"
    dataset = GenImageDataset(data_file=data_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        prefetch_factor=None,
        drop_last=False,
        collate_fn=gen_image_dataset_collate,
    )
