# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random

import numpy as np
import pandas as pd
from torchvision import transforms

from .real_datasets import (
    CC12MValidationRealAttributeDataset,
    CC12MValidationRealImageDataset,
    COCORealAttributeDataset,
    COCORealImageDataset,
    GeoDERealAttributeDataset,
    GeoDERealImageDataset,
    ImageNetValidationRealAttributeDataset,
    ImageNetValidationRealImageDataset,
)


class COCO15KRealImageDataset(COCORealImageDataset):
    def __init__(self, img_size=256):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.labels = np.load(self.ANNOT_VAL_PATH, allow_pickle=True)
        self.labels = random.Random(0).sample(list(self.labels), 14749)
        self.img_key = "image_id"
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])


class COCO15KRealAttributeDataset(COCORealAttributeDataset):
    def __init__(self, crop_scale=0.9, extra_args=None):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.labels = np.load(self.ANNOT_VAL_PATH, allow_pickle=True)
        self.labels = random.Random(0).sample(list(self.labels), 14749)
        self.img_key = "image_id"


class ImageNetValidation15KRealImageDataset(ImageNetValidationRealImageDataset):
    def __init__(self, img_size=(256, 256)):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        with open(self.ANNOT_VAL_PATH) as f:
            self.labels = json.load(f)
        self.labels["items"] = random.Random(0).sample(list(self.labels["items"]), 14749)

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])


class ImageNetValidation15KRealAttributeDataset(ImageNetValidationRealAttributeDataset):
    def __init__(self):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        with open(self.ANNOT_VAL_PATH) as f:
            self.labels = json.load(f)
        self.labels["items"] = random.Random(0).sample(list(self.labels["items"]), 14749)

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])


## CC12M15K
class CC12MValidation15KRealImageDataset(CC12MValidationRealImageDataset):
    def __init__(self, img_size=(256, 256)):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.labels = np.load(self.ANNOT_VAL_PATH, allow_pickle=True)
        self.labels = random.Random(0).sample(list(self.labels), 14749)

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])


class CC12MValidation15KRealAttributeDataset(CC12MValidationRealAttributeDataset):
    def __init__(self):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.labels = np.load(self.ANNOT_VAL_PATH, allow_pickle=True)
        self.labels = random.Random(0).sample(list(self.labels), 14749)


class GeoDE15KRealImageDataset(GeoDERealImageDataset):
    def __init__(self, img_size=(256, 256)):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.geode_df = pd.read_csv(self.ANNOT_VAL_PATH)
        self.geode_df = self.geode_df[self.geode_df["tree_tag"] == "no"]
        indices = []
        for i in range(0, 27):
            for k in range(0, 6):
                indices.extend(range((i * 1080) + (k * 180), (i * 1080) + (k * 180) + 90))
        self.geode_df = self.geode_df.iloc[indices]
        self.labels = list(self.geode_df["object"])
        self.regions = list(self.geode_df["region"])
        self.image_paths = [f"{self.IMG_ROOT_PATH}{file_path}" for file_path in self.geode_df["file_path"]]

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])


class GeoDE15KRealAttributeDataset(GeoDERealAttributeDataset):
    def __init__(self):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.geode_df = pd.read_csv(self.ANNOT_VAL_PATH)
        self.geode_df = self.geode_df[self.geode_df["tree_tag"] == "no"]
        indices = []
        for i in range(0, 27):
            for k in range(0, 6):
                indices.extend(range((i * 1080) + (k * 180), (i * 1080) + (k * 180) + 90))
        self.geode_df = self.geode_df.iloc[indices]
        self.labels = list(self.geode_df["object"])
        self.regions = list(self.geode_df["region"])
        self.image_paths = [f"{self.IMG_ROOT_PATH}{file_path}" for file_path in self.geode_df["file_path"]]

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])
