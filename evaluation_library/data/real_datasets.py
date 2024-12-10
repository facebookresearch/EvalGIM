# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

from . import imagenet_classes
from .data_classes import (
    GenImageDatapoint,
    RealAttributeDatapoint,
    RealImageDatapoint,
)

StrOrBytesPath: TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
FileDescriptorOrPath: TypeAlias = int | StrOrBytesPath


def pil_loader_v2(fp: FileDescriptorOrPath, max_size: tuple | None = None, mode: str = "RGB") -> Image.Image:
    if max_size is None:
        if isinstance(fp, str | Path):
            with open(fp, "rb") as f:
                return Image.open(f).convert(mode)
        elif isinstance(fp, bytes):
            return Image.open(BytesIO(fp)).convert(mode)
        else:
            raise ValueError("Invalid input type for fp.")

    if isinstance(fp, str | Path):
        with open(fp, "rb") as f:
            image = Image.open(BytesIO(f.read()))
    elif isinstance(fp, bytes):
        image = Image.open(fp)
    else:
        raise ValueError("Invalid input type for fp.")
    image.draft("RGB", (max_size[0], max_size[1]))
    return image


def real_image_dataset_collate(original_batch):
    image = []
    class_label = []
    group = []

    for item in original_batch:
        image.append(item.image)
        class_label.append(item.class_label)
        group.append(item.group)

    return {"image": image, "class_label": class_label, "group": group}


def real_attribute_dataset_collate(original_batch):
    new_batch = {
        "prompt": [],
        "condition": {"class_id": []},
        "class_label": [],
        "group": [],
        "dsg_questions": [],
        "dsg_children": [],
        "dsg_parents": [],
    }

    for item in original_batch:
        new_batch["prompt"].append(item.prompt)
        new_batch["condition"]["class_id"].append(item.condition["class_id"])
        new_batch["class_label"].append(item.class_label)
        new_batch["group"].append(item.group)
        new_batch["dsg_questions"].append(item.dsg_questions)
        new_batch["dsg_children"].append(item.dsg_children)
        new_batch["dsg_parents"].append(item.dsg_parents)

    if torch.is_tensor(new_batch["condition"]["class_id"][0]):
        new_batch["condition"]["class_id"] = torch.stack(new_batch["condition"]["class_id"]).to(device="cuda")

    assert len(new_batch["condition"].keys()) == len(
        original_batch[0].condition.keys()
    ), "not all conditions have been added"

    return new_batch


def gen_image_dataset_collate(original_batch):
    new_batch = real_attribute_dataset_collate(original_batch)
    images = [item.image for item in original_batch]
    new_batch["image"] = torch.stack(images, dim=0)  # assuming all images have the same size
    return new_batch


class RealImageDataset(ABC):
    """Dataset of real images, used for computing marginal metrics
    Does not require a 1:1 mapping of images and prompts with
    RealAttributeDataset.
    """

    @abstractmethod
    def __getitem__(self, idx) -> RealImageDatapoint:
        """
        Returns RealImageDatapoint containing
            image: Tensor
            class_label: Optional[str]
            group: Optional[List[str]]
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns dataset length
        """
        pass


class RealAttributeDataset(ABC):
    """Dataset of prompts and metadata, used for generating images
    and computing metrics
    Does not require a 1:1 mapping of images and prompts with
    RealImageDataset.
    """

    @abstractmethod
    def __getitem__(self, idx) -> RealAttributeDatapoint:
        """
        Returns RealAttributeDatapoint containing
            prompt: str
            condition: Condition
            class_label: Optional[str]
            group: Optional[List[str]]
            dsg_questions: Optional[List[str]]
            dsg_children: Optional[List[str]]
            dsg_parents: Optional[List[str]]
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns dataset length
        """
        pass


class GenImageDataset:
    def __init__(self, data_file: str | Path):
        with open(data_file) as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        example = self.data[idx]

        img = pil_loader_v2(example["image_path"])
        img = to_tensor(img)

        gen_kwargs = example.copy()  # FIXME: load attributes from disk based on example_id
        gen_kwargs.pop("image_path")

        return GenImageDatapoint(image=img, **gen_kwargs)

    def __len__(self):
        return len(self.data)


class COCORealImageDataset(RealImageDataset):
    def __init__(self, img_size=256):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()
        with open(self.ANNOT_VAL_PATH) as f:
            data = json.load(f)
        self.ids = list(
            pd.DataFrame(data["annotations"]).groupby(["image_id"]).first().reset_index()["image_id"]
        )
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["COCO"]
        return paths["IMG_ROOT_VAL"], paths["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealImageDatapoint:
        id = self.ids[idx]
        img = pil_loader_v2(
            Path(self.IMG_ROOT_PATH) / f"COCO_val2014_000000{id:06}.jpg", max_size=self.img_size
        )

        img = img.resize(self.img_size).convert("RGB")
        img_cropped = img.resize(self.img_size)

        datapoint = RealImageDatapoint(image=self.to_tensor(img_cropped))
        return datapoint

    def __len__(self):
        return len(self.ids)


class COCORealAttributeDataset(RealAttributeDataset):
    def __init__(self):
        _, self.ANNOT_VAL_PATH = self.get_dataset_path()

        with open(self.ANNOT_VAL_PATH) as f:
            data = json.load(f)

        self.labels = list(
            pd.DataFrame(data["annotations"]).groupby(["image_id"]).first().reset_index()["caption"]
        )

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["COCO"]
        return paths["IMG_ROOT_VAL"], paths["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealAttributeDatapoint:
        datapoint = RealAttributeDatapoint(
            prompt=self.labels[idx],
            condition={
                "class_id": self.labels[idx],
            },
        )
        return datapoint

    def __len__(self):
        return len(self.labels)


class ImageNetValidationRealImageDataset(RealImageDataset):
    """ImageNet validation return RealImageDatapoints with tensor"""

    def __init__(self, img_size=(256, 256)):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        with open(self.ANNOT_VAL_PATH) as f:
            self.labels = json.load(f)

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["ImageNet1k"]
        return paths["IMG_ROOT"], paths["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealImageDatapoint:
        item = self.labels["items"][idx]
        img = pil_loader_v2(item["image_path"], max_size=self.img_size)

        img_cropped = img.resize(self.img_size).convert("RGB")
        img_tensor = self.to_tensor(img_cropped)

        datapoint = RealImageDatapoint(image=img_tensor, class_label=None)
        return datapoint

    def __len__(self) -> int:
        return len(self.labels["items"])


class ImageNetValidationRealAttributeDataset(RealAttributeDataset):
    def __init__(self):
        _, self.ANNOT_VAL_PATH = self.get_dataset_path()

        with open(self.ANNOT_VAL_PATH) as f:
            self.labels = json.load(f)

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["ImageNet1k"]
        return paths["IMG_ROOT"], paths["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealAttributeDatapoint:
        item = self.labels["items"][idx]

        datapoint = RealAttributeDatapoint(
            prompt=imagenet_classes.id2txt[item["class_id"]],
            condition={
                "class_id": imagenet_classes.id2txt[item["class_id"]],
            },
        )
        return datapoint

    def __len__(self) -> int:
        return len(self.labels["items"])


class CC12MValidationRealImageDataset(RealImageDataset):
    """CC12M validation return RealImageDatapoints with tensor"""

    def __init__(self, img_size=(256, 256)):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.labels = np.load(self.ANNOT_VAL_PATH, allow_pickle=True)

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT_VAL, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["CC12M"]
        return paths["IMG_ROOT_VAL"], paths["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealImageDatapoint:
        item = self.labels[idx]
        # img = pil_loader(item["image_path"])
        img = pil_loader_v2(Path(self.IMG_ROOT_PATH) / str(item["image_id"]), max_size=self.img_size)

        img_cropped = img.resize(self.img_size).convert("RGB")
        img_tensor = self.to_tensor(img_cropped)

        datapoint = RealImageDatapoint(image=img_tensor, class_label=item["captions"][0])
        return datapoint

    def __len__(self):
        return len(self.labels)


class CC12MValidationRealAttributeDataset(RealAttributeDataset):
    def __init__(self):
        _, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.labels = np.load(self.ANNOT_VAL_PATH, allow_pickle=True)

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT_VAL, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["CC12M"]
        return paths["IMG_ROOT_VAL"], paths["ANNOT_VAL"]

    def __getitem__(self, idx):
        item = self.labels[idx]
        datapoint = RealAttributeDatapoint(
            prompt=item["captions"][0],
            condition={
                "class_id": item["captions"][0],
            },
        )
        return datapoint

    def __len__(self):
        return len(self.labels)


class GeoDERealImageDataset(RealImageDataset):
    """GeoDE return RealImageDatapoints with tensor"""

    def __init__(self, img_size=(256, 256)):
        self.IMG_ROOT_PATH, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.geode_df = pd.read_csv(self.ANNOT_VAL_PATH)
        self.geode_df = self.geode_df[self.geode_df["tree_tag"] == "no"]
        self.labels = list(self.geode_df["object"])
        self.regions = list(self.geode_df["region"])
        self.image_paths = [f"{self.IMG_ROOT_PATH}{file_path}" for file_path in self.geode_df["file_path"]]

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        assert len(self.img_size) == 2

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["GeoDE"]
        return paths["IMG_ROOT_VAL"], paths["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealImageDatapoint:
        img = pil_loader_v2(self.image_paths[idx], max_size=self.img_size)

        img_cropped = img.resize(self.img_size).convert("RGB")
        img_tensor = self.to_tensor(img_cropped)

        datapoint = RealImageDatapoint(
            image=img_tensor, class_label=self.labels[idx], group=[self.regions[idx]]
        )
        return datapoint

    def __len__(self) -> int:
        return len(self.labels)


class GeoDERealAttributeDataset(RealAttributeDataset):
    def __init__(self):
        _, self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.geode_df = pd.read_csv(self.ANNOT_VAL_PATH)
        self.geode_df = self.geode_df[self.geode_df["tree_tag"] == "no"]
        self.labels = list(self.geode_df["object"])
        self.regions = list(self.geode_df["region"])
        self.image_paths = [f"{self.IMG_ROOT_PATH}{file_path}" for file_path in self.geode_df["file_path"]]

        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        paths = data["GeoDE"]
        return paths["IMG_ROOT_VAL"], paths["ANNOT_VAL"]

    def get_region_reformatted(self, region) -> str:
        return (
            region.replace("SouthEastAsia", "Southeast Asia")
            .replace("EastAsia", "East Asia")
            .replace("WestAsia", "West Asia")
            .replace("Americas", "the Americas")
        )

    def __getitem__(self, idx) -> RealAttributeDatapoint:
        prompt = f"{self.labels[idx].replace('_', ' ')} in {self.get_region_reformatted(self.regions[idx])}"

        datapoint = RealAttributeDatapoint(
            prompt=prompt,
            condition={
                "class_id": prompt,
            },
            class_label=self.labels[idx],
            group=[self.regions[idx]],
        )
        return datapoint

    def __len__(self) -> int:
        return len(self.labels)


class TIFA160RealAttributeDataset(RealAttributeDataset):
    def __init__(self):
        _, self.ANNOT_VAL_PATH = self.get_dataset_path()
        self.labels = pd.read_csv(self.ANNOT_VAL_PATH)
        self.item_ids = self.labels["item_id"].unique()

    def get_dataset_path(self) -> tuple[str, str]:
        """Returns IMG_ROOT, ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return "", data["TIFA160"]["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealAttributeDatapoint:
        item_id = self.item_ids[idx]
        item = self.labels[self.labels.item_id == item_id]
        prompt = item.iloc[0]["text"]

        datapoint = RealAttributeDatapoint(
            prompt=prompt,
            dsg_questions=item["question_natural_language"].to_list(),
            dsg_children=item["proposition_id"].to_list(),
            dsg_parents=item["dependency"].to_list(),
            condition={"class_id": item_id},
        )
        return datapoint

    def __len__(self) -> int:
        return len(self.labels)
