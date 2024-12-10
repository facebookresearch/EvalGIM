# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import pandas as pd
import yaml

from .data_classes import (
    RealAttributeDatapoint,
)
from .real_datasets import RealAttributeDataset


class T2ICompBenchRealAttributeDataset(RealAttributeDataset):
    """
    Creates a RealAttributeDataset from T2I CompBench: https://github.com/openai/dalle3-eval-samples/tree/main/prompts/t2i%20compbench
    """

    def __init__(self):
        self.ANNOT_VAL_PATH = self.get_dataset_path()

        self.prompts = []
        self.groups = []
        for i in os.listdir(self.ANNOT_VAL_PATH):
            data = open(f"{self.ANNOT_VAL_PATH}{i}")
            data = data.read().split("\n")
            self.prompts.extend(data)
            self.groups.extend([i.split(".txt")[0]] * len(data))

    def get_dataset_path(self) -> str:
        """Returns ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return data["T2I-COMPBENCH"]["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealAttributeDatapoint:
        prompt = self.prompts[idx]
        datapoint = RealAttributeDatapoint(
            prompt=prompt,
            condition={
                "class_id": prompt,
            },
            group=[self.groups[idx]],
        )
        return datapoint

    def __len__(self):
        return len(self.prompts)


class PartiPromptsRealAttributeDataset(RealAttributeDataset):
    """
    Creates a RealAttributeDataset from PartiPrompts: https://github.com/google-research/parti

    """

    def __init__(self, img_size=512, crop_scale=0.9):
        self.ANNOT_VAL_PATH = self.get_dataset_path()
        self.data_df = pd.read_csv(self.ANNOT_VAL_PATH, sep="\t")
        self.prompts = list(self.data_df["Prompt"])

    def get_dataset_path(self) -> str:
        """Returns ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return data["PARTI_PROMPTS"]["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealAttributeDatapoint:
        prompt = self.prompts[idx]
        datapoint = RealAttributeDatapoint(
            prompt=prompt,
            condition={
                "class_id": prompt,
            },
        )
        return datapoint

    def __len__(self):
        return len(self.prompts)


class DrawBenchRealAttributeDataset(RealAttributeDataset):
    """
    Creates a RealAttributeDataset from DrawBench: https://github.com/openai/dalle3-eval-samples/blob/main/prompts/drawbench.txt
    """

    def __init__(self):
        self.ANNOT_VAL_PATH = self.get_dataset_path()
        data = open(self.ANNOT_VAL_PATH)
        data = data.read()
        self.prompts = data.split("\n")

    def get_dataset_path(self) -> str:
        """Returns ANNOT_VAL"""
        config_path = os.path.join(os.path.dirname(__file__), "paths.yaml")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return data["DRAWBENCH"]["ANNOT_VAL"]

    def __getitem__(self, idx) -> RealAttributeDatapoint:
        prompt = self.prompts[int(idx / 3)]

        datapoint = RealAttributeDatapoint(
            prompt=prompt,
            condition={
                "class_id": prompt,
            },
        )
        return datapoint

    def __len__(self):
        return len(self.prompts)
