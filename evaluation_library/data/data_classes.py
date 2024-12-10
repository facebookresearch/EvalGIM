# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, NamedTuple

import torch

Condition = Any


class Datapoint(NamedTuple):
    pixel_values: torch.Tensor
    condition: Condition


class RealImageDatapoint(NamedTuple):
    image: torch.Tensor
    class_label: str | None = None
    group: list[str] | None = None


class RealAttributeDatapoint(NamedTuple):
    prompt: str
    condition: Condition
    class_label: str | None = None
    group: list[str] | None = None
    dsg_questions: list[str] | None = None
    dsg_children: list[str] | None = None
    dsg_parents: Condition | None = None


class GenImageDatapoint(NamedTuple):
    image: torch.Tensor
    prompt: str
    condition: Condition
    class_label: str | None = None
    group: list[str] | None = None
    dsg_questions: list[str] | None = None
    dsg_children: list[str] | None = None
    dsg_parents: Condition | None = None
