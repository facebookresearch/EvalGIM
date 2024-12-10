# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s

import argparse
import hashlib
import json
import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import submitit
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from .data.data_classes import RealAttributeDatapoint
from .data.dataset_evaluation import get_attribute_datasets_evaluation
from .data.real_datasets import real_attribute_dataset_collate

torch.backends.cuda.matmul.allow_tf32 = True


DATASET_SIZE = {
    "cc12m_validation_dataset": 14749,
    "coco_txt_dataset": 40504,
    "imagenet_validation_dataset": 50000,
    "geode_dataset": 29160,
    "tifa160_dataset": 160,
    "evaluation_library.data.real_datasets_balanced.COCO15K": 14749,
    "evaluation_library.data.real_datasets_balanced.ImageNetValidation15K": 14749,
    "evaluation_library.data.real_datasets_balanced.CC12MValidation15K": 14749,
    "evaluation_library.data.real_datasets_balanced.GeoDE15K": 14580,
}


class ImageGenerationPipeline:
    @abstractmethod
    def sample(
        self, batch: RealAttributeDatapoint, seed: int = None, num_images_per_prompt: int = 1
    ) -> torch.Tensor | list[Image.Image]:
        raise NotImplementedError()


class DiffusersPipeline(ImageGenerationPipeline):
    def __init__(self, model_id: str, cfg_scale: float, device: int | str = "balanced"):
        self.model_id = model_id
        self.accelerator = Accelerator(mixed_precision="no")
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map=device
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.guidance_scale = cfg_scale

    @property
    def num_channels_latents(self):
        if hasattr(self.pipeline, "unet"):
            return self.pipeline.unet.config.in_channels
        elif hasattr(self.pipeline, "transformer"):
            return self.pipeline.transformer.config.in_channels
        else:
            raise Exception("could not extract num_channels_latents")

    @property
    def height(self):
        if hasattr(self.pipeline, "unet"):
            return self.pipeline.unet.config.sample_size
        elif hasattr(self.pipeline, "transformer"):
            return self.pipeline.default_sample_size
        else:
            raise Exception("could not extract height")

    @property
    def width(self):
        if hasattr(self.pipeline, "unet"):
            return self.pipeline.unet.config.sample_size
        elif hasattr(self.pipeline, "transformer"):
            return self.pipeline.default_sample_size
        else:
            raise Exception("could not extract width")

    def prepare_latents(self, batch_size: int, seed: int, num_images_per_prompt: int) -> torch.Tensor:
        """
        Create random latents which are different among the images of a prompt but equal across prompts.
        """
        generator = torch.Generator(self.pipeline.device).manual_seed(seed) if seed is not None else None
        return torch.randn(
            num_images_per_prompt,
            self.num_channels_latents,
            self.height,
            self.width,
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=generator,
        ).repeat(batch_size, 1, 1, 1)

    @torch.inference_mode()
    def sample(
        self, batch: dict[str, Any], seed: int = None, num_images_per_prompt: int = 1
    ) -> list[Image.Image]:
        prompts = batch["prompt"]
        if num_images_per_prompt > 1:
            prompts = np.array(prompts).repeat(num_images_per_prompt, axis=0).tolist()
        gen_kwargs = {"guidance_scale": self.guidance_scale}
        images = self.pipeline(
            prompt=prompts,
            # uncomment to create random latents which are different among the images of a prompt but equal
            # across prompts
            # latents=self.prepare_latents(len(prompts), seed, num_images_per_prompt),
            **gen_kwargs,
        ).images
        if num_images_per_prompt > 1:
            images = [
                images[i : i + num_images_per_prompt] for i in range(0, len(images), num_images_per_prompt)
            ]
        return images


def main(args: argparse.Namespace, data_file: Path):
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

    pipeline = DiffusersPipeline(args.model_id, args.cfg_scale)
    print(args.cfg_scale)

    set_seed(seed=args.seed + task_id, device_specific=True)

    with open(data_file) as f:
        dataset = json.load(f)
    dataset = dataset[task_id::num_tasks]

    def collate_fn(original_batch):
        for i in range(len(original_batch)):
            original_batch[i] = SimpleNamespace(**original_batch[i])
        new_batch = real_attribute_dataset_collate(original_batch)
        new_batch["image_path"] = [item.image_path for item in original_batch]
        return new_batch

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.debug else 4,
        prefetch_factor=None,
        drop_last=False,
        collate_fn=collate_fn,
    )

    for batch in tqdm(dataloader, desc="Generating images"):
        try:
            images = pipeline.sample(batch)
            for i, image in enumerate(images):
                if isinstance(image, torch.Tensor):
                    image = to_pil_image(image)
                image.save(batch["image_path"][i])
        except KeyboardInterrupt:
            break


def get_datapoint_id(datapoint: RealAttributeDatapoint) -> str:
    h = hashlib.sha256()
    h.update(datapoint.prompt.encode("utf-8"))
    h.update(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f").encode("utf-8")
    )  # adding timestamp since prompts might not be unique
    return h.hexdigest()


def args_to_str(args: argparse.Namespace) -> str:
    model_id = args.model_id.replace("/", "--")
    return f"{model_id}__{args.dataset_name}__cfg{args.cfg_scale}"


def launch(args: argparse.Namespace):
    output_path = Path(args.output_path) / args_to_str(args)
    images_path = output_path / "images"
    if images_path.exists():
        if len([name for name in images_path.iterdir()]) == DATASET_SIZE[args.dataset_name]:
            print(f"SKIPPING {str(output_path)} as all images have been generated!")
            return
        else:
            raise Exception(
                f"{str(images_path)} is not empty and some images are missing. "
                "Please delete or rename the directory to regenerate all images."
            )
    images_path.mkdir(parents=True, exist_ok=True)

    dataset = get_attribute_datasets_evaluation(dataset_name=args.dataset_name).dataset
    dataset = Subset(dataset, indices=range(args.num_samples))

    ds = []
    for datapoint in tqdm(dataset, desc="Converting dataset"):
        id = get_datapoint_id(datapoint)
        dp = datapoint._asdict()
        dp["image_path"] = str((images_path / f"{id}.png").resolve())
        ds.append(dp)

    data_file = output_path / "index.json"
    with open(data_file, "w") as f:
        json.dump(ds, f, indent=4)

    if args.debug or args.local:
        main(args, data_file)
    else:
        executor = submitit.AutoExecutor(folder=output_path / "submitit_logs")

        slurm_params = {  
            "slurm_partition": "", #TODO
            "slurm_constraint": "", #TODO
        }
        executor.update_parameters(
            **slurm_params,
            slurm_nodes=1,
            slurm_mem="16G",
            slurm_ntasks_per_node=1,
            slurm_gpus_per_node=1,
            slurm_cpus_per_task=8,
            slurm_time=3 * 24 * 60,
            slurm_name="generate",
        )

        jobs = []
        with executor.batch():
            for _ in range(args.num_jobs):
                job = executor.submit(main, args, data_file)
                jobs.append(job)
        print(f"Submitted {len(jobs)} jobs")


def sweep(args: argparse.Namespace):
    for dataset_n in [args.dataset_name]:
        args.dataset_name = dataset_n
        for cfg_scale in [2.0, 5.0, 7.5]:
            args.cfg_scale = cfg_scale
            launch(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="coco_txt_dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--output_path", type=str, default="./projects/generated")
    parser.add_argument("--num_jobs", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local", action="store_true", help="runs on a local gpu")
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        sweep(args)
    else:
        launch(args)
