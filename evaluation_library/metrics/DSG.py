# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
import torch
from torchmetrics import Metric
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor


class DSG(Metric):
    def __init__(
        self,
        model_name_or_path: str = "Salesforce/instructblip-flan-t5-xl",
        max_new_tokens: int = 100,
        **kwargs: Any,
    ):
        """
        Args:
            model_name_or_path: path to a HuggingFace Transformers vision-language model for VQA
            max_new_tokens: number of new tokens that can be generated during inference
        """

        super().__init__(**kwargs)

        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16
        ).eval()
        self.processor = InstructBlipProcessor.from_pretrained(model_name_or_path)

        self.max_new_tokens = max_new_tokens
        self.add_state("scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, generated_images_batch: torch.Tensor, real_attribute_datapoint_batch: dict[Any]) -> None:
        num_generated_images = generated_images_batch.shape[0]
        batch_questions = real_attribute_datapoint_batch["dsg_questions"]
        batch_children = real_attribute_datapoint_batch["dsg_children"]
        batch_parents = real_attribute_datapoint_batch["dsg_parents"]

        assert batch_questions is not None and batch_parents is not None
        assert len(batch_questions) == len(batch_children) == len(batch_parents)

        for idx in range(num_generated_images):
            questions = batch_questions[idx]
            children = batch_children[idx]
            dependencies = batch_parents[idx]
            assert questions is not None and len(questions) > 0, "No questions provided"

            generated_image = generated_images_batch[idx].unsqueeze(0)
            generated_image_repeated = generated_image.expand(len(questions), -1, -1, -1)

            formatted_questions = []
            for q in questions:
                formatted_questions.append(self._format_question_for_vqa_input(q))
            processed_inputs = self.processor(
                images=generated_image_repeated,
                text=formatted_questions,
                padding=True,
                truncation=True,
                return_tensors="pt",
                do_rescale=False,
            ).to(self.model.device)

            # -- run VQA to generate predicted answers
            generated_ids = self.model.generate(
                **processed_inputs,
                max_new_tokens=self.max_new_tokens,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            # -- score predictions
            score = self._score_single_example(generated_text, children, dependencies)
            self.scores += score
            self.n_samples += 1

    def _format_question_for_vqa_input(self, question: str) -> str:
        question = f"{question} Answer yes or no."
        return question

    def _score_single_example(
        self, predicted_answers: list[str], children: dict[str, list[str]], parents: dict[str, list[str]]
    ) -> int:
        """
        This function scores the model's predictions.
        We have a set of questions, predictions and ground truth answers.
        For each question, we have a child and parent(s), which are the dependencies.
        For DSG, a question is correct *if and only if* the prediction matches the target
        and the parent questions are all correct as well.
        """

        questionid2scores = {}
        for i, (pred) in enumerate(predicted_answers):
            # question ids are 1-indexed, so add 1 to our current index
            questionid2scores[i + 1] = float(pred.strip().lower() == "yes")

        # convert dependencies to dict
        assert len(children) == len(parents)
        for child_id, parent_ids in zip(children, parents, strict=False):
            parent_ids = [int(p) for p in parent_ids.split(",")]

            # first check if any parent questions were wrong
            any_parent_is_wrong = False
            for p in parent_ids:
                if p == 0:  # case where there's no dependency
                    continue

                if questionid2scores[p] == 0:
                    any_parent_is_wrong = True
                    break

            # now zero-out score if any parent was wrong
            if any_parent_is_wrong:
                questionid2scores[child_id] = 0

        avg_score = np.mean(list(questionid2scores.values()))
        return avg_score.item()

    def compute(self) -> dict:
        return {"dsg": self.scores / self.n_samples}
