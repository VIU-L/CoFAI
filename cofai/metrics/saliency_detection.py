"""Saliency detection meters."""

from __future__ import annotations

import torch
from torch import nn


class SaliencyDetectionMeter:
    def __init__(self, ignore_index: int = 255, threshold_step: float | None = None, beta_squared: float = 1):
        if threshold_step is None:
            raise ValueError("SaliencyDetectionMeter requires threshold_step")
        self.ignore_index = int(ignore_index)
        self.beta_squared = float(beta_squared)
        self.thresholds = torch.arange(float(threshold_step), 1, float(threshold_step))
        self.true_positives = torch.zeros(len(self.thresholds))
        self.predicted_positives = torch.zeros(len(self.thresholds))
        self.actual_positives = torch.zeros(len(self.thresholds))

    @torch.no_grad()
    def update(self, preds, target) -> None:
        preds = preds.float() / 255.0

        if target.shape[1] == 1:
            target = target.squeeze(1)

        if len(preds.shape) == len(target.shape) + 1:
            if preds.shape[1] != 2:
                raise ValueError("Expected 2-class logits in preds when preds has extra channel dim")
            preds = nn.functional.softmax(preds, dim=1)[:, 1, :, :]
        else:
            preds = torch.sigmoid(preds)

        if len(preds.shape) != len(target.shape):
            raise ValueError(
                "preds and target must have same number of dimensions, or preds one more"
            )

        valid_mask = target != self.ignore_index

        for idx, thresh in enumerate(self.thresholds):
            f_preds = (preds >= thresh).long()
            f_target = target.long()

            f_preds = torch.masked_select(f_preds, valid_mask)
            f_target = torch.masked_select(f_target, valid_mask)

            self.true_positives[idx] += torch.sum(f_preds * f_target).cpu()
            self.predicted_positives[idx] += torch.sum(f_preds).cpu()
            self.actual_positives[idx] += torch.sum(f_target).cpu()

    def compute(self) -> dict[str, float]:
        precision = self.true_positives.float() / self.predicted_positives
        recall = self.true_positives.float() / self.actual_positives

        num = (1 + self.beta_squared) * precision * recall
        denom = self.beta_squared * precision + recall
        fscore = num / denom
        fscore[fscore != fscore] = 0

        return {"maxF": float(fscore.max().item() * 100.0)}

