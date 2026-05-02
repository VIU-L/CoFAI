"""Edge detection meters."""

from __future__ import annotations

import torch
from cofai.losses.loss_functions import BalancedBinaryCrossEntropyLoss


class EdgeDetectionMeter:
    def __init__(self, pos_weight: float, ignore_index: int):
        self.loss = 0.0
        self.n = 0
        self.loss_function = BalancedBinaryCrossEntropyLoss(
            pos_weight=pos_weight, ignore_index=ignore_index
        )
        self.ignore_index = int(ignore_index)

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred = pred.squeeze()
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)

        gt = gt.squeeze()
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)

        valid_mask = gt != self.ignore_index
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        pred = pred.float().squeeze() / 255.0
        loss = float(self.loss_function(pred, gt).item())
        numel = int(gt.numel())
        self.n += numel
        self.loss += numel * loss

    def reset(self) -> None:
        self.loss = 0.0
        self.n = 0

    def compute(self) -> dict[str, float]:
        if self.n <= 0:
            return {"loss": 0.0}
        return {"loss": float(self.loss) / float(self.n)}

