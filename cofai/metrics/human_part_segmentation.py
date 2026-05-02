"""Human part segmentation meters."""

from __future__ import annotations

import numpy as np
import torch


PART_CATEGORY_NAMES = ["background", "head", "torso", "uarm", "larm", "uleg", "lleg"]


class HumanPartSegmentationMeter:
    def __init__(self, database: str, ignore_idx: int = 255):
        if database != "PASCALContext":
            raise NotImplementedError
        self.database = database
        self.cat_names = PART_CATEGORY_NAMES
        self.n_parts = 6
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)
        self.ignore_idx = int(ignore_idx)

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = gt != self.ignore_idx

        for i_part in range(self.n_parts + 1):
            tmp_gt = gt == i_part
            tmp_pred = pred == i_part
            self.tp[i_part] += int(torch.sum(tmp_gt & tmp_pred & valid).item())
            self.fp[i_part] += int(torch.sum(~tmp_gt & tmp_pred & valid).item())
            self.fn[i_part] += int(torch.sum(tmp_gt & ~tmp_pred & valid).item())

    def reset(self) -> None:
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)

    def compute(self) -> dict[str, float]:
        jac = [0.0] * (self.n_parts + 1)
        for i_part in range(0, self.n_parts + 1):
            denom = float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part])
            jac[i_part] = float(self.tp[i_part]) / max(denom, 1e-8)
        return {"mIoU": float(np.mean(jac) * 100.0)}

