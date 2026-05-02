"""Surface normals estimation meters."""

from __future__ import annotations

import torch


def _normalize_tensor(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    norm = torch.norm(input_tensor, p="fro", dim=dim, keepdim=True)
    zero_mask = norm == 0
    norm = norm.clone()
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


class SurfaceNormalsEstimationMeter:
    def __init__(self, ignore_index: int = 255):
        self.sum_deg_diff = 0.0
        self.total = 0
        self.ignore_index = int(ignore_index)

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
        pred = 2 * pred / 255 - 1  # reverse post-processing
        valid_mask = (gt != self.ignore_index).all(dim=1)

        pred = _normalize_tensor(pred, dim=1)
        gt = _normalize_tensor(gt, dim=1)
        deg_diff = torch.rad2deg(
            2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1))
        )
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += float(torch.sum(deg_diff).cpu().item())
        self.total += int(deg_diff.numel())

    def compute(self) -> dict[str, float]:
        if self.total <= 0:
            return {"mean": 0.0}
        return {"mean": float(self.sum_deg_diff) / float(self.total)}

