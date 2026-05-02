"""Depth estimation meters."""

from __future__ import annotations

import numpy as np
import torch


class DepthEstimationMeter:
    def __init__(self, max_depth: float | None = None, min_depth: float | None = None):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.abs_rel = 0.0
        self.sq_rel = 0.0

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred, gt = pred.squeeze(), gt.squeeze()
        if self.max_depth is None or self.min_depth is None:
            raise ValueError("DepthEstimationMeter requires max_depth and min_depth")

        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)
        self.n_valid += float(mask.float().sum().item())

        gt = gt.clone()
        pred = pred.clone()
        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += float(log_rmse_tmp.sum().item())

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += float(rmse_tmp.sum().item())

        self.abs_rel += float((torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item())
        self.sq_rel += float((((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item())

    def compute(self) -> dict[str, float]:
        if self.n_valid <= 0:
            return {"rmse": 0.0, "log_rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0}
        return {
            "rmse": float(np.sqrt(self.total_rmses / self.n_valid)),
            "log_rmse": float(np.sqrt(self.total_log_rmses / self.n_valid)),
            "abs_rel": float(self.abs_rel / self.n_valid),
            "sq_rel": float(self.sq_rel / self.n_valid),
        }


class DepthEstimationMeterLegacy:
    def __init__(self, ignore_index: int = 255):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.ignore_index = int(ignore_index)
        self.abs_rel = 0.0
        self.sq_rel = 0.0

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred, gt = pred.squeeze(), gt.squeeze()
        mask = (gt != self.ignore_index).bool()
        self.n_valid += float(mask.float().sum().item())

        gt = gt.clone()
        pred = pred.clone()
        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += float(log_rmse_tmp.sum().item())

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += float(rmse_tmp.sum().item())

        self.abs_rel += float((torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item())
        self.sq_rel += float((((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item())

    def compute(self) -> dict[str, float]:
        if self.n_valid <= 0:
            return {"rmse": 0.0, "log_rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0}
        return {
            "rmse": float(np.sqrt(self.total_rmses / self.n_valid)),
            "log_rmse": float(np.sqrt(self.total_log_rmses / self.n_valid)),
            "abs_rel": float(self.abs_rel / self.n_valid),
            "sq_rel": float(self.sq_rel / self.n_valid),
        }

