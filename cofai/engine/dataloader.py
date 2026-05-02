"""Build DataLoader that yields `EvalBatch`."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cofai.engine.schema import EvalBatch

__all__ = ["build_dataloader", "collate_fn"]


def collate_fn(batch: List[Dict[str, Any]]) -> EvalBatch:
    """Collate eval samples: ``torch.stack`` images → ``inputs[\"img\"]``, list ``samples`` (no ``img``).

    Requires identical spatial size per item (plain ``stack``).
    """

    if not batch:
        raise ValueError("Empty batch")
    xs: List[torch.Tensor] = []
    samples: List[Dict[str, Any]] = []
    for item in batch:
        xs.append(_to_chw(item["img"]))
        s = dict(item)
        s.pop("img")
        samples.append(s)
    return EvalBatch(inputs={"img": torch.stack(xs, dim=0)}, samples=samples)


def build_dataloader(cfg: Any, dataset: Any) -> DataLoader:
    """Return DataLoader producing EvalBatch."""

    raw = OmegaConf.to_container(OmegaConf.select(cfg, "dataloader", default={}) or {}, resolve=True)
    kw = raw if isinstance(raw, dict) else {}
    kw.pop("dataset", None)
    kw.pop("collate_fn", None)
    return DataLoader(dataset, collate_fn=collate_fn, **kw)


def _to_chw(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        # PIL or ndarray: use ToTensor lazily to avoid importing torchvision here
        from torchvision.transforms import ToTensor

        t = ToTensor()(x)
    if t.dim() == 4 and t.size(0) == 1:
        t = t[0]
    if t.dim() != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape={tuple(t.shape)}")
    return t
