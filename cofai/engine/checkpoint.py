from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from cofai.utils.utils import rename_key_by_rules


@dataclass(frozen=True)
class LoadConfig:
    path: str
    strict: bool = False
    rules: Optional[list] = None
    # Common checkpoint containers: "state_dict" (DDP/Lightning style) or raw state dict.
    state_dict_key: str = "state_dict"
    weights_only: bool = True
    map_location: str = "cpu"


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        from omegaconf import OmegaConf

        out = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        return dict(out) if isinstance(out, dict) else {}
    except Exception:
        try:
            return dict(cfg)
        except Exception:
            return {}


def _extract_state_dict(ckpt: Any, *, state_dict_key: str) -> Mapping[str, Any]:
    if isinstance(ckpt, Mapping):
        if state_dict_key in ckpt and isinstance(ckpt[state_dict_key], Mapping):
            return ckpt[state_dict_key]  # type: ignore[return-value]
        # Common alternates
        for k in ("model", "model_state_dict", "net", "state"):
            if k in ckpt and isinstance(ckpt[k], Mapping):
                return ckpt[k]  # type: ignore[return-value]
    if isinstance(ckpt, Mapping):
        return ckpt  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint type for state_dict: {type(ckpt)!r}")


def _apply_rename_rules(state_dict: Mapping[str, Any], rules: Optional[list]) -> Dict[str, Any]:
    if not rules:
        return dict(state_dict)
    out: Dict[str, Any] = {}
    for original_key in sorted(state_dict.keys()):
        new_key = rename_key_by_rules(str(original_key), rules)
        if new_key != "":
            out[new_key] = state_dict[original_key]
    return out


def load_checkpoint_into_model(model: torch.nn.Module, load_cfg: Any) -> None:
    """Load a checkpoint into `model` using a flexible config dict.

    Recommended config location: `plan.load` (engine-level).
    Supported fields:
    - path (str, required)
    - strict (bool, default false)
    - rules (list[[type, pattern, replacement]], optional)  # rename_key_by_rules
    - state_dict_key (str, default "state_dict")
    - weights_only (bool, default true)
    - map_location (str, default "cpu")
    """

    cfgd = _as_dict(load_cfg)
    path = str(cfgd.get("path") or "").strip()
    if not path:
        return

    lc = LoadConfig(
        path=path,
        strict=bool(cfgd.get("strict", False)),
        rules=cfgd.get("rules"),
        state_dict_key=str(cfgd.get("state_dict_key", "state_dict")),
        weights_only=bool(cfgd.get("weights_only", True)),
        map_location=str(cfgd.get("map_location", "cpu")),
    )

    ckpt = torch.load(lc.path, map_location=lc.map_location, weights_only=lc.weights_only)
    state_dict = _extract_state_dict(ckpt, state_dict_key=lc.state_dict_key)
    state_dict = _apply_rename_rules(state_dict, lc.rules)
    model.load_state_dict(state_dict, strict=lc.strict)

