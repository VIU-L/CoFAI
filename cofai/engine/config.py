from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf
from pathlib import Path


def _ensure_hydra_runtime_resolver() -> None:
    # Many configs use `${hydra:runtime.cwd}` as a fallback, but HydraConfig is not
    # initialized when using Hydra's compose API (e.g. in unit tests). Provide a
    # resolver that falls back to the current working directory in that case.
    def _hydra_resolver(path: str) -> Any:
        try:
            from hydra.core.hydra_config import HydraConfig

            return OmegaConf.select(HydraConfig.get(), path)
        except Exception:
            if str(path) == "runtime.cwd":
                return str(Path.cwd())
            raise

    OmegaConf.register_new_resolver("hydra", _hydra_resolver, replace=True)


def resolve_plan(cfg: Any) -> tuple[str, Any]:
    _ensure_hydra_runtime_resolver()
    # Strict: root cfg is the plan.
    plan_cfg = cfg
    plan_cfg = OmegaConf.create(OmegaConf.to_container(plan_cfg, resolve=True))
    plan_key = str(getattr(plan_cfg, "name", None) or "")
    if not plan_key:
        raise ValueError("plan.name is required")
    return plan_key, plan_cfg


def validate_plan(plan_cfg: Any, *, plan_key: str) -> None:
    if plan_cfg is None:
        raise ValueError("plan is required")
    # Strict: no implicit fallback for required plan fields.
    if getattr(plan_cfg, "name", None) is None:
        raise ValueError("plan.name is required")
    if getattr(plan_cfg, "description", None) is None:
        raise ValueError("plan.description is required")
    if getattr(plan_cfg, "dataset", None) is None:
        raise ValueError("plan.dataset is required")
    if getattr(plan_cfg, "task_configs", None) is None:
        raise ValueError("plan.task_configs is required")
    if getattr(plan_cfg, "model", None) is None:
        raise ValueError("plan.model is required")

    ds = getattr(plan_cfg, "dataset", None)
    ds_dict = OmegaConf.to_container(ds, resolve=True) if not isinstance(ds, dict) else dict(ds)
    if not ds_dict.get("type"):
        raise ValueError("plan.dataset.type is required")

    tcs = getattr(plan_cfg, "task_configs", None)
    tcs_list = list(tcs) if not isinstance(tcs, list) else tcs
    for d in tcs_list:
        dd = OmegaConf.to_container(d, resolve=True) if not isinstance(d, dict) else dict(d)
        label = dd.get("label")
        if not label:
            raise ValueError("task_configs.*.label is required")
        meter_cfg = dd.get("meter")
        if not isinstance(meter_cfg, dict) or "type" not in meter_cfg:
            raise ValueError(f"task_configs.{label}.meter must be dict with type")


