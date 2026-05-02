from __future__ import annotations

from typing import Any, List

import torch
from omegaconf import OmegaConf

from cofai.engine.evaluator import MultiTaskEvaluator, TaskConfig, infer_kind
from cofai.engine.registry import instantiate_class
from cofai.engine.checkpoint import load_checkpoint_into_model

__all__ = [
    "build_transforms",
    "build_dataset",
    "build_tasks",
    "build_model",
]


def build_transforms(transforms_cfg: Any):
    """Build eval-time transforms from a list of config dicts."""

    if transforms_cfg is None:
        return None
    transforms_list = list(transforms_cfg) if not isinstance(transforms_cfg, list) else transforms_cfg
    from torchvision.transforms import Compose

    transforms = [instantiate_class(transform_cfg) for transform_cfg in transforms_list]
    return Compose(transforms)


def build_dataset(*, plan_cfg: Any) -> Any:
    dataset_cfg = plan_cfg.dataset
    dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)  # type: ignore

    test_tf = getattr(plan_cfg, "test_transforms", None)
    if test_tf is not None:
        dataset_cfg["transform"] = build_transforms(test_tf)
    elif "transform" in dataset_cfg:
        raise ValueError(
            "Unsupported `plan.dataset.transform` in eval config. "
            "Use `plan.test_transforms: List[dict]` instead."
        )

    ds_type = dataset_cfg.pop("type")
    return instantiate_class({"type": ds_type, **dataset_cfg})


def build_tasks(*, plan_cfg: Any) -> tuple[List[str], MultiTaskEvaluator, List[dict]]:
    task_cfg_dicts = plan_cfg.task_configs
    if not isinstance(task_cfg_dicts, list):
        task_cfg_dicts = list(task_cfg_dicts)

    task_configs: List[TaskConfig] = []
    tasks: List[str] = []
    # NOTE: task_specs are intentionally plain dicts (not TaskSpec objects).
    # This keeps model-side logic simple and avoids mixed dict/object runtime shapes.
    task_specs: List[dict] = []
    for task_cfg in task_cfg_dicts:
        task_cfg_dict = OmegaConf.to_container(task_cfg, resolve=True) if not isinstance(task_cfg, dict) else dict(task_cfg)
        label = str(task_cfg_dict.get("label"))
        kind = infer_kind(label=label, kind=task_cfg_dict.get("kind"))
        task_specs.append({"label": label, "kind": kind, "params": task_cfg_dict.get("params")})
        meter_cfg = task_cfg_dict.get("meter")
        meter = instantiate_class(meter_cfg)
        task_configs.append(TaskConfig(label=label, meter=meter))
        tasks.append(label)

    return tasks, MultiTaskEvaluator(task_configs), task_specs


def build_model(*, cfg: Any, plan_cfg: Any, tasks: List[str], task_specs: List[dict], device: torch.device) -> Any:
    model_cfg = plan_cfg.model
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True) if not isinstance(model_cfg, dict) else dict(model_cfg)

    try:
        model = instantiate_class(
            model_cfg,
            eval_tasks=list(tasks),
            task_specs=list(task_specs),
            device=str(device),
        ).to(device)
    except TypeError:
        try:
            model = instantiate_class(model_cfg, task_specs=list(task_specs), device=str(device)).to(device)
        except TypeError:
            model = instantiate_class(model_cfg).to(device)

    # Recommended location: plan.load (per-eval run).
    plan_load = getattr(plan_cfg, "load", None)
    if plan_load is not None:
        load_checkpoint_into_model(model, plan_load)

    model.eval()
    if hasattr(model, "update"):
        model.update()
    return model