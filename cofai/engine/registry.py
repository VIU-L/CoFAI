"""Registry and dynamic class instantiation for the engine layer.

Resolution order for ``get_obj_from_str``:

1. Fully-qualified name: ``"package.module.Class"`` via ``importlib`` (``sys.modules`` first).
2. Short name: global registry (classes registered with ``@register("Name")``).
3. Short name fallback: scan common cofai modules (legacy builder behavior).

Notes:

- No wildcard imports; no implicit module scanning for FQDNs.
- Prefer fully-qualified names in configs for clarity.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Callable, Dict, Optional, Type

__all__ = [
    "Registry",
    "register",
    "list_registered_classes",
    "get_obj_from_str",
    "instantiate_class",
    "instantiate_transforms",
]


class Registry:
    """Simple name -> class registry."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type] = {}

    def register(self, name: str) -> Callable[[Type], Type]:
        """Decorator to register a class under the given name."""

        def decorator(cls: Type) -> Type:
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Optional[Type]:
        """Return the class by name, or None if not found."""
        return self._registry.get(name)

    def list_registered(self) -> list:
        """Return a list of all registered names."""
        return list(self._registry.keys())


_cofai_registry = Registry()


def register(name: str) -> Callable[[Type], Type]:
    """Decorator to register a class in the global registry."""
    return _cofai_registry.register(name)


def list_registered_classes() -> list:
    """List all names currently registered in the global registry."""
    return _cofai_registry.list_registered()


def _resolve_short_name(string: str) -> Type:
    """Resolve a bare class name via registry then legacy module scan."""
    registered_cls = _cofai_registry.get(string)
    if registered_cls is not None:
        return registered_cls

    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError:
        pass

    common_modules = [
        "cofai.datasets",
        "cofai.transforms",
        "cofai.metrics",
        "cofai.heads",
        "cofai.models",
        "cofai.backbone",
        "cofai.engine.builder",
        "cofai.engine.run_eval",
    ]
    import_errors: list[tuple[str, BaseException]] = []
    for module_name in common_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            import_errors.append((module_name, e))
            continue
        if hasattr(module, string):
            try:
                return getattr(module, string)
            except AttributeError:
                pass

    msg = (
        f"'{string}' not found in registry or common modules. Use a fully-qualified name "
        "like 'package.module.Class' or register it via @register(name)."
    )
    if import_errors:
        detail = "\n".join(f"  - {name}: {exc}" for name, exc in import_errors)
        msg += f"\nImport failures while scanning candidate packages:\n{detail}"
        raise ImportError(msg) from import_errors[0][1]
    raise ImportError(msg)


def get_obj_from_str(string: str, reload: bool = False) -> Type:
    """Resolve a class object from string.

    Supported formats:

    1. Fully-qualified: ``"package.module.Class"`` (preferred).
    2. Short name: resolved from the global registry, then from a fixed list of cofai modules.
    """
    if "." in string:
        module_name, class_name = string.rsplit(".", 1)
        module_imp = sys.modules.get(module_name)
        if module_imp is None:
            module_imp = importlib.import_module(module_name)

        if module_imp is None:
            raise ImportError(f"Module '{module_name}' cannot be imported")
        if reload:
            importlib.reload(module_imp)
        return getattr(module_imp, class_name)
    return _resolve_short_name(string)


def instantiate_class(config: Dict[str, Any], **kwargs) -> Any:
    """Instantiate a class from a config dict with key ``type``."""
    if not isinstance(config, dict):
        try:
            from omegaconf import OmegaConf  # type: ignore

            config = OmegaConf.to_container(config, resolve=True)  # type: ignore
        except Exception:
            config = dict(config)
    config = dict(config)
    if "type" not in config:
        raise KeyError(f"Expected key 'type' to instantiate. Got config: {config}")

    cls_name = config.pop("type")
    try:
        cls = get_obj_from_str(cls_name)
        return cls(**config, **kwargs)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import class '{cls_name}': {e}")


def instantiate_transforms(config: Dict[str, Any], **kwargs) -> Any:
    """Instantiate a transform pipeline; optional nested ``transforms`` list."""
    if "type" not in config:
        raise KeyError("Expected key 'type' to instantiate.")

    config = dict(config)
    cls = config.pop("type")
    transform_cls = get_obj_from_str(cls)

    if "transforms" in config:
        items = config.pop("transforms")
        transforms = [instantiate_class(t) for t in items]
        return transform_cls(transforms=transforms, **config, **kwargs)
    return transform_cls(**config, **kwargs)
