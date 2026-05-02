"""Engine-level skeleton (v0: eval-only).

`cofai.engine` is the long-term home for runner/config/builder contracts.
CLI 入口迁移到项目根目录 `test.py`。
"""

from __future__ import annotations

from cofai.engine.checkpoint import load_checkpoint_into_model

__all__ = []

"""
Engine skeleton (minimal, eval-first).

This package is intentionally lightweight: it provides a small set of primitives
(`registry`, `config`, `logging`, `loop`, `runner`) that can be evolved toward a
more general train/val/test engine over time.
"""

__all__: list[str] = []

