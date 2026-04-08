# Migration Guide: mpcompress -> cofai

## Breaking Change

This release performs a hard rename from `mpcompress` to `cofai`.

- Old import path: `mpcompress.*`
- New import path: `cofai.*`
- No compatibility shim is provided in this release.

## Quick Replacement

Update imports in your downstream projects:

```bash
rg -l "mpcompress" . | xargs sed -i 's/mpcompress/cofai/g'
```

If you also reference the project name in docs/scripts:

```bash
rg -l "MPCompress" . | xargs sed -i 's/MPCompress/CoFAI/g'
```

## Typical Changes

- `from mpcompress.models import ...` -> `from cofai.models import ...`
- `from mpcompress.backbone import ...` -> `from cofai.backbone import ...`
- Config values such as `_target_: mpcompress.xxx` -> `_target_: cofai.xxx`
- Documentation symbol paths:
  - `::: mpcompress.models` -> `::: cofai.models`

## Validation Checklist

- `python -c "import cofai"` succeeds in your environment
- no `mpcompress` residue in code/config/docs:
  - `rg "mpcompress|MPCompress" .`
- examples/scripts use `cofai` imports only

## Checkpoint Loading Note

If any of your custom checkpoints rely on class/module path strings, update those paths to `cofai.*` accordingly in your own loading or conversion scripts.

