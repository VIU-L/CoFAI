# AI M2510 PCA-align

PCA-align compresses Qwen3-VL-8B-Instruct visual features at slot 9. It combines
learned gains in the original and PCA channel spaces with lightweight depthwise
analysis/synthesis transforms, a hyperprior and eight space-channel context groups.
The proposal configuration uses `h_dim=1152`, `y_dim=256`, `z_dim=48`, with 65
learned inference QPs from 0 through 64 in one checkpoint.

Source: **AI M2510: 2026年8月,《基于PCA通道加权的Qwen3-VL特征压缩》**,
revision `AI M2510_v3.docx`, Shanghai Jiao Tong University.

The implementation reuses CoFAI's Qwen backbone, feature codec model, MMStar
dataset/metric, checkpoint loader and evaluation engine. Tokens use `(1,H*W,C)`
with `token_res=(H,W)`; evaluation requires batch size 1.

## Resources

Follow the [Qwen3-VL CTC guide](../ctc/README-QWEN3VL.md) for the MMStar dataset,
Qwen3-VL-8B-Instruct backbone and `PROJECT_ROOT` setup. Then download the codec:

```bash
poetry run cofai-download examples/pca_align/pca-align.manifest.txt
```

The checkpoint is saved under `${PROJECT_ROOT}/weights/pca_align/`; no extraction
is needed. It contains the author's epoch-5 codec parameters, learned QP gains,
PCA row-basis, mean and entropy tables under `qwen_codec.*`. The original training
state was exported after the author's forced quantile/CDF update, preserving
inference behavior while omitting optimizer/RNG state and unused training modules.
Separate `U-fix.npy` and `mean-fix.npy` files are not required.

The plan uses the common engine loader with `weights_only: true`. It sets
`strict: false` because the pretrained backbone is loaded separately; all codec
parameters and buffers are present in the inference checkpoint.

## MMStar evaluation

The default evaluates all 1500 questions at QPs 0, 8, 16, 32, 48 and 64 using real
entropy coding and the existing CTC backbone, preprocessing and generation settings:

```bash
poetry run cofai-eval examples/pca_align/plan/mmstar__qwen3vl-8b-ins-slot9__pca-align__vqa.yaml
```

For one quality point:

```bash
poetry run cofai-eval examples/pca_align/plan/mmstar__qwen3vl-8b-ins-slot9__pca-align__vqa.yaml \
  args.multi_run=false args.quality=32
```

Multi-run writes per-QP results and `summary.json` under `logs/<plan-name>/`.
`args.real=true` counts the actual `y` and `z` payload bytes. `args.real=false`
uses likelihood estimates; the author's forward and real-coding paths can also
produce different reconstructed features and task predictions.

Use the existing CTC plan for the uncompressed reference:

```bash
poetry run cofai-eval conf/plan/mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa.yaml
```
