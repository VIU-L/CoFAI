# VQFC

当前集成了论文 **“Transform-Free Feature Coding via Entropy-Constrained Vector Quantization” (VQFC)** 中的基线实验（基于 DINOv2 的特征压缩）。

## cofai-eval 评测

**cofai-eval** 是基于 Hydra 配置的统一评测入口，通过 **`poetry run cofai-eval`** 命令串联数据集、模型与指标（详见 [docs/engine.md](../../docs/engine.md)）。VQFC 对应 plan 位于 `conf/plan/`，可与下方脚本评测对照使用。

```bash
# VOC2012 sel20 语义分割
CUDA_VISIBLE_DEVICES=0 poetry run cofai-eval \
  conf/plan/voc2012-sel20--VQFC-dinov2-giant-seg.yaml \
  args.real=true args.multi_run=true

# ImageNet sel100 分类
CUDA_VISIBLE_DEVICES=0 poetry run cofai-eval \
  conf/plan/imagenet-sel100--VQFC-dinov2-giant-cls.yaml \
  args.real=true args.multi_run=true
```

> 注意：对于语义分割任务，这里的 plan 暂不支持 VQFC 的滑动窗口推理，而是使用 Resize 到固定尺寸的预处理，因此测下来结果会比较差，这里仅作为参考。为获得更好的结果，可以使用下边的 run_eval_slide 进行推理测试。

## 集成测试

请按照项目根目录 `README.md` 的说明完成环境配置：安装 Poetry、创建 `.env` 文件（含 `PROJECT_ROOT` 变量），并准备测试数据与权重。

其中 segmentation 任务使用了滑动窗口推理，classification 任务使用了整图推理。

**VQFC 推理示例**：

```bash 
CUDA_VISIBLE_DEVICES=0 python examples/vqfc/run_eval_slide.py \
    --config examples/vqfc/config/eval_base.yaml examples/vqfc/config/dino_orig_slide_giant_seg_vqfc_64.yaml \
    --preset voc2012_sel20_seg \
    --head voc2012_seg_giant_last1 \
    --quality 0 \
    --cuda --real \
    --output_dir logs/voc2012_sel20_seg_dino_orig_slide_giant_seg_vqfc_64

CUDA_VISIBLE_DEVICES=0 python examples/vqfc/run_eval_slide.py \
      --config examples/vqfc/config/eval_base.yaml examples/vqfc/config/dino_orig_slide_giant_cls_vqfc_512.yaml \
      --preset imagenet_sel100_cls \
      --head imagenet_cls_giant_last1 \
      --quality 0 \
      --cuda --real \
      --output_dir logs/imagenet_sel100_cls_dino_orig_slide_giant_cls_vqfc_512
```

