# VQFC

当前集成了论文 **“Transform-Free Feature Coding via Entropy-Constrained Vector Quantization” (VQFC)** 中的基线实验（基于 DINOv2 的特征压缩）。

## 集成测试

请按照项目根目录 `README.md` 的说明完成环境配置：安装 Poetry、创建 `.env` 文件（含 `PROJECT_ROOT` 变量），并准备测试数据与权重。

其中 segmentation 任务使用了滑动窗口推理，classification 任务使用了整图推理。

**VQFC 推理示例**：

```bash 
CUDA_VISIBLE_DEVICES=0 python examples/vqfc/run_eval_slide.py \
    --config examples/vqfc/config/eval_base.yaml examples/vqfc/config/dino_orig_slide_giant_seg_vqfc_64.yaml \
    --preset voc2012_sel20_seg \
    --head voc2012_seg_giant_last1 \
    --quality 1.0 \
    --cuda --output_dir eval_test --real

CUDA_VISIBLE_DEVICES=0 python examples/vqfc/run_eval_slide.py \
      --config examples/vqfc/config/eval_base.yaml examples/vqfc/config/dino_orig_slide_giant_cls_vqfc_512.yaml \
      --preset imagenet_sel100_cls \
      --head imagenet_cls_giant_last1 \
      --quality 1.0 \
      --cuda --output_dir eval_test --real
```

