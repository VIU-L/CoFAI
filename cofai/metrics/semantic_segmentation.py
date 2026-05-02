"""Semantic segmentation meters."""

from __future__ import annotations

import numpy as np
import torch


VOC_CATEGORY_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

NYU_CATEGORY_NAMES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refridgerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]

CITY_CATEGORY_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


class SemanticSegmentationMeter:
    def __init__(self, database: str, ignore_idx: int = 255):
        if database == "PASCALContext":
            n_classes = 20
            cat_names = VOC_CATEGORY_NAMES
            has_bg = True
        elif database == "NYUD":
            n_classes = 40
            cat_names = NYU_CATEGORY_NAMES
            has_bg = False
        elif database == "CityScape":
            n_classes = 19
            cat_names = CITY_CATEGORY_NAMES
            has_bg = False
        else:
            raise NotImplementedError

        self.n_classes = int(n_classes) + int(has_bg)
        self.cat_names = cat_names
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes
        self.ignore_idx = int(ignore_idx)

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred = pred.squeeze()
        gt = gt.squeeze()
        valid = gt != self.ignore_idx
        for i_part in range(0, self.n_classes):
            tmp_gt = gt == i_part
            tmp_pred = pred == i_part
            self.tp[i_part] += int(torch.sum(tmp_gt & tmp_pred & valid).item())
            self.fp[i_part] += int(torch.sum(~tmp_gt & tmp_pred & valid).item())
            self.fn[i_part] += int(torch.sum(tmp_gt & ~tmp_pred & valid).item())

    def reset(self) -> None:
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    def compute(self) -> dict[str, float]:
        jac = [0.0] * self.n_classes
        for i_part in range(self.n_classes):
            denom = float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part])
            jac[i_part] = float(self.tp[i_part]) / max(denom, 1e-8)
        return {"mIoU": float(np.mean(jac) * 100.0)}

