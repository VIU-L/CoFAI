"""Scene classification meters."""

from __future__ import annotations

import numpy as np
import torch


class SceneClassificationMeter:
    def __init__(self, database: str):
        if database == "NYUD":
            self.n_classes = 27
            self.class_names = [
                "basement",
                "bathroom",
                "bedroom",
                "bookstore",
                "cafe",
                "classroom",
                "computer_lab",
                "conference_room",
                "dinette",
                "dining_room",
                "excercise_room",
                "foyer",
                "furniture_store",
                "home_office",
                "home_storage",
                "indoor_balcony",
                "kitchen",
                "laundry_room",
                "living_room",
                "office",
                "office_kitchen",
                "playroom",
                "printer_room",
                "reception_room",
                "student_lounge",
                "study",
                "study_room",
            ]
        else:
            raise NotImplementedError

        self.confusion_matrix: np.ndarray
        self.total_samples: int
        self.reset()

    def reset(self) -> None:
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self.total_samples = 0

    @torch.no_grad()
    def update(self, pred, target) -> None:
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred_classes = pred
        for t, p in zip(target, pred_classes):
            self.confusion_matrix[int(t), int(p)] += 1
        self.total_samples += int(len(target))

    def compute(self) -> dict[str, float]:
        if self.total_samples <= 0:
            return {"accuracy": 0.0}
        tp = np.diag(self.confusion_matrix)
        accuracy = float(np.sum(tp) / float(self.total_samples)) * 100.0
        return {"accuracy": accuracy}

