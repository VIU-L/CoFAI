"""
MLoRE Dataset Adapters for CoFAI Framework

This module provides dataset wrappers for PASCAL-Context and NYUD datasets,
adapted to work with the CoFAI framework while maintaining compatibility
with RFC's original data loading.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from cofai.transforms import get_mlore_transforms  # noqa: E402


__all__ = [
    "MLoREImageDataset",
    "PASCALContextDataset",
    "NYUDDataset",
    "get_mlore_dataset",
    "collate_mlore",
]


class MLoREImageDataset(data.Dataset):
    """
    Base class for MLoRE multi-task image datasets.
    
    Provides a unified interface returning (img, img_meta) format
    as specified by the CoFAI framework.
    
    Args:
        root: Dataset root directory
        split: Data split ('train' or 'val')
        tasks: List of tasks to load
        transform: Optional transform pipeline
        img_size: Target image size (H, W)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'val',
        tasks: List[str] = None,
        transform=None,
        img_size: Tuple[int, int] = (512, 512),
    ):
        self.root = root
        self.split = split
        from cofai.engine.evaluator import ALLOWED_KINDS

        self.tasks = tasks or ['semseg', 'edge']
        bad = [t for t in self.tasks if str(t) not in ALLOWED_KINDS]
        if bad:
            raise ValueError(
                f"Invalid tasks for {self.__class__.__name__}: {bad}. "
                f"Allowed kinds: {sorted(ALLOWED_KINDS)}"
            )
        self.transform = transform
        self.img_size = img_size
        
        # To be implemented by subclasses
        self.images = []
        self.im_ids = []
        self.labels = {}  # Dict mapping task -> list of label paths
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Load image
        img = self._load_image(index)
        
        # Create img_meta following CoFAI format
        img_meta = {
            "img_path": self.images[index],
            "img_name": self.im_ids[index],
            "ori_size": img.shape[:2] if isinstance(img, np.ndarray) else img.size[::-1],
        }
        
        # Create sample dict
        sample = {"img": img, "meta": img_meta}
        
        # Load labels for each task
        for task in self.tasks:
            if task in self.labels and len(self.labels[task]) > index:
                label = self._load_label(index, task)
                if label is not None:
                    sample[task] = label
                    img_meta[f"{task}_label_path"] = self.labels[task][index]
        
        # Apply transforms if provided
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _load_image(self, index):
        """Load image at given index."""
        img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return img
    
    def _load_label(self, index, task):
        """Load label for given task at given index."""
        raise NotImplementedError


class PASCALContextDataset(MLoREImageDataset):
    """
    PASCAL-Context dataset adapter for CoFAI.
    
    Supports tasks:
    - semseg: Semantic segmentation (21 classes)
    - edge: Edge detection (binary)
    - human_parts: Human part segmentation (7 classes)
    - normals: Surface normal estimation (3 channels)
    - sal: Saliency detection (binary)
    
    Args:
        root: Dataset root directory
        split: Data split ('train' or 'val')
        tasks: List of tasks to load
        transform: Optional transform pipeline
        download: Whether to download if not exists
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'val',
        tasks: List[str] = None,
        transform=None,
        download: bool = False,
        **kwargs
    ):
        super().__init__(root, split, tasks, transform, **kwargs)
        
        # Import original dataset class
        from cofai.datasets.rfcdata.pascal_context import PASCALContext
        
        # Map tasks to dataset flags
        task_flags = {
            'semseg': 'do_semseg',
            'edge': 'do_edge',
            'human_parts': 'do_human_parts',
            'normals': 'do_normals',
            'sal': 'do_sal',
        }
        
        flags = {task_flags[t]: True for t in self.tasks if t in task_flags}
        
        # Create underlying dataset
        self._dataset = PASCALContext(
            root=root,
            download=download,
            split=[split] if isinstance(split, str) else split,
            transform=None,  # We handle transforms ourselves
            retname=True,
            **flags
        )
        
        self.images = self._dataset.images
        self.im_ids = self._dataset.im_ids
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        sample = self._dataset[index]
        sample["meta"]["img_path"] = self.images[index]
        sample["meta"]["ori_size"] = sample["meta"]["img_size"]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class NYUDDataset(MLoREImageDataset):
    """
    NYUD (NYU Depth V2) dataset adapter for CoFAI.
    
    Supports tasks:
    - semseg: Semantic segmentation (13 classes)
    - edge: Edge detection (binary)
    - normals: Surface normal estimation (3 channels)
    - depth: Depth estimation (1 channel)
    - scene: Scene classification (13 classes)
    
    Args:
        root: Dataset root directory
        split: Data split ('train' or 'val')
        tasks: List of tasks to load
        transform: Optional transform pipeline
        download: Whether to download if not exists
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'val',
        tasks: List[str] = None,
        transform=None,
        download: bool = False,
        **kwargs
    ):
        super().__init__(root, split, tasks, transform, **kwargs)
        
        # Import original dataset class
        from cofai.datasets.rfcdata.nyud import NYUD_MT
        
        # Map tasks to dataset flags
        task_flags = {
            'semseg': 'do_semseg',
            'edge': 'do_edge',
            'normals': 'do_normals',
            'depth': 'do_depth',
            'scene': 'do_scene',
        }
        
        flags = {task_flags[t]: True for t in self.tasks if t in task_flags}
        
        # Create underlying dataset
        self._dataset = NYUD_MT(
            root=root,
            download=download,
            split=split,
            transform=None,
            **flags
        )
        
        self.images = self._dataset.images
        self.im_ids = self._dataset.im_ids if hasattr(self._dataset, 'im_ids') else [
            Path(p).stem for p in self.images
        ]
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        sample = self._dataset[index]
        sample["meta"]["img_path"] = self.images[index]
        sample["meta"]["ori_size"] = sample["meta"]["img_size"]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_mlore_dataset(
    dataset_name: str,
    root: str,
    split: str = 'val',
    tasks: List[str] = None,
    transform=None,
    p=None,
    **kwargs
):
    """
    Factory function to create MLoRE datasets.
    
    Args:
        dataset_name: Dataset name ('PASCALContext' or 'NYUD')
        root: Dataset root directory
        split: Data split
        tasks: List of tasks
        transform: Optional transform (auto-created if None)
        p: Configuration dict
        **kwargs: Additional dataset arguments
        
    Returns:
        Dataset instance
    """
    if transform is None:
        transform = get_mlore_transforms(p, split)
    
    if dataset_name == 'PASCALContext':
        return PASCALContextDataset(
            root=root,
            split=split,
            tasks=tasks,
            transform=transform,
            **kwargs
        )
    elif dataset_name == 'NYUD':
        return NYUDDataset(
            root=root,
            split=split,
            tasks=tasks,
            transform=transform,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collate_mlore(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for MLoRE datasets.
    
    Handles variable-sized images and task labels by stacking tensors
    and collecting metadata.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Collated batch dict
    """
    # Separate image, labels, and meta
    images = []
    metas = []
    task_labels = {}
    
    for sample in batch:
        images.append(sample["img"])
        metas.append(sample['meta'])
        
        for key in sample:
            if key not in ['img', 'meta']:
                if key not in task_labels:
                    task_labels[key] = []
                task_labels[key].append(sample[key])
    
    # Stack images
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)
    else:
        images = torch.stack([torch.from_numpy(img) for img in images], dim=0)
    
    # Stack task labels
    for key in task_labels:
        labels = task_labels[key]
        if isinstance(labels[0], torch.Tensor):
            task_labels[key] = torch.stack(labels, dim=0)
        else:
            task_labels[key] = torch.stack([torch.from_numpy(arr) for arr in labels], dim=0)
    
    # Build result
    result = {
        'img': images,
        'meta': metas,
        **task_labels
    }
    
    return result


# Wrapper to use RFC's original common_config functions
class RFCDatasetWrapper:
    """
    Wrapper to use RFC's original dataset creation functions.
    
    This provides exact compatibility with RFC's data loading pipeline.
    """
    
    @staticmethod
    def get_train_dataset(p, transforms=None):
        """Get training dataset using RFC's function."""
        from cofai.utils.mlore_common_config import get_train_dataset
        return get_train_dataset(p, transforms)
    
    @staticmethod
    def get_test_dataset(p, transforms=None):
        """Get test dataset using RFC's function."""
        from cofai.utils.mlore_common_config import get_test_dataset
        return get_test_dataset(p, transforms)
    
    @staticmethod
    def get_transformations(p):
        """Get transforms using RFC's function."""
        from cofai.utils.mlore_common_config import get_transformations
        return get_transformations(p)
    
    @staticmethod
    def get_train_dataloader(p, dataset, sampler=None):
        """Get training dataloader using RFC's function."""
        from cofai.utils.mlore_common_config import get_train_dataloader
        return get_train_dataloader(p, dataset, sampler)
    
    @staticmethod
    def get_test_dataloader(p, dataset):
        """Get test dataloader using RFC's function."""
        from cofai.utils.mlore_common_config import get_test_dataloader
        return get_test_dataloader(p, dataset)




