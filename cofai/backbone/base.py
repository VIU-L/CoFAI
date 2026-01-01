import torch
from typing import Protocol


def parse_dtype(dtype):
    """Convert string or torch.dtype to torch.dtype object.

    Args:
        dtype (str or torch.dtype): Data type to parse. Can be a string like "float32",
                                   "torch.float16", etc., or a torch.dtype object.

    Returns:
        torch.dtype: The parsed torch.dtype object.

    Raises:
        ValueError: If dtype is not a supported string format or torch.dtype.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        # Support common string formats
        dtype_mapping = {
            "torch.float": torch.float,
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.double": torch.double,
            "torch.int": torch.int,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "float": torch.float,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "double": torch.double,
        }
        if dtype in dtype_mapping:
            return dtype_mapping[dtype]
        else:
            raise ValueError(
                f"Unsupported data type: {dtype}. Supported types: {list(dtype_mapping.keys())}"
            )
    else:
        raise ValueError(
            f"autocast_dtype must be a string or torch.dtype, got: {type(dtype)}"
        )


class BackboneProtocol(Protocol):
    """Protocol defining the interface for backbone models."""

    patch_size: int
    hidden_size: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to intermediate features.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)

        Returns:
            h (torch.Tensor): Encoded features of shape (B, N, C)
        """
        ...

    def decode(self, h: torch.Tensor, tasks: list[str]) -> torch.Tensor:
        """Decode encoded features for downstream tasks.

        Args:
            h (torch.Tensor): Encoded features from encode method, shape (B, N, C)
            tasks (list[str]): List of tasks to decode for, e.g. ["rec", "cls", "seg"]
        Returns:
            task_feats (dict): Dictionary of task-specific features
        """
