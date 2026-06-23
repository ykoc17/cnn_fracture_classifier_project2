"""Device selection shared by training and evaluation code."""

from typing import Literal

import torch

DeviceChoice = Literal["auto", "cpu", "cuda"]


def resolve_device(device: DeviceChoice | str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available and otherwise to CPU."""
    if isinstance(device, torch.device):
        requested = device
    else:
        name = str(device).strip().lower()
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        if name not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        requested = torch.device(name)

    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available in this PyTorch environment")
    return requested
