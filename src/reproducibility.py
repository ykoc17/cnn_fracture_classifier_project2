"""Reproducibility helpers for Python, NumPy, and PyTorch."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed supported RNGs and enable practical deterministic CUDA behavior."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside a PyTorch DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded generator for deterministic DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
