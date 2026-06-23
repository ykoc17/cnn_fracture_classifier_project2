"""Strict NPZ loading for the graded NT and UT datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .paths import PROCESSED_DATA_ROOT, resolve_repo_path
from .reproducibility import make_generator, seed_worker

ALLOWED_DATASETS = frozenset({"NT", "UT"})
ALLOWED_SPLITS = frozenset({"train", "val", "test"})
REQUIRED_KEYS = frozenset({"images", "labels"})
IMAGE_SHAPE = (1, 128, 128)


@dataclass(frozen=True)
class LoadedSplit:
    """Validated tensors and provenance for one fixed dataset split."""

    dataset: str
    split: str
    path: Path
    images: torch.Tensor
    labels: torch.Tensor

    def as_dataset(self) -> TensorDataset:
        return TensorDataset(self.images, self.labels)


def validate_dataset_name(dataset: str) -> str:
    """Return a canonical graded dataset name and reject ASB explicitly."""
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    canonical = dataset.strip().upper()
    if canonical not in ALLOWED_DATASETS:
        raise ValueError(
            "graded scripts accept only NT or UT; ASB is prohibited because its labels are systematically wrong"
        )
    return canonical


def _validate_split_name(split: str) -> str:
    if not isinstance(split, str):
        raise TypeError("split must be a string")
    canonical = split.strip().lower()
    if canonical not in ALLOWED_SPLITS:
        raise ValueError("split must be one of: train, val, test")
    return canonical


def _split_path(dataset: str, split: str, data_root: str | Path | None) -> Path:
    root = PROCESSED_DATA_ROOT if data_root is None else resolve_repo_path(data_root)
    return root / dataset / f"{split}.npz"


def _validate_arrays(images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    if images.dtype != np.float32:
        raise ValueError(f"{path}: images dtype must be float32, got {images.dtype}")
    if labels.dtype != np.int64:
        raise ValueError(f"{path}: labels dtype must be int64, got {labels.dtype}")
    if images.ndim != 4 or tuple(images.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"{path}: images shape must be (N, 1, 128, 128), got {images.shape}")
    if labels.ndim != 1:
        raise ValueError(f"{path}: labels shape must be (N,), got {labels.shape}")
    if images.shape[0] == 0 or images.shape[0] != labels.shape[0]:
        raise ValueError(f"{path}: images and labels must have the same non-zero sample count")
    if not np.isfinite(images).all():
        raise ValueError(f"{path}: images contain NaN or infinite values")
    pixel_min = float(images.min())
    pixel_max = float(images.max())
    if pixel_min < 0.0 or pixel_max > 1.0:
        raise ValueError(f"{path}: pixels must be in [0, 1], got [{pixel_min}, {pixel_max}]")
    label_values = np.unique(labels)
    if not np.isin(label_values, (0, 1)).all():
        raise ValueError(f"{path}: labels must contain only 0 or 1, got {label_values.tolist()}")


def load_split(
    dataset: str,
    split: str,
    *,
    data_root: str | Path | None = None,
) -> LoadedSplit:
    """Load and validate one immutable train, validation, or test NPZ split."""
    dataset_name = validate_dataset_name(dataset)
    split_name = _validate_split_name(split)
    path = _split_path(dataset_name, split_name, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"dataset split not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        keys = frozenset(archive.files)
        if keys != REQUIRED_KEYS:
            raise ValueError(f"{path}: NPZ keys must be exactly {sorted(REQUIRED_KEYS)}, got {sorted(keys)}")
        images = archive["images"]
        labels = archive["labels"]

    _validate_arrays(images, labels, path)
    return LoadedSplit(
        dataset=dataset_name,
        split=split_name,
        path=path,
        images=torch.from_numpy(images),
        labels=torch.from_numpy(labels),
    )


def _make_loader(
    loaded: LoadedSplit,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    return DataLoader(
        loaded.as_dataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def create_training_loaders(
    dataset: str,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    data_root: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders; test data is intentionally excluded."""
    train = load_split(dataset, "train", data_root=data_root)
    validation = load_split(dataset, "val", data_root=data_root)
    return (
        _make_loader(
            train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        _make_loader(
            validation,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


def create_test_loader(
    dataset: str,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    data_root: str | Path | None = None,
) -> DataLoader:
    """Create a separate, non-shuffled test loader for final evaluation only."""
    test = load_split(dataset, "test", data_root=data_root)
    return _make_loader(
        test,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_train_channel_stats(
    dataset: str,
    *,
    data_root: str | Path | None = None,
) -> dict[str, list[float]]:
    """Compute normalization statistics from the training split only."""
    train = load_split(dataset, "train", data_root=data_root)
    mean = train.images.mean(dim=(0, 2, 3))
    std = train.images.std(dim=(0, 2, 3), unbiased=False)
    return {"mean": mean.tolist(), "std": std.tolist()}
