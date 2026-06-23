"""Validated, repository-rooted model checkpoint persistence."""

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .data import validate_dataset_name
from .device import DeviceChoice, resolve_device
from .models import build_model
from .paths import resolve_repo_path

REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
        "format_version",
        "model_state_dict",
        "architecture",
        "dataset",
        "seed",
        "preprocessing",
        "epoch",
        "validation_metric",
    }
)


def _architecture_for(model: nn.Module, architecture: Mapping[str, Any] | None) -> dict[str, Any]:
    if architecture is not None:
        return dict(architecture)
    provider = getattr(model, "architecture_config", None)
    if provider is None or not callable(provider):
        raise ValueError("architecture metadata is required for models without architecture_config()")
    return dict(provider())


def _validate_metric(metric: Mapping[str, float]) -> dict[str, float]:
    if not metric:
        raise ValueError("validation_metric must contain at least one named value")
    validated = {str(name): float(value) for name, value in metric.items()}
    if not all(math.isfinite(value) for value in validated.values()):
        raise ValueError("validation metric values must be finite")
    return validated


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    dataset: str,
    seed: int,
    preprocessing: Mapping[str, Any],
    epoch: int,
    validation_metric: Mapping[str, float],
    architecture: Mapping[str, Any] | None = None,
) -> Path:
    """Save all metadata needed to reproduce and restore a trained model."""
    dataset_name = validate_dataset_name(dataset)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if not isinstance(preprocessing, Mapping):
        raise TypeError("preprocessing must be a mapping")

    destination = resolve_repo_path(path, create_parent=True)
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "architecture": _architecture_for(model, architecture),
        "dataset": dataset_name,
        "seed": seed,
        "preprocessing": dict(preprocessing),
        "epoch": epoch,
        "validation_metric": _validate_metric(validation_metric),
    }
    # File objects avoid Windows path-encoding issues in PyTorch's zip writer.
    with destination.open("wb") as checkpoint_file:
        torch.save(payload, checkpoint_file)
    return destination


def _validate_checkpoint(checkpoint: Any, path: Path) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: checkpoint must contain a dictionary")
    missing = REQUIRED_CHECKPOINT_KEYS - checkpoint.keys()
    if missing:
        raise ValueError(f"{path}: checkpoint is missing required keys: {sorted(missing)}")
    checkpoint["dataset"] = validate_dataset_name(checkpoint["dataset"])
    if not isinstance(checkpoint["architecture"], Mapping):
        raise ValueError(f"{path}: architecture must be a mapping")
    if not isinstance(checkpoint["preprocessing"], Mapping):
        raise ValueError(f"{path}: preprocessing must be a mapping")
    checkpoint["validation_metric"] = _validate_metric(checkpoint["validation_metric"])
    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    device: DeviceChoice | str | torch.device = "auto",
) -> dict[str, Any]:
    """Load and validate a checkpoint without constructing its model."""
    source = resolve_repo_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    map_location = resolve_device(device)
    with source.open("rb") as checkpoint_file:
        try:
            checkpoint = torch.load(checkpoint_file, map_location=map_location, weights_only=True)
        except TypeError:  # Compatibility with older supported PyTorch releases.
            checkpoint_file.seek(0)
            checkpoint = torch.load(checkpoint_file, map_location=map_location)
    return _validate_checkpoint(checkpoint, source)


def restore_model(
    path: str | Path,
    *,
    device: DeviceChoice | str | torch.device = "auto",
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a registered architecture and load its saved model weights."""
    resolved_device = resolve_device(device)
    checkpoint = load_checkpoint(path, device=resolved_device)
    model = build_model(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint
