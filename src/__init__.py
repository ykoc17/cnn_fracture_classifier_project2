"""Shared, graded-safe infrastructure for Tasks 0--5."""

from .checkpoints import load_checkpoint, restore_model, save_checkpoint
from .data import (
    ALLOWED_DATASETS,
    LoadedSplit,
    compute_train_channel_stats,
    create_test_loader,
    create_training_loaders,
    load_split,
    validate_dataset_name,
)
from .device import resolve_device
from .evaluation import EvaluationResult, classification_metrics, evaluate_model
from .models import SimpleCNN, SimpleCNNConfig, build_model
from .paths import PROCESSED_DATA_ROOT, REPO_ROOT, repo_relative_path, resolve_repo_path
from .reproducibility import seed_everything
from .training import EpochRecord, fit, train_one_epoch

__all__ = [
    "ALLOWED_DATASETS",
    "EpochRecord",
    "EvaluationResult",
    "LoadedSplit",
    "PROCESSED_DATA_ROOT",
    "REPO_ROOT",
    "SimpleCNN",
    "SimpleCNNConfig",
    "build_model",
    "classification_metrics",
    "compute_train_channel_stats",
    "create_test_loader",
    "create_training_loaders",
    "evaluate_model",
    "fit",
    "load_checkpoint",
    "load_split",
    "resolve_device",
    "repo_relative_path",
    "resolve_repo_path",
    "restore_model",
    "save_checkpoint",
    "seed_everything",
    "train_one_epoch",
    "validate_dataset_name",
]
