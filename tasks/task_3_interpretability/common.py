"""Shared frozen-model helpers for Task 3 interpretability scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from src import PROCESSED_DATA_ROOT, load_split, resolve_repo_path, restore_model

DEFAULT_CHECKPOINT = Path("tasks/task_0_baseline_cnn/results/best_model.pt")
DEFAULT_SAMPLE_INDICES = (1, 0, 88, 7)
SELECTION_ROLES = {
    1: "correct_class_0",
    0: "correct_class_1",
    88: "false_positive",
    7: "false_negative",
}
LABEL_NAMES = {0: "no fracture", 1: "fracture"}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_task0(
    checkpoint_path: str | Path,
    data_root: str | Path | None,
    device: torch.device,
):
    checkpoint_file = resolve_repo_path(checkpoint_path)
    model, checkpoint = restore_model(checkpoint_file, device=device)
    if checkpoint["dataset"] != "NT":
        raise ValueError(f"Task 3 requires the frozen Task 0 NT checkpoint, got {checkpoint['dataset']}")
    root = PROCESSED_DATA_ROOT if data_root is None else resolve_repo_path(data_root)
    test_data = load_split("NT", "test", data_root=root)
    return model, checkpoint, checkpoint_file, test_data


def preprocess(images: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
    normalization = metadata.get("normalization")
    if normalization is None:
        return images
    if not isinstance(normalization, Mapping) or not {"mean", "std"} <= normalization.keys():
        raise ValueError(f"unsupported checkpoint normalization metadata: {normalization!r}")
    mean = torch.as_tensor(normalization["mean"], dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(normalization["std"], dtype=images.dtype).view(1, -1, 1, 1)
    return (images - mean) / std


def validate_indices(indices: list[int], sample_count: int) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError("sample indices must not contain duplicates")
    invalid = [index for index in indices if index < 0 or index >= sample_count]
    if invalid:
        raise IndexError(f"sample indices out of range: {invalid}")


def selection_role(index: int, true_label: int, predicted_label: int) -> str:
    expected = SELECTION_ROLES.get(index)
    if expected is not None:
        return expected
    if true_label == predicted_label:
        return f"correct_class_{true_label}"
    if true_label == 0:
        return "false_positive"
    return "false_negative"


def hook_counts(model: torch.nn.Module) -> dict[str, int]:
    modules = list(model.modules())
    return {
        "forward": sum(len(module._forward_hooks) for module in modules),
        "backward": sum(len(module._backward_hooks) for module in modules),
    }


def normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().cpu()
    minimum = values.min()
    maximum = values.max()
    if float(maximum - minimum) <= 1e-12:
        return torch.zeros_like(values)
    return (values - minimum) / (maximum - minimum)
