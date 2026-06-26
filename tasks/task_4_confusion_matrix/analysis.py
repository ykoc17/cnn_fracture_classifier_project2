"""Analyze NT/test errors for the frozen Task 0 checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Support direct execution from the repository root on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

from src import evaluate_model, load_split, resolve_device, resolve_repo_path, restore_model

DEFAULT_CHECKPOINT = Path("tasks/task_0_baseline_cnn/results/best_model.pt")
DEFAULT_REFERENCE_METRICS = Path("tasks/task_0_baseline_cnn/results/metrics.json")
DEFAULT_OUTPUT_DIR = Path("tasks/task_4_confusion_matrix/results")
CLASS_NAMES = {0: "class 0", 1: "class 1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-metrics", type=Path, default=DEFAULT_REFERENCE_METRICS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-per-error",
        type=int,
        default=6,
        help="Maximum examples of each error type to include in the gallery.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.max_per_error < 1:
        raise ValueError("--max-per-error must be positive")


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


def apply_checkpoint_preprocessing(
    images: torch.Tensor,
    preprocessing: Mapping[str, Any],
) -> torch.Tensor:
    """Apply exactly the normalization declared by the checkpoint."""
    normalization = preprocessing.get("normalization")
    if normalization is None:
        return images
    if not isinstance(normalization, Mapping) or not {"mean", "std"} <= normalization.keys():
        raise ValueError(f"unsupported checkpoint normalization metadata: {normalization!r}")
    mean = torch.as_tensor(normalization["mean"], dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(normalization["std"], dtype=images.dtype).view(1, -1, 1, 1)
    if mean.shape[1] != images.shape[1] or std.shape[1] != images.shape[1]:
        raise ValueError("normalization metadata does not match the image channel count")
    if torch.any(std <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return (images - mean) / std


def calculate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float | int]], dict[str, float | int], dict[str, float | int]]:
    raw_matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    row_totals = raw_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        raw_matrix,
        row_totals,
        out=np.zeros_like(raw_matrix, dtype=float),
        where=row_totals != 0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(precision[label]),
            "recall": float(recall[label]),
            "f1": float(f1[label]),
            "support": int(support[label]),
        }
        for label in (0, 1)
    }

    averages: dict[str, dict[str, float | int]] = {}
    for average in ("macro", "weighted"):
        average_precision, average_recall, average_f1, _ = precision_recall_fscore_support(
            targets, predictions, labels=[0, 1], average=average, zero_division=0
        )
        averages[average] = {
            "precision": float(average_precision),
            "recall": float(average_recall),
            "f1": float(average_f1),
            "support": int(targets.size),
        }
    return raw_matrix, normalized_matrix, per_class, averages["macro"], averages["weighted"]


def write_predictions_csv(
    path: Path,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    fields = [
        "sample_index",
        "true_label",
        "predicted_label",
        "class_0_probability",
        "class_1_probability",
        "confidence",
        "correct",
        "error_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for index, (target, prediction, probability) in enumerate(
            zip(targets, predictions, probabilities, strict=True)
        ):
            is_correct = bool(target == prediction)
            error_type = "correct"
            if not is_correct:
                error_type = "false_positive" if target == 0 else "false_negative"
            writer.writerow(
                {
                    "sample_index": index,
                    "true_label": int(target),
                    "predicted_label": int(prediction),
                    "class_0_probability": float(probability[0]),
                    "class_1_probability": float(probability[1]),
                    "confidence": float(probability[prediction]),
                    "correct": is_correct,
                    "error_type": error_type,
                }
            )


def write_metrics_csv(
    path: Path,
    per_class: Mapping[str, Mapping[str, float | int]],
    macro_average: Mapping[str, float | int],
    weighted_average: Mapping[str, float | int],
    accuracy: float,
    test_size: int,
) -> None:
    fields = ["scope", "label", "precision", "recall", "f1", "support", "accuracy"]
    rows = [
        {"scope": "class", "label": label, **per_class[label], "accuracy": ""}
        for label in ("0", "1")
    ]
    rows.extend(
        [
            {"scope": "average", "label": "macro", **macro_average, "accuracy": ""},
            {"scope": "average", "label": "weighted", **weighted_average, "accuracy": ""},
            {
                "scope": "overall",
                "label": "all",
                "precision": "",
                "recall": "",
                "f1": "",
                "support": test_size,
                "accuracy": accuracy,
            },
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(path: Path, matrix: np.ndarray, *, normalized: bool) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Row proportion" if normalized else "Number of samples")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
        xlabel="Predicted label",
        ylabel="True label",
        title=(
            "NT/test row-normalized confusion matrix"
            if normalized
            else "NT/test confusion matrix (raw counts)"
        ),
    )
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            label = f"{value:.1%}" if normalized else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=14,
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_misclassified_gallery(
    path: Path,
    images: torch.Tensor,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    false_positive_indices: list[int],
    false_negative_indices: list[int],
) -> None:
    groups = [
        ("False positives\n(true 0 → predicted 1)", false_positive_indices),
        ("False negatives\n(true 1 → predicted 0)", false_negative_indices),
    ]
    columns = max(len(indices) for _, indices in groups)
    figure, axes = plt.subplots(2, columns, figsize=(2.55 * columns, 7.2), squeeze=False)
    for row, (group_name, indices) in enumerate(groups):
        for column in range(columns):
            axis = axes[row, column]
            axis.set_xticks([])
            axis.set_yticks([])
            if column >= len(indices):
                axis.axis("off")
                continue
            index = indices[column]
            prediction = int(predictions[index])
            confidence = float(probabilities[index, prediction])
            axis.imshow(images[index, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(
                f"index {index}\ntrue {int(targets[index])} → pred {prediction}\nconf. {confidence:.3f}",
                fontsize=9,
            )
        axes[row, 0].set_ylabel(group_name, fontsize=10, labelpad=12)
    figure.suptitle(
        "Misclassified NT/test examples from the frozen Task 0 model", fontsize=13, y=0.97
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.06, wspace=0.22, hspace=0.65)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    reference_metrics_path = resolve_repo_path(args.reference_metrics)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = restore_model(checkpoint_path, device=device)
    if checkpoint["dataset"] != "NT":
        raise ValueError(f"Task 4 requires an NT checkpoint, got {checkpoint['dataset']!r}")
    test_split = load_split("NT", "test", data_root=args.data_root)
    raw_images = test_split.images
    model_images = apply_checkpoint_preprocessing(raw_images, checkpoint["preprocessing"])
    loader = DataLoader(
        TensorDataset(model_images, test_split.labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    evaluation = evaluate_model(model, loader, device=device)
    targets = evaluation.targets.numpy()
    predictions = evaluation.predictions.numpy()
    probabilities = evaluation.probabilities.numpy()
    accuracy = float(accuracy_score(targets, predictions))
    raw_matrix, normalized_matrix, per_class, macro_average, weighted_average = calculate_metrics(
        targets, predictions
    )

    with reference_metrics_path.open("r", encoding="utf-8") as input_file:
        reference_metrics = json.load(input_file)
    reference_accuracy = float(reference_metrics["accuracy"])
    reference_matrix = np.asarray(reference_metrics["confusion_matrix"], dtype=int)
    test_size = int(targets.size)
    count_total_matches = int(raw_matrix.sum()) == test_size == len(test_split.labels)
    accuracy_from_matrix = float(np.trace(raw_matrix) / raw_matrix.sum())
    matrix_accuracy_matches = bool(np.isclose(accuracy, accuracy_from_matrix, atol=1e-12, rtol=0.0))
    task_0_accuracy_matches = bool(np.isclose(accuracy, reference_accuracy, atol=1e-12, rtol=0.0))
    task_0_matrix_matches = bool(np.array_equal(raw_matrix, reference_matrix))
    if not count_total_matches:
        raise RuntimeError("confusion-matrix counts do not equal the NT/test size")
    if not matrix_accuracy_matches:
        raise RuntimeError("accuracy does not agree with the confusion-matrix diagonal")
    if not task_0_accuracy_matches or not task_0_matrix_matches:
        raise RuntimeError("fresh evaluation does not reproduce the frozen Task 0 metrics")

    false_positive_indices = np.flatnonzero((targets == 0) & (predictions == 1)).tolist()
    false_negative_indices = np.flatnonzero((targets == 1) & (predictions == 0)).tolist()
    selected_false_positives = false_positive_indices[: args.max_per_error]
    selected_false_negatives = false_negative_indices[: args.max_per_error]

    write_predictions_csv(output_dir / "predictions.csv", targets, predictions, probabilities)
    write_metrics_csv(
        output_dir / "metrics.csv",
        per_class,
        macro_average,
        weighted_average,
        accuracy,
        test_size,
    )
    plot_confusion_matrix(output_dir / "confusion_matrix_counts.png", raw_matrix, normalized=False)
    plot_confusion_matrix(
        output_dir / "confusion_matrix_normalized.png", normalized_matrix, normalized=True
    )
    plot_misclassified_gallery(
        output_dir / "misclassified_gallery.png",
        raw_images,
        targets,
        predictions,
        probabilities,
        selected_false_positives,
        selected_false_negatives,
    )

    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_metric": checkpoint["validation_metric"],
        "preprocessing": checkpoint["preprocessing"],
        "dataset": "NT",
        "split": "test",
        "test_size": test_size,
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "accuracy": accuracy,
        "confusion_matrix": {
            "labels": [0, 1],
            "raw_counts": raw_matrix.tolist(),
            "row_normalized": normalized_matrix.tolist(),
        },
        "per_class": per_class,
        "macro_average": macro_average,
        "weighted_average": weighted_average,
        "errors": {
            "false_positive_count": len(false_positive_indices),
            "false_negative_count": len(false_negative_indices),
            "all_false_positive_indices": false_positive_indices,
            "all_false_negative_indices": false_negative_indices,
            "gallery_false_positive_indices": selected_false_positives,
            "gallery_false_negative_indices": selected_false_negatives,
        },
        "verification": {
            "confusion_matrix_total": int(raw_matrix.sum()),
            "nt_test_size": len(test_split.labels),
            "count_total_matches": count_total_matches,
            "accuracy_from_confusion_matrix": accuracy_from_matrix,
            "matrix_accuracy_matches": matrix_accuracy_matches,
            "task_0_reference_accuracy": reference_accuracy,
            "task_0_accuracy_matches": task_0_accuracy_matches,
            "task_0_confusion_matrix_matches": task_0_matrix_matches,
        },
    }
    write_json(output_dir / "metrics.json", metrics_payload)

    print(f"Device: {device}")
    print(f"NT/test samples: {test_size}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Confusion matrix: {raw_matrix.tolist()}")
    print(f"False positives: {len(false_positive_indices)}; false negatives: {len(false_negative_indices)}")
    print(f"Task 0 metrics reproduced: {task_0_accuracy_matches and task_0_matrix_matches}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
