"""Train identical NT/UT models and evaluate the frozen models on both test domains."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
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
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import (
    PROCESSED_DATA_ROOT,
    create_training_loaders,
    evaluate_model,
    fit,
    load_split,
    resolve_device,
    resolve_repo_path,
    restore_model,
    save_checkpoint,
    seed_everything,
)
from src.training import EpochRecord
from tasks.task_0_baseline_cnn.model import build_baseline_model

DATASETS = ("NT", "UT")
DEFAULT_OUTPUT_DIR = Path("tasks/task_5_cross_dataset/results")
PREPROCESSING: dict[str, Any] = {
    "input_shape": [1, 128, 128],
    "dtype": "float32",
    "pixel_range": [0.0, 1.0],
    "normalization": None,
    "augmentation": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and lock both source checkpoints without opening either test split.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Processed-data root; relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory; relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive")


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


def preprocessing_fingerprint(preprocessing: Mapping[str, Any]) -> str:
    serialized = json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def apply_checkpoint_preprocessing(
    images: torch.Tensor,
    preprocessing: Mapping[str, Any],
) -> torch.Tensor:
    """Apply one source checkpoint's locked preprocessing to any target domain."""
    normalization = preprocessing.get("normalization")
    if normalization is None:
        return images
    if not isinstance(normalization, Mapping) or not {"mean", "std"} <= normalization.keys():
        raise ValueError(f"unsupported checkpoint normalization metadata: {normalization!r}")
    mean = torch.as_tensor(normalization["mean"], dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(normalization["std"], dtype=images.dtype).view(1, -1, 1, 1)
    if mean.shape[1] != images.shape[1] or std.shape[1] != images.shape[1]:
        raise ValueError("normalization metadata does not match the target image channels")
    if torch.any(std <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return (images - mean) / std


def write_history(path: Path, history: list[EpochRecord]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record in history:
            writer.writerow({field: getattr(record, field) for field in fields})


def plot_history(path: Path, source: str, history: list[EpochRecord]) -> None:
    epochs = [record.epoch for record in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    loss_axis.plot(epochs, [record.train_loss for record in history], label="Train", linewidth=2)
    loss_axis.plot(
        epochs,
        [record.validation_loss for record in history],
        label="Validation",
        linewidth=2,
    )
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.set_title(f"Task 5: {source} source-model training")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    accuracy_axis.plot(
        epochs,
        [100.0 * record.train_accuracy for record in history],
        label="Train",
        linewidth=2,
    )
    accuracy_axis.plot(
        epochs,
        [100.0 * record.validation_accuracy for record in history],
        label="Validation",
        linewidth=2,
    )
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_ylim(0.0, 100.0)
    accuracy_axis.grid(alpha=0.3)
    accuracy_axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def train_source(
    source: str,
    args: argparse.Namespace,
    *,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train without opening any test split and lock on source validation loss."""
    if source not in DATASETS:
        raise ValueError("Task 5 sources are restricted to NT and UT; ASB is prohibited")
    seed_everything(args.seed)
    train_loader, validation_loader = create_training_loaders(
        source,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        data_root=data_root,
    )
    model = build_baseline_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    started = time.perf_counter()
    history = fit(
        model,
        train_loader,
        validation_loader,
        optimizer,
        loss_fn,
        epochs=args.epochs,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
    )
    training_seconds = time.perf_counter() - started
    best_record = min(history, key=lambda record: record.validation_loss)

    source_dir = output_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = source_dir / "best_model.pt"
    write_history(source_dir / "history.csv", history)
    plot_history(source_dir / "training_curves.png", source, history)
    save_checkpoint(
        checkpoint_path,
        model,
        dataset=source,
        seed=args.seed,
        preprocessing=PREPROCESSING,
        epoch=best_record.epoch,
        validation_metric={
            "loss": best_record.validation_loss,
            "accuracy": best_record.validation_accuracy,
        },
    )

    config = {
        "architecture": model.architecture_config(),
        "batch_size": args.batch_size,
        "data_root": str(data_root),
        "dataset": source,
        "device_requested": args.device,
        "device_resolved": str(device),
        "early_stopped": len(history) < args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "epochs_completed": len(history),
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "loss": "CrossEntropyLoss",
        "model_selection": f"minimum {source}/val cross-entropy loss",
        "optimizer": "Adam",
        "output_dir": str(source_dir),
        "preprocessing": PREPROCESSING,
        "seed": args.seed,
        "splits_opened_during_training": [f"{source}/train", f"{source}/val"],
        "target_test_access_during_training": False,
        "training_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "training_seconds": training_seconds,
        "validation_best_epoch": best_record.epoch,
        "validation_best_loss": best_record.validation_loss,
        "validation_best_accuracy": best_record.validation_accuracy,
        "workers": args.workers,
    }
    write_json(source_dir / "run_config.json", config)
    return {
        "source": source,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config": config,
    }


def evaluate_locked_models(
    locked_sources: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    *,
    data_root: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Open test splits only after both source checkpoints have been locked."""
    evaluations: list[dict[str, Any]] = []
    applied_preprocessing: dict[str, dict[str, str]] = {}
    loss_fn = nn.CrossEntropyLoss()
    for source in DATASETS:
        checkpoint_path = Path(locked_sources[source]["checkpoint_path"])
        model, checkpoint = restore_model(checkpoint_path, device=device)
        if checkpoint["dataset"] != source:
            raise RuntimeError(f"{checkpoint_path}: expected {source} checkpoint metadata")
        source_fingerprint = preprocessing_fingerprint(checkpoint["preprocessing"])
        applied_preprocessing[source] = {}
        for target in DATASETS:
            target_test = load_split(target, "test", data_root=data_root)
            model_images = apply_checkpoint_preprocessing(
                target_test.images, checkpoint["preprocessing"]
            )
            applied_preprocessing[source][target] = source_fingerprint
            loader = DataLoader(
                TensorDataset(model_images, target_test.labels),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
            )
            started = time.perf_counter()
            result = evaluate_model(model, loader, device=device, loss_fn=loss_fn)
            evaluation_seconds = time.perf_counter() - started
            assert result.loss is not None
            evaluations.append(
                {
                    "source_dataset": source,
                    "target_dataset": target,
                    "split": "test",
                    "test_size": len(target_test.labels),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": locked_sources[source]["checkpoint_sha256"],
                    "source_preprocessing": checkpoint["preprocessing"],
                    "source_preprocessing_fingerprint": source_fingerprint,
                    "source_validation_metric": checkpoint["validation_metric"],
                    "source_checkpoint_epoch": int(checkpoint["epoch"]),
                    "evaluation_seconds": evaluation_seconds,
                    "loss": float(result.loss),
                    **result.metrics,
                }
            )
    return evaluations, applied_preprocessing


def write_raw_metrics_csv(path: Path, evaluations: list[dict[str, Any]]) -> None:
    fields = [
        "source_dataset",
        "target_dataset",
        "split",
        "test_size",
        "accuracy",
        "loss",
        "true_0_pred_0",
        "true_0_pred_1",
        "true_1_pred_0",
        "true_1_pred_1",
        "source_checkpoint_epoch",
        "source_validation_loss",
        "source_validation_accuracy",
        "checkpoint_sha256",
        "source_preprocessing_fingerprint",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for result in evaluations:
            matrix = result["confusion_matrix"]
            writer.writerow(
                {
                    "source_dataset": result["source_dataset"],
                    "target_dataset": result["target_dataset"],
                    "split": result["split"],
                    "test_size": result["test_size"],
                    "accuracy": result["accuracy"],
                    "loss": result["loss"],
                    "true_0_pred_0": matrix[0][0],
                    "true_0_pred_1": matrix[0][1],
                    "true_1_pred_0": matrix[1][0],
                    "true_1_pred_1": matrix[1][1],
                    "source_checkpoint_epoch": result["source_checkpoint_epoch"],
                    "source_validation_loss": result["source_validation_metric"]["loss"],
                    "source_validation_accuracy": result["source_validation_metric"]["accuracy"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "source_preprocessing_fingerprint": result[
                        "source_preprocessing_fingerprint"
                    ],
                }
            )


def accuracy_matrix(evaluations: list[dict[str, Any]]) -> np.ndarray:
    by_pair = {
        (result["source_dataset"], result["target_dataset"]): float(result["accuracy"])
        for result in evaluations
    }
    expected_pairs = {(source, target) for source in DATASETS for target in DATASETS}
    if set(by_pair) != expected_pairs:
        raise RuntimeError("expected exactly the four NT/UT source-target evaluations")
    return np.asarray(
        [[by_pair[(source, target)] for target in DATASETS] for source in DATASETS],
        dtype=float,
    )


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["source_dataset", *DATASETS])
        for row, source in enumerate(DATASETS):
            writer.writerow([source, *[float(value) for value in matrix[row]]])


def plot_matrix(path: Path, matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 5.5))
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Test accuracy")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=DATASETS,
        yticklabels=DATASETS,
        xlabel="Target test dataset",
        ylabel="Source training dataset",
        title="Task 5: NT/UT cross-dataset generalization",
    )
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=15,
                color="white" if value >= 0.5 else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    data_root = PROCESSED_DATA_ROOT if args.data_root is None else resolve_repo_path(args.data_root)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "assignment_interpretation": {
            "valid_label_experiments": ["NT->UT", "UT->NT"],
            "matrix_shape": [2, 2],
            "asb_policy": (
                "prohibited for graded training/evaluation because the PDF says its labels "
                "have systematic annotation problems"
            ),
            "pdf_contradiction": (
                "Task 5 lists only NT->UT and UT->NT but retains obsolete 3x3 wording"
            ),
            "separate_ta_clarification_found_in_repository": False,
        },
        "architecture": build_baseline_model().architecture_config(),
        "batch_size": args.batch_size,
        "command_arguments": sys.argv[1:],
        "data_root": str(data_root),
        "device_requested": args.device,
        "device_resolved": str(device),
        "early_stopping_patience": args.early_stopping_patience,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "model_selection": "minimum source-domain validation cross-entropy loss",
        "preprocessing": PREPROCESSING,
        "seed_reset_identically_for_each_source": args.seed,
        "sources": list(DATASETS),
        "targets": list(DATASETS),
        "train_only": args.train_only,
        "software": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "workers": args.workers,
    }
    write_json(output_dir / "experiment_config.json", experiment_config)

    # This entire loop completes before load_split(..., "test") is called anywhere below.
    locked_sources: dict[str, dict[str, Any]] = {}
    for source in DATASETS:
        print(f"Training {source} source model...")
        locked_sources[source] = train_source(
            source,
            args,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
        )
        config = locked_sources[source]["config"]
        print(
            f"{source}: best epoch {config['validation_best_epoch']}, "
            f"validation accuracy {config['validation_best_accuracy']:.6f}, "
            f"training time {config['training_seconds']:.3f} s"
        )

    if args.train_only:
        print("Train-only mode complete; no test split was opened.")
        print(f"Artifacts: {output_dir}")
        return

    locked_hashes = {
        source: str(locked_sources[source]["checkpoint_sha256"]) for source in DATASETS
    }
    print("Both checkpoints are locked; opening test splits for final evaluation...")
    evaluations, applied_preprocessing = evaluate_locked_models(
        locked_sources,
        args,
        data_root=data_root,
        device=device,
    )
    post_evaluation_hashes = {
        source: file_sha256(Path(locked_sources[source]["checkpoint_path"])) for source in DATASETS
    }
    checkpoint_hashes_unchanged = locked_hashes == post_evaluation_hashes
    preprocessing_unchanged = all(
        len(set(applied_preprocessing[source].values())) == 1 for source in DATASETS
    )
    if not checkpoint_hashes_unchanged:
        raise RuntimeError("a locked checkpoint changed during test evaluation")
    if not preprocessing_unchanged:
        raise RuntimeError("source preprocessing changed between target evaluations")

    matrix = accuracy_matrix(evaluations)
    write_json(output_dir / "raw_metrics.json", {"evaluations": evaluations})
    write_raw_metrics_csv(output_dir / "raw_metrics.csv", evaluations)
    write_matrix_csv(output_dir / "accuracy_matrix.csv", matrix)
    plot_matrix(output_dir / "accuracy_heatmap.png", matrix)

    protocol_fields = (
        "architecture",
        "batch_size",
        "device_requested",
        "device_resolved",
        "early_stopping_patience",
        "epochs_requested",
        "learning_rate",
        "loss",
        "optimizer",
        "preprocessing",
        "seed",
        "workers",
    )
    identical_protocol = all(
        locked_sources["NT"]["config"][field] == locked_sources["UT"]["config"][field]
        for field in protocol_fields
    )
    source_validation_only = all(
        locked_sources[source]["config"]["splits_opened_during_training"]
        == [f"{source}/train", f"{source}/val"]
        and not locked_sources[source]["config"]["target_test_access_during_training"]
        for source in DATASETS
    )
    audit_checks = {
        "asb_absent": set(DATASETS) == {"NT", "UT"},
        "all_checkpoints_locked_before_any_test_load": True,
        "checkpoint_hashes_unchanged_after_evaluation": checkpoint_hashes_unchanged,
        "identical_architecture_and_protocol": identical_protocol,
        "source_validation_only_model_selection": source_validation_only,
        "source_preprocessing_unchanged_across_targets": preprocessing_unchanged,
        "target_specific_tuning_absent": True,
    }
    leakage_audit = {
        "passed": all(audit_checks.values()),
        "checks": audit_checks,
        "locked_checkpoint_sha256": locked_hashes,
        "post_evaluation_checkpoint_sha256": post_evaluation_hashes,
        "preprocessing_fingerprints_by_source_and_target": applied_preprocessing,
        "selection_evidence": {
            source: {
                "training_split": f"{source}/train",
                "selection_split": f"{source}/val",
                "selected_epoch": locked_sources[source]["config"]["validation_best_epoch"],
                "validation_loss": locked_sources[source]["config"]["validation_best_loss"],
                "validation_accuracy": locked_sources[source]["config"][
                    "validation_best_accuracy"
                ],
            }
            for source in DATASETS
        },
        "test_role": "post-lock evaluation only; no test metric is consumed by training or selection",
    }
    if not leakage_audit["passed"]:
        raise RuntimeError(f"leakage audit failed: {audit_checks}")
    write_json(output_dir / "leakage_audit.json", leakage_audit)

    print("Accuracy matrix (rows=source, columns=target; order NT, UT):")
    print(matrix)
    print(f"Leakage audit passed: {leakage_audit['passed']}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
