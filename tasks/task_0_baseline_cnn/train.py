"""Train and evaluate the Task 0 baseline CNN on the fixed NT splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

# Support both direct execution and ``python -m`` from the repository root.
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

from src import (
    PROCESSED_DATA_ROOT,
    create_test_loader,
    create_training_loaders,
    evaluate_model,
    fit,
    resolve_device,
    resolve_repo_path,
    restore_model,
    save_checkpoint,
    seed_everything,
)
from src.training import EpochRecord
from tasks.task_0_baseline_cnn.model import build_baseline_model

DATASET = "NT"
DEFAULT_OUTPUT_DIR = Path("tasks/task_0_baseline_cnn/results")
PREPROCESSING = {
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
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Processed-data directory; relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory; relative paths are resolved from the repository root.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def write_history(path: Path, history: list[EpochRecord]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow({field: getattr(record, field) for field in fieldnames})


def plot_history(path: Path, history: list[EpochRecord]) -> None:
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
    loss_axis.set_title("Task 0: NT baseline training")
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
    figure.savefig(path, dpi=160)
    plt.close(figure)


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


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    data_root = PROCESSED_DATA_ROOT if args.data_root is None else resolve_repo_path(args.data_root)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader = create_training_loaders(
        DATASET,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        data_root=data_root,
    )
    model = build_baseline_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    training_start = time.perf_counter()
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
    training_seconds = time.perf_counter() - training_start
    best_record = min(history, key=lambda record: record.validation_loss)

    history_path = output_dir / "history.csv"
    plot_path = output_dir / "training_curves.png"
    checkpoint_path = output_dir / "best_model.pt"
    config_path = output_dir / "run_config.json"
    metrics_path = output_dir / "metrics.json"
    write_history(history_path, history)
    plot_history(plot_path, history)
    save_checkpoint(
        checkpoint_path,
        model,
        dataset=DATASET,
        seed=args.seed,
        preprocessing=PREPROCESSING,
        epoch=best_record.epoch,
        validation_metric={
            "loss": best_record.validation_loss,
            "accuracy": best_record.validation_accuracy,
        },
    )

    run_config = {
        "architecture": model.architecture_config(),
        "batch_size": args.batch_size,
        "command_arguments": sys.argv[1:],
        "data_root": str(data_root),
        "dataset": DATASET,
        "device_requested": args.device,
        "device_resolved": str(device),
        "early_stopped": len(history) < args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "epochs_completed": len(history),
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "loss": "CrossEntropyLoss",
        "model_selection": "minimum NT/val cross-entropy loss",
        "optimizer": "Adam",
        "output_dir": str(output_dir),
        "preprocessing": PREPROCESSING,
        "seed": args.seed,
        "software": {
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "splits": {
            "training": "NT/train",
            "model_selection": "NT/val",
            "final_evaluation": "NT/test",
        },
        "training_seconds": training_seconds,
        "validation_best_epoch": best_record.epoch,
        "workers": args.workers,
    }
    write_json(config_path, run_config)

    # Test data is first opened only after preprocessing, selection, and checkpointing are final.
    restored_model, restored_checkpoint = restore_model(checkpoint_path, device=device)
    if restored_checkpoint["preprocessing"] != PREPROCESSING:
        raise RuntimeError("checkpoint preprocessing metadata does not match the locked configuration")
    test_loader = create_test_loader(
        DATASET,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        data_root=data_root,
    )
    evaluation_start = time.perf_counter()
    test_result = evaluate_model(restored_model, test_loader, device=device, loss_fn=loss_fn)
    evaluation_seconds = time.perf_counter() - evaluation_start
    assert test_result.loss is not None
    metrics = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_validation_metric": restored_checkpoint["validation_metric"],
        "dataset": DATASET,
        "evaluation_seconds": evaluation_seconds,
        "loss": test_result.loss,
        "split": "test",
        **test_result.metrics,
    }
    write_json(metrics_path, metrics)

    print(f"Best validation epoch: {best_record.epoch}")
    print(f"Training time: {training_seconds:.3f} s")
    print(f"NT/test loss: {test_result.loss:.6f}")
    print(f"NT/test accuracy: {test_result.metrics['accuracy']:.6f}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
