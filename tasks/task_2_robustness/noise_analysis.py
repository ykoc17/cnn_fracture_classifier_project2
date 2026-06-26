"""Evaluate the frozen Task 0 NT checkpoint under deterministic Gaussian noise."""

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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import (
    PROCESSED_DATA_ROOT,
    evaluate_model,
    load_split,
    repo_relative_path,
    resolve_device,
    resolve_repo_path,
    restore_model,
    seed_everything,
)

SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
DEFAULT_CHECKPOINT = Path("tasks/task_0_baseline_cnn/results/best_model.pt")
DEFAULT_REFERENCE_METRICS = Path("tasks/task_0_baseline_cnn/results/metrics.json")
DEFAULT_OUTPUT_DIR = Path("tasks/task_2_robustness/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-metrics", type=Path, default=DEFAULT_REFERENCE_METRICS)
    parser.add_argument("--realizations", type=int, default=5)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gallery-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--failure-drop",
        type=float,
        default=0.20,
        help="Failure when mean accuracy drops by at least this absolute amount from clean accuracy.",
    )
    parser.add_argument(
        "--chance-threshold",
        type=float,
        default=0.60,
        help="Failure when mean accuracy is at or below this near-chance threshold.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.realizations < 1:
        raise ValueError("--realizations must be positive")
    if args.noise_seed < 0:
        raise ValueError("--noise-seed must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if not 0.0 <= args.failure_drop <= 1.0:
        raise ValueError("--failure-drop must be in [0, 1]")
    if not 0.0 <= args.chance_threshold <= 1.0:
        raise ValueError("--chance-threshold must be in [0, 1]")
    if len(set(args.gallery_indices)) != len(args.gallery_indices):
        raise ValueError("--gallery-indices must not contain duplicates")


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


def apply_checkpoint_normalization(
    images: torch.Tensor,
    preprocessing: Mapping[str, Any],
) -> torch.Tensor:
    """Apply normalization only after noise and clamping in the original image domain."""
    normalization = preprocessing.get("normalization")
    if normalization is None:
        return images
    if not isinstance(normalization, Mapping) or not {"mean", "std"} <= normalization.keys():
        raise ValueError(f"unsupported checkpoint normalization metadata: {normalization!r}")
    mean = torch.as_tensor(normalization["mean"], dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(normalization["std"], dtype=images.dtype).view(1, -1, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("normalization standard deviations must be positive")
    return (images - mean) / std


def realization_seed(base_seed: int, sigma_index: int, realization: int) -> int:
    return base_seed + sigma_index * 10_000 + realization


def add_noise(images: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Add deterministic Gaussian noise in [0,1] space and clamp before normalization."""
    if sigma == 0.0:
        return images.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
    return torch.clamp(images + sigma * noise, 0.0, 1.0)


def evaluate_images(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[float, float]:
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    result = evaluate_model(model, loader, device=device, loss_fn=nn.CrossEntropyLoss())
    assert result.loss is not None
    return float(result.metrics["accuracy"]), float(result.loss)


def write_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["sigma", "realization", "noise_seed", "accuracy", "loss"]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    raw_rows: list[dict[str, Any]],
    *,
    clean_accuracy: float,
    failure_drop: float,
    chance_threshold: float,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for sigma in SIGMAS:
        rows = [row for row in raw_rows if row["sigma"] == sigma]
        accuracies = np.asarray([row["accuracy"] for row in rows], dtype=float)
        losses = np.asarray([row["loss"] for row in rows], dtype=float)
        mean_accuracy = float(accuracies.mean())
        drop = clean_accuracy - mean_accuracy
        near_chance = mean_accuracy <= chance_threshold
        excessive_drop = drop >= failure_drop
        reasons: list[str] = []
        if near_chance:
            reasons.append("near_chance")
        if excessive_drop:
            reasons.append("accuracy_drop")
        summary.append(
            {
                "sigma": sigma,
                "realizations": len(rows),
                "mean_accuracy": mean_accuracy,
                "std_accuracy": float(accuracies.std(ddof=1)) if len(rows) > 1 else 0.0,
                "min_accuracy": float(accuracies.min()),
                "max_accuracy": float(accuracies.max()),
                "mean_loss": float(losses.mean()),
                "std_loss": float(losses.std(ddof=1)) if len(rows) > 1 else 0.0,
                "absolute_accuracy_drop": drop,
                "failed": bool(reasons),
                "failure_reasons": reasons,
            }
        )
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sigma",
        "realizations",
        "mean_accuracy",
        "std_accuracy",
        "min_accuracy",
        "max_accuracy",
        "mean_loss",
        "std_loss",
        "absolute_accuracy_drop",
        "failed",
        "failure_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["failure_reasons"] = ";".join(row["failure_reasons"])
            writer.writerow(serialized)


def plot_accuracy(path: Path, summary: list[dict[str, Any]], chance_threshold: float) -> None:
    sigmas = [row["sigma"] for row in summary]
    means = np.asarray([row["mean_accuracy"] for row in summary]) * 100.0
    standard_deviations = np.asarray([row["std_accuracy"] for row in summary]) * 100.0
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        sigmas,
        means,
        yerr=standard_deviations,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Mean accuracy ± 1 SD",
    )
    axis.axhline(chance_threshold * 100.0, color="tab:red", linestyle="--", label="Near-chance threshold")
    axis.set_xlabel("Gaussian noise sigma in [0,1] pixel domain")
    axis.set_ylabel("NT/test accuracy (%)")
    axis.set_title("Task 2: robustness of frozen Task 0 checkpoint")
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_gallery(
    path: Path,
    raw_images: torch.Tensor,
    labels: torch.Tensor,
    indices: list[int],
    base_seed: int,
) -> None:
    figure, axes = plt.subplots(
        len(indices),
        len(SIGMAS),
        figsize=(2.0 * len(SIGMAS), 2.1 * len(indices)),
        squeeze=False,
    )
    for column, sigma in enumerate(SIGMAS):
        seed = realization_seed(base_seed, column, 0)
        noisy = add_noise(raw_images[indices], sigma, seed)
        for row, index in enumerate(indices):
            axis = axes[row, column]
            axis.imshow(noisy[row, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            if row == 0:
                axis.set_title(f"σ={sigma:g}")
            if column == 0:
                axis.set_ylabel(f"index {index}\nlabel {int(labels[index])}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Fixed NT/test samples under deterministic Gaussian noise", y=1.01)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.noise_seed)
    device = resolve_device(args.device)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    reference_metrics_path = resolve_repo_path(args.reference_metrics)
    data_root = PROCESSED_DATA_ROOT if args.data_root is None else resolve_repo_path(args.data_root)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = restore_model(checkpoint_path, device=device)
    if checkpoint["dataset"] != "NT":
        raise ValueError(f"Task 2 requires the frozen Task 0 NT checkpoint, got {checkpoint['dataset']}")
    preprocessing = checkpoint["preprocessing"]
    loaded = load_split("NT", "test", data_root=data_root)
    invalid_gallery_indices = [
        index for index in args.gallery_indices if index < 0 or index >= len(loaded.labels)
    ]
    if invalid_gallery_indices:
        raise IndexError(f"gallery indices out of range: {invalid_gallery_indices}")

    raw_rows: list[dict[str, Any]] = []
    for sigma_index, sigma in enumerate(SIGMAS):
        realization_count = 1 if sigma == 0.0 else args.realizations
        for realization in range(realization_count):
            noise_seed = realization_seed(args.noise_seed, sigma_index, realization)
            noisy_raw = add_noise(loaded.images, sigma, noise_seed)
            model_inputs = apply_checkpoint_normalization(noisy_raw, preprocessing)
            accuracy, loss = evaluate_images(
                model,
                model_inputs,
                loaded.labels,
                batch_size=args.batch_size,
                workers=args.workers,
                device=device,
            )
            raw_rows.append(
                {
                    "sigma": sigma,
                    "realization": realization,
                    "noise_seed": noise_seed,
                    "accuracy": accuracy,
                    "loss": loss,
                }
            )

    clean_accuracy = next(row["accuracy"] for row in raw_rows if row["sigma"] == 0.0)
    with reference_metrics_path.open("r", encoding="utf-8") as input_file:
        reference_accuracy = float(json.load(input_file)["accuracy"])
    if abs(clean_accuracy - reference_accuracy) > 1e-12:
        raise RuntimeError(
            f"sigma=0 accuracy {clean_accuracy} does not reproduce Task 0 accuracy {reference_accuracy}"
        )

    summary = summarize(
        raw_rows,
        clean_accuracy=clean_accuracy,
        failure_drop=args.failure_drop,
        chance_threshold=args.chance_threshold,
    )
    first_failure = next((row for row in summary if row["failed"]), None)
    write_raw_csv(output_dir / "raw_results.csv", raw_rows)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_json(
        output_dir / "summary.json",
        {
            "checkpoint": repo_relative_path(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "dataset": "NT",
            "split": "test",
            "preprocessing": preprocessing,
            "sigma_zero_reproduced": True,
            "task0_reference_accuracy": reference_accuracy,
            "noise_seed": args.noise_seed,
            "nonzero_realizations": args.realizations,
            "failure_definition": {
                "near_chance_mean_accuracy_at_or_below": args.chance_threshold,
                "absolute_accuracy_drop_at_or_above": args.failure_drop,
            },
            "first_failure_sigma": None if first_failure is None else first_failure["sigma"],
            "summary": summary,
        },
    )
    plot_accuracy(output_dir / "accuracy_vs_sigma.png", summary, args.chance_threshold)
    plot_gallery(
        output_dir / "noise_gallery.png",
        loaded.images,
        loaded.labels,
        args.gallery_indices,
        args.noise_seed,
    )

    for row in summary:
        print(
            f"sigma={row['sigma']:g} accuracy={row['mean_accuracy']:.6f} "
            f"std={row['std_accuracy']:.6f} drop={row['absolute_accuracy_drop']:.6f}"
        )
    print(f"First failure sigma: {None if first_failure is None else first_failure['sigma']}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
