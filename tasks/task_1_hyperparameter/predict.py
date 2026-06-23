"""Predict the fracture label of one image with the Task 0 best checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct execution from the repository root on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from src import resolve_device, resolve_repo_path, restore_model

DEFAULT_CHECKPOINT = Path("tasks/task_0_baseline_cnn/results/best_model.pt")
LABEL_NAMES = {0: "no fracture", 1: "fracture"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        help="Image path relative to the repository root, or an absolute path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path relative to the repository root, or an absolute path.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_image(image_path: str | Path) -> torch.Tensor:
    """Load one image using the preprocessing locked by the Task 0 run."""
    resolved_path = resolve_repo_path(image_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"image not found: {resolved_path}")

    with Image.open(resolved_path) as image:
        grayscale = image.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


def validate_checkpoint_preprocessing(checkpoint: dict) -> None:
    preprocessing = checkpoint["preprocessing"]
    expected = {
        "input_shape": [1, 128, 128],
        "dtype": "float32",
        "pixel_range": [0.0, 1.0],
        "normalization": None,
        "augmentation": None,
    }
    if preprocessing != expected:
        raise ValueError(
            "checkpoint preprocessing is incompatible with this prediction script: "
            f"expected {expected}, got {preprocessing}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = restore_model(args.checkpoint, device=device)
    if checkpoint["dataset"] != "NT":
        raise ValueError(f"Task 0 prediction requires an NT checkpoint, got {checkpoint['dataset']}")
    validate_checkpoint_preprocessing(checkpoint)

    image = load_image(args.image).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0]
    label = int(probabilities.argmax().item())

    print(f"Predicted label: {label} ({LABEL_NAMES[label]})")
    print(f"Class 0 probability: {float(probabilities[0]):.6f}")
    print(f"Class 1 probability: {float(probabilities[1]):.6f}")


if __name__ == "__main__":
    main()
