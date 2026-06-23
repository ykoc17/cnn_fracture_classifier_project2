"""Export selected NT NPZ samples as PNG images for manual prediction tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct execution from the repository root on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from src import load_split, resolve_repo_path

DEFAULT_OUTPUT_DIR = Path("tasks/task_0_baseline_cnn/imported_images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indices", type=int, nargs="+", help="One or more zero-based sample indices.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
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
        help="PNG output directory; relative paths are resolved from the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("indices must not contain duplicates")

    loaded = load_split("NT", args.split, data_root=args.data_root)
    sample_count = len(loaded.labels)
    invalid_indices = [index for index in args.indices if index < 0 or index >= sample_count]
    if invalid_indices:
        raise IndexError(
            f"indices out of range for NT/{args.split} with {sample_count} samples: {invalid_indices}"
        )

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in args.indices:
        label = int(loaded.labels[index].item())
        pixels = loaded.images[index, 0].numpy()
        image = Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8), mode="L")
        output_path = output_dir / f"nt_{args.split}_{index:04d}_label{label}.png"
        image.save(output_path)
        print(f"index={index} label={label} path={output_path}")


if __name__ == "__main__":
    main()
