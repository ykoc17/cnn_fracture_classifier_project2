"""
Export .mat files to PyTorch-friendly NPZ format with train/val/test splits.
"""

import scipy.io
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Paths
DATA_DIR = Path("data/mmc1")
OUTPUT_DIR = Path("data/processed")

# Dataset categories
CATEGORIES = ["ASB", "NT", "UT"]


def load_mat_data(category: str) -> tuple[np.ndarray, np.ndarray]:
    """Load images and labels from .mat file."""
    mat_path = DATA_DIR / f"{category}_TestSet_Imgs.mat"
    data = scipy.io.loadmat(str(mat_path))
    
    # Images: (128, 128, N) -> (N, 1, 128, 128) for PyTorch (channel-first)
    images = data["TestImgs"].transpose(2, 0, 1)  # (N, 128, 128)
    images = images[:, np.newaxis, :, :]  # (N, 1, 128, 128)
    images = images.astype(np.float32)
    
    # Labels: (N, 1) -> (N,)
    labels = data["YTest"].flatten().astype(np.int64)
    
    return images, labels


def split_data(images: np.ndarray, labels: np.ndarray) -> dict:
    """Split data into train/val/test sets with stratification."""
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=labels
    )
    
    # Second split: val vs test (50-50 of the remaining)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )
    
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }


def save_split(images: np.ndarray, labels: np.ndarray, path: Path):
    """Save images and labels to NPZ file."""
    np.savez_compressed(
        path,
        images=images,
        labels=labels
    )
    print(f"  Saved: {path} | images: {images.shape}, labels: {labels.shape}")


def export_category(category: str):
    """Export a single category to train/val/test NPZ files."""
    print(f"\nProcessing {category}...")
    
    # Load data
    images, labels = load_mat_data(category)
    print(f"  Loaded: {images.shape[0]} samples")
    
    # Create output directory
    output_dir = OUTPUT_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    splits = split_data(images, labels)
    
    # Save each split
    for split_name, (X, y) in splits.items():
        save_path = output_dir / f"{split_name}.npz"
        save_split(X, y, save_path)
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        dist = ", ".join([f"class {u}: {c}" for u, c in zip(unique, counts)])
        print(f"    Distribution: {dist}")


def main():
    print("=" * 60)
    print("Exporting dataset to PyTorch-friendly format")
    print("=" * 60)
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Split ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"Random seed: {RANDOM_SEED}")
    
    for category in CATEGORIES:
        export_category(category)
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
