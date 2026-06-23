import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Direct execution adds tests/, rather than the repository root, to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoints import restore_model, save_checkpoint
from src.data import load_split
from src.evaluation import classification_metrics
from src.models import SimpleCNN
from src.paths import REPO_ROOT, resolve_repo_path
from src.reproducibility import seed_everything

SCRATCH_ROOT = Path(__file__).resolve().parent / "_scratch"


class DataLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_root = SCRATCH_ROOT
        self.split_directory = self.data_root / "NT"
        self.split_path = self.split_directory / "train.npz"
        self.split_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.split_path.unlink(missing_ok=True)

    @staticmethod
    def valid_arrays() -> tuple[np.ndarray, np.ndarray]:
        images = np.linspace(0.0, 1.0, 4 * 128 * 128, dtype=np.float32).reshape(4, 1, 128, 128)
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        return images, labels

    def test_loads_valid_npz_split(self) -> None:
        images, labels = self.valid_arrays()
        np.savez(self.split_path, images=images, labels=labels)

        loaded = load_split("NT", "train", data_root=self.data_root)

        self.assertEqual(loaded.dataset, "NT")
        self.assertEqual(loaded.split, "train")
        self.assertEqual(tuple(loaded.images.shape), (4, 1, 128, 128))
        self.assertEqual(loaded.images.dtype, torch.float32)
        self.assertEqual(loaded.labels.dtype, torch.int64)

    def test_rejects_asb(self) -> None:
        with self.assertRaisesRegex(ValueError, "ASB is prohibited"):
            load_split("ASB", "train", data_root=self.data_root)

    def test_rejects_invalid_npz_content(self) -> None:
        valid_images, valid_labels = self.valid_arrays()
        invalid_cases = {
            "keys": {"images": valid_images},
            "shape": {"images": valid_images[:, 0], "labels": valid_labels},
            "image dtype": {"images": valid_images.astype(np.float64), "labels": valid_labels},
            "label dtype": {"images": valid_images, "labels": valid_labels.astype(np.int32)},
            "label value": {
                "images": valid_images,
                "labels": np.array([0, 1, 2, 1], dtype=np.int64),
            },
            "pixel range": {"images": valid_images + np.float32(0.1), "labels": valid_labels},
            "finite pixels": {
                "images": np.full_like(valid_images, np.nan),
                "labels": valid_labels,
            },
        }

        for name, arrays in invalid_cases.items():
            with self.subTest(name=name):
                np.savez(self.split_path, **arrays)
                with self.assertRaises(ValueError):
                    load_split("NT", "train", data_root=self.data_root)


class ModelAndMetricTests(unittest.TestCase):
    def test_model_input_output_shape(self) -> None:
        model = SimpleCNN()
        output = model(torch.zeros(4, 1, 128, 128))
        self.assertEqual(tuple(output.shape), (4, 2))

    def test_metric_calculation(self) -> None:
        metrics = classification_metrics(
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 1, 1]),
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["per_class"]["0"]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["1"]["precision"], 2.0 / 3.0)


class ReproducibilityAndCheckpointTests(unittest.TestCase):
    def test_seed_reproducibility(self) -> None:
        seed_everything(17)
        first_numpy = np.random.rand(3)
        first_torch = torch.rand(3)
        seed_everything(17)
        np.testing.assert_array_equal(first_numpy, np.random.rand(3))
        torch.testing.assert_close(first_torch, torch.rand(3), rtol=0.0, atol=0.0)

    def test_checkpoint_round_trip(self) -> None:
        seed_everything(23)
        model = SimpleCNN()
        preprocessing = {"pixel_range": [0.0, 1.0], "normalization": None}

        checkpoint_path = SCRATCH_ROOT / "model.pt"
        checkpoint_path.unlink(missing_ok=True)
        try:
            saved_path = save_checkpoint(
                checkpoint_path,
                model,
                dataset="UT",
                seed=23,
                preprocessing=preprocessing,
                epoch=4,
                validation_metric={"accuracy": 0.75},
            )
            restored, checkpoint = restore_model(saved_path, device="cpu")
        finally:
            checkpoint_path.unlink(missing_ok=True)

        self.assertFalse(restored.training)
        self.assertEqual(checkpoint["dataset"], "UT")
        self.assertEqual(checkpoint["seed"], 23)
        self.assertEqual(checkpoint["preprocessing"], preprocessing)
        self.assertEqual(checkpoint["epoch"], 4)
        self.assertEqual(checkpoint["validation_metric"], {"accuracy": 0.75})
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, restored.state_dict()[name])

    def test_relative_paths_are_repository_rooted(self) -> None:
        expected = (REPO_ROOT / "tasks" / "task_0_baseline_cnn" / "results" / "model.pt").resolve()
        actual = resolve_repo_path("tasks/task_0_baseline_cnn/results/model.pt")
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
