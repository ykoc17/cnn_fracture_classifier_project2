"""Create final-convolution Grad-CAM figures for fixed NT/test samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional
from torch import nn

from src import resolve_device, resolve_repo_path
from tasks.task_3_interpretability.common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SAMPLE_INDICES,
    LABEL_NAMES,
    file_sha256,
    hook_counts,
    load_frozen_task0,
    normalize_map,
    preprocess,
    selection_role,
    validate_indices,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path("tasks/task_3_interpretability/results/gradcam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_SAMPLE_INDICES))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def find_final_convolution(model: nn.Module) -> tuple[str, nn.Conv2d]:
    convolution_layers = [
        (name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)
    ]
    if not convolution_layers:
        raise RuntimeError("model contains no convolutional layer")
    return convolution_layers[-1]


def compute_gradcam(
    model: nn.Module,
    model_input: torch.Tensor,
    layer: nn.Conv2d,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, bool]]:
    activation: torch.Tensor | None = None
    gradient: torch.Tensor | None = None

    def capture_activation(_module, _inputs, output):
        nonlocal activation
        activation = output.detach().clone()

    def capture_gradient(_module, _grad_input, grad_output):
        nonlocal gradient
        gradient = grad_output[0].detach().clone()

    counts_before = hook_counts(model)
    was_training = model.training
    model.eval()
    forward_handle = layer.register_forward_hook(capture_activation)
    backward_handle = layer.register_full_backward_hook(capture_gradient)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(model_input)
            probabilities = torch.softmax(logits.detach(), dim=1)[0]
            predicted_label = int(probabilities.argmax())
            logits[0, predicted_label].backward()
        if activation is None or gradient is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam,
            size=model_input.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = normalize_map(cam)
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    counts_after = hook_counts(model)
    hooks_removed = counts_after == counts_before
    gradients_cleared = all(parameter.grad is None for parameter in model.parameters())
    mode_restored = model.training == was_training
    if not hooks_removed or not gradients_cleared or not mode_restored:
        raise RuntimeError("Grad-CAM hook, gradient, or model-mode cleanup failed")
    return probabilities.cpu(), cam, predicted_label, {
        "hooks_removed": hooks_removed,
        "gradients_cleared": gradients_cleared,
        "evaluation_mode_used": True,
        "original_mode_restored": mode_restored,
    }


def plot_sample(
    path: Path,
    raw_image: torch.Tensor,
    cam: torch.Tensor,
    index: int,
    true_label: int,
    predicted_label: int,
    confidence: float,
) -> None:
    image = raw_image[0].cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original image")
    axes[1].imshow(cam.numpy(), cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(cam.numpy(), cmap="jet", vmin=0.0, vmax=1.0, alpha=0.45)
    axes[2].set_title(f"Model-localized evidence\nfor predicted class {predicted_label}")
    for axis in axes:
        axis.axis("off")
    correctness = "correct" if true_label == predicted_label else "misclassified"
    figure.suptitle(
        f"Grad-CAM — NT/test index {index} — true {true_label} ({LABEL_NAMES[true_label]}), "
        f"predicted {predicted_label} ({LABEL_NAMES[predicted_label]}), {correctness}, "
        f"confidence {confidence:.3f}",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, checkpoint_path, test_data = load_frozen_task0(
        args.checkpoint, args.data_root, device
    )
    validate_indices(args.indices, len(test_data.labels))
    layer_name, layer = find_final_convolution(model)
    entries = []
    verification = None
    for index in args.indices:
        raw_image = test_data.images[index : index + 1]
        model_input = preprocess(raw_image, checkpoint["preprocessing"]).to(device)
        probabilities, cam, predicted_label, verification = compute_gradcam(model, model_input, layer)
        true_label = int(test_data.labels[index])
        confidence = float(probabilities[predicted_label])
        role = selection_role(index, true_label, predicted_label)
        filename = f"gradcam_index_{index:04d}_{role}.png"
        plot_sample(
            output_dir / filename,
            raw_image[0],
            cam,
            index,
            true_label,
            predicted_label,
            confidence,
        )
        entries.append(
            {
                "index": index,
                "selection_role": role,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "target_class": predicted_label,
                "confidence": confidence,
                "class_probabilities": probabilities.tolist(),
                "correct": true_label == predicted_label,
                "figure": filename,
                "cam_min": float(cam.min()),
                "cam_max": float(cam.max()),
            }
        )
        print(f"index={index} role={role} figure={filename}")
    write_json(
        output_dir / "gradcam_metadata.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "dataset": "NT",
            "split": "test",
            "sample_indices": args.indices,
            "selection_policy": "fixed correct class 0, correct class 1, false positive, false negative",
            "preprocessing": checkpoint["preprocessing"],
            "target_layer": layer_name,
            "target_definition": "the model's predicted class for each sample",
            "interpretation_scope": "model-localized evidence, not causal proof",
            "verification": verification,
            "samples": entries,
        },
    )


if __name__ == "__main__":
    main()
