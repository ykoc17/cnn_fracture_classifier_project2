"""Create multi-depth activation-map figures for fixed NT/test samples."""

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

DEFAULT_OUTPUT_DIR = Path("tasks/task_3_interpretability/results/activation_maps")
ROLE_FILENAME_SLUGS = {
    "correct_class_0": "c0",
    "correct_class_1": "c1",
    "false_positive": "fp",
    "false_negative": "fn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_SAMPLE_INDICES))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def capture_activations(
    model: nn.Module,
    model_input: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, bool]]:
    relu_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.ReLU)]
    if len(relu_layers) < 3:
        raise RuntimeError(f"expected at least three ReLU activation depths, found {len(relu_layers)}")
    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    counts_before = hook_counts(model)
    was_training = model.training
    model.eval()
    try:
        for name, module in relu_layers:
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, layer_name=name: activations.__setitem__(
                        layer_name, output.detach().clone()
                    )
                )
            )
        with torch.no_grad():
            logits = model(model_input)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    counts_after = hook_counts(model)
    hooks_removed = counts_after == counts_before
    mode_restored = model.training == was_training
    if not hooks_removed or not mode_restored:
        raise RuntimeError("activation hook cleanup or model-mode restoration failed")
    return logits, activations, {
        "hooks_removed": hooks_removed,
        "evaluation_mode_used": True,
        "original_mode_restored": mode_restored,
    }


def plot_sample(
    path: Path,
    raw_image: torch.Tensor,
    true_label: int,
    predicted_label: int,
    probabilities: torch.Tensor,
    activations: dict[str, torch.Tensor],
    index: int,
) -> list[dict]:
    layer_items = list(activations.items())
    figure, axes = plt.subplots(1 + len(layer_items), 4, figsize=(12, 3 * (1 + len(layer_items))))
    axes[0, 0].imshow(raw_image[0].cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Original NT/test image")
    axes[0, 0].axis("off")
    axes[0, 1].bar(["Class 0", "Class 1"], probabilities.cpu().numpy(), color=["tab:blue", "tab:orange"])
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Softmax scores")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    axes[0, 2].text(
        0.0,
        0.75,
        f"Index: {index}\nTrue: {true_label} ({LABEL_NAMES[true_label]})\n"
        f"Predicted: {predicted_label} ({LABEL_NAMES[predicted_label]})\n"
        f"Confidence: {float(probabilities[predicted_label]):.3f}",
        fontsize=12,
        va="top",
    )

    layer_metadata: list[dict] = []
    for row, (layer_name, activation) in enumerate(layer_items, start=1):
        channel_scores = activation[0].mean(dim=(1, 2))
        top_channels = torch.topk(channel_scores, k=min(3, activation.shape[1])).indices.tolist()
        maps = [activation[0].mean(dim=0), *[activation[0, channel] for channel in top_channels]]
        titles = [f"{layer_name}: channel mean", *[f"channel {channel}" for channel in top_channels]]
        for column, (feature_map, title) in enumerate(zip(maps, titles)):
            axes[row, column].imshow(normalize_map(feature_map).numpy(), cmap="magma", vmin=0.0, vmax=1.0)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
        layer_metadata.append(
            {
                "layer": layer_name,
                "activation_shape": list(activation.shape),
                "displayed_top_channels": top_channels,
            }
        )
    correctness = "correct" if true_label == predicted_label else "misclassified"
    figure.suptitle(
        f"Activation maps — index {index} — true {true_label}, predicted {predicted_label} "
        f"({correctness}), confidence {float(probabilities[predicted_label]):.3f}\n"
        "Each activation panel is independently min-max scaled",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return layer_metadata


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, checkpoint_path, test_data = load_frozen_task0(
        args.checkpoint, args.data_root, device
    )
    validate_indices(args.indices, len(test_data.labels))
    entries = []
    verification = None
    for index in args.indices:
        raw_image = test_data.images[index : index + 1]
        model_input = preprocess(raw_image, checkpoint["preprocessing"]).to(device)
        logits, activations, verification = capture_activations(model, model_input)
        probabilities = torch.softmax(logits.detach().cpu(), dim=1)[0]
        predicted_label = int(probabilities.argmax())
        true_label = int(test_data.labels[index])
        role = selection_role(index, true_label, predicted_label)
        filename = f"act_{index:04d}_{ROLE_FILENAME_SLUGS[role]}.png"
        layer_metadata = plot_sample(
            output_dir / filename,
            raw_image[0],
            true_label,
            predicted_label,
            probabilities,
            activations,
            index,
        )
        entries.append(
            {
                "index": index,
                "selection_role": role,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": float(probabilities[predicted_label]),
                "class_probabilities": probabilities.tolist(),
                "correct": true_label == predicted_label,
                "figure": filename,
                "layers": layer_metadata,
            }
        )
        print(f"index={index} role={role} figure={filename}")
    write_json(
        output_dir / "activation_metadata.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "dataset": "NT",
            "split": "test",
            "sample_indices": args.indices,
            "selection_policy": "fixed correct class 0, correct class 1, false positive, false negative",
            "preprocessing": checkpoint["preprocessing"],
            "visualization": "mean activation and three highest-spatial-mean channels at every ReLU depth",
            "verification": verification,
            "samples": entries,
        },
    )


if __name__ == "__main__":
    main()
