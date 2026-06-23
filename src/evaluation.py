"""Binary classification metrics and model evaluation."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn

from .device import DeviceChoice, resolve_device


@dataclass(frozen=True)
class EvaluationResult:
    loss: float | None
    metrics: dict[str, Any]
    targets: torch.Tensor
    predictions: torch.Tensor
    probabilities: torch.Tensor


def classification_metrics(
    targets: Iterable[int] | np.ndarray | torch.Tensor,
    predictions: Iterable[int] | np.ndarray | torch.Tensor,
) -> dict[str, Any]:
    """Calculate deterministic binary metrics using an explicit class order."""
    y_true = np.asarray(torch.as_tensor(targets).cpu()).reshape(-1)
    y_pred = np.asarray(torch.as_tensor(predictions).cpu()).reshape(-1)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must have the same non-zero length")
    if not np.isin(y_true, (0, 1)).all() or not np.isin(y_pred, (0, 1)).all():
        raise ValueError("targets and predictions must contain only class labels 0 and 1")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in (0, 1)
        },
    }


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: DeviceChoice | str | torch.device = "auto",
    loss_fn: nn.Module | None = None,
) -> EvaluationResult:
    """Evaluate a model without gradients; callers decide whether the loader is val or test."""
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    was_training = model.training
    model.eval()

    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for inputs, batch_targets in loader:
            inputs = inputs.to(resolved_device, non_blocking=True)
            batch_targets = batch_targets.to(resolved_device, non_blocking=True)
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"model must return logits with shape (N, 2), got {tuple(logits.shape)}")
            batch_probabilities = torch.softmax(logits, dim=1)
            batch_predictions = batch_probabilities.argmax(dim=1)
            batch_size = batch_targets.shape[0]
            if loss_fn is not None:
                loss_sum += float(loss_fn(logits, batch_targets).item()) * batch_size
            sample_count += batch_size
            targets.append(batch_targets.cpu())
            predictions.append(batch_predictions.cpu())
            probabilities.append(batch_probabilities.cpu())

    model.train(was_training)
    if sample_count == 0:
        raise ValueError("cannot evaluate an empty DataLoader")
    all_targets = torch.cat(targets)
    all_predictions = torch.cat(predictions)
    return EvaluationResult(
        loss=None if loss_fn is None else loss_sum / sample_count,
        metrics=classification_metrics(all_targets, all_predictions),
        targets=all_targets,
        predictions=all_predictions,
        probabilities=torch.cat(probabilities),
    )
