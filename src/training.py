"""Training helpers that use train and validation data only."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from .device import DeviceChoice, resolve_device
from .evaluation import evaluate_model


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    device: DeviceChoice | str | torch.device = "auto",
) -> dict[str, float]:
    """Train for one epoch and return sample-weighted loss and accuracy."""
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for inputs, targets in loader:
        inputs = inputs.to(resolved_device, non_blocking=True)
        targets = targets.to(resolved_device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("cannot train on an empty DataLoader")
    return {"loss": loss_sum / sample_count, "accuracy": correct / sample_count}


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    epochs: int,
    device: DeviceChoice | str | torch.device = "auto",
    early_stopping_patience: int | None = None,
) -> list[EpochRecord]:
    """Fit using only train/validation loaders and restore the best validation-loss state."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None")

    resolved_device = resolve_device(device)
    history: list[EpochRecord] = []
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=resolved_device,
        )
        validation = evaluate_model(
            model,
            validation_loader,
            device=resolved_device,
            loss_fn=loss_fn,
        )
        assert validation.loss is not None
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                validation_loss=validation.loss,
                validation_accuracy=validation.metrics["accuracy"],
            )
        )

        if validation.loss < best_validation_loss:
            best_validation_loss = validation.loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
