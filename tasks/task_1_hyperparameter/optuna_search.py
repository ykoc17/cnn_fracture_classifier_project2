"""Reproducible Optuna search and final NT-only retraining for Task 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
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
import optuna
import sklearn
import torch
from optuna.trial import FrozenTrial, TrialState
from torch import nn

from src import (
    PROCESSED_DATA_ROOT,
    SimpleCNN,
    SimpleCNNConfig,
    create_test_loader,
    create_training_loaders,
    evaluate_model,
    resolve_device,
    resolve_repo_path,
    restore_model,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from src.training import EpochRecord

DATASET = "NT"
DEFAULT_OUTPUT_DIR = Path("tasks/task_1_hyperparameter/results")
PREPROCESSING = {
    "input_shape": [1, 128, 128],
    "dtype": "float32",
    "pixel_range": [0.0, 1.0],
    "normalization": None,
    "augmentation": None,
}
SEARCH_SPACE = {
    "learning_rate": "log-uniform [1e-4, 5e-2]",
    "batch_size": [32, 64, 128],
    "optimizer": ["Adam", "AdamW", "SGD"],
    "weight_decay": "log-uniform [1e-7, 1e-3]",
    "depth": "integer [2, 4]",
    "base_channels": [8, 12, 16],
    "dropout": "[0.0, 0.4] in steps of 0.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--trial-epochs", type=int, default=5)
    parser.add_argument("--min-completed-trials", type=int, default=20)
    parser.add_argument("--final-epochs", type=int, default=30)
    parser.add_argument("--final-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--study-name", default="task1_nt_optuna")
    parser.add_argument(
        "--study-only",
        action="store_true",
        help="Run search artifacts only; do not retrain or open NT/test (used for smoke studies).",
    )
    parser.add_argument(
        "--reset-study",
        action="store_true",
        help="Delete the selected output directory's existing Optuna database before starting.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "trials",
        "trial_epochs",
        "final_epochs",
        "final_patience",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_completed_trials < 0:
        raise ValueError("--min-completed-trials cannot be negative")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def sample_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "depth": trial.suggest_int("depth", 2, 4),
        "base_channels": trial.suggest_categorical("base_channels", [8, 12, 16]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.1),
    }


def model_from_params(params: Mapping[str, Any]) -> SimpleCNN:
    channels = tuple(int(params["base_channels"]) * (2**index) for index in range(int(params["depth"])))
    return SimpleCNN(
        SimpleCNNConfig(
            in_channels=1,
            channels=channels,
            num_classes=2,
            dropout=float(params["dropout"]),
        )
    )


def optimizer_from_params(model: nn.Module, params: Mapping[str, Any]) -> torch.optim.Optimizer:
    common = {
        "lr": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
    }
    name = params["optimizer"]
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), **common)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), **common)
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **common)
    raise ValueError(f"unsupported optimizer: {name}")


def make_objective(
    args: argparse.Namespace,
    *,
    data_root: Path,
    device: torch.device,
):
    def objective(trial: optuna.Trial) -> float:
        # Reset every trial to the same initialization/shuffle seed to reduce comparison noise.
        seed_everything(args.seed)
        params = sample_parameters(trial)
        train_loader, validation_loader = create_training_loaders(
            DATASET,
            batch_size=int(params["batch_size"]),
            seed=args.seed,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            data_root=data_root,
        )
        model = model_from_params(params).to(device)
        optimizer = optimizer_from_params(model, params)
        loss_fn = nn.CrossEntropyLoss()
        best_accuracy = -math.inf
        best_loss = math.inf
        best_epoch = 0

        for epoch in range(1, args.trial_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, loss_fn, device=device)
            validation = evaluate_model(
                model,
                validation_loader,
                device=device,
                loss_fn=loss_fn,
            )
            assert validation.loss is not None
            accuracy = float(validation.metrics["accuracy"])
            if accuracy > best_accuracy or (accuracy == best_accuracy and validation.loss < best_loss):
                best_accuracy = accuracy
                best_loss = validation.loss
                best_epoch = epoch
            trial.report(accuracy, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")

        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_validation_loss", best_loss)
        trial.set_user_attr("channels", list(model.config.channels))
        trial.set_user_attr(
            "parameter_count",
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        )
        return best_accuracy

    return objective


def completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def write_trials_csv(path: Path, study: optuna.Study) -> None:
    parameter_names = sorted({name for trial in study.trials for name in trial.params})
    fields = [
        "number",
        "state",
        "validation_accuracy",
        "duration_seconds",
        "best_epoch",
        "best_validation_loss",
        "parameter_count",
        *parameter_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            duration = None if trial.duration is None else trial.duration.total_seconds()
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "validation_accuracy": trial.value,
                "duration_seconds": duration,
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "best_validation_loss": trial.user_attrs.get("best_validation_loss"),
                "parameter_count": trial.user_attrs.get("parameter_count"),
            }
            row.update(trial.params)
            writer.writerow(row)


def plot_optimization_history(path: Path, study: optuna.Study) -> None:
    complete = sorted(completed_trials(study), key=lambda trial: trial.number)
    numbers = [trial.number for trial in complete]
    values = np.asarray([float(trial.value) for trial in complete])
    running_best = np.maximum.accumulate(values)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(numbers, values, label="Completed trial", alpha=0.8)
    axis.plot(numbers, running_best, label="Best validation accuracy so far", linewidth=2)
    axis.set_xlabel("Optuna trial number")
    axis.set_ylabel("Best NT/val accuracy")
    axis.set_title("Task 1 optimization history (test data excluded)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parameter_importances(study: optuna.Study) -> dict[str, float]:
    try:
        return {
            name: float(value)
            for name, value in optuna.importance.get_param_importances(study).items()
        }
    except (RuntimeError, ValueError, ZeroDivisionError):
        return {}


def plot_parameter_importance(path: Path, importances: Mapping[str, float]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    if importances:
        ordered = sorted(importances.items(), key=lambda item: item[1])
        axis.barh([name for name, _ in ordered], [value for _, value in ordered])
        axis.set_xlabel("fANOVA importance")
    else:
        axis.text(0.5, 0.5, "Insufficient completed trials for importance estimation", ha="center")
        axis.set_axis_off()
    axis.set_title("Task 1 validation-objective parameter importance")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_history(path: Path, history: list[EpochRecord]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record in history:
            writer.writerow({field: getattr(record, field) for field in fields})


def plot_final_history(path: Path, history: list[EpochRecord]) -> None:
    epochs = [record.epoch for record in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    loss_axis.plot(epochs, [record.train_loss for record in history], label="Train")
    loss_axis.plot(epochs, [record.validation_loss for record in history], label="Validation")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()
    accuracy_axis.plot(epochs, [record.train_accuracy for record in history], label="Train")
    accuracy_axis.plot(epochs, [record.validation_accuracy for record in history], label="Validation")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.grid(alpha=0.3)
    accuracy_axis.legend()
    figure.suptitle("Task 1 selected configuration retraining")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_selected_configuration(
    params: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    data_root: Path,
    device: torch.device,
) -> tuple[SimpleCNN, list[EpochRecord], EpochRecord, float]:
    seed_everything(args.seed)
    train_loader, validation_loader = create_training_loaders(
        DATASET,
        batch_size=int(params["batch_size"]),
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        data_root=data_root,
    )
    model = model_from_params(params).to(device)
    optimizer = optimizer_from_params(model, params)
    loss_fn = nn.CrossEntropyLoss()
    history: list[EpochRecord] = []
    best_record: EpochRecord | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    start = time.perf_counter()

    for epoch in range(1, args.final_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
        )
        validation = evaluate_model(
            model,
            validation_loader,
            device=device,
            loss_fn=loss_fn,
        )
        assert validation.loss is not None
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_accuracy=train_metrics["accuracy"],
            validation_loss=validation.loss,
            validation_accuracy=validation.metrics["accuracy"],
        )
        history.append(record)
        improved = best_record is None or record.validation_accuracy > best_record.validation_accuracy
        tied_but_lower_loss = (
            best_record is not None
            and record.validation_accuracy == best_record.validation_accuracy
            and record.validation_loss < best_record.validation_loss
        )
        if improved or tied_but_lower_loss:
            best_record = record
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.final_patience:
                break

    training_seconds = time.perf_counter() - start
    assert best_record is not None and best_state is not None
    model.load_state_dict(best_state)
    return model, history, best_record, training_seconds


def reset_database(db_path: Path) -> None:
    for suffix in ("", "-journal", "-shm", "-wal"):
        candidate = Path(f"{db_path}{suffix}")
        candidate.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    device = resolve_device(args.device)
    data_root = PROCESSED_DATA_ROOT if args.data_root is None else resolve_repo_path(args.data_root)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "optuna_study.db"
    if args.reset_study:
        reset_database(db_path)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    startup_trials = min(5, max(1, args.trials))
    warmup_steps = min(2, args.trial_epochs)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=startup_trials,
        n_warmup_steps=warmup_steps,
    )
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    objective = make_objective(args, data_root=data_root, device=device)
    study.optimize(objective, n_trials=args.trials)

    maximum_total_trials = max(args.trials, args.min_completed_trials * 4)
    while len(completed_trials(study)) < args.min_completed_trials:
        if len(study.trials) >= maximum_total_trials:
            raise RuntimeError(
                f"only {len(completed_trials(study))} completed trials after "
                f"{len(study.trials)} total trials"
            )
        study.optimize(objective, n_trials=1)

    complete = completed_trials(study)
    if not complete:
        raise RuntimeError("the study produced no completed trials")
    write_trials_csv(output_dir / "trials.csv", study)
    plot_optimization_history(output_dir / "optimization_history.png", study)
    importances = parameter_importances(study)
    plot_parameter_importance(output_dir / "parameter_importance.png", importances)

    state_counts = {
        state.name: sum(trial.state == state for trial in study.trials)
        for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    }
    best_summary = {
        "best_params": study.best_trial.params,
        "best_trial_number": study.best_trial.number,
        "best_validation_accuracy": study.best_value,
        "parameter_importances": importances,
        "search_space": SEARCH_SPACE,
        "seed": args.seed,
        "state_counts": state_counts,
        "total_trials": len(study.trials),
        "trial_epochs": args.trial_epochs,
    }
    write_json(output_dir / "best_params.json", best_summary)
    print(
        f"Study complete: {state_counts['COMPLETE']} completed, "
        f"{state_counts['PRUNED']} pruned, best NT/val accuracy={study.best_value:.6f}"
    )
    if args.study_only:
        print("Study-only mode: NT/test was not opened.")
        return

    selected_params = study.best_trial.params
    model, history, best_record, training_seconds = train_selected_configuration(
        selected_params,
        args,
        data_root=data_root,
        device=device,
    )
    write_history(output_dir / "final_history.csv", history)
    plot_final_history(output_dir / "final_training_curves.png", history)
    checkpoint_path = output_dir / "final_model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        dataset=DATASET,
        seed=args.seed,
        preprocessing=PREPROCESSING,
        epoch=best_record.epoch,
        validation_metric={
            "accuracy": best_record.validation_accuracy,
            "loss": best_record.validation_loss,
        },
    )
    study_config = {
        "architecture": model.architecture_config(),
        "best_trial_number": study.best_trial.number,
        "data_root": str(data_root),
        "dataset": DATASET,
        "device_requested": args.device,
        "device_resolved": str(device),
        "final_early_stopped": len(history) < args.final_epochs,
        "final_epochs_completed": len(history),
        "final_epochs_requested": args.final_epochs,
        "final_patience": args.final_patience,
        "final_training_seconds": training_seconds,
        "final_validation_best_epoch": best_record.epoch,
        "objective": "maximize best NT/val accuracy; NT/test excluded",
        "output_dir": str(output_dir),
        "preprocessing": PREPROCESSING,
        "search_space": SEARCH_SPACE,
        "seed": args.seed,
        "selected_params": selected_params,
        "software": {
            "numpy": np.__version__,
            "optuna": optuna.__version__,
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "split_roles": {
            "trials": ["NT/train", "NT/val"],
            "final_retraining": ["NT/train", "NT/val"],
            "final_evaluation": "NT/test after checkpoint lock",
        },
        "state_counts": state_counts,
        "trial_epochs": args.trial_epochs,
        "workers": args.workers,
    }
    write_json(output_dir / "study_config.json", study_config)

    # NT/test is first opened here, after search, retraining, selection, and checkpoint saving.
    restored_model, restored_checkpoint = restore_model(checkpoint_path, device=device)
    test_loader = create_test_loader(
        DATASET,
        batch_size=int(selected_params["batch_size"]),
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        data_root=data_root,
    )
    test_result = evaluate_model(
        restored_model,
        test_loader,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
    )
    assert test_result.loss is not None
    final_metrics = {
        "exploratory_search": {
            "best_trial_number": study.best_trial.number,
            "best_validation_accuracy": study.best_value,
            "completed_trials": state_counts["COMPLETE"],
            "pruned_trials": state_counts["PRUNED"],
            "trial_epochs": args.trial_epochs,
        },
        "final_model_selection": {
            "checkpoint_epoch": restored_checkpoint["epoch"],
            "validation_metric": restored_checkpoint["validation_metric"],
        },
        "final_test": {
            "dataset": DATASET,
            "loss": test_result.loss,
            "split": "test",
            **test_result.metrics,
        },
    }
    write_json(output_dir / "final_metrics.json", final_metrics)
    print(f"Final retraining time: {training_seconds:.3f} s")
    print(f"Final NT/test accuracy: {test_result.metrics['accuracy']:.6f}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
