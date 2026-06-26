"""Generate LaTeX result macros and an input hash manifest from saved task artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent

DATA_INPUTS = {
    "task0_metrics": "tasks/task_0_baseline_cnn/results/metrics.json",
    "task0_config": "tasks/task_0_baseline_cnn/results/run_config.json",
    "task1_params": "tasks/task_1_hyperparameter/results/best_params.json",
    "task1_metrics": "tasks/task_1_hyperparameter/results/final_metrics.json",
    "task1_config": "tasks/task_1_hyperparameter/results/study_config.json",
    "task2_summary": "tasks/task_2_robustness/results/summary.json",
    "task3_activations": (
        "tasks/task_3_interpretability/results/activation_maps/activation_metadata.json"
    ),
    "task3_gradcam": "tasks/task_3_interpretability/results/gradcam/gradcam_metadata.json",
    "task4_metrics": "tasks/task_4_confusion_matrix/results/metrics.json",
    "task5_metrics": "tasks/task_5_cross_dataset/results/raw_metrics.json",
    "task5_nt_config": "tasks/task_5_cross_dataset/results/NT/run_config.json",
    "task5_ut_config": "tasks/task_5_cross_dataset/results/UT/run_config.json",
    "task5_audit": "tasks/task_5_cross_dataset/results/leakage_audit.json",
}

FIGURES = [
    "tasks/task_0_baseline_cnn/results/training_curves.png",
    "tasks/task_1_hyperparameter/results/optimization_history.png",
    "tasks/task_1_hyperparameter/results/parameter_importance.png",
    "tasks/task_2_robustness/results/accuracy_vs_sigma.png",
    "tasks/task_2_robustness/results/noise_gallery.png",
    (
        "tasks/task_3_interpretability/results/activation_maps/"
        "activation_index_0088_false_positive.png"
    ),
    "tasks/task_3_interpretability/results/gradcam/gradcam_index_0000_correct_class_1.png",
    "tasks/task_3_interpretability/results/gradcam/gradcam_index_0007_false_negative.png",
    "tasks/task_3_interpretability/results/gradcam/gradcam_index_0088_false_positive.png",
    "tasks/task_4_confusion_matrix/results/confusion_matrix_normalized.png",
    "tasks/task_4_confusion_matrix/results/misclassified_gallery.png",
    "tasks/task_5_cross_dataset/results/accuracy_heatmap.png",
]


def read_json(relative_path: str) -> dict[str, Any]:
    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required report input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def decimal(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def command(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def require_close(left: float, right: float, description: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"inconsistent saved metrics for {description}: {left} != {right}")


def main() -> None:
    values = {name: read_json(path) for name, path in DATA_INPUTS.items()}
    missing_figures = [path for path in FIGURES if not (REPO_ROOT / path).is_file()]
    if missing_figures:
        raise FileNotFoundError(f"required report figures are missing: {missing_figures}")

    t0 = values["task0_metrics"]
    t0_config = values["task0_config"]
    t1_params = values["task1_params"]
    t1 = values["task1_metrics"]
    t1_config = values["task1_config"]
    t2 = values["task2_summary"]
    t3 = values["task3_gradcam"]
    t4 = values["task4_metrics"]
    t5_rows = values["task5_metrics"]["evaluations"]
    t5_nt = values["task5_nt_config"]
    t5_ut = values["task5_ut_config"]
    t5_audit = values["task5_audit"]

    t5 = {(row["source_dataset"], row["target_dataset"]): row for row in t5_rows}
    if set(t5) != {(source, target) for source in ("NT", "UT") for target in ("NT", "UT")}:
        raise ValueError("Task 5 must contain exactly the four NT/UT evaluations")
    require_close(t0["accuracy"], t2["task0_reference_accuracy"], "Task 0 / Task 2")
    require_close(t0["accuracy"], t4["accuracy"], "Task 0 / Task 4")
    require_close(t0["accuracy"], t5[("NT", "NT")]["accuracy"], "Task 0 / Task 5")
    if not t2["sigma_zero_reproduced"] or not t4["verification"]["task_0_accuracy_matches"]:
        raise ValueError("saved Task 0 reproduction checks did not pass")
    if not t5_audit["passed"] or not all(t5_audit["checks"].values()):
        raise ValueError("saved Task 5 leakage audit did not pass")

    t0_total = sum(item["support"] for item in t0["per_class"].values())
    t0_correct = sum(t0["confusion_matrix"][index][index] for index in (0, 1))
    t1_test = t1["final_test"]
    t1_selected = t1_params["best_params"]
    t1_correct = sum(t1_test["confusion_matrix"][index][index] for index in (0, 1))
    t1_total = sum(item["support"] for item in t1_test["per_class"].values())
    t3_by_role = {sample["selection_role"]: sample for sample in t3["samples"]}

    lines = [
        "% Generated by report/generate_values.py; do not edit metrics by hand.",
        command("TaskZeroAccuracy", pct(t0["accuracy"])),
        command("TaskZeroLoss", decimal(t0["loss"], 4)),
        command("TaskZeroCorrect", t0_correct),
        command("TaskZeroTestSize", t0_total),
        command("TaskZeroValAccuracy", pct(t0["checkpoint_validation_metric"]["accuracy"])),
        command("TaskZeroBestEpoch", t0_config["validation_best_epoch"]),
        command("TaskZeroEpochsRun", t0_config["epochs_completed"]),
        command("TaskZeroTrainingSeconds", decimal(t0_config["training_seconds"], 2)),
        command("TaskZeroCMZeroZero", t0["confusion_matrix"][0][0]),
        command("TaskZeroCMZeroOne", t0["confusion_matrix"][0][1]),
        command("TaskZeroCMOneZero", t0["confusion_matrix"][1][0]),
        command("TaskZeroCMOneOne", t0["confusion_matrix"][1][1]),
        command("TaskOneTrials", t1_params["total_trials"]),
        command("TaskOneComplete", t1_params["state_counts"]["COMPLETE"]),
        command("TaskOnePruned", t1_params["state_counts"]["PRUNED"]),
        command("TaskOneTrialEpochs", t1_params["trial_epochs"]),
        command("TaskOneBestTrial", t1_params["best_trial_number"]),
        command("TaskOneValAccuracy", pct(t1_params["best_validation_accuracy"])),
        command("TaskOneTestAccuracy", pct(t1_test["accuracy"])),
        command("TaskOneTestLoss", decimal(t1_test["loss"], 4)),
        command("TaskOneCorrect", t1_correct),
        command("TaskOneTestSize", t1_total),
        command("TaskOneGain", f"{100.0 * (t1_test['accuracy'] - t0['accuracy']):.2f}"),
        command("TaskOneBestEpoch", t1["final_model_selection"]["checkpoint_epoch"]),
        command("TaskOneTrainingSeconds", decimal(t1_config["final_training_seconds"], 2)),
        command("TaskOneLearningRate", f"{t1_selected['learning_rate']:.6f}"),
        command("TaskOneBatchSize", t1_selected["batch_size"]),
        command("TaskOneOptimizer", t1_selected["optimizer"]),
        command("TaskOneWeightDecay", f"{t1_selected['weight_decay']:.2e}"),
        command("TaskOneDepth", t1_selected["depth"]),
        command("TaskOneChannels", t1_selected["base_channels"]),
        command("TaskOneDropout", f"{t1_selected['dropout']:.1f}"),
        command(
            "TaskOneLRImportance",
            pct(t1_params["parameter_importances"]["learning_rate"]),
        ),
        command(
            "TaskOneBatchImportance",
            pct(t1_params["parameter_importances"]["batch_size"]),
        ),
        command("TaskTwoFirstFailure", f"{t2['first_failure_sigma']:.2f}"),
        command("TaskTwoCleanAccuracy", pct(t2["summary"][0]["mean_accuracy"])),
        command("TaskTwoRealizations", t2["nonzero_realizations"]),
        command("TaskTwoFailureDrop", pct(t2["failure_definition"]["absolute_accuracy_drop_at_or_above"])),
        command("TaskTwoChanceThreshold", pct(t2["failure_definition"]["near_chance_mean_accuracy_at_or_below"])),
        command("TaskThreeCorrectZeroIndex", t3_by_role["correct_class_0"]["index"]),
        command("TaskThreeCorrectOneIndex", t3_by_role["correct_class_1"]["index"]),
        command("TaskThreeFalsePositiveIndex", t3_by_role["false_positive"]["index"]),
        command("TaskThreeFalseNegativeIndex", t3_by_role["false_negative"]["index"]),
        command("TaskThreeFalsePositiveConfidence", pct(t3_by_role["false_positive"]["confidence"])),
        command("TaskThreeFalseNegativeConfidence", pct(t3_by_role["false_negative"]["confidence"])),
        command("TaskFourAccuracy", pct(t4["accuracy"])),
        command("TaskFourFalsePositives", t4["errors"]["false_positive_count"]),
        command("TaskFourFalseNegatives", t4["errors"]["false_negative_count"]),
        command("TaskFourMacroFOne", pct(t4["macro_average"]["f1"])),
        command("TaskFourClassZeroPrecision", pct(t4["per_class"]["0"]["precision"])),
        command("TaskFourClassZeroRecall", pct(t4["per_class"]["0"]["recall"])),
        command("TaskFourClassZeroFOne", pct(t4["per_class"]["0"]["f1"])),
        command("TaskFourClassZeroSupport", t4["per_class"]["0"]["support"]),
        command("TaskFourClassOnePrecision", pct(t4["per_class"]["1"]["precision"])),
        command("TaskFourClassOneRecall", pct(t4["per_class"]["1"]["recall"])),
        command("TaskFourClassOneFOne", pct(t4["per_class"]["1"]["f1"])),
        command("TaskFourClassOneSupport", t4["per_class"]["1"]["support"]),
        command("TaskFiveNTNT", pct(t5[("NT", "NT")]["accuracy"])),
        command("TaskFiveNTUT", pct(t5[("NT", "UT")]["accuracy"])),
        command("TaskFiveUTNT", pct(t5[("UT", "NT")]["accuracy"])),
        command("TaskFiveUTUT", pct(t5[("UT", "UT")]["accuracy"])),
        command("TaskFiveNTUTLoss", decimal(t5[("NT", "UT")]["loss"], 4)),
        command("TaskFiveUTNTLoss", decimal(t5[("UT", "NT")]["loss"], 4)),
        command(
            "TaskFiveNTUTSourceDrop",
            f"{100.0 * (t5[('NT', 'NT')]['accuracy'] - t5[('NT', 'UT')]['accuracy']):.2f}",
        ),
        command(
            "TaskFiveNTUTTargetGap",
            f"{100.0 * (t5[('UT', 'UT')]['accuracy'] - t5[('NT', 'UT')]['accuracy']):.2f}",
        ),
        command("TaskFiveNTBestEpoch", t5_nt["validation_best_epoch"]),
        command("TaskFiveUTBestEpoch", t5_ut["validation_best_epoch"]),
        command("TaskFiveNTTrainingSeconds", decimal(t5_nt["training_seconds"], 2)),
        command("TaskFiveUTTrainingSeconds", decimal(t5_ut["training_seconds"], 2)),
        command("TaskFiveNTUTClassZeroErrors", t5[("NT", "UT")]["confusion_matrix"][0][1]),
        command("TaskFiveNTUTClassOneErrors", t5[("NT", "UT")]["confusion_matrix"][1][0]),
    ]

    task_two_rows = [r"\newcommand{\TaskTwoRows}{%"]
    for row in t2["summary"]:
        failed = r"\textbf{Yes}" if row["failed"] else "No"
        task_two_rows.append(
            f"{row['sigma']:.2f} & {pct(row['mean_accuracy'])} & "
            f"{100.0 * row['std_accuracy']:.2f} & "
            f"{100.0 * row['absolute_accuracy_drop']:.2f} & {failed} \\\\" 
        )
    task_two_rows.append("}")
    lines.extend(task_two_rows)

    task_three_rows = [r"\newcommand{\TaskThreeRows}{%"]
    roles = (
        ("correct_class_0", "Correct class 0"),
        ("correct_class_1", "Correct class 1"),
        ("false_positive", "False positive"),
        ("false_negative", "False negative"),
    )
    for key, role in roles:
        sample = t3_by_role[key]
        task_three_rows.append(
            f"{sample['index']} & {role} & {sample['true_label']} & "
            f"{sample['predicted_label']} & {pct(sample['confidence'])} \\\\"
        )
    task_three_rows.append("}")
    lines.extend(task_three_rows)

    output_path = REPORT_DIR / "results_values.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_paths = [*DATA_INPUTS.values(), *FIGURES]
    manifest = {
        "generator": "report/generate_values.py",
        "inputs": {
            path: {
                "sha256": sha256(REPO_ROOT / path),
                "size_bytes": (REPO_ROOT / path).stat().st_size,
            }
            for path in manifest_paths
        },
        "output": "report/results_values.tex",
        "cross_checks": {
            "task0_matches_task2_sigma_zero": True,
            "task0_matches_task4": True,
            "task0_matches_task5_nt_to_nt": True,
            "task5_leakage_audit_passed": True,
        },
    }
    (REPORT_DIR / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {output_path}")
    print(f"Validated {len(manifest_paths)} inputs and 4 cross-task consistency checks")


if __name__ == "__main__":
    main()
