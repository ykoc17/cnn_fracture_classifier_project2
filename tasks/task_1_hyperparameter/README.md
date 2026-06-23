# Task 1: Optuna Hyperparameter Study

## Objective and data protocol

This task optimizes a configurable CNN using only `NT/train` for gradient updates and
`NT/val` for pruning, trial scoring, hyperparameter selection, and final early stopping.
`NT/test` is not loaded during any Optuna trial. It is evaluated once, after the selected
configuration is retrained from scratch, locked by validation accuracy, saved, and reloaded.

All trials use seed `42`, the same supplied train/validation split, and deterministic data
shuffling. The Optuna objective is the best NT validation accuracy reached within five epochs.

## Search space

| Parameter | Search space |
|---|---|
| Learning rate | Log-uniform `[1e-4, 5e-2]` |
| Batch size | `{32, 64, 128}` |
| Optimizer | `{Adam, AdamW, SGD}`; SGD uses momentum `0.9` and Nesterov |
| Weight decay | Log-uniform `[1e-7, 1e-3]` |
| Convolutional depth | Integer `[2, 4]` |
| Base channel width | `{8, 12, 16}`; later layers double the width |
| Dropout | `{0.0, 0.1, 0.2, 0.3, 0.4}` |

The study uses a seeded TPE sampler and a median pruner with five startup trials and two
warm-up epochs.

## Commands

Run from the repository root on Windows.

Smoke study (two completed trials, no retraining and no NT/test access):

```powershell
conda run -n cnn_project python .\tasks\task_1_hyperparameter\optuna_search.py --trials 2 --trial-epochs 2 --min-completed-trials 2 --seed 42 --device cpu --workers 0 --output-dir tasks/task_1_hyperparameter/smoke_results --study-name task1_nt_smoke --study-only --reset-study
```

Full study and final evaluation:

```powershell
conda run -n cnn_project python .\tasks\task_1_hyperparameter\optuna_search.py --trials 24 --trial-epochs 5 --min-completed-trials 20 --final-epochs 30 --final-patience 6 --seed 42 --device cpu --workers 0 --output-dir tasks/task_1_hyperparameter/results --study-name task1_nt_optuna --reset-study
```

Because pruning reduced the completed-trial count, the script automatically added trials until
20 had completed. The final database contains 53 trials: 20 completed, 33 pruned, and 0 failed.

## Exploratory Optuna results

These are validation scores from short exploratory trials, not test results.

| Trial | Best NT/val accuracy | LR | Batch | Optimizer | Weight decay | Depth | Base channels | Dropout |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 42 | **96.4413%** | 0.0118121 | 32 | AdamW | 9.39e-7 | 3 | 16 | 0.2 |
| 16 | 95.7295% | 0.0123810 | 32 | AdamW | 1.21e-4 | 3 | 16 | 0.1 |
| 31 | 95.3737% | 0.0189530 | 32 | AdamW | 5.68e-6 | 3 | 16 | 0.2 |
| 5 | 95.0178% | 0.00291555 | 64 | Adam | 1.05e-7 | 4 | 16 | 0.0 |
| 17 | 95.0178% | 0.0129032 | 32 | AdamW | 1.53e-4 | 3 | 16 | 0.1 |

Trial 42 supplied the selected configuration:

```json
{
  "learning_rate": 0.011812093209924242,
  "batch_size": 32,
  "optimizer": "AdamW",
  "weight_decay": 9.394678769587759e-07,
  "depth": 3,
  "base_channels": 16,
  "dropout": 0.2
}
```

## Parameter importance

The inspected fANOVA importance estimate was:

| Parameter | Importance |
|---|---:|
| Learning rate | 0.4168 |
| Batch size | 0.3745 |
| Base channels | 0.1141 |
| Weight decay | 0.0643 |
| Dropout | 0.0152 |
| Optimizer | 0.0112 |
| Depth | 0.0039 |

Within this search, learning rate and batch size explain most of the estimated variation in
short-run validation accuracy. Base width and weight decay have smaller effects, while depth,
optimizer identity, and dropout add little estimated importance. This is a local association
from an adaptively sampled and pruned study; it is not a causal or universal ranking.

## Final locked-model result

The selected configuration was retrained from scratch for up to 30 epochs. Validation-accuracy
early stopping with patience 6 stopped after 11 epochs and restored epoch 5.

- Final validation accuracy at selected epoch: **96.4413%**
- Final validation loss: `0.11438820338758285`
- Final retraining time: `47.397 s`
- Final NT test accuracy: **95.3901%** (`269/282` correct)
- Final NT test loss: `0.1473176213549384`
- Test confusion matrix (`rows=true`, `columns=predicted`): `[[138, 4], [9, 131]]`

A separate fresh load of `final_model.pt` reproduced the stored test accuracy and loss exactly.
The 96.4413% exploratory/validation score and 95.3901% final test score are deliberately kept
separate in `final_metrics.json`.

## Artifacts

- `optuna_search.py` — Reproducible search, pruning, retraining, and final evaluation CLI.
- `import_nt_images.py` — Export selected NT NPZ samples as labeled PNG files.
- `predict.py` — Predict a single image with the validation-locked Task 1 model.
- `results/optuna_study.db` — Persistent Optuna SQLite study with all 53 trials.
- `results/trials.csv` — Flattened completed/pruned trial records and parameters.
- `results/best_params.json` — Search space, best trial, selected parameters, and importances.
- `results/optimization_history.png` — Completed-trial validation scores and best-so-far curve.
- `results/parameter_importance.png` — Inspected fANOVA importance plot.
- `results/final_model.pt` — Validation-locked final checkpoint and metadata.
- `results/final_history.csv` — Final retraining train/validation metrics.
- `results/final_training_curves.png` — Final retraining loss and accuracy curves.
- `results/final_metrics.json` — Separated exploratory, final-validation, and final-test results.
- `results/study_config.json` — Full environment, split, search, and retraining configuration.

## Predicting with the selected Task 1 model

The supplied NT samples are stored inside NPZ archives. Export one or more images by passing
their zero-based indices to `import_nt_images.py`. The default split is `test`, and each PNG
filename includes its known label.

Run this exact sequence from the repository root on Windows:

```powershell
# 1. Export NT/test image 1 into the Task 1 imported_images directory.
conda run -n cnn_project python .\tasks\task_1_hyperparameter\import_nt_images.py 1 --split test

# 2. Predict it with results/final_model.pt from the Optuna-selected configuration.
conda run -n cnn_project python .\tasks\task_1_hyperparameter\predict.py tasks\task_1_hyperparameter\imported_images\nt_test_0001_label0.png

# 3. Optionally remove the exported image.
Remove-Item .\tasks\task_1_hyperparameter\imported_images\nt_test_0001_label0.png
```

Label `0` means **no fracture** and label `1` means **fracture**. Image `1` has known label `0`,
and the saved Task 1 model predicts label `0`. The command also prints both class softmax scores.

Multiple images can be exported in one command:

```powershell
conda run -n cnn_project python .\tasks\task_1_hyperparameter\import_nt_images.py 0 1 2 --split test
```

For an external image, pass its repository-relative or absolute path directly:

```powershell
conda run -n cnn_project python .\tasks\task_1_hyperparameter\predict.py path\to\image.png --device auto
```

The script converts the input to grayscale, resizes it to `128 x 128`, scales it to `[0, 1]`,
validates checkpoint preprocessing metadata, and defaults to `results/final_model.pt`. Predictions
are meaningful only for images sufficiently similar to the NT microscopy data. Softmax scores are
not calibrated probabilities.

## Limitations

- Twenty completed trials are meaningful for this project scale but still a modest search.
- Each trial received only five epochs, favoring configurations that learn quickly.
- The TPE sampler and median pruning make fANOVA importances dependent on adaptive sampling.
- Results use one fixed seed; robustness across seeds has not been measured.
- No augmentation or additional normalization was searched.
- The final evaluation covers NT only and does not establish UT generalization.
- CPU timings are machine-dependent.
