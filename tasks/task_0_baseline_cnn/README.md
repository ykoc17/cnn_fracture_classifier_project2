# Task 0: Baseline CNN Classifier

## Objective

Train a simple convolutional neural network for binary fracture classification using only
the graded NT dataset. `NT/train` is used for optimization, `NT/val` for model selection
and early stopping, and `NT/test` only for the final locked-checkpoint evaluation.

## Reproduction command

Run from the repository root on Windows:

```powershell
conda run -n cnn_project python .\tasks\task_0_baseline_cnn\train.py --seed 42 --epochs 30 --batch-size 64 --learning-rate 0.001 --device cpu --workers 0 --early-stopping-patience 6 --output-dir tasks/task_0_baseline_cnn/results
```

## Architecture

The baseline has 23,426 trainable parameters:

```text
Input (1 x 128 x 128)
  -> Conv2d(1, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)
  -> Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
  -> Conv2d(32, 64, 3, padding=1) -> ReLU -> AdaptiveAvgPool2d(1)
  -> Flatten -> Linear(64, 2)
```

Images are used in their supplied `float32` `[0, 1]` range. No augmentation or additional
normalization is applied. Training uses Adam, learning rate `0.001`, batch size `64`, and
cross-entropy loss. Loss values are accumulated per batch and divided by the number of
samples, so partial final batches are weighted correctly.

## Actual run results

- Seed: `42`
- Device: CPU (`torch 2.12.1+cpu`)
- Requested epochs: `30`
- Completed epochs: `19` (early stopping patience `6`)
- Best validation-loss epoch: `13`
- Best validation loss: `0.1768424599514313`
- Validation accuracy at the selected epoch: `90.3915%`
- Training time: `112.893 s` (`1 min 52.9 s`)
- Final NT test loss: `0.18987287718353543`
- Final NT test accuracy: `92.1986%` (`260/282` correct)
- Test confusion matrix (`rows=true`, `columns=predicted`): `[[134, 8], [14, 126]]`

The test set was first loaded after preprocessing and validation-based model selection were
locked and `best_model.pt` had been saved. A separate fresh checkpoint load reproduced the
reported test accuracy and loss exactly.

## Artifacts

- `model.py` — Task 0 model definition.
- `train.py` — Reproducible training and final evaluation CLI.
- `predict.py` — Single-image prediction CLI using the saved best model.
- `results/best_model.pt` — Best validation-loss model and complete checkpoint metadata.
- `results/history.csv` — Per-epoch train/validation loss and accuracy.
- `results/metrics.json` — Final NT test loss, accuracy, per-class metrics, and confusion matrix.
- `results/run_config.json` — Full run, environment, preprocessing, and split configuration.
- `results/training_curves.png` — Train/validation loss and accuracy curves.

## Predicting a new image

Provide an image path relative to the repository root. The script converts it to grayscale,
resizes it to `128 x 128`, scales pixels to `[0, 1]`, loads `results/best_model.pt`, and prints
the predicted label and both class probabilities.

```powershell
conda run -n cnn_project python .\tasks\task_0_baseline_cnn\predict.py path\to\image.png
```

Label `0` means **no fracture** and label `1` means **fracture**. An alternative checkpoint or
device can be selected explicitly:

```powershell
conda run -n cnn_project python .\tasks\task_0_baseline_cnn\predict.py path\to\image.png --checkpoint tasks\task_0_baseline_cnn\results\best_model.pt --device auto
```

The prediction is meaningful only for microscopy images sufficiently similar to the NT data
used for training. The printed softmax probabilities are model confidence scores, not calibrated
probabilities.

## Limitations

- This is a deliberately small baseline, not a hyperparameter-optimized model.
- Results are from one fixed seed; variability across seeds has not been measured.
- No data augmentation, class weighting, or additional normalization was studied.
- Evaluation is limited to NT; cross-dataset generalization to UT is not established here.
- CPU timing is machine-dependent.
