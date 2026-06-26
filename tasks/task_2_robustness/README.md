# Task 2: Robustness Analysis with Gaussian Noise

## Objective and protocol

This task measures how the frozen Task 0 NT model degrades under additive Gaussian noise.
`noise_analysis.py` reconstructs the model from
`tasks/task_0_baseline_cnn/results/best_model.pt` and uses the preprocessing metadata stored
inside that checkpoint; it does not redefine or retrain the network.

For each nonzero noise level, five deterministic realizations are evaluated on the complete
`NT/test` split. Noise is added to the original `float32` `[0, 1]` images, the result is clamped
to `[0, 1]`, and only then is checkpoint normalization applied. The Task 0 checkpoint records
no additional normalization, so the clamped tensors are passed directly to the model.

## Exact command

Run from the repository root on Windows:

```powershell
conda run -n cnn_project python .\tasks\task_2_robustness\noise_analysis.py --checkpoint tasks/task_0_baseline_cnn/results/best_model.pt --reference-metrics tasks/task_0_baseline_cnn/results/metrics.json --realizations 5 --noise-seed 2026 --batch-size 64 --device cpu --workers 0 --gallery-indices 0 1 2 --failure-drop 0.20 --chance-threshold 0.60 --output-dir tasks/task_2_robustness/results
```

## Objective failure definition

The model is considered failed at a noise level when either condition holds:

1. Mean accuracy is at or below `60%`, representing performance close to binary chance.
2. Mean accuracy has dropped by at least `20` absolute percentage points from clean accuracy.

The first failure occurs at **σ = 0.1**, where both conditions are satisfied.

## Actual results

Accuracy standard deviations are sample standard deviations across five realizations. Clean
accuracy uses one evaluation because σ=0 has no random noise realization.

| Sigma | Mean accuracy | Accuracy SD | Drop from clean | Failed? |
|---:|---:|---:|---:|:---:|
| 0.00 | 92.1986% | 0.0000 pp | 0.0000 pp | No |
| 0.05 | 79.0071% | 0.2967 pp | 13.1915 pp | No |
| 0.10 | 50.7801% | 0.1586 pp | 41.4184 pp | **Yes** |
| 0.15 | 50.3546% | 0.0000 pp | 41.8440 pp | **Yes** |
| 0.20 | 50.3546% | 0.0000 pp | 41.8440 pp | **Yes** |
| 0.30 | 50.3546% | 0.0000 pp | 41.8440 pp | **Yes** |
| 0.50 | 50.3546% | 0.0000 pp | 41.8440 pp | **Yes** |

The σ=0 result exactly reproduces the Task 0 test accuracy
`0.9219858156028369`. The checkpoint SHA-256 recorded in `summary.json` is
`9f021dd15f7d3f4ef172b64a54c26bc4c8e6ee14a512a68e53d7cb601b86806b`.

## Interpretation of inspected figures

The accuracy curve shows moderate degradation at σ=0.05, followed by a sharp transition to
near-chance accuracy at σ=0.1. Accuracy remains at 50.3546% from σ=0.15 onward. This plateau is
not robustness: a checked σ=0.15 realization produced confusion matrix `[[142, 0], [140, 0]]`,
meaning the model predicted class 0 for every test image.

In the fixed-sample gallery, σ=0.05 preserves the main specimen textures and shapes while adding
visible grain. At σ=0.1, fine structure is substantially masked. At σ=0.3 and σ=0.5, the images
are dominated by high-frequency speckle and the original morphology is difficult to distinguish.
This visual deterioration is consistent with the sharp accuracy loss, although the collapse at
σ=0.1 occurs while some coarse structure remains visually present.

The five realizations have little accuracy variation. At high sigma, the zero standard deviation
results from the same all-class-0 prediction collapse across realizations, not from reliable noise
invariance.

## Determinism verification

The analysis was rerun with the same seed and arguments. `raw_results.csv`, `summary.csv`, and
`summary.json` were byte-identical between runs, and σ=0 reproduced the stored Task 0 metric.

## Artifacts

- `noise_analysis.py` — Deterministic noise evaluation and plotting CLI.
- `results/raw_results.csv` — Accuracy and loss for every sigma/realization pair.
- `results/summary.csv` — Per-sigma mean, standard deviation, range, drop, and failure status.
- `results/summary.json` — Machine-readable configuration, checkpoint hash, thresholds, and results.
- `results/accuracy_vs_sigma.png` — Accuracy curve with ±1 standard-deviation error bars.
- `results/noise_gallery.png` — Fixed NT/test samples at every evaluated sigma.

## Limitations

- Five realizations provide a useful estimate but not a high-precision uncertainty analysis.
- Additive Gaussian noise is synthetic and may not represent real microscopy acquisition errors.
- The gallery contains only three fixed samples and is not representative by itself.
- Very large sigma values may be unrealistic for the actual imaging process.
- This evaluates one frozen Task 0 model and one seed; it does not compare robust-training methods.
- Results apply to NT only and do not establish robustness on UT.
