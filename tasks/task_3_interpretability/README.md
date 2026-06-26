# Task 3: Activation Maps and Grad-CAM

## Objective and frozen model

This task inspects the frozen Task 0 NT model with two complementary techniques:

1. **Activation maps** after each of the three convolution/ReLU depths.
2. **Grad-CAM** from the final convolutional layer (`features.6`).

Both scripts reconstruct the architecture and weights from
`tasks/task_0_baseline_cnn/results/best_model.pt` and apply the checkpoint's preprocessing
metadata. They do not retrain or redefine the model.

Grad-CAM is treated as **model-localized evidence for the predicted class**, not causal proof
that a highlighted region physically caused a fracture or uniquely determined the prediction.

## Fixed sample selection

The following stable `NT/test` indices cover correct predictions from both classes and both
available error directions:

| Index | Selection role | True label | Predicted label | Confidence |
|---:|---|---:|---:|---:|
| 1 | Correct class 0 | 0 (no fracture) | 0 | 0.9786 |
| 0 | Correct class 1 | 1 (fracture) | 1 | 0.6462 |
| 88 | False positive | 0 (no fracture) | 1 | 0.9786 |
| 7 | False negative | 1 (fracture) | 0 | 0.6194 |

Exact indices, probabilities, labels, layer names, checkpoint hash, and output filenames are
stored in `activation_metadata.json` and `gradcam_metadata.json`. Activation-map figures use
compact names of the form `act_<index>_<role>.png`, where `role` is `c0`, `c1`, `fp`, or `fn`.

## Exact reproduction commands

Run from the repository root on Windows:

```powershell
conda run -n cnn_project python .\tasks\task_3_interpretability\activation_maps.py --checkpoint tasks/task_0_baseline_cnn/results/best_model.pt --indices 1 0 88 7 --device cpu --output-dir tasks/task_3_interpretability/results/activation_maps

conda run -n cnn_project python .\tasks\task_3_interpretability\gradcam.py --checkpoint tasks/task_0_baseline_cnn/results/best_model.pt --indices 1 0 88 7 --device cpu --output-dir tasks/task_3_interpretability/results/gradcam
```

## Activation-map observations

Each activation figure contains the original image, true and predicted labels, confidence,
class scores, the channel-mean activation, and the three channels with the highest spatial mean
at every depth. Each activation panel is independently min-max scaled, so color intensity should
be compared spatially within a panel, not numerically between panels.

- **Early depth (`features.1`, 16 channels, 128×128):** maps retain fine-grained edges,
  high-frequency specimen texture, and local contrast. Some channels respond to differently
  oriented texture; sample 7 also shows a strong response along the specimen/background edge.
- **Middle depth (`features.4`, 32 channels, 64×64):** responses are smoother and combine local
  texture into broader patches. Individual fine pixels are less prominent, but specimen structure
  and large contrast transitions remain visible.
- **Late depth (`features.7`, 64 channels, 32×32):** maps are coarse and spatially selective.
  The correct class-1 sample and false positive share localized bright patches in the same highly
  active late channels, showing that similar internal evidence can support both a correct fracture
  prediction and a confident error.
- The false-negative sample has a large dark background region and a strong curved boundary. Its
  late activations emphasize that boundary and a few interior patches, but these features result
  in class 0 rather than the true class 1.

## Grad-CAM observations

Grad-CAM targets each sample's predicted class.

- **Correct class 1, index 0:** evidence is concentrated in several compact hotspots along the
  central/lower portion and near the upper-right region rather than covering the entire image.
  Confidence is only 0.646, so the localization accompanies a comparatively uncertain decision.
- **Correct class 0, index 1:** class-0 evidence is more distributed, with a strong upper-right
  hotspot and several weaker peripheral regions. A class-0 heatmap is positive evidence used for
  “no fracture”; it should not be read as a map of where a fracture is absent.
- **False positive, index 88:** the model predicts fracture with 0.979 confidence, but evidence is
  diffuse across multiple textured regions and lacks one uniquely convincing localized structure.
  This is a clear case where confident attention is misleading.
- **False negative, index 7:** class-0 evidence is strongest around parts of the specimen boundary
  and several interior patches while much of the central dark region receives little weight. The
  0.619 confidence indicates ambiguity, and the highlighted areas do not explain why the true
  fracture evidence was missed.

## Interpretability questions from the project brief

### What shapes or patterns is the CNN learning?

The inspected outputs show a progression from oriented fine texture and contrast edges, through
broader texture regions, to coarse localized patches. The final model appears sensitive to local
texture density, brightness/contrast transitions, and specimen boundaries. No single late map
consistently represents a clean human-recognizable fracture shape across all samples.

### How does the CNN understand a fracture or defect?

It does not reason about fracture mechanics. It builds a hierarchy of visual correlations learned
from labeled NT images: early filters detect local texture/edges, deeper filters combine these
responses, and the classifier weights the pooled representation. Grad-CAM shows where the final
convolutional evidence for the chosen class is spatially strongest, but not a physical mechanism.

### Are the learned features interpretable?

They are **partially interpretable**. Early and middle activations visibly track texture and
boundaries, and Grad-CAM provides coarse localization. Interpretation becomes ambiguous at the
late layers: maps are low resolution, independently scaled, and similar hotspots occur in correct
and incorrect cases. The confident false positive demonstrates that plausible-looking attention
can still support a wrong decision. These figures are useful diagnostics, not complete explanations.

## Implementation verification

- All inference is performed in evaluation mode.
- Every forward/backward hook is removed in a `finally` block.
- Grad-CAM parameter gradients are cleared after every sample.
- Scripts compare hook counts and model mode before/after use and raise an error on leakage.
- Metadata records `hooks_removed=true`, `original_mode_restored=true`, and, for Grad-CAM,
  `gradients_cleared=true`.
- Figures are saved at 300 DPI with original image, labels, prediction, confidence, and method output.

## Artifacts

- `activation_maps.py` — Multi-depth activation capture and publication-ready figures.
- `gradcam.py` — Final-convolution Grad-CAM and overlays.
- `common.py` — Frozen-checkpoint loading, preprocessing, fixed selection, and verification helpers.
- `results/activation_maps/activation_metadata.json` — Machine-readable activation-map provenance.
- `results/activation_maps/act_*.png` — Four activation-map figures.
- `results/gradcam/gradcam_metadata.json` — Machine-readable Grad-CAM provenance.
- `results/gradcam/*.png` — Four Grad-CAM figures.

## Limitations

- Four deliberately selected samples cannot represent the full NT/test distribution.
- Top activation channels are selected by spatial mean, which favors broadly active channels.
- Independent min-max scaling hides absolute activation magnitude differences between panels.
- Grad-CAM is limited by the final feature map's 32×32 spatial resolution and interpolation.
- Softmax confidence is not calibrated uncertainty.
- Results explain one frozen Task 0 checkpoint on NT and may not transfer to other models or UT.
