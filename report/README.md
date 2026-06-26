# Reproducible LaTeX report

The report covers completed Tasks 0, 1, 2, 3, 4, and 5. Metrics are generated from saved
task artifacts rather than copied manually.

From the repository root, regenerate the values and manifest:

```powershell
conda run -n cnn_project python .\report\generate_values.py
```

Compile from the report directory so all relative figure paths resolve consistently:

```powershell
Set-Location .\report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
Set-Location ..
```

Files:

- `report.tex` — editable report source.
- `generate_values.py` — artifact reader, cross-task consistency checks, and macro generator.
- `results_values.tex` — generated values consumed by LaTeX.
- `artifact_manifest.json` — SHA-256 hashes and sizes of every metric/figure input.
- `report.pdf` — compiled final report.

Task sections use explicit `\clearpage` boundaries and start/end page labels. The introduction's
task table reports their page ranges after the second LaTeX pass.
