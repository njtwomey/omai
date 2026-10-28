# Decision theory and calibration (part 4, principles of learning)

Decisions from predictions: expected loss and the Bayes decision, cost-sensitive learning, ROC analysis, proper scoring rules, classifier calibration (Platt, isotonic, temperature, binning) and its evaluation, calibration for regression, conformal prediction and uncertainty quantification, as sessions over `metrics`, `learn` (calibrators) and `stats`.

## Capabilities

- Decision theory: the Bayes decision on known class densities, the cost matrix as knobs, the threshold moving.
- Calibration: a calibrator fitted on held-out scores (Platt's logistic, isotonic, temperature scaling, histogram binning) with the reliability diagram before and after; options: the base model, the calibrator, the bins.
- Evaluation: expected calibration error and its binning sensitivity, reliability over time.
- Conformal: split conformal intervals and sets, coverage as data arrive, online conformal under shift (`trust/` has these).
- Uncertainty: ensembles, dropout at test time, Bayesian last layers; aleatoric against epistemic on a 1-D regression.

## Examples to make

- The cost matrix as knobs: the optimal threshold sliding on the score densities.
- Reliability diagrams before and after each calibrator, on an overconfident network.
- Conformal coverage as the calibration set grows, with the planted miscoverage rate.
- Epistemic uncertainty widening away from the data on a 1-D fit.

## Rendering

- Calibration views (exist), conformal views (exist), score densities with a threshold, bands.
- Missing: the before-and-after comparison panel.

## Shared

- `shared/`: the held-out score slot shared with `evaluation-metrics`, the calibrator slot table.

## Problems

- `trust/` calibration, conformal and uncertainty move here.
