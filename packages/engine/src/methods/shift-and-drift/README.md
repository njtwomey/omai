# Shift and drift (part 3, data)

Distribution shift planted and met: covariate, label and concept shift, subpopulation shift and shortcuts, drift over a stream, their detection and the adaptation to them, as sessions over `datasets` modifiers, `metrics`, `stats` (two-sample tests) and the models of `supervised-learning` and `online-learning`.

## Capabilities

- Planting: a shift of each kind between a source and a target, with its size as the knob.
- Detection: two-sample tests (KS, MMD, classifier two-sample), drift detectors on a stream (ADWIN, DDM, Page-Hinkley) stepped as data arrive.
- Effect: a model's score under each shift; importance weighting by density ratio (`unsupervised-learning/density-ratio`).
- Adaptation: reweighting, domain adaptation (`transfer-learning`), retraining windows.

## Examples to make

- A stream drifting at a planted point: the detector's statistic, the alarm, the model's error before and after.
- Covariate shift corrected by importance weights: the weighted fit against the unweighted.
- A shortcut feature that stops working on the target: the model's reliance shown by attributions.

## Rendering

- Curves over a stream with the planted change marked, two-sample histograms, decision surfaces on source and target side by side.
- Missing: the side-by-side comparison panel (source against target) under one cursor.

## Shared

- `shared/`: the shift modifiers with truth (shared with `data-preparation`), the stream base (data arriving step by step) shared with `online-learning` and `time-series`.

## Problems

- New chapter; `trust/` shift methods move here.
