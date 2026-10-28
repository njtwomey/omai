# Anomaly detection (part 5, learning paradigms)

Outliers scored: statistical methods (z-scores, Mahalanobis, elliptic envelopes), density and distance methods (kNN distances, local outlier factor), isolation and ensembles (isolation forest, extended variants), one-class methods (one-class SVM, SVDD), deep one-class and reconstruction methods (autoencoders), anomalies in time series, and the evaluation of detectors, as sessions over `stats`, `kernels`, `trees` (isolation), `nn` and `metrics`.

## Capabilities

- Every detector with the contamination and the threshold rule as knobs, scores as a probe over the plane, and the ranking of the data's rows; log the scores and the threshold each step where the method iterates (isolation trees grown one at a time, SVDD's dual steps, an autoencoder's epochs).
- Density-based: LOF's reachability and local densities as probes for a chosen point.
- Isolation forest: the forest growing tree by tree, the path length of a point as a probe.
- One-class: SVDD's sphere and one-class SVM's boundary as the kernel's bandwidth changes.
- Reconstruction: an autoencoder's error map over the plane and per row.
- Time series: anomalies in a planted series by matrix profile (`time-series`), by residuals of a forecast, by a reconstruction window.
- Evaluation: precision at the contamination rate, ROC and PR with the planted anomalies as truth.

## Examples to make

- Isolation forest's score surface sharpening tree by tree, a point's isolation path drawn.
- LOF for a chosen point: its neighbourhood and reachability distances, k as the knob.
- The contamination as the knob: the threshold line on the score histogram, the flagged rows.
- An autoencoder's reconstruction error growing away from the training manifold.

## Rendering

- Score surfaces (`anomalyScores` view exists), rows with scores as colour and flags as markers, threshold explorers, histograms of scores.
- Missing: a per-point explanation probe (the neighbourhood, the isolation path) drawn on the rows.

## Shared

- `shared/`: the outlier detector's standard probes (`methods/shared/kinds`: the score surface, the ranking, the threshold), the planted-anomaly data with truth, the contamination-to-threshold rule.

## Problems

- `anomaly/` moves here. Isolation forests and LOF are not in `lib` yet.
