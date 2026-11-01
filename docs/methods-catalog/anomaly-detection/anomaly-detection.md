# Anomaly detection (part 5, learning paradigms)

Outliers scored: statistical methods (z-scores, Mahalanobis and the elliptic envelope, PCA reconstruction, histogram-based scores), distance and density methods (k-nearest-neighbour distances, the local outlier factor), isolation forests, one-class methods (the one-class SVM, support vector data description), reconstruction by autoencoders, anomalies in series, the combination of scores and the threshold rules, and the evaluation of detectors against planted anomalies, as sessions over `stats`, `distances`, `numerics/spatial`, `numerics/linalg`, `kernels`, `optim/smo`, `nn` and `metrics`, with the isolation forest and the local outlier factor moving to `lib`.

## Structure

Every detector scores rows and flags the ones above a threshold, so the methods live under `fit/detect/` (section 3 of `methods/README.md`), one folder per family; the combination of scores and the threshold rules are a folder of their own because every detector ends in them; the evaluation is a study over detectors. Anomalies in series are `time-series`' (the matrix profile's discords, forecast residuals, change points) and have a `see` row.

```
fit/detect/
  statistical/        z-scores (classical and robust), Mahalanobis with the covariance as a mode (empirical, shrunk, MCD), PCA reconstruction, HBOS
  distance/           the k-nearest-neighbour distance, the local outlier factor, the connectivity-based outlier factor
  isolation/          the isolation forest and its extended variant
  one-class/          the one-class SVM and SVDD with the kernel slot
  reconstruction/     autoencoders, deep SVDD
  series/             see time-series
  combine/            score normalisation and combination, the threshold rules (contamination, quantile, extreme-value)
  evaluate/           precision at the contamination, ROC and PR against the planted anomalies, the threshold explorer
shared/               the planted-anomaly data slot, the threshold rule, the score surface and the per-point evidence probes (below)
```

Nothing is built; the old plan's `anomaly/` folder (isolation, neighbours, statistical, sequences, combine) maps onto the folders above.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. Every detector is of the `outlier` kind on the iterative base and logs `fit/iteration/end` e, `model/params` p and `model/scores` p (the score of every training row); the chapter adds `model/scoreSurface` p (the score over the plane, from `decisionFunction`), `model/flagged` p (the rows above the threshold) and `model/evidence` p (why a point scored: its neighbourhood, its isolation path, its reconstruction) through `shared/probes.ts`, and `eval/averagePrecision` s against the planted anomalies; the tables name what a detector adds beyond these. The step of a detector that iterates is a tree, a pair, an epoch or a concentration step; a closed-form detector runs as one step. The status is `built <path>`, `partly` or empty; library readiness is in the needs column (a function named there exists; `TODO:` names what is missing and its proposed home).

### fit/detect/statistical/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `z-score` | Per-feature z-scores with the location and scale as the mode (mean and standard deviation, or median and MAD: the robust one does not let the outliers inflate the scale), the combination (max, sum of squares) as the knob; on one feature the picture is the histogram with the flagged tail; the masking of one outlier by another as the lesson. | `model/featureScores` p, `model/histogram` p | `stats/descriptive.ts`: `zscore`; `stats/quantiles.ts`: `median`, `medianAbsoluteDeviation`; `stats/histograms.ts`: `histogram` | |
| `mahalanobis` | The squared Mahalanobis distance with the covariance as the mode (empirical, Ledoit-Wolf, OAS, the minimum covariance determinant: the elliptic envelope), the contour at the threshold (a chi-squared quantile at the contamination) as the picture; the empirical ellipse swallowing the outliers (masking) against the MCD's; a planted non-Gaussian cluster where every ellipse fails. | `model/ellipse` p, `model/distances` p, `solver/determinant` s (the MCD mode's concentration steps) | `distances/distances.ts`: `mahalanobisDistance`; `stats/covariance.ts`: `empiricalCovariance`, `ledoitWolf`, `oas`; `numerics/special/cdf.ts` (the chi-squared quantile); TODO fast MCD in `lib/covariance/robust.ts` (as `unsupervised-learning` asks) | |
| `pca-reconstruction` | The reconstruction error from the top-k components as the score (k as the knob), the residual subspace drawn, the rows far from the principal plane; against Mahalanobis (which it approximates with the small components' variances flattened). | `model/components` p, `model/reconstructionError` p, `study/kCurve` proj (average precision against k) | `numerics/linalg/svd.ts`: `svd`, `lowRankApproximation` | |
| `hbos` | A histogram per feature, the score as the sum of negative log bin heights: fast, scale-free, blind to interactions (the planted interaction outlier it misses, which Mahalanobis finds); the bin rule as the knob. | `model/featureHistograms` p, `model/featureScores` p | `stats/histograms.ts`: `histogram`, `histogramBinEdges` | |

### fit/detect/distance/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `knn-distance` | The distance to the k-th neighbour (or the mean over k) as the score, k as the knob, the metric from `distances`; a point's neighbourhood drawn; the index as a knob (brute, kd-tree) with the query cost; the failure on clusters of differing density that the local outlier factor fixes. | `query/neighbours` p (a `neighbourRecord`), `query/cost` s, `study/kCurve` proj | `numerics/spatial/brute.ts`: `kNearest`; `numerics/spatial/trees.ts`: `kdTree`, `ballTree`; `distances/distances.ts`: `distances`; `exposition/neighbours.ts`: `neighbourRecord` | |
| `local-outlier-factor` | Reachability distances and local reachability densities, the factor as the ratio of a point's density to its neighbours' (a probe for a chosen point: its neighbourhood, the reach distances, the densities), k as the knob; the planted dense and sparse clusters where the kNN distance flags the sparse cluster and LOF does not; the connectivity-based outlier factor as the mode. | `query/reachability` p, `query/localDensity` p, `model/factors` p, `study/kCurve` proj | `numerics/spatial/brute.ts`: `kNearest`; `exposition/neighbours.ts`: `neighbourRecord`; TODO `lib/anomaly/neighbours.ts`: `localOutlierFactor`, `connectivityOutlierFactor` | |

### fit/detect/isolation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `isolation-forest` | One tree per step (random feature, random split on a subsample), the average path length as the score normalised by the expected length c(n), the score surface sharpening tree by tree, a point's isolation path drawn through its tree, the subsample size and the tree count as knobs; the extended variant (random hyperplanes) as the mode fixing the axis-parallel bands of the plain one. | `forest/trees` s, `forest/scoreChange` s, `model/isolationPath` p (for a point), `model/depthHistogram` p, `study/treesCurve` proj (average precision against trees) | `random`: `bootstrapIndices`; `exposition/anomaly.ts`: `anomalyScores`; TODO `lib/anomaly/isolation.ts`: `isolationForestAlgorithm` (its trees have no criterion, so they need no `lib/trees`), `extendedIsolationForest` | |

### fit/detect/one-class/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `one-class-svm` | The kernel slot, ν as the knob (the fraction allowed outside), SMO on the one-class dual one pair per step, the support vectors on the boundary appearing; SVDD (the smallest sphere in feature space, C as its knob) as the mode; γ changing how tightly the boundary wraps the data; the margin record of `supervised-learning` reused. | `solver/pair` m, `solver/dualObjective` s, `solver/supportVectors` s, `model/boundary` p (a `marginRecord`) | `optim/smo.ts`: `smoAlgorithm`; `kernels/kernel.ts`: `gram`; `kernels/stationary.ts`: `rbfKernel`; `exposition/margins.ts`: `marginRecord`; TODO the one-class and SVDD duals beside `svcDual` and `svrDual` in `optim/smo.ts` | |

### fit/detect/reconstruction/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `autoencoder` | A small autoencoder trained on the normal rows one epoch per step, the reconstruction error as the score, the error map over the plane growing away from the training manifold, the bottleneck as the knob (too wide and it reconstructs the outliers too); deep SVDD (the encoder trained to pull the normal rows to a centre) as the mode; a variational autoencoder's likelihood is `generative-models`'. | `training/loss` s, `training/epoch` s, `model/errorMap` p, `model/reconstruction` p (for a point), `model/params` p | `nn/module/containers.ts`: `mlp`, `Sequential`; `nn/training/fit.ts`: `fitAlgorithm`; `losses/regression.ts`: `mse`; `data/batching.ts`: `epochBatches`; the trainer base of `methods/shared/bases/` | |

### fit/detect/series/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `series-anomalies` | Discords by the matrix profile, forecast residuals above a band, change points. | | `dynamics/filtering/changepoint.ts`: `bocpd`, `pelt`; `signal/mass.ts`: `distanceProfile` | see `time-series` (`process/signals/series/matrix-profile/discovery`, `change-points/`, `fit/forecast/practice/`) |

### fit/detect/combine/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `score-combination` | Scores from several detectors normalised (z, min-max, Gaussian scaling as modes) and combined (average, maximum, maximum of average, the feature-bagging ensemble) as modes; the rank correlation between detectors as a probe; the combined precision against the best single detector. | `model/normalisedScores` p, `model/rankCorrelation` p, `eval/averagePrecision` s | `stats/descriptive.ts`: `zscore`; `metrics/correlation-agreement.ts` (the rank correlation); TODO `lib/anomaly/combine.ts`: the normalisations and combinations | |
| `threshold-rule` | The threshold as the mode: a contamination rate (the quantile of the scores), a fixed quantile, extreme-value theory on the score tail (a generalised Pareto fitted over a high threshold, the exceedance probability as the rule); the threshold line on the score histogram, the flagged rows, the precision and recall at every threshold as a sweep. | `threshold/value` s, `threshold/flagged` s, `model/scoreHistogram` p, `study/sweep` proj (a `thresholdSweep`) | `stats/quantiles.ts`: `quantile`; `stats/extremes.ts`: `fitGpd`, `peaksOverThreshold`; `exposition/thresholds.ts`: `thresholdSweep`; TODO the contamination-to-threshold rule in `shared/threshold.ts` | |

### fit/detect/evaluate/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `detector-evaluation` | A study over detectors on one dataset with planted anomalies (the contamination and the kind of anomaly as knobs: global, local, clustered): precision at the contamination rate, ROC and PR curves with their areas, the ranking of detectors by average precision, the threshold explorer over any detector's scores; which detector finds which kind. | `detector/averagePrecision` s, `detector/rocAuc` s, `detector/precisionAtContamination` s, `model/prCurve` p, `study/ranking` proj | `metrics/curves.ts`: `rocCurve`, `rocAucScore`; `metrics/precision-recall.ts`: `precisionRecallCurve`, `averagePrecisionScore`; `metrics/operating-points.ts`: `precisionAtFixedRecall`; `exposition/thresholds.ts`: `thresholdSweep`; `datasets/outliers.ts`: `blobsWithOutliers`, `moonsWithOutliers`, `linearOutliers`; `data/modifiers/features.ts`: `withOutliers` | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `data.ts` | The planted-anomaly data slot: generators with the outlier mask as the truth and the contamination and the kind of anomaly (global, local, clustered, an interaction outlier) as knobs, the modifier that plants outliers in any dataset, the real sets with a labelled rare class; from `datasets/outliers.ts` and `data/modifiers/features.ts` | every group |
| `threshold.ts` | The contamination-to-threshold rule and the score normalisation shared by every detector: the flagged set at the viewed step, the threshold as a knob on every page | every detector, `combine/` |
| `probes.ts` | The chapter's probes beyond the kind's: the score surface over the plane (`decisionSurfaceOf` with `decisionFunction`), the ranking of the rows, the per-point evidence (the neighbourhood, the isolation path, the reconstruction, the Mahalanobis contour) drawn on the rows, `eval/averagePrecision` at a schedule | every detector |
| `records.ts` | The `anomalyScores` record with its evidence, a score histogram with the threshold line, an isolation path through a tree | every group |

## Examples to make

- The isolation forest's score surface sharpening tree by tree, a point's isolation path drawn.
- The local outlier factor for a chosen point: its neighbourhood and reachability distances, k as the knob; the kNN distance beside it on clusters of differing density.
- The contamination as the knob: the threshold line on the score histogram, the flagged rows, precision and recall following it.
- An autoencoder's reconstruction error growing away from the training manifold, epoch by epoch.
- Mahalanobis with the empirical covariance against the MCD: masking on the planted outliers.

## Rendering

- Score surfaces (`anomalyScores` view exists), rows with scores as colour and flags as markers, threshold explorers, histograms of scores, margins, neighbours.
- Missing: a per-point explanation probe (the neighbourhood, the isolation path, the reconstruction) drawn on the rows; a detector-comparison panel (several score surfaces under one threshold).

## Problems

- `anomaly/` moves here. The library gap is `lib/anomaly/` (the isolation forest, the local outlier factor, the combinations) plus the one-class duals in `optim/smo.ts` and fast MCD in `lib/covariance/`; the statistical, PCA, histogram and kNN-distance detectors are a few lines each over pieces that exist.
- The threshold is a knob of every detector rather than a method, which is why `combine/` is a folder of rules and `shared/threshold.ts` serves every page; whether `evaluate/` belongs under `analyse/evaluate/` instead is the same question `unsupervised-learning` records for its validity indices.
- Series anomalies are `time-series`' and the autoencoder's generative cousins are `generative-models`'; this chapter keeps the detectors that score rows.
