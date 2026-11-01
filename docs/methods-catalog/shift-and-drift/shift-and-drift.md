# Shift and drift (part 3, data)

Distribution shift planted and met: covariate, label and concept shift, subpopulation shift and shortcuts between a source and a target, drift over a stream, their detection (two-sample tests, label-shift estimation, stream detectors, monitoring), their effect on a model, and the adaptation to them (importance weighting by density ratios, prior adjustment, retraining windows, conformal intervals under shift), as sessions over the `data` modifiers, `stats` (two-sample and kernel tests), `learn` (conformal, splitters, sample weights), `explain` (attributions) and the models of `supervised-learning` and `online-learning`.

## Structure

The chapter lives under `data/shift/` (section 3 of `methods/README.md`), one folder per activity, with the shift kind a mode of the data, the test and the detector modes of their rows, and the adaptation a mode where several fix the same shift. The modifiers with their truths join the planted-problem catalogue of `data/shared/`; the stream base is `methods/shared/streams.ts` (section 4), shared with `online-learning`, `time-series` and `signal-processing`.

```
data/shift/
  planting/     a shift of each kind between a source and a target (the modifier as a mode), a stream with drift
  detection/    two-sample tests, label-shift estimation, drift detectors on a stream, monitoring a model's error and coverage
  effect/       a model's score under each shift, the shortcut it learned shown by its attributions
  adaptation/   importance weighting by density ratios, prior adjustment, retraining windows, conformal intervals under shift
  shared/       the shift slot, the side-by-side and detector-trace records (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a source-against-target study is the shift's size or one batch of samples; the step of anything on the stream is one arriving point or batch, with the planted drift an event.

### planting/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `shift-modifiers` | A source and a target from one generator with the shift as a mode: covariate shift (the inputs' density moved, the posterior kept), label shift (the priors changed, the class-conditionals kept), concept drift (the posterior changed), subpopulation shift (a group under-represented in the source), a shortcut (a nuisance feature correlated with the label in the source only); the size as the knob; the two datasets side by side with the true posterior of each; the shifted split. | `data/source` t!, `data/target` t!, `shift/size` s, `truth/sourcePosterior` p, `truth/targetPosterior` p, `data/sideBySide` p | `data/modifiers/shift.ts`: `withCovariateShift`, `withConceptDrift`, `shiftedSplit`; `data/modifiers/labels.ts`: `withLabelShift`, `withPrevalence`, `withClassImbalance`; `data/modifiers/features.ts`: `withNuisanceFeatures`, `withCorrelatedFeatures`; `data/truth.ts`: `bayesPredict`; TODO `data/modifiers/shift.ts`: `withSubpopulationShift`, `withShortcut` | |
| `stream` | Data arriving one point or one batch per step with drift planted (abrupt at a point, gradual over a window, recurring; the kind as a mode): the true posterior at each step, the drift marked as an event; the base every detector and learner below runs on. | `stream/t` s, `stream/batch` t, `truth/drift` e, `truth/posterior` p | `methods/shared/streams.ts` over `data/modifiers/shift.ts`: `withConceptDrift`; `datasets/corpus.ts`: `driftingTopics`; `datasets/changepoints.ts`: `changepointSeries` | |

### detection/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `two-sample-tests` | Source against target with the test as a mode (Kolmogorov-Smirnov per feature, Cramér-von Mises, Anderson-Darling, the energy and Wasserstein distances with permutation nulls, Epps-Singleton, MMD with the median-heuristic bandwidth, the classifier two-sample test whose accuracy is the statistic): the statistic against the shift's size (a sweep) and against n (one batch per step), the power curve of each test, the per-feature tests corrected for multiplicity; which features shifted. | `test/statistic` s, `test/pValue` s, `test/n` s, `features/pValues` t!, `null/statistics` t!, `study/power` proj | `stats/tests/two-sample.ts`: `ks2samp`, `cramervonmises2samp`, `andersonKsamp`, `energyTest`, `wassersteinTest`, `eppsSingleton`, `classifierTwoSampleTest`; `stats/tests/kernel.ts`: `mmdTest`, `mmd2`, `medianHeuristic`; `stats/tests/multiple.ts`: `benjaminiHochberg` | |
| `label-shift-estimation` | The target's class priors estimated without target labels: black-box shift estimation from the source confusion matrix and the target's predicted-label histogram, Saerens's EM on the posteriors (one iteration per step, the priors converging); the estimate against the planted priors; the posteriors adjusted and the accuracy recovered. | `estimate/priors` t!, `estimate/error` s, `em/iteration` s, `em/change` s, `eval/adjusted` s, `truth/priors` p | `exposition/evaluation.ts`: `confusionMatrix`; TODO `learn/shift.ts`: `blackBoxShiftEstimate`, `emPriorAdjustmentAlgorithm`, `adjustPriors` (the private `priorShifted` of `data/modifiers/_shared.ts` made public) | |
| `drift-detectors` | On the stream, the detector as a mode (DDM and EDDM on the error rate, ADWIN's adaptive window with its cut as the event, KSWIN's sliding KS test, Page-Hinkley and CUSUM on a statistic): the statistic climbing to its threshold (knob), the alarm against the planted drift (delay, false alarms), ADWIN's window size; the delay against the false-alarm rate as a study. | `detector/statistic` s, `detector/threshold` s, `detector/alarm` e, `detector/delay` s, `detector/window` s, `truth/drift` e, `study/delayAgainstFalseAlarms` proj | TODO `learn/drift.ts`: `ddm`, `eddm`, `adwin`, `kswin` as `Algorithm`s, with `cusum` and `pageHinkley` shared with the time-series chapter's proposed `lib/timeseries/online-detection.ts`; `stats/tests/two-sample.ts`: `ks2samp` for KSWIN | |
| `monitoring` | A model deployed on the stream: its error on the labelled points, the coverage of its conformal intervals falling under shift and tracked back by adaptive conformal inference, the drift of its score distribution without labels (the population stability index, the KS statistic of scores); the detector above triggering a refit. | `monitor/error` s, `monitor/coverage` s, `monitor/scoreShift` s, `monitor/psi` s, `monitor/alarm` e, `model/refit` e | `metrics/classification.ts`: `accuracyScore`; `learn/conformal.ts`: `adaptiveConformalInference`, `intervalCoverage`, `splitConformalClassifier`; `stats/tests/two-sample.ts`: `ks2samp`; TODO `learn/drift.ts`: `populationStabilityIndex` | |

### effect/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `model-under-shift` | One model (the registry's) fitted on the source and scored on the target under each shift (the mode) and size (the knob): the score falling, the decision surface on the source and the target side by side with each one's true posterior, the excess risk against the target's Bayes rate; label shift hurting calibration, covariate shift hurting where the source was sparse. | `eval/source` s, `eval/target` s, `eval/excessRisk` s, `model/surface` p, `data/sideBySide` p, `study/bySize` proj | `data/truth.ts`: `bayesPredict`, `excessRisk`, `bayesErrorEstimate`; `methods/shared/options/models.ts` (the model slot); `exposition/surfaces.ts`: `decisionSurface` | |
| `shortcut-reliance` | A shortcut feature that works on the source and not on the target: the model's reliance shown by permutation importance and SHAP on each, the accuracy gap; the shortcut removed, reweighted or trained against with a worst-group objective, and the reliance moving. | `importance/source` t!, `importance/target` t!, `eval/source` s, `eval/target` s, `eval/worstGroup` s | `explain/inspection.ts`: `permutationImportance`; `explain/shapley.ts`: `samplingShapley`; `data/modifiers/features.ts`: `withCorrelatedFeatures`; `fairness/in-processing.ts`: `exponentiatedGradientReduction` (a worst-group constraint); TODO `withShortcut` as above | |

### adaptation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `importance-weighting` | Covariate shift corrected by density ratios: the ratio estimated by a classifier between source and target (its probabilities turned into weights, the clipping as the knob), KLIEP and uLSIF as the direct modes (one iteration per step, the ratio over the plane converging to the truth the generator gives); the weighted fit against the unweighted on the target (the score, the surfaces); the effective sample size of the weights as the price. | `ratio/estimate` t!, `ratio/error` s, `weights/ess` s, `solver/objective` s, `eval/weighted` s, `eval/unweighted` s, `ratio/surface` p | `stats/tests/two-sample.ts`: `classifierTwoSampleTest` (the classifier); `learn/validation.ts`: `sampleWeightsOf` (weights into `fit`); `sampling/importance.ts`: `importanceEss`; TODO `learn/density-ratio.ts`: `classifierDensityRatio`, `kliepAlgorithm`, `ulsif` (shared with `unsupervised-learning`'s density-ratio row) | |
| `label-shift-correction` | The priors from `label-shift-estimation` applied: the posteriors re-weighted by the prior ratio, the decision threshold moved, accuracy and calibration on the target before and after; against retraining with importance weights by class. | `eval/before` s, `eval/after` s, `calibration/before` p, `calibration/after` p, `priors/ratio` t | TODO `learn/shift.ts`: `adjustPriors`; `learn/calibration.ts`: `plattScaling` (recalibration on the target); `exposition/trust.ts`: `calibrationRecord` | |
| `retraining` | On the stream, the policy as a mode (never, at fixed intervals, when the detector alarms; on a sliding window whose length is the knob, or an expanding one): the error over time under each, the cost in refits, forgetting against stability on recurring drift. | `model/refit` e, `monitor/error` s, `policy/refits` s, `window/size` s, `study/byPolicy` proj | `learn/model-selection/blocked-splitters.ts`: `slidingWindowSplit`, `expandingWindowSplit`; `methods/shared/streams.ts`; the detectors above | |
| `conformal-under-shift` | Coverage promised at 1 − α on the source and broken on the target; weighted conformal prediction with the density ratio restoring it under covariate shift; adaptive conformal inference tracking the miscoverage online under any shift (its step size as the knob), the interval width adjusting. | `interval/coverage` s, `interval/width` s, `interval/alpha` s, `interval/covered` e, `study/coverageAgainstShift` proj | `learn/conformal.ts`: `splitConformalRegressor`, `adaptiveConformalInference`, `intervalCoverage`, `conformalQuantile`; TODO `learn/conformal.ts`: `weightedConformalRegressor` | |
| `see transfer-learning` | Domain adaptation that changes the representation (alignment, adversarial, self-training on the target) is the transfer chapter's; here adaptation reweights or refits. | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `shifts.ts` | The shift slot: each modifier with the truth it leaves (the source and target posteriors, the priors, the drift point, the shortcut column), the size as one shared knob; part of the planted-problem catalogue `data/shared/problems.ts` shared with `data-preparation` and `data-centric-learning` | every group |
| `methods/shared/streams.ts` | The one-point-per-step base with planted drift (abrupt, gradual, recurring), section 4's `streams.ts`, shared with `online-learning`, `time-series` and `signal-processing` | `planting/stream`, `detection/`, `adaptation/retraining` |
| `records.ts` | The side-by-side record (source and target with their surfaces under one cursor), the detector trace with the planted drift marked, the power curve | every group |

## Examples to make

- A stream drifting at a planted point: the detector's statistic, the alarm, the model's error before and after; retraining on the alarm.
- Covariate shift corrected by importance weights: the weighted fit against the unweighted, the ratio against the truth.
- A shortcut feature that stops working on the target: the model's reliance shown by attributions.
- The classifier two-sample test's power against the shift's size, beside MMD's.

## Rendering

- Curves over a stream with the planted change marked, two-sample histograms, decision surfaces on source and target side by side.
- Missing: the side-by-side comparison panel under one cursor; a power-curve panel; the detector trace with events.

## Problems

- New chapter; the `trust/` shift study of the old plan moves here; nothing built. The library lacks the stream detectors (ADWIN, DDM, KSWIN, Page-Hinkley), label-shift estimation, the density-ratio estimators and weighted conformal prediction: one `learn/drift.ts`, one `learn/shift.ts`, one `learn/density-ratio.ts`. The tests and the modifiers exist.
- The side-by-side comparison (source against target, two policies on one stream) is the comparison panel of `methods/README.md` section 5 that twenty chapters ask for; this chapter cannot show its point without it.
- The stream base is shared four ways; it belongs in `methods/shared/streams.ts` rather than in any chapter, and the time-series chapter's `shared/stream.ts` is the same thing.
