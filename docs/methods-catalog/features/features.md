# Feature engineering and selection (part 3, data)

Features built and chosen: construction (interactions, polynomials, splines, bins, cyclical and date features, aggregates over time and groups), automated synthesis, selection by filter, wrapper and embedded methods, importance measures compared, and the stability and inference of a selection (stability selection, knockoffs, post-selection inference), as sessions over `learn` (preprocessing, feature selection), `explain` (inspection, Shapley), `linear` (paths) and the model slots of `supervised-learning`. Every selection is a study that nests fits on a planted design whose relevant set is known.

## Structure

The chapter lives under `data/features/` (section 3 of `methods/README.md`), one folder per activity, with the selection strategy a mode of its row (the score of a filter, the direction of a wrapper, the penalty of an embedded selector) and the importance measures one row that compares them. The nested fits run on a selection-study base in `shared/`.

```
data/features/
  construction/     interactions and polynomials, splines and bins, cyclical and date features, aggregates over time and groups
  synthesis/        automated feature synthesis: a search over transformations scored by a model
  selection/
    filter/         univariate scores, mutual information, relevance against redundancy (mRMR), variance
    wrapper/        forward and backward selection, recursive feature elimination, with cross-validation
    embedded/       penalties (the lasso path), select-from-model
  importance/       permutation, drop-column, conditional and Shapley importances compared on a planted design
  stability/        stability selection, knockoffs, post-selection inference
  shared/           the importance record and feature-set strip, the selection-study base, the planted designs (below)
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a selection is one feature added, dropped or scored; of a path, one penalty; of a stability study, one subsample; `eval/score` is the cross-validated score of the model slot on the current set.

### construction/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `feature-construction` | Features added one per step (interactions, polynomial terms of a degree, spline bases, bins, cyclical sine-cosine pairs for hours and angles, date parts) with a linear model refitted each time: its fit over the plane, the held-out score rising then overfitting as the degree climbs (the knob), the design matrix's width and condition number. | `features/count` s, `features/added` m, `design/condition` s, `eval/score` s, `model/fit` p, `study/byDegree` proj | `learn/preprocessing/features.ts`: `polynomialFeatures`, `interactionFeatures`, `splineFeatures`, `kBinsDiscretizer`, `functionTransformer` (cyclical and date parts are a `functionTransformer` each; TODO `cyclicalFeatures`, `dateFeatures` beside them); `numerics/linalg/basics.ts`: `cond` | |
| `aggregates` | Rolling and expanding aggregates over a series (mean, standard deviation, quantiles, lags) with the window as the knob, and group aggregates over a key (count, mean, last) joined back to the rows: the features drawn against the target, their importance after a fit; leakage when a window looks ahead. | `window/size` s, `features/matrix` t, `features/importance` t!, `eval/score` s, `features/picture` p | `stats/running.ts`: `momentsOf`, `ewmMean`, `p2Quantile`; `stats/smoothing/smoothers.ts`: `runningStatistic`, `runningMean`; `signal/sliding.ts`: `rollingMeanStd`; `explain/inspection.ts`: `permutationImportance`; TODO `learn/preprocessing/aggregates.ts`: `rollingFeatures`, `groupAggregates` (a few lines over the kernels above) | |

### synthesis/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `feature-synthesis` | A search over transformations (unary: log, square, reciprocal; binary: products, ratios, differences; aggregates across a relation) one candidate per step, each scored by a model's cross-validated gain under a budget (the knob), the beam of the best features so far and the depth of their expressions; against the hand-built set; the explosion of candidates as the depth grows. | `candidate/expression` m, `candidate/gain` s, `beam/features` t!, `beam/depth` s, `eval/score` s, `search/evaluated` s | `learn/preprocessing/features.ts`: `functionTransformer`; `learn/model-selection/cross-validation.ts`: `crossValScore`; `symbolic` for the candidates' expressions and names; TODO `learn/feature-synthesis.ts`: `synthesiseFeaturesAlgorithm` (the candidate grammar and the beam) | |

### selection/filter/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `filter-selection` | The score as a mode (ANOVA F, chi-square, the regression F and r, mutual information, variance), the features ranked and the top k or the top percentile kept (k as the step): a model's score against k, correlated features both kept (redundancy) where mRMR drops one, Relief's neighbour-based weights as the mode that sees interactions; the scores as bars at the viewed step on a planted design whose relevant set is known. | `scores/values` t!, `selection/kept` t!, `selection/k` s, `selection/recall` s (of the planted set), `eval/score` s, `study/byK` proj | `learn/feature-selection.ts`: `fClassif`, `chi2`, `fRegression`, `rRegression`, `mutualInfoClassif`, `mutualInfoRegression`, `selectKBest`, `selectPercentile`, `selectByPValue`, `varianceThreshold`; `datasets/make-classification.ts`: `makeClassification` (informative, redundant and noise columns tracked); TODO `learn/feature-selection.ts`: `mrmr`, `relief` | |

### selection/wrapper/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `wrapper-selection` | The strategy as a mode: forward selection (one feature added per step, the cross-validated score of every candidate), backward elimination, recursive feature elimination (the model's weakest feature dropped per step), each with its cross-validated curve against the set size and the one-standard-error choice; the surviving set as a strip over steps; the nested cost counted (fits per step). | `step/index` s, `step/candidates` t, `step/chosen` m, `selection/set` t!, `eval/score` s, `eval/scoreStd` s, `study/curve` proj, `study/fits` s | `learn/feature-selection-wrappers.ts`: `sequentialFeatureSelector` (forward and backward), `rfe`, `rfecv` (each an `Algorithm` inside; expose them); `learn/model-selection/criteria.ts`: `oneStandardErrorRule`; `learn/model-selection/cross-validation.ts`: `crossValScore` | |

### selection/embedded/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `embedded-selection` | A penalty doing the selecting: the lasso path with features entering as α falls (the step), the elastic net keeping correlated groups where the lasso picks one, select-from-model at a threshold on the coefficients or importances; the selected set against the planted one along the path (recall and false discoveries). | `path/alpha` s, `path/coefficients` t!, `path/active` t!, `selection/recall` s, `selection/falseDiscoveries` s, `study/path` proj | `linear/paths.ts`: `lassoPath`, `enetPath`, `alphaGrid`; `learn/feature-selection.ts`: `selectFromModel`; `exposition/linear.ts`: `regularisationPath` | |
| `see supervised-learning` | The regularisation paths and their cross-validation curves are `fit/regress/` and `fit/classify/`; the row above reads them as a selector. | | | |

### importance/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `importance-measures` | Permutation importance (repeats as the knob, one repeat per step with the estimate's band narrowing), drop-column importance (a refit per feature), conditional permutation importance, the coefficient size of a scaled linear model, SHAP's mean absolute value and SAGE, impurity importance where a tree is the model: the measures disagreeing on correlated features (a planted pair that shares the signal), the bars side by side at the viewed step; which measures the planted design's truth favours. | `importance/permutation` t!, `importance/dropColumn` t!, `importance/conditional` t!, `importance/shap` t!, `importance/repeat` s, `importance/table` p (the `importances` record) | `explain/inspection.ts`: `permutationImportance` (an `Algorithm` inside; expose it), `dropColumnImportance`, `conditionalPermutationImportance`; `explain/shapley.ts`: `samplingShapley`, `linearShap`, `sage`; `exposition/attributions.ts`: `importances`; `data/modifiers/features.ts`: `withCorrelatedFeatures`; impurity importance needs the trees of `supervised-learning` (TODO there) | |

### stability/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `stability-selection` | Subsamples one per step: the lasso (or any selector) fitted on each, the selection frequency of every feature rising, the path of frequencies against the penalty, the threshold (knob) and the expected number of false selections it bounds; the stable set against the planted one. | `subsample/index` s, `subsample/selected` t, `frequency/values` t!, `frequency/path` t!, `selection/stable` t!, `bound/falseSelections` s | `learn/feature-selection-wrappers.ts`: `stabilitySelection` (an `Algorithm` inside; expose it); `linear/paths.ts`: `lassoPath` | |
| `knockoffs` | Model-X knockoffs for a Gaussian design: the knockoff copies constructed (equi-correlated or SDP as modes), the statistics W per feature (lasso coefficient differences), the knockoff filter's threshold at the chosen false discovery rate (the knob), the selected set; the realised FDR over repeats against the target. | `knockoffs/statistics` t!, `knockoffs/threshold` s, `selection/set` t!, `selection/fdr` s, `selection/power` s, `study/byQ` proj | `linear/paths.ts`: `lassoPath`; `numerics/linalg/cholesky.ts`: `cholesky` (the Gaussian knockoffs); TODO `learn/knockoffs.ts`: `gaussianKnockoffs`, `knockoffStatistics`, `knockoffFilter` | |
| `post-selection-inference` | Naive p-values after selection are wrong: the same data selects and tests, the null rejected far too often on pure noise (one simulation per step, the realised Type I rate climbing); data splitting (select on one half, test on the other) and the selective-inference adjustment for the lasso as the fixes; the p-value histograms side by side. | `simulation/index` s, `naive/pValues` t!, `split/pValues` t!, `naive/typeI` s, `split/typeI` s, `pvalues/histogram` p | `linear/inference.ts`: `waldTable`, `olsSummary`; `data/rows.ts`: `trainTestSplit`; TODO `learn/selective-inference.ts`: `lassoSelectiveInference` (the polyhedral lemma) | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `records.ts` | The importance record (several measures of one feature set in one table, from `exposition/attributions.ts`'s `importances`), the feature-set strip (which features are in over steps), the selection-frequency path | every group |
| `study.ts` | The selection-study base: a loop of fits with a changing feature set, the inner fit summarised (its score and its coefficients), the cross-validated score with its band, the fits counted; nested by `wrapper/`, `stability/` and `synthesis/` | `synthesis/`, `selection/wrapper/`, `stability/` |
| `designs.ts` | Planted designs with a known relevant set: `makeClassification`'s informative, redundant, repeated and noise columns, `sparseUncorrelated`, `friedman1`'s five of ten, a correlated pair sharing the signal, groups for the elastic net; recall and false discoveries against them as probes | every group |

## Examples to make

- Recursive feature elimination one feature at a time, the score and the surviving set.
- Importance measures disagreeing on correlated features (permutation against drop-column against conditional).
- Stability selection: selection frequencies rising with the penalty.
- Knockoffs: the knockoff statistics and the threshold at a chosen false discovery rate.
- Forward selection against the lasso path on one planted design: which recovers the set first.

## Rendering

- Bars for importances at the viewed step (`importances`), a path of selection frequencies, the score against the set size, regularisation paths.
- Missing: a feature-set strip (which features are in, over steps); a bars-side-by-side panel for several measures of one set.

## Problems

- New chapter; nothing built. Selection studies nest model fits: `shared/study.ts` is the sweep base summarising the inner tapes, the hard case of `methods/README.md` problem 11.
- Several routines are `Algorithm`s inside a function (`rfe`, `sequentialFeatureSelector`, `stabilitySelection`, `permutationImportance`); exposing them is a line each. Missing outright: mRMR and Relief, knockoffs, selective inference, the synthesis search, and impurity importance (with the trees of `supervised-learning`).
- `importance/` is shared with `trustworthy-machine-learning` (global explanations); it lives here because its comparison is on a planted design, and that chapter carries a `see` row.
