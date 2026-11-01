# Data-centric learning (part 3, data)

Learning that changes the data rather than the model: labelling (annotation budgets, crowd labels, label noise planted), augmentation and synthetic data, data valuation (Shapley and Banzhaf values of points, influence functions, the removal curves that validate them), data selection (coresets, prototypes, value-driven subsets) and error analysis (error slices, label errors by confident learning, confusion by subgroup), as sessions over `learn` (active learning, resampling, model selection), `explain` (valuation, examples, surrogates), the `data` modifiers, `datasets` (weak labels) and the model slots of `supervised-learning`.

## Structure

The chapter is a theme rather than a folder: its methods fall under four folders of `data/` (section 3 of `methods/README.md`) and one of `analyse/evaluate/`, where a study examines a fitted model's errors on its data. The label models and the query strategies in depth belong to `weak-supervision` and `active-learning` and appear here as `see` rows; what is here is what is done to the data. The chapter's helpers go to `data/shared/` because the chapter spans folders.

```
data/
  labelling/        annotation budgets, crowd labels aggregated, label noise planted and measured
  augmentation/     augmentations on images, tables and audio (the augmentation as a mode) and their effect; synthetic rows
  valuation/        Shapley and Banzhaf values of points, influence functions and TracIn, the removal curves that validate them
  selection/        coresets and subset selection, prototypes and criticisms, value-driven selection
  shared/           planted label noise and annotators, the per-point value record, the budgeted-fit base (below)
analyse/evaluate/
  errors/           error slices by a surrogate tree, label errors by confident learning, confusion by subgroup
```

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a budget is one label bought; of a valuation, one permutation or one solver iteration; of a selection, one point chosen or removed; of an error analysis, one split or one ranked suspect; `rows/values` is the per-point value overlay every valuation and selection draws.

### labelling/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `annotation-budget` | Labels bought one per step under a budget: random against uncertainty, margin, entropy and query-by-committee strategies (the mode) on a pool, the accuracy against labels spent, the points chosen drawn on the plane; the stream variant with a labelling rate; the base case of every data-centric page: which data to label. | `budget/spent` s, `query/index` s, `query/score` s, `eval/accuracy` s, `pool/picture` p, `study/curve` proj | `learn/active-learning.ts`: `activeLearning`, `streamActiveLearning` (each an `Algorithm` inside), `uncertaintySampling`, `marginSampling`, `entropySampling`, `voteEntropySampling`, `expectedErrorReduction`, `rankedBatchSampling` | |
| `see active-learning` | The strategies in depth (their disagreement, their failure modes) are that chapter's; the row above is the budget as a data question. | | | |
| `crowd-labels` | Votes from annotators with planted confusion matrices (their number, accuracy and coverage as knobs): majority vote against the exact posterior the generator gives, the annotators' accuracies recovered from agreement; the label model (Dawid-Skene by EM) is `weak-supervision`'s and runs here on this data. | `votes/matrix` t!, `labels/majority` t!, `labels/accuracy` s, `annotators/accuracy` t!, `truth/posterior` p | `datasets/weak-labels.ts`: `crowdLabels`, `labellingFunctions` (the exact posterior of the true class in the truth) | |
| `see weak-supervision` | Dawid-Skene, the generative label model and their EM are that chapter's. | | | |
| `label-noise` | Noise planted (symmetric or pair-flip, the rate as the knob, the transition matrix as the truth): a model's accuracy and its surface against the clean fit, the noise rate estimated from the model's confident predictions, the memorisation of wrong labels over training (the loss on clean against noisy points per epoch). | `noise/rate` s, `noise/transition` t, `eval/clean` s, `eval/noisy` s, `estimate/rate` s, `training/lossClean` s, `training/lossNoisy` s | `data/modifiers/labels.ts`: `withLabelNoise`, `noiseTransition`, `labelNoiseModifier`; `nn/training/fit.ts`: `fitAlgorithm` (the memorisation curve) | |

### augmentation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `image-augmentation` | On the shapes and glyphs, the augmentation as a mode (rotation, shift, zoom, flips, cutout, Gaussian noise, mixup and CutMix between pairs) with its strength as the knob, a batch drawn as a grid at the viewed step; a small classifier's accuracy with and without, the decision surface of a 2-D analogue smoothing under mixup. | `augment/batch` p (a grid of images), `augment/strength` s, `eval/withAugmentation` s, `eval/without` s, `model/surface` p | `signal/image/geometry.ts`: `rotate`, `shift`, `zoom`, `affineTransform`; `nn/functional/regularisation.ts`: `cutout`, `gaussianNoise`; `datasets/images.ts`: `shapes`, `digitGlyphs`; TODO `nn/functional/regularisation.ts`: `mixup`, `cutmix` | |
| `tabular-augmentation` | Rows synthesised by SMOTE-style interpolation, by noise, or by sampling a generator fitted to the data (a Gaussian mixture, a kernel density): the synthetic rows against the real ones on the plane, a model trained on synthetic and tested on real (the utility), the nearest real neighbour of each synthetic row (the privacy proxy). | `synthetic/rows` t!, `synthetic/count` s, `eval/trainSyntheticTestReal` s, `privacy/nearestDistance` t!, `rows/picture` p | `learn/preprocessing/resampling.ts`: `smote`, `adasyn`; `mixture/gaussian.ts`: `gaussianEmAlgorithm`; `stats/density.ts`: `gaussianKde`; `numerics/spatial/brute.ts`: `kNearest` | |
| `audio-augmentation` | SpecAugment's time and frequency masks, time stretching and pitch shifting, mixing at an SNR, as modes on a clip: the spectrogram before and after, a keyword spotter's accuracy with and without. | `augment/spectrogram` t!, `augment/factor` s, `eval/withAugmentation` s, `eval/without` s | `signal/audio/augmentation.ts`: `specAugment`, `timeStretch`, `pitchShift`, `mixAtSnr`, `addNoise` | |
| `see generative-models` | Synthetic data from a trained generator (a VAE or GAN, its samples' fidelity and diversity) is that chapter's. | | | |

### valuation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `data-shapley` | The value of each training point as its average marginal contribution to the model's score: Monte Carlo over permutations one per step (the estimates settling, their standard error), truncated when the score stops moving (the tolerance as the knob), beta Shapley and Banzhaf as the weightings that differ, exact Shapley for k-nearest neighbours in closed form; the most valuable and most harmful points marked on the rows, the planted label errors among the harmful. | `permutation/index` s, `values/estimate` t!, `values/standardError` t!, `values/converged` s, `rows/values` p, `values/ranking` t! | `explain/valuation.ts`: `dataShapley` (an `Algorithm` inside; expose it), `betaShapley`, `dataBanzhaf`, `knnShapley`, `knnUtility`, `subsetUtility`, `totalLoss` | |
| `influence` | Influence functions on a convex model: the Hessian inverted (or conjugate gradients, one iteration per step), the influence of each training point on the loss of a chosen test point (the probe), the most influential points drawn; TracIn from checkpoints as the training-time mode; representer values for a linear last layer; against leave-one-out retraining on a small set (the truth). | `influence/values` t!, `influence/error` s (against leave-one-out), `solver/residual` s, `test/point` m, `rows/values` p | `explain/valuation.ts`: `influenceFunctions`, `tracIn`, `representerValues`; `autodiff`: `hessian`, `grad`; `numerics/sparse/iterative.ts`: `cgAlgorithm` | |
| `removal-curves` | Values validated by what removing points does: the highest-valued removed first (the score falling fastest), the lowest first (the score rising as the harmful go), random removal as the baseline; one removal per step with a refit; the curves for Shapley, Banzhaf, influence and random overlaid. | `removal/count` s, `removal/order` m, `eval/score` s, `study/curves` proj | `explain/valuation.ts`: `subsetUtility`; `explain/evaluation.ts`: `removeAndRetrain` (removes features today; TODO `removeRowsAndRetrain` beside it) | |

### selection/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `coresets` | A subset standing in for the set; the method as a mode (uniform, stratified, k-center greedy one centre per step with its covering radius falling, sensitivity sampling for k-means, Bayesian coresets' weights): the model on the subset against the full set as the size is the knob, the coreset drawn over the data. | `subset/size` s, `subset/indices` t!, `subset/radius` s, `subset/weights` t!, `eval/subset` s, `eval/full` s, `subset/picture` p, `study/bySize` proj | `data/rows.ts`: `take`, `stratifiedIndices`; `numerics/spatial/brute.ts`: `kNearest`; `cluster/kmeans.ts`: `squaredDistancesToCentres`; TODO `learn/coresets.ts`: `kCenterGreedyAlgorithm`, `sensitivitySampling`, `bayesianCoresetAlgorithm` | |
| `prototypes-and-criticisms` | MMD-critic: prototypes chosen greedily one per step to match the data's distribution (the MMD falling), then criticisms where the prototypes fail; the nearest examples of a chosen point as the probe; the prototypes as a grid of images on the glyphs. | `prototype/index` s, `prototype/mmd` s, `prototypes/set` t!, `criticisms/set` t!, `prototypes/picture` p | `explain/examples.ts`: `mmdCritic`, `nearestExamples`; `kernels/matrices.ts`: `mmdSquared` | |
| `value-driven-selection` | Training on the top-valued points of `valuation/` against a coreset of the same size against random: the learning curve against the subset size for each, the fits counted; where valuation's cost pays. | `subset/size` s, `eval/byValue` s, `eval/coreset` s, `eval/random` s, `study/curves` proj | the rows above; `learn/model-selection/cross-validation.ts`: `learningCurve` | |

### analyse/evaluate/errors/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `error-slices` | A fitted model's errors explained: a surrogate tree fitted to the error indicator (its depth as the knob, one split per step), the leaves as slices with their error rate and size, the subgroup report; the worst slices' rows marked on the data; the planted hard subgroup recovered. | `tree/split` m, `tree/depth` s, `slices/table` t!, `slices/worst` m, `slices/report` p, `rows/errors` p | `explain/surrogates.ts`: `surrogateTree`; `explain/rules.ts`: `treeRules`, `ruleMask`; `exposition/subgroups.ts`: `subgroupReport`, `subgroupGrid`; `exposition/trees.ts`: `treeRecord` | |
| `label-errors` | Confident learning: out-of-sample predicted probabilities, the confident joint between given and true labels (the thresholds per class), the suspected label errors ranked by self-confidence against the planted noise (precision at k), the noise transition estimated against the planted one; trust scores as the simpler ranking. | `joint/matrix` t!, `errors/ranked` t!, `errors/precisionAtK` s, `estimate/transition` t!, `truth/transition` p, `rows/suspects` p | `learn/model-selection/cross-validation.ts`: `crossValPredict`; `explain/examples.ts`: `trustScores`; `data/modifiers/labels.ts`: `withLabelNoise`; TODO `learn/label-noise.ts`: `confidentJoint`, `findLabelIssues`, `estimateNoiseTransition` | |
| `confusion-by-subgroup` | The confusion matrix per group and slice (a protected attribute, a bin of a feature), the rates' disparities, calibration per group; the group report at the viewed threshold. | `groups/confusion` t!, `groups/rates` t!, `groups/disparities` t, `groups/report` p | `exposition/evaluation.ts`: `confusionMatrix`; `exposition/fairness.ts`: `groupReport`, `groupDisparities`, `groupReportAt`; `fairness/group-metrics.ts`: `groupRates`, `calibrationByGroup` | |
| `see trustworthy-machine-learning` | Robust losses, co-teaching and the training-time handling of noise are that chapter's label-noise section; here the errors are found and the data fixed. | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `noise.ts` | Planted label noise (symmetric, pair-flip; instance-dependent TODO) and annotators with their confusion matrices, with the truth (which labels are wrong, the transition, the posterior); part of the planted-problem catalogue `data/shared/problems.ts`, shared with `weak-supervision` and `trustworthy-machine-learning` | `labelling/`, `valuation/`, `errors/` |
| `values.ts` | The per-point value record: a value, a rank and a standard error per row, drawn as colour and markers on the rows panel with a legend scale; the removal curve as a projection | `valuation/`, `selection/`, `errors/` |
| `budget.ts` | The budgeted-fit base: a study whose step refits a model on a changing subset (one permutation, one removal, one prototype), the inner fit summarised to its score, the fits counted against the budget; `methods/shared/sweeps.ts` with a subset as the knob | `labelling/annotation-budget`, `valuation/`, `selection/` |

## Examples to make

- Data Shapley by Monte Carlo: the values settling, the most valuable and most harmful points marked on the rows; the removal curves validating them.
- Confident learning: the suspected label errors ranked against the planted noise.
- Mixup as the knob: the decision surface smoothing.
- k-center greedy choosing a coreset one centre per step, the covering radius falling, the model on it against the full set.
- An annotation budget spent by uncertainty sampling against random, the accuracy against labels bought.

## Rendering

- Rows with per-point values as colour (`rows` overlays), confusion matrices (`ContingencyTable`), decision surfaces, bars by slice, grids of images.
- Missing: a per-point value overlay with a legend scale; a removal-curve panel with several orders overlaid; an image-grid panel for augmented batches.

## Problems

- New chapter; nothing built. It spans five folders (`data/labelling`, `data/augmentation`, `data/valuation`, `data/selection`, `analyse/evaluate/errors`): the chapter is a theme, and its README could split along those lines once building starts.
- Data Shapley nests thousands of fits: the budgeted base must keep only the inner scores, and the models must be the cheap ones (k-nearest neighbours in closed form, a linear model); the worker producer (`methods/README.md` problem 7) before anything larger.
- Library gaps: mixup and CutMix, coresets, confident learning, row-removal curves; `dataShapley`'s inner algorithm exposed. Everything else named here exists.
- Label models and query strategies are other chapters'; the `see` rows keep this chapter to what it does to the data.
