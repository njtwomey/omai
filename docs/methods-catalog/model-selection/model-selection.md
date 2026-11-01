# Model selection (part 4, principles of learning)

Choosing among models: cross-validation and its splitters, selection bias and nested cross-validation, information criteria and the evidence, the regularisation path with its cross-validated choice, hyperparameter search (grid, random, quasi-random, Bayesian, TPE, evolutionary, successive halving and Hyperband, population-based), learning curves, the comparison of learning algorithms and cost-sensitive selection, as sessions over `learn/model-selection`, `tune`, `metrics` and `stats` driving the task methods of other chapters. Every method here is a study: a method over an inner method, taking that method as a slot.

## Structure

A study is a method whose run drives other methods, so everything here lives under `study/` (section 3 of `methods/README.md`), by what the study does: `cross-validate/` for the splitters and the scores they produce, `sweep/` for one or more knobs swept with a sampler (grid, random, Bayesian, multi-fidelity), `learning-curve/` for the sample size swept, `cost/` for a cost matrix over a grid (section 7.6). Each takes an inner slot: a task method of `fit/` (its model registry entry with its modes, section 7.3) or any `Method` with a configuration space, plus a data slot and a scorer. The regularisation path itself is `supervised-learning`'s (`methods/linear/lassoPath.ts`); its cross-validated choice is here.

```
study/
  cross-validate/
    splitters/        k-fold, stratified, grouped, repeated, leave-one-out, shuffle splits, time-series and purged splits drawn fold by fold
    scores/           cross-validation scored fold by fold, the validation curve with the one-standard-error rule, the permutation test of a score
    nested/           nested cross-validation against the naive loop: the optimism exposed
    criteria/         AIC, BIC, Mallows' Cp and GCV against cross-validation on a model sequence; the evidence, WAIC and PSIS-LOO; stacking
    compare/          paired tests over folds, the corrected resampled t-test, Friedman and Nemenyi with the critical-difference diagram
  sweep/
    grid/             grid and random search, quasi-random designs, the sweep panel's data
    bayesian/         the surrogate and its acquisition (built), TPE, CMA-ES and evolutionary samplers, the sampler as the mode
    multi-fidelity/   successive halving, Hyperband and ASHA as brackets, the pruners, population-based training
    analysis/         hyperparameter importance (fANOVA, permutation), slices and contours of a finished study
  learning-curve/     the score against the training size, the gap between training and validation, the data-starved regime
  cost/               a cost matrix over a grid: the chosen threshold and configuration under the costs
  shared/             the inner-method slot, the scorer slot, the trials and folds records, the sweep harness (below)
```

The built `methods/tune/bayesianOptimisation.ts` moves to `study/sweep/bayesian/`; `methods/linear/lassoPath.ts` stays with `supervised-learning` (`fit/regress/`) and is the inner method of `path-selection` below.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a cross-validation is one fold; the step of a sweep is one trial (one configuration, fitted and scored); the step of a learning curve is one training size; the inner fits run as sub-runs whose tapes are kept or summarised by the harness, and every study shares `trial/config` m, `trial/score` s, `trial/best` s and `study/table` proj (every trial, as `exposition/searches.ts`: `searchTrials`), named as `bayesianOptimisation` names its `x`, `value` and `best`.

### cross-validate/splitters/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `splitter` | The splitter as the mode (k-fold, stratified, grouped, stratified-grouped, repeated, leave-one-out, leave-p-out, shuffle splits, predefined, holdout) drawn fold by fold on the rows: the fold's rows as a board (train, validation, held out), the class balance per fold (stratification working or not), groups kept together, the disjointness check; k as the knob. | `fold/index` s, `fold/trainSize` s, `fold/classBalance` t, `fold/board` p (rows coloured by their role at the viewed fold), `fold/groups` p | `learn/model-selection/splitters.ts`: `kFold`, `stratifiedKFold`, `groupKFold`, `stratifiedGroupKFold`, `leaveOneOut`, `leavePOut`, `repeatedKFold`, `shuffleSplit`, `stratifiedShuffleSplit`, `groupShuffleSplit`, `predefinedSplit`, `holdout`, `foldsAreDisjoint`; `exposition/boards.ts`: `boardSteps` | |
| `dependent-data-splits` | Time-series, sliding and expanding windows, blocked and purged k-fold with an embargo, spatial and cluster blocks as modes: the leakage a random split would allow (the autocorrelation across the cut), the gap and embargo drawn on the series or the map; the score under a random split against the honest one on the same model. | `fold/index` s, `fold/trainRange` t, `fold/gap` s, `fold/leakage` s, `fold/picture` p, `study/scoreRandomVsBlocked` proj | `learn/model-selection/blocked-splitters.ts`: `timeSeriesSplit`, `slidingWindowSplit`, `expandingWindowSplit`, `blockedTimeSeriesSplit`, `purgedKFold`, `spatialBlockSplit`, `clusterBlockSplit`; `sampling/diagnostics/autocorrelation.ts`: `autocorrelation` for the leakage measure | |

### cross-validate/scores/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `cross-validation` | An inner task method scored one fold per step: the fold's train and validation scores, the running mean and its standard error, the per-fold predictions assembled into the out-of-fold prediction (and its confusion table or residuals); the scorer and the splitter as slots; the variance across folds against the number of folds (k as the knob, with the bias and variance of the estimate). | `fold/index` s, `fold/trainScore` s, `fold/validationScore` s, `fold/fitTime` s, `study/mean` proj, `study/standardError` proj, `study/outOfFold` p (the assembled predictions), `fold/innerTape` m (a reference to the sub-run) | `learn/model-selection/cross-validation.ts`: `crossValidate`, `crossValScore`, `crossValPredict`; `learn/scoring.ts`: `makeScorer`, `getScorer`, `scorersOf`; `trace/nesting.ts`: `subRun`; TODO `methods/shared/sweeps.ts` (the harness over an inner method, section 4 of `methods/README.md`) | |
| `validation-curve` | One knob of the inner method swept (the step is one value), each value cross-validated: the training and validation curves with their bands, the gap opening as the model overfits, the best value and the one-standard-error rule's simpler choice marked; the knob as the slot (any entry of the method's space, including a mode's own knobs through the union). | `sweep/value` s, `sweep/trainScore` s, `sweep/validationScore` s, `sweep/band` s, `sweep/chosen` m, `sweep/oneStandardError` m, `study/curve` proj (`crossValidationCurve`) | `learn/model-selection/cross-validation.ts`: `validationCurve`; `learn/model-selection/criteria.ts`: `oneStandardErrorRule`; `exposition/linear.ts`: `crossValidationCurve`; `lib/space`: the method's space for the knob's range | |
| `path-selection` | The regularisation path (the inner method: `lassoPath`, or an elastic-net or ridge path) with its cross-validation curve: the coefficients against the strength above, the CV error with its band beneath, the minimum and the one-standard-error choice as two vertical lines, the model at the chosen strength as a probe; the path's own tape reused, not refitted. | `path/alpha` s, `path/coefficients` t!, `path/cvError` s, `path/band` s, `path/chosen` m, `model/params` p (at the chosen alpha), `study/curve` proj | `linear/paths.ts`: `lassoPath`, `enetPath`, `ridgePath`, `cvPath`, `alphaGrid`, `curveOf`; `exposition/linear.ts`: `regularisationPath`, `crossValidationCurve`; `methods/linear/lassoPath.ts` as the inner method (`supervised-learning`'s) | |
| `permutation-test-score` | The score's significance: the labels permuted one permutation per step, the inner method refitted and cross-validated, the permutation distribution of the score forming against the real score, the p-value's running estimate; a model that learns nothing scoring well by chance on a tiny dataset. | `permutation/score` s, `permutation/pValue` s, `study/distribution` proj (with the real score marked) | `learn/model-selection/cross-validation.ts`: `permutationTestScore`; `random/permutation.ts`: `permutation` | |

### cross-validate/nested/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `nested-cross-validation` | The optimism of selecting and scoring on the same folds: the naive loop's best score against the nested loop's outer score, one outer fold per step (its inner search as a sub-run, its chosen configuration as a message); noise features added as the knob, the gap between the two growing with them; the chosen configurations varying across outer folds. | `outer/fold` s, `outer/chosen` m, `outer/score` s, `outer/naiveScore` s, `study/optimism` proj, `outer/innerTape` m | `learn/model-selection/search.ts`: `nestedCrossValidation`, `gridSearchCV`; `trace/nesting.ts`: `subRun`; `datasets/make-classification.ts`: `makeClassification` (noise features by `nInformative` against `nFeatures`) | |
| `selection-bias` | Many random models (or one model on many random feature subsets) scored on one validation set, one per step: the maximum score climbing with the count tried (the winner's curse), the chosen model's held-out score flat; the adaptive-data-analysis cures (Thresholdout, the Ladder) as modes that keep the leaderboard honest. Shared with `learning-theory`. | `trial/score` s, `trial/max` s, `trial/heldOut` s, `study/curve` proj (max against trials), `leaderboard/answer` s | `learn/model-selection/splitters.ts`: `holdout`; `random/permutation.ts`: `choice`; TODO `learn/model-selection/adaptive.ts`: `thresholdout`, `ladder` (a few lines each) | |

### cross-validate/criteria/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `information-criteria` | A model sequence (polynomial degree, the number of components, the lag order; the step adds one) with AIC, AICc, BIC, Mallows' Cp and GCV against the cross-validated error on one axis: where they agree, where BIC's heavier penalty picks the smaller model, the asymptotic equivalence of AIC and leave-one-out seen on a linear smoother; the bootstrap .632 estimate beside them. | `model/size` s, `criteria/values` t (one per criterion), `criteria/cv` s, `criteria/loo` s, `criteria/chosen` m (per criterion), `study/curve` proj | `learn/model-selection/criteria.ts`: `aic`, `aicc`, `bic`, `mallowsCp`, `gcv`, `linearSmootherLoo`, `bootstrap632`, `gaussianLogLikelihood` | |
| `bayesian-model-selection` | The evidence (marginal likelihood) of the conjugate linear model against its degree (Occam's razor from the integral, the evidence peaking at the true degree), maximised over the hyperparameters; WAIC and PSIS-LOO on posteriors with their k-hat warnings; Bayes factors between two models; stacking weights over a set. Shared with `probabilistic-inference` (its model criticism). | `model/size` s, `evidence/logEvidence` s, `evidence/hyperparameters` t, `loo/elpd` s, `loo/kHat` t!, `compare/weights` p, `study/curve` proj | `linear/bayesian.ts`: `maximiseEvidence`, `bayesianLinearFit`, `polynomialBasis`; `probprog/criticism.ts`: `waic`, `psisLoo`, `compareModels`, `stackingWeights`; `stats/tests/bayesian.ts`: `bayesFactorTTest` | |

### cross-validate/compare/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `paired-comparison` | Two inner methods on the same folds (the same splits and keys, so the difference is the method): the per-fold differences, the paired t-test and Wilcoxon on them, McNemar on the disagreements of two classifiers, DeLong on two AUCs, the paired bootstrap of any metric; the inflated Type I error of the naive t-test over folds against the corrected resampled test, by repeated experiments. | `fold/scoreA` s, `fold/scoreB` s, `fold/difference` s, `test/results` p (every test's statistic and p-value), `study/typeIError` proj | `stats/tests/means.ts`: `ttestRel`; `stats/tests/nonparametric.ts`: `wilcoxon`; `metrics/foundations.ts`: `mcnemarTest`, `delongTest`, `pairedBootstrap`, `withInterval`; TODO `learn/model-selection/compare.ts`: `correctedResampledTTest` (Nadeau-Bengio), `fiveByTwoCv` | |
| `many-datasets` | Several methods over several datasets (the dataset registry's query as the slot: one dataset per step, each method cross-validated on it): the rank table, the Friedman test on the ranks, Nemenyi's post hoc and the critical-difference diagram, the average ranks with their cliques; the warning when the datasets are few. | `dataset/name` m, `dataset/scores` t (one per method), `dataset/ranks` t, `study/averageRanks` proj, `study/criticalDifference` proj | `stats/tests/nonparametric.ts`: `friedman`, `nemenyiFriedman`, `nemenyi`; `datasets/registry.ts`: `whereDatasets`, `loadDataset`; `datasets/collections.ts`: `datasetCollections` | |

### sweep/grid/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `grid-and-random-search` | The sampler as the mode (grid, random, Sobol, Halton, Latin hypercube) over the inner method's space (two knobs for the picture): one trial per step, the points on the plane coloured by score, the best so far, the budget as the knob; random beating grid when one knob matters (the projection onto each axis), the conditional knobs of a mode appearing only with it (section 7.1 of `methods/README.md`); the sweep panel's data (`searchTrials`). | `trial/config` m, `trial/score` s, `trial/best` s, `trial/foldScores` t, `study/table` proj, `study/surface` p (the scores over the two knobs), `study/marginals` proj (score against each knob alone) | `learn/model-selection/search.ts`: `gridSearchCV`, `randomizedSearchCV`, `parameterGrid`, `parameterSampler`; `tune/samplers.ts`: `gridSampler`, `randomSampler`, `sobolSampler`, `haltonSampler`, `latinHypercubeSampler`; `tune/tuning.ts`: `studyAlgorithm` (one trial per step); `tune/estimators.ts`: `estimatorObjective`; `exposition/searches.ts`: `searchTrials` | |

### sweep/bayesian/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `bayesian-optimisation` | A one-dimensional function (Forrester, a damped sine, Gramacy-Lee) with the acquisition as the mode (expected improvement, lower confidence bound, probability of improvement): a few random trials, then each step maximises the acquisition over a Gaussian-process surrogate of the trials so far; the surrogate's mean and band with the acquisition beneath as a probe, the noise as the knob. Any method's space TODO as the objective, with the surrogate over its encoding. | `x` s, `value` s, `best` s, `model/params` p, `tune/posterior` p | `tune/surrogate.ts`: `fitGpSurrogate`, `gpPosterior`; `tune/bayesian.ts`: `expectedImprovement`, `lowerConfidenceBound`, `probabilityOfImprovement`, `acquisitions`, `maximizeAcquisition`, `gpSampler`; `datasets/objectives.ts`: `forrester`, `dampedSine`, `gramacyLee` | built `methods/tune/bayesianOptimisation.ts` |
| `model-based-samplers` | The sampler as the mode on one inner method and budget: the GP sampler, TPE (the good and bad Parzen densities over each knob at the viewed step, l(x)/g(x) as the acquisition), CMA-ES (the ellipse in the knob plane), the evolutionary sampler, Nelder-Mead; the best-so-far curves of every sampler overlaid under one key; the exploration against exploitation visible in where each puts its trials. | `trial/config` m, `trial/score` s, `trial/best` s, `sampler/model` p (the GP band or the Parzen densities or the CMA ellipse), `study/table` proj, `race/best` t | `tune/tpe.ts`: `tpeSampler`; `tune/parzen.ts`: `parzenEstimator`, `tpeAcquisition`; `tune/bayesian.ts`: `gpSampler`; `tune/samplers.ts`: `cmaesSampler`, `evolutionarySampler`, `nelderMeadSampler`; `tune/tuning.ts`: `studyAlgorithm`; `tune/study.ts`: `createStudy`, `bestTrial` | |
| `multi-objective-search` | Two objectives (accuracy and model size, or two metrics) with the Pareto front forming one trial per step, the hypervolume on the tape, the sampler as the mode (random, TPE's multi-objective form, NSGA-II); the trade-off chosen by a weight as a probe. | `trial/objectives` t, `front/members` t!, `front/hypervolume` s, `study/front` proj (`paretoFront`), `front/chosen` m | `tune/objectives.ts`: `multiObjective`; `tune/tpe.ts`: `hypervolumeSubsetSelection`; `tune/study.ts`: `bestTrials`, `studyHypervolume`; `optim/multiobjective.ts`: `paretoFront`, `hypervolume`; `optim/nsga2.ts`: `nsga2Algorithm`; `exposition/frontiers.ts`: `paretoFront` | |

### sweep/multi-fidelity/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `successive-halving` | n configurations at a small budget, the worst half dropped each rung (an event per rung with who was cut), the survivors continued: the learning curves of every configuration with the cuts drawn, the budget per rung on the tape, the winner; the reduction factor η and the number of rungs as knobs; the trap of a slow starter cut early. | `rung/index` s, `rung/budget` s, `rung/cut` e (the configurations dropped), `config/curve` t! (every configuration's score so far), `trial/best` s, `study/bracket` proj | `tune/schedules.ts`: `successiveHalvingSchedule`, `halvingPlan`, `nextAssignment`; `tune/pruners.ts`: `successiveHalvingPruner`; `tune/wrappers.ts`: `halvingGridSearchCv`, `halvingRandomSearchCv`; `learn/model-selection/search.ts` | |
| `hyperband` | Several brackets of successive halving with different starts (many cheap configurations against few expensive ones), the bracket as the step's context, the asynchronous form (ASHA) promoting without waiting; the brackets' outcomes compared, the total budget on the tape; against random search at the same budget. | `bracket/index` s, `rung/index` s, `rung/cut` e, `config/curve` t!, `trial/best` s, `study/brackets` proj, `race/bestAtBudget` t | `tune/schedules.ts`: `hyperbandSchedule`, `ashaSchedule`; `tune/pruners.ts`: `hyperbandPruner`, `hyperbandBracketOf` | |
| `pruning` | The pruner as the mode on one search (median, percentile, threshold, patient, none): the intermediate values of every trial as curves, the pruned ones ending early as events, the budget saved against the best score lost; the patience as the knob. | `trial/intermediate` t!, `trial/pruned` e, `trial/score` s, `study/budgetSaved` proj, `study/table` proj | `tune/pruners.ts`: `medianPruner`, `percentilePruner`, `thresholdPruner`, `patientPruner`, `nopPruner`; `tune/analysis.ts`: `intermediateValues` | |
| `population-based-training` | A population of trainers (the inner method: a network's trainer with its learning rate and regularisation as knobs) stepping together: every few steps the worst copy the best (exploit, an event) and perturb their knobs (explore), so the schedule is found while training; the knobs' trajectories per member, the population's best. | `member/score` t, `member/knobs` t!, `population/exploit` e, `population/explore` e, `population/best` s, `study/knobTrajectories` proj | `tune/tuning.ts`: `studyAlgorithm`; `nn/training/fit.ts`: `fitAlgorithm`; TODO `tune/population.ts`: `populationBasedTraining` (the exploit and explore steps over a set of running sub-runs); `trace/nesting.ts`: `subRun` | |

### sweep/analysis/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `hyperparameter-importance` | A finished study (the slot) analysed: which knobs mattered by fANOVA (a random forest over the trials, the variance decomposed, interactions), by permutation, by correlation; the slice of the score against each knob, the contour over two, the parallel coordinates of every trial; the optimisation history with the best so far. | `importance/values` t!, `importance/interactions` t, `analysis/slices` p, `analysis/contour` p (`hyperparameterSurface`), `analysis/parallelCoordinates` p, `analysis/history` p | `tune/importance.ts`: `fanovaImportance`, `permutationImportance`, `correlationImportance`, `pedAnovaImportance`; `tune/analysis.ts`: `trialsTable`, `optimisationHistory`, `sliceData`, `contourData`, `parallelCoordinates`; `exposition/posteriors.ts`: `hyperparameterSurface` | |

### learning-curve/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `learning-curve` | The training size as the step (a geometric schedule), each size cross-validated on the inner method: the training and validation scores converging (the gap is the variance, the level is the bias), the asymptote estimated by fitting a power law (how much more data would help), two models crossing (the simpler winning when data is scarce); the inner fits as sub-runs. | `size/n` s, `size/trainScore` s, `size/validationScore` s, `size/band` s, `size/fitTime` s, `study/curve` proj (`learningCurve`), `study/asymptote` proj | `learn/model-selection/cross-validation.ts`: `learningCurve`; `exposition/searches.ts`: `learningCurve`; `optim/minimize/least-squares.ts`: `curveFit` for the power-law extrapolation | |
| `learning-curves-raced` | Several inner methods' learning curves under one key and one schedule, overlaid: which model wins at which size; the bias-variance reading of the gaps. Shared with `learning-theory` (its rates). | `size/n` s, `race/validationScores` t, `race/gaps` t, `study/curves` proj | the `learning-curve` method; the race harness (`shared/harness.ts`, below) | |

### cost/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `cost-sensitive-selection` | The exemplar study of section 8 of `methods/README.md`: a grid over an inner classifier's knobs, a cost matrix (its entries as knobs), the threshold chosen per configuration by the minimum expected cost over the threshold sweep, the configuration chosen by the cost rather than the accuracy; the chosen threshold moving as the costs change, the cost curve of the best configuration; class weights on the objective slot as the other mode. | `trial/config` m, `trial/costs` t (over thresholds), `trial/bestThreshold` s, `trial/expectedCost` s, `trial/best` s, `study/table` proj, `study/costCurve` p | `metrics/contingency.ts`: `tableCost`, `thresholdCosts`, `analyseThresholds`; `metrics/foundations.ts`: `expectedCost`, `costThreshold`; `metrics/operating-points.ts`: `minimumExpectedCost`, `costCurve`; `exposition/thresholds.ts`: `thresholdSweep`; `learn/model-selection/search.ts`: `gridSearchCV` with a cost scorer (`learn/scoring.ts`: `makeScorer`); the `classWeight` option of the task methods' objective slot (TODO, section 7.6) | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `inner.ts` | The inner-method slot: any `Method` with a configuration space (a task method's registry entry with its modes, a trainer, a procedure), its data slot and its scorer, as one object the harness drives; the key tree fixed so every trial and fold shares the data and the start (problem 6 of `methods/README.md`) | every group |
| `harness.ts` | The sweep harness: run the inner method as a sub-run per trial or fold, keep its tape or a summary (`keep: 'all' | 'last' | n`), log the trial's configuration and score as the step, assemble the `searchTrials`, `crossValidationCurve` and `learningCurve` records as projections; one session driving several for the races; a thin layer over `methods/shared/sweeps.ts` once it exists | every group |
| `scorers.ts` | The scorer slot: every `metrics` metric as a scorer with its direction and what it needs (labels, scores, probabilities), the cost scorer from a cost matrix, the multi-metric scorer for the sweep panel | `cross-validate/`, `sweep/`, `cost/` |
| `folds.ts` | The fold record: the rows' roles at the viewed fold as a board, the per-fold scores and times, the out-of-fold predictions | `cross-validate/` |
| `brackets.ts` | The bracket record: configurations against rungs with their budgets, the cuts, the survivors' curves, for the halving and Hyperband panels | `sweep/multi-fidelity/` |

## Examples to make

- Nested CV against naive CV on noise features: the optimistic score exposed, the gap growing with the noise features.
- The CV curve with the one-standard-error rule picking the simpler model, on the lasso path's own tape.
- Successive halving: configurations dropped round by round, the survivors' curves with the cuts drawn.
- Bayesian optimisation's surrogate and acquisition over one knob (exists), then over two knobs of a real inner method.
- A critical-difference diagram over the registry's tabular classification sets for three classifiers.
- The cost-sensitive selection page: a grid, a cost matrix as knobs, the chosen threshold and configuration moving with the costs.

## Rendering

- CV curves with bands (`crossValidationCurve`), search trials (`searchTrials`), hyperparameter surfaces (`hyperparameterSurface`), learning curves (`learningCurve`), Pareto fronts (`paretoFront`), boards for folds (`boardSteps`), the threshold explorer for the cost study.
- Missing: the sweep panel (one knob on the x axis, a curve per metric, click a point to open that trial's run); a bracket diagram for halving and Hyperband; a critical-difference diagram; a fold board with the groups and the embargo drawn on a series.

## Problems

- `tune/` moves here. Hyperparameter optimisation over any method means a method is the data of another: the nested-session problem in full (problem 11 of `methods/README.md`); `shared/harness.ts` is the stopgap until `methods/shared/sweeps.ts` lands, and the `keep` policy on sub-run tapes decides what a sweep of a thousand trials costs.
- The built `bayesianOptimisation` takes a one-dimensional test function, not a method's space; making the objective a slot (any method, its space encoded for the surrogate) is the first change when it moves.
- The library gaps are small and named: population-based training (`tune/population.ts`), Thresholdout and the Ladder (`learn/model-selection/adaptive.ts`), the corrected resampled t-test and 5×2 CV (`learn/model-selection/compare.ts`), and the `classWeight` option on the task methods' objective slot.
- Model criticism (WAIC, PSIS-LOO, stacking) is shared with `probabilistic-inference`, where the posteriors are made; here it is one criterion among others on a model sequence.
