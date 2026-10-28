# Causal inference (a methods chapter; the encyclopedia splits it between experimental design, causal discovery and the trustworthy part)

Causes estimated: identification on a causal graph (d-separation, back-door and front-door sets, instruments, the do-operator by graph surgery), estimators of average treatment effects (regression adjustment, propensity scores with inverse probability weighting, stratification and matching, doubly robust and double machine learning), quasi-experimental designs (difference in differences and event studies, synthetic control, regression discontinuity, instrumental variables), heterogeneous effects and uplift (the meta-learners, causal forests, Qini curves), counterfactuals on a structural causal model, and sensitivity analysis and refutation, as sessions over `pgm` (causal graphs and structural models), `datasets` (generators with both potential outcomes), `linear`, `learn` and `stats`. Causal discovery is `exploration-and-discovery`'s and experiments are `online-experimentation`'s; both get a row here.

## Structure

Every method here is done to data under a causal model, so the chapter lives under `analyse/` (section 3 of `methods/README.md`) as `analyse/causal/`: `identification/` for the questions a graph answers before any data, `effects/` for the estimators of one average effect, `quasi-experimental/` for the designs that find their own instrument in time or a threshold, `heterogeneous/` for effects that vary, `counterfactuals/` for the unit-level question, `sensitivity/` for what an unobserved confounder would do. The library has the graphs, the structural models, the identification and the counterfactuals (`pgm/causal`) and the generators with their truths (`datasets/causal.ts`); the estimators are the gap, proposed as a `lib/causal/` domain over `learn` and `linear`.

```
analyse/causal/
  identification/    graph queries: d-separation, adjustment sets, instruments, the ID algorithm's cases, the do-operator; Simpson's paradox
  effects/           regression adjustment, IPW and the propensity score, stratification, matching, AIPW, double machine learning
  quasi-experimental/  difference in differences and event studies, synthetic control, regression discontinuity, instrumental variables
  heterogeneous/     S, T, X, R and DR learners, causal forests; uplift and Qini curves, policy value
  counterfactuals/   abduction, action, prediction; twin networks; probabilities of necessity and sufficiency
  sensitivity/       an unobserved confounder swept, E-values, Rosenbaum bounds, refutation tests
  shared/            the structural model slot with its truth, the nuisance-model slot, the estimate record, the graph overlay (below)
```

The old `causal/` area of `docs/structure.md` 5.5 is this folder; its discovery half (PC, GES, LiNGAM, NOTEARS, FCI) is `exploration-and-discovery`'s.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of an estimator is one batch of data added (so the estimate converges, or fails to, as n grows) or one fold of cross-fitting; of an identification query one candidate set; of a counterfactual one of its three stages; of a sensitivity study one strength of the hidden confounder. Status is about the method alone; what the library has is in the needs column.

### identification/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `graph-queries` | The causal DAG as the slot (the sprinkler, Simpson's confounder, the front-door model, an instrument, a collider trap, M-bias, a random linear model) with a treatment and an outcome chosen: the back-door sets (all of them, then the minimal), the front-door sets, the instruments, what the ID algorithm says is identifiable, one candidate set per step with the paths it opens and blocks lit on the graph (the Bayes-ball picture); the adjustment formula evaluated on the network's observational distribution against the interventional truth by graph surgery (the same number when the set is valid, a different one when a collider is conditioned on). | `query/candidate` m (the set and whether it is valid), `query/paths` p (the structure with open and blocked paths), `query/identified` s, `estimate/adjusted` s, `truth/interventional` s, `query/sets` p | `pgm/causal/identification.ts`: `backdoorAdjustmentSets`, `minimalAdjustmentSet`, `isBackdoorAdjustmentSet`, `frontdoorAdjustmentSets`, `backdoorAdjustment`, `frontdoorAdjustment`, `identifyEffect`, `instrumentalVariables`, `isInstrument`; `pgm/causal/interventions.ts`: `doIntervention`, `interventionalDistribution`, `averageCausalEffect`; `pgm/models/bayes-net.ts`: `dSeparated`, `activeTrailNodes`; `pgm/causal/scm.ts`: `scm`, `scmToBayesNet`, `scmStructure`; ready | |
| `simpsons-paradox` | A structural model with a confounder whose strength is the knob: the naive difference between treated and untreated and the adjusted estimate, one batch of data per step with both intervals narrowing (on the wrong and the right answer), the reversal of sign as the strength crosses the point where it happens, the stratified picture (the effect within each stratum against the pooled one). | `data/n` s, `estimate/naive` s, `estimate/adjusted` s, `estimate/intervals` t, `truth/ate` s, `strata/effects` p, `estimate/reversed` e | `datasets/causal.ts`: `treatmentEffects` (the `confounding` knob; the truth has the ATE and both potential outcomes); `pgm/causal/interventions.ts`: `averageCausalEffect`; `stats/resampling.ts`: `bootstrapAlgorithm`; the adjusted estimator from `effects/` | |

### effects/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `treatment-effect` | The estimator as the mode on data from the structural model slot: the naive difference, regression adjustment (the outcome model from the nuisance slot), inverse probability weighting on an estimated propensity (the overlap histograms, the weights' effective sample size, trimming as the knob), stratification on the propensity (the strata count as the knob), matching (nearest neighbour on the covariates or on the propensity, with or without replacement, a caliper; the pairs drawn), AIPW and double machine learning (cross-fitting one fold per step, the orthogonal score); the estimate with its bootstrap interval converging as n grows (the step adds data) against the ATE and ATT truths, the bias of each under a misspecified outcome model (the nonlinear setting as the knob: adjustment fails, the doubly robust ones hold), the covariate balance before and after as standardised mean differences. | `data/n` s, `estimate/ate` s, `estimate/interval` t, `estimate/error` s, `truth/ate` s, `truth/att` s, `weights/ess` s, `weights/max` s, `fold/index` s (DML), `propensity/overlap` p (the two histograms), `balance/smd` p (before and after), `matching/pairs` p, `model/params` p | `datasets/causal.ts`: `treatmentEffects` (the knobs `confounding`, `overlap`, `heterogeneity`, the nonlinear setting; the truth with the ATE, ATT, CATE and both potential outcomes); `linear/model.ts`: `linearModel`, `fitLinear` (regression adjustment and the logistic propensity); `numerics/spatial/brute.ts`: `kNearest` (matching); `learn/model-selection/splitters.ts`: `kFold` (cross-fitting); `stats/resampling.ts`: `bootstrapAlgorithm`; `rl/off-policy.ts`: `ips`, `doublyRobust` (the bandit forms of the same estimators); TODO `lib/causal/estimation.ts`: `regressionAdjustment`, `ipw`, `stratifiedEstimate`, `matchingEstimate`, `aipw`, `doubleMachineLearningAlgorithm` (traced, one fold per step), `balanceTable`, `overlap`, each on the `learn` contract for the nuisance models | |

### quasi-experimental/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `difference-in-differences` | Two groups over two periods, then many (an event study: one period per step, the leads as the placebo check that parallel trends hold, the lags as the dynamic effect), the trends planted parallel or not (the knob that breaks the design), the two-by-two estimate and the two-way fixed-effects regression agreeing, staggered adoption as the mode where they do not (the forbidden comparisons' bias). | `period/t` s, `period/means` t (by group), `estimate/did` s, `estimate/interval` t, `eventStudy/coefficients` t!, `truth/effect` s, `trends/picture` p | `linear/model.ts`: `fitLinear` (the fixed-effects regression); TODO `datasets/causal.ts`: `panelEffects` (a panel generator with the trend and adoption knobs); TODO `lib/causal/quasi-experimental.ts`: `differenceInDifferences`, `eventStudy` | |
| `synthetic-control` | One treated unit and a pool of donors: the donor weights found by constrained least squares to match the pre-period (the weights as bars, the fit's error), the synthetic unit's path against the treated one's and the gap after treatment, the placebo-in-space test one donor per step (each donor treated in turn, the real gap's rank among the placebos as the p-value). | `fit/weights` t!, `fit/preError` s, `path/treated` t!, `path/synthetic` t!, `gap/value` s, `placebo/donor` m, `placebo/gap` s, `placebo/rank` s | `stats/resampling.ts`: `permutationTestAlgorithm` (the placebo logic); `optim/minimize` for the constrained weights; TODO `lib/causal/quasi-experimental.ts`: `syntheticControl` (simplex-constrained least squares), `placeboInSpace` | |
| `regression-discontinuity` | A running variable with a cutoff: local linear fits on each side (the bandwidth as the knob, a sweep over it showing the bias-variance trade and the optimal bandwidth's choice), the jump at the cutoff as the estimate with its interval against the truth, McCrary's density test for manipulation of the running variable, the fuzzy design as the instrumental-variable mode when the cutoff only shifts the probability of treatment. | `fit/bandwidth` s, `fit/estimate` s, `fit/interval` t, `fit/error` s, `truth/jump` s, `fit/curves` p (the two local fits over the points), `density/test` p, `study/bandwidthCurve` proj | `stats/smoothing/smoothers.ts`: `loess` (the local fits); TODO `datasets/causal.ts`: `discontinuityDesign` (sharp and fuzzy, with the jump as truth); TODO `lib/causal/quasi-experimental.ts`: `regressionDiscontinuity`, `rddBandwidth`, `densityTest` | |
| `instrumental-variables` | Two-stage least squares on data with an unobserved confounder and an instrument: the first stage's strength (its F statistic; the instrument weakened by a knob until the weak-instrument bias appears), the OLS bias against the IV estimate as n grows with both intervals, the Wald estimate for a binary instrument, the exclusion restriction broken by a knob (the instrument given a direct path) and the estimate drifting; the graph's instrument check beside it. | `data/n` s, `estimate/ols` s, `estimate/iv` s, `estimate/intervals` t, `firstStage/f` s, `firstStage/coefficient` s, `truth/effect` s, `truth/olsBias` s, `graph/instrument` p | `datasets/causal.ts`: `instrumentalVariables` (the truth has the effect, the OLS bias and the first stage); `linear/model.ts`: `fitLinear`; `pgm/causal/identification.ts`: `isInstrument`; TODO `lib/causal/quasi-experimental.ts`: `twoStageLeastSquares`, `firstStageF`, `waldEstimate` | |

### heterogeneous/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `cate-learners` | The learner as the mode (S, T, X, R and DR learners composed over the base-learner slot, any regressor or classifier of `fit/`; a causal forest) on data with a planted heterogeneous effect (its strength and shape as knobs): the CATE surface over the covariate plane against the planted τ(x), the PEHE error on the tape as n grows, the X-learner's advantage when the treated are few (the treated share as the knob), the S-learner's regularisation bias towards zero; the individual effects as a probe over the rows. | `data/n` s, `estimate/pehe` s, `estimate/ateFromCate` s, `truth/ate` s, `estimate/cateSurface` p (a decision surface over the plane), `truth/cateSurface` p, `rows/effects` p, `model/params` p | `datasets/causal.ts`: `treatmentEffects` (the truth has `cate`, `ite` and both potential outcomes); the regressors of `fit/` (`linear/model.ts`: `linearModel`; `kernels`; `nn`); TODO `lib/causal/heterogeneous.ts`: `sLearner`, `tLearner`, `xLearner`, `rLearner`, `drLearner` (compositions on the `learn` contract), `causalForest` (waits on the trees of `methods/README.md` section 6, problem 12), `pehe` | |
| `uplift` | A CATE model as a targeting rule: the units ranked by predicted uplift and treated decile by decile (the step adds the next), the uplift and Qini curves against the random and the oracle targeting, the area under the uplift curve, the policy's value at a budget (the knob) and where it crosses the treat-everyone line. | `decile/index` s, `decile/uplift` s, `curve/qini` t!, `curve/uplift` t!, `curve/auuc` s, `policy/value` s, `policy/budget` s | the learners above; TODO `metrics/uplift.ts`: `qiniCurve`, `upliftCurve`, `auuc`, `policyValue` | |

### counterfactuals/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `counterfactual` | Pearl's three steps one per step on a discrete or linear structural model from the slot: abduction (the noise terms' posterior given what was observed), action (the surgery on the twin network, drawn), prediction (the counterfactual outcome's distribution); "would the patient have recovered without the drug" against the interventional answer for a new patient (the two differ, and the page shows by how much); the probabilities of necessity and sufficiency from the model; the closed form for a linear Gaussian model against the sampled one. | `stage/name` m, `noise/posterior` t!, `counterfactual/distribution` t!, `counterfactual/mean` s, `interventional/mean` s, `model/twinNetwork` p (a `diagram` record), `probabilities/necessity` s, `probabilities/sufficiency` s | `pgm/causal/counterfactuals.ts`: `abductActPredict`, `abduction`, `counterfactual`, `twinNetwork`, `causationProbabilities`, `linearCounterfactual`; `pgm/causal/scm.ts`: `scm`, `scmSample`; `pgm/inference/elimination.ts`: `variableElimination`; ready | |

### sensitivity/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `sensitivity-analysis` | A confounder hidden from the estimator (a covariate of the generator withheld, the knob) and its strength on the treatment and on the outcome swept as two axes (one point of the grid per step): the adjusted estimate's contour where it crosses zero (how strong a confounder would have to be, against the strengths of the observed covariates as benchmarks), the E-value, Rosenbaum bounds for a matched design as the mode; refutation tests as the other mode, each a study over the estimator (a placebo treatment, a random common cause added, a random subset, the estimate expected not to move or to vanish). | `sweep/strengths` t, `sweep/estimate` s, `sweep/crossesZero` e, `study/contour` proj, `eValue/value` s, `bounds/gamma` s, `refute/test` m, `refute/estimate` s, `refute/passed` s | `datasets/causal.ts`: `treatmentEffects` (the hidden-covariate knob TODO as `hidden`); the estimators of `effects/`; `stats/resampling.ts`: `bootstrapAlgorithm`; TODO `lib/causal/sensitivity.ts`: `omittedVariableBias` (the contours), `eValue`, `rosenbaumBounds`, `refute` (placebo treatment, random common cause, data subset) | |

### see elsewhere

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `causal-discovery` | see exploration-and-discovery (PC, GES, LiNGAM, NOTEARS and FCI on data from a known graph: `pgm/causal/discovery.ts`: `gesAlgorithm`, `directLingam`, `notearsAlgorithm`; `pgm/causal/fci.ts`: `fci`; `pgm/learning/structure.ts`: `pcAlgorithm`; the structural model slot shared with this chapter) | | | |
| `experiments` | see online-experimentation (A/B tests with CUPED and sequential testing on `datasets/experiments.ts`: `abTest`) and statistics (design: randomisation, blocking, factorial designs, power from `stats/power.ts`) | | | |
| `off-policy-evaluation` | see reinforcement-learning (`act/deep/offline/off-policy-evaluation`: IPS and doubly robust on logged bandit data, the same estimators as `effects/` under a known logging policy) | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `models.ts` | The structural model slot with its truth: the named graphs (the sprinkler, Simpson's confounder, the front door, an instrument, a collider trap, M-bias) as `pgm/causal/scm.ts` structural models with their interventional truths, and the generators `datasets/causal.ts`: `treatmentEffects`, `instrumentalVariables`, `linearScm` with their knobs (confounding, overlap, heterogeneity, the nonlinear setting, a hidden covariate); the panel and discontinuity generators to add; shared with `exploration-and-discovery` (discovery) and `statistics` (experiments) | every group |
| `nuisance.ts` | The nuisance-model slot: outcome and propensity models drawn from the registries of `fit/regress` and `fit/classify` with their modes, and the cross-fitting folds (`learn/model-selection/splitters.ts`: `kFold`) that double machine learning and the DR learner need | `effects/`, `heterogeneous/`, `sensitivity/` |
| `estimates.ts` | The estimate record: an estimate with its interval converging over n against a truth line, several estimators on one axis; the balance table (standardised mean differences before and after); the overlap histograms; the CATE surface against the truth; the Qini and uplift curves | `effects/`, `quasi-experimental/`, `heterogeneous/` |
| `graphs.ts` | The query overlay: a `Structure` of the DAG (`pgm/causal/scm.ts`: `scmStructure`) with the treatment, the outcome, the adjustment set and the open and blocked paths at the viewed step, and the twin network for counterfactuals | `identification/`, `counterfactuals/`, `sensitivity/` |

## Examples to make

- Simpson's paradox as a structural model: the naive and adjusted estimates with the confounder's strength as the knob, the sign reversing.
- Propensity-score overlap: the two groups' score histograms, the weights and their effective sample size, the balance before and after.
- Double machine learning's cross-fitting stepped fold by fold, the estimate's interval against the truth, beside regression adjustment failing on the nonlinear outcome.
- Two-stage least squares as the instrument weakens: the first-stage F falling, the IV estimate's interval widening past the OLS bias.
- Uplift over the plane from a planted heterogeneous effect, the Qini curve against the oracle's.
- A counterfactual in three steps on the twin network: abduction, action, prediction.

## Rendering

- Graphs with the adjustment set highlighted (structure view), estimates with intervals converging, surfaces of effect over the plane, histograms of overlap, the twin network (`diagram`).
- Missing: a graph query overlay (paths blocked and open at the viewed step); an estimates panel (several estimators on one axis against a truth line, converging over n); a balance table panel; a contour panel for the sensitivity sweep.

## Problems

- The library gap is the estimators: nothing in `lib` computes a treatment effect from data today beyond the graph-side adjustment formula on a network and the bandit-side `rl/off-policy.ts`. A `lib/causal/` domain (estimation, quasi-experimental, heterogeneous, sensitivity) on the `learn` contract is proposed; `pgm/causal/` is the alternative home, kept for the graph-side pieces it already holds so the layering stays simple (`causal` imports `pgm`, `learn`, `linear` and `stats`; nothing imports it).
- The generators are nearly there: `treatmentEffects` has every truth an estimator page needs; the panel (difference in differences), discontinuity and hidden-confounder generators are additions to `datasets/causal.ts`.
- Causal forests wait on trees, which the library lacks (`methods/README.md` section 6, problem 12); the meta-learners do not.
- One chapter for effects and another for discovery is the split the plan made; the structural model slot is shared, so a page that discovers the graph and then estimates on it spans both and belongs to `study/`.
