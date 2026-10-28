# Online learning (part 5, learning paradigms)

Learning as data arrive: online convex optimisation (online gradient descent, follow the regularised leader, the online Newton step, mirror descent, with the regret against the best fixed point), prediction with expert advice (weighted majority, Hedge, follow the leader and its perturbed form), incremental learners (the perceptron and its kernel and averaged forms, passive-aggressive, the SGD classifier and regressor, Hoeffding trees, online naive Bayes, recursive least squares, streaming k-means), as sessions over `optim/online`, `optim`, `cluster`, `linear` and the models of `supervised-learning`, with a stream as the data slot and a planted drift as its knob. Continual learning is `transfer-learning`'s, drift detection `shift-and-drift`'s, online active learning `active-learning`'s and bandits `online-experimentation`'s; each has a `see` row.

## Structure

An incremental classifier or regressor is the task method with a stream as its data slot, so those live as `online/` sections of the task folders (`fit/classify/online/`, `fit/regress/online/`, `fit/cluster/online/`; section 3 of `methods/README.md`). Online convex optimisation and prediction with expert advice belong to no task: they are sequential prediction over a sequence of losses, and the layout has no home for them, so this chapter proposes `fit/online/` as a section of `fit/` beside the tasks (the alternative, `study/theory/` for the regret studies and the `optimisation` chapter for the learners, is recorded under Problems). The stream base itself (data arriving one row or one batch per step, with drift and recurrence planted) is `methods/shared/streams.ts` (section 4), shared with `shift-and-drift`, `time-series` and `online-experimentation`.

```
fit/online/
  oco/                online gradient descent, FTRL, the online Newton step, mirror descent and exponentiated gradient, with the regret
  experts/            weighted majority and Hedge, follow the leader and follow the perturbed leader, switching experts
fit/classify/online/  the perceptron (plain, averaged, voted, kernel, pocket), passive-aggressive, the SGD classifier, Hoeffding trees, online naive Bayes
fit/regress/online/   recursive least squares with forgetting, the SGD regressor, online kernel regression with a budget
fit/cluster/online/   mini-batch and streaming k-means, online dictionary learning
shared/               the stream slot (losses, experts, drifting rows), the regret record, progressive validation (below)
```

Nothing is built here; the conjugate Bayesian regression one row per step (`methods/linear/bayesianRegression.ts`) is the built online Bayesian learner and is `probabilistic-inference`'s.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of every method here is one round: one row, one batch or one loss arriving. The incremental learners on the iterative base log `fit/iteration/end` e, `model/params` p and the kind's probes (`model/decision-surface` p at the viewed step); every method of this chapter also logs the stream's `stream/t` s and `stream/drift` e (a planted change) from `shared/stream.ts`, and `eval/progressiveError` s (every row predicted before it is learnt) from `shared/progressive.ts`. The status is `built <path>`, `partly` or empty; library readiness is in the needs column (a function named there exists; `TODO:` names what is missing and its proposed home).

### fit/online/oco/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `online-gradient-descent` | The learner as the mode (online gradient descent with its step-size rule: constant, 1/√t, adaptive; follow the regularised leader with its regulariser; the online Newton step for exp-concave losses; mirror descent with the entropic mirror, which is exponentiated gradient on the simplex), the losses one per round from the slot (stochastic, adversarial, alternating, drifting): the point played, the round loss, the cumulative loss and the regret against the best fixed point in hindsight with the √T bound drawn; the comparison: the same losses against two learners, and the alternating losses on which FTL fails. | `round/loss` s, `round/cumulativeLoss` s, `round/regret` s, `round/play` t!, `round/bound` s, `round/comparator` p (the best fixed point so far), `study/regretCurve` proj | `optim/online.ts`: `onlineGradientDescent`, `ftrl`, `onlineNewtonStep`, `exponentiatedGradient`, `onlineAlgorithm`, `runOnline`, `linearLosses`; `optim/minimize/convex.ts`: `mirrorDescentAlgorithm`, `entropicMirror`, `euclideanMirror`; `optim/sparse-updates.ts`: `ftrlProximal` (the sparse FTRL of ad models); TODO the loss-sequence generators in `shared/stream.ts` | |

### fit/online/experts/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `prediction-with-experts` | Hedge (the learning rate as the knob; the weights over experts as bars, one expert turning bad and its weight collapsing), weighted majority with its mistake bound on binary predictions, follow the leader against follow the perturbed leader as modes; the regret against the best expert and the √(T ln N) bound; the experts as a slot (constant predictors, the forecasters of `fit/forecast/`, the models of a registry); switching experts (fixed share) when the best expert changes (a planted switch). Shared with `games` (regret matching). | `round/weights` t!, `round/loss` s, `round/regret` s, `round/bestExpert` s, `round/bound` s, `round/switched` e, `model/expertPredictions` p | `optim/online.ts`: `hedge`, `onlineAlgorithm`, `runOnline` (with experts); TODO weighted majority, follow the leader, follow the perturbed leader and fixed share as `OnlineLearner`s in `optim/online.ts` (a few lines each) | |

### fit/classify/online/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `perceptron` | One row per step, a mistake moves the boundary (events), the mistake count against Novikoff's bound (the margin of the planted separator as the knob); the averaged and voted perceptron, the kernel perceptron (the support set growing) and the pocket algorithm on inseparable data as modes; the batch perceptron loss is a mode of `fit/classify/linear/`. | `stream/mistake` e, `stream/mistakes` s, `stream/bound` s, `solver/weights` t!, `model/supportSet` p (the kernel mode) | `losses/margin.ts`: `perceptron`, `hinge`; `kernels/kernel.ts`: `crossGram`; TODO `lib/linear/online.ts`: `perceptronAlgorithm` (plain, averaged, voted, kernel, pocket) | |
| `passive-aggressive` | PA, PA-I and PA-II as modes (C as the knob): the margin violation and the closed-form step that fixes it exactly (passive when the margin holds, aggressive otherwise), against the perceptron on a noisy stream (PA-I's cap on the step); the regressor with its ε-tube as the regression mode. | `stream/violation` s, `stream/stepSize` s, `solver/weights` t! | `losses/margin.ts`: `hinge`; TODO `lib/linear/online.ts`: `passiveAggressiveAlgorithm` | |
| `sgd-classifier` | Any loss of the objective table with any penalty, one row or one minibatch per step, the learning-rate schedule as the knob, averaging (ASGD) and the cumulative L1 penalty (sparsity on a stream) as modes; the progressive validation error against the training error; the same model as `fit/classify/linear/` in its gradient mode with batch 1, so the two pages share one method. | `stream/loss` s, `solver/coef` t!, `solver/learningRate` s, `solver/nonZero` s (the L1 mode) | `optim/first-order.ts`: `sgd`, `asgd`; `optim/schedules.ts`: `inverseTimeDecay`, `constantSchedule`; `optim/sparse-updates.ts`: `sgdCumulativeL1`, `truncatedGradient`; `methods/shared/options`: `classificationLosses`, `penalties`, `schedules`; `data/batching.ts`: `batches` | |
| `hoeffding-tree` | The very fast decision tree: statistics per leaf accumulating one row per step, a split made when the Hoeffding bound separates the two best candidates (δ and the tie threshold as knobs; the gap and the bound at each leaf as a probe), the tree growing on a stream with a planted concept drift; the adaptive variant with a drift detector at the nodes as the mode; against a batch tree refitted on windows. | `tree/split` m, `tree/leaves` s, `tree/bound` p (per leaf: the gap and the bound), `model/structure` p (a `treeRecord`) | `information/discrete.ts`: `informationGain`; `exposition/trees.ts`: `treeRecord`; TODO `lib/trees/hoeffding.ts`: `hoeffdingTreeAlgorithm` | |
| `online-naive-bayes` | The counts updating one row per step, the class-conditional tables as probes: the natural streaming classifier. | | | as `fit/classify/naive-bayes/naive-bayes` in its online mode (`supervised-learning`) |
| `online-bayesian` | Bayes point machines by EP and online probit regression; the conjugate linear regression one row per step is built. | | `linear/bayesian.ts`: `bayesianLinearAlgorithm` | see `probabilistic-inference` (`infer/models/online-bayesian-classifiers`; built `methods/linear/bayesianRegression.ts`) |

### fit/regress/online/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `recursive-least-squares` | RLS with a forgetting factor (λ as the knob) tracking a drifting regression function (the coefficients following the planted drift, the covariance inflating), against online ridge by SGD on the same stream; the Kalman filter as the same recursion (`infer/filters/`), the conjugate update as its Bayesian reading. | `solver/coef` t!, `solver/covarianceTrace` s, `stream/error` s, `model/fit` p | `linear/bayesian.ts`: `bayesianUpdate`, `bayesianLinearAlgorithm` (RLS is the conjugate update with a flat prior); `signal/adaptive.ts`: `adaptiveFilterAlgorithm` (the LMS and RLS filters on signals); TODO the forgetting factor as an option of `bayesianUpdate` | |
| `sgd-regressor` | As `sgd-classifier` with the regression losses (squared, Huber, ε-insensitive, quantile), the fitted curve moving as rows arrive. | as `sgd-classifier`, with `model/fit` p | `methods/shared/options/objectives.ts`: `regressionLosses`; `optim/first-order.ts`: `sgd`, `asgd` | |
| `online-kernel-regression` | Kernel recursive least squares with a budget: the dictionary of kept rows (the approximate linear dependence test as the knob), the fit between arrivals, the budget's cost in accuracy; the kernel perceptron's regression cousin. | `solver/dictionary` s, `solver/added` e, `model/fit` p, `model/dictionary` p | `kernels/kernel.ts`: `gram`, `crossGram`; TODO `lib/kernel-methods/online.ts`: `krlsAlgorithm` | |

### fit/cluster/online/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `streaming-k-means` | Mini-batch k-means one batch per step with the per-centre learning rates falling as centres accumulate counts, the centres following a drifting cluster (the planted drift), against Lloyd refitted on windows; online dictionary learning as the sibling for atoms instead of centres. | `solver/centres` t!, `solver/inertia` s, `solver/counts` t, `model/centres` p, `model/assignments` p | `cluster/kmeans.ts`: `miniBatchKMeansAlgorithm`, `initialCentres`; `optim/dictionary.ts`: `onlineDictionaryLearningAlgorithm`; TODO a drifting-cluster generator in `shared/stream.ts` | |

### cross-chapter

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `drift-detection` | CUSUM, Page-Hinkley, ADWIN and DDM alarming on the progressive error, the change-point detectors on the stream. | | `dynamics/filtering/changepoint.ts`: `bocpd`, `pelt`; `stats/tests/sequential.ts`: `sprtAlgorithm`; TODO ADWIN and DDM | see `shift-and-drift` (and `time-series` for `process/signals/series/change-points/online-detectors`) |
| `continual-learning` | Tasks in sequence, catastrophic forgetting, replay, elastic weight consolidation. | | | see `transfer-learning` |
| `stream-active-learning` | A row arrives, the strategy decides to ask or not, with a budget. | | `learn/active-learning.ts`: `streamActiveLearning` | see `active-learning` (`data/selection/active/stream/`) |
| `adaptive-conformal` | Intervals on a stream whose miscoverage is tracked online. | | `learn/conformal.ts`: `adaptiveConformalInference` | see `time-series` (`fit/forecast/practice/conformal-forecast`) |
| `bandits` | Exploration against exploitation one pull per step. | | | see `online-experimentation` and `reinforcement-learning` |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `stream.ts` | The stream slot: rows arriving one or a batch per step from any generator with a planted drift (sudden, gradual, recurring) and its change points as the truth, built on `methods/shared/streams.ts` and `data/modifiers/shift.ts` (`withConceptDrift`, `conceptDriftModifier`); loss sequences for online convex optimisation (stochastic, adversarial, alternating, drifting); experts as a slot with their predictions per round | every group |
| `regret.ts` | The regret record: the round loss, the cumulative loss, the comparator (the best fixed point or expert so far) and the bound curve, as a projection over any tape with `round/loss` | `oco/`, `experts/` |
| `progressive.ts` | Progressive (prequential) validation: every row predicted before it is learnt, the error at a schedule as `eval/progressiveError`, with a window as the knob so drift shows; shared with `shift-and-drift` | every incremental learner |

## Examples to make

- The perceptron on a stream: each mistake and the boundary's move, the mistake count against the bound.
- Hedge's weights over experts as one expert turns bad, the regret curve against its bound; then a planted switch and fixed share.
- A Hoeffding tree splitting on a stream, the bound and the gap at each candidate, a drift arriving.
- Online gradient descent against follow the leader on alternating losses: the regret linear for one, √T for the other.
- Mini-batch k-means following a drifting cluster.

## Rendering

- Curves of cumulative loss and regret with the bound, decision surfaces at the viewed step, weight bars over experts, trees growing, fits moving.
- Missing: a stream strip (the rows so far, the drift marked, the prequential error beneath) as a panel; a bars panel that follows the cursor for the experts' weights.

## Problems

- New chapter (the old hub had `online/`); `active-learning` keeps its own chapter rather than nesting under online learning as the encyclopedia does.
- The layout of section 3 has no home for online convex optimisation and the experts: `fit/online/` is proposed here as a section beside the tasks. The alternatives are `study/theory/` (the regret studies belong to `learning-theory` as much as here) and the `optimisation` chapter (online gradient descent is an optimiser); the pieces are in `optim/online.ts` whatever the folder.
- The incremental classifiers are the batch models in their gradient mode with batch 1 (`sgd-classifier` is one method with `fit/classify/linear/`); the stream pages should reuse the method and change the data slot and the evaluation (progressive), not fork it. The perceptron and passive-aggressive need their own updates (`lib/linear/online.ts`) because they are not gradient steps on a loss.
- The stream base (`methods/shared/streams.ts`) is shared infrastructure not yet built; every row of this chapter waits on it, and `shift-and-drift` and `time-series` ask for the same thing.
