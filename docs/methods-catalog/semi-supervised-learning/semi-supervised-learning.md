# Semi-supervised learning (part 5, learning paradigms)

Learning from a few labels and many unlabelled points: self-training and co-training over any classifier, graph-based methods (label propagation and spreading, Laplacian regularisation), generative models with unlabelled data (EM over a mixture with labels pinning components), low-density separation (the transductive SVM, entropy minimisation) and consistency regularisation on small networks (the Π-model, mean teacher, pseudo-labelling, FixMatch), as sessions over the models of `supervised-learning`, `mixture`, `graph`, `numerics/spatial` and `nn`, with the label fraction as the knob every method shares and the hidden labels as the truth every probe compares with.

## Structure

A semi-supervised method is a classifier whose data slot is partially labelled, so the methods live under `fit/classify/semi-supervised/` (section 3 of `methods/README.md`), one folder per family; the consistency trainers are problem methods of the torch world and live under `train/networks/consistency/`; the label-hiding modifier and the page that shows what it hides are a data matter under `data/labels/`, shared with `weak-supervision` (labels weak) and `active-learning` (labels bought); the label-budget curve is a study under `study/learning-curve/`.

```
fit/classify/semi-supervised/
  wrappers/           self-training, co-training and tri-training over any base classifier
  graph/              label propagation and spreading on a neighbour graph, Laplacian-regularised models
  generative/         EM on a mixture with labelled rows pinned
  margin/             the transductive SVM, entropy minimisation
train/networks/consistency/   the Π-model, mean teacher, pseudo-label, FixMatch and MixMatch on a small network with augmentations
data/labels/          the partial-label modifier with its truth: which rows keep labels, stratified or not
study/learning-curve/ the label fraction as the x-axis for every method above against the supervised fit on the labelled rows alone
shared/               the partially labelled data slot, the neighbour graph, the augmentation slot, the pseudo-label records (below)
```

Nothing is built; the old plan's `semi-supervised/` folder (self-training, graph-based, labels, margin-based) maps onto the first four folders above.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. Every method here is a classifier on the iterative base and logs `fit/iteration/end` e, `model/params` p, `model/decision-surface` p and `model/predictions` p; the tables name what it adds. The hidden labels are the truth, so `eval/accuracy` s is on the unlabelled rows unless a row says otherwise. The status is `built <path>`, `partly` or empty; library readiness is in the needs column (a function named there exists; `TODO:` names what is missing and its proposed home).

### fit/classify/semi-supervised/wrappers/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `self-training` | The base classifier as a slot, the confidence threshold (or the top-k per round) as the knob, one round per step: fit, score the unlabelled rows, add the confident ones as pseudo-labels (events), refit; the pseudo-labels against the hidden truth (the error among what was added, round by round), the decision surface per round; the failure as a planted scenario: one early wrong pseudo-label spreading through its cluster. | `round/added` s, `round/pseudoError` s, `round/threshold` s, `round/pseudoLabelled` e, `model/pseudoLabels` p (with correctness), `eval/accuracy` s | `data/modifiers/shift.ts`: `semiSupervisedSplit`; `learn/estimator.ts`: `cloneEstimator`; `learn/explained.ts`: `decisionSurfaceOf`; TODO `lib/learn/semi-supervised.ts`: `selfTrainingAlgorithm` | |
| `co-training` | Two views (a feature split as the knob, with planted conditional independence between the views), each classifier labelling the rows the other is unsure of; the disagreement between the views on the tape, the gain over self-training when the views are independent and its loss when they are not; tri-training with three classifiers and majority agreement as the mode. | `round/added` s, `round/disagreement` s, `round/pseudoError` s, `model/viewSurfaces` p, `eval/accuracy` s | as `self-training`; `learn/validation.ts`: `takeColumns` (the views); TODO `coTrainingAlgorithm`, `triTrainingAlgorithm` in `lib/learn/semi-supervised.ts` | |

### fit/classify/semi-supervised/graph/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `label-propagation` | A neighbour graph (k, or an RBF affinity with its width, as knobs), the labels diffusing one iteration per step: the harmonic solution with the labelled rows clamped (propagation) or softly held with α (spreading) as the mode, the label mass per node as colour on the graph, convergence as an event, the closed form as the check; two moons with four labels against the supervised fit on the four alone. The graph's bandwidth is the knob that breaks it (too wide and the clusters merge). | `iteration/labels` t!, `iteration/change` s, `iteration/converged` e, `model/graph` p (a `Structure` with label mass), `model/confidence` p, `eval/accuracy` s | `numerics/spatial/brute.ts`: `kNeighboursGraph`; `kernels/stationary.ts`: `rbfKernel`; `graph/matrices.ts`: `normalisedLaplacian`, `transitionMatrix`, `normalisedAdjacency`; `graph/spectral.ts`: `propagate` (one propagation step with a teleport term); `datasets/classification.ts`: `moons`; TODO `lib/learn/semi-supervised.ts`: `labelPropagationAlgorithm`, `labelSpreadingAlgorithm` (named apart from the community `graph/community.ts: labelPropagationAlgorithm`) | |
| `laplacian-regularisation` | A smoothness penalty over the graph (the Dirichlet energy of the decision function) on a linear or kernel model, LapRLS and LapSVM as modes, γ as the knob: the boundary bending around the unlabelled clusters as γ grows; the penalty's value on the tape. | `solver/objective` s, `solver/dirichletEnergy` s, `model/graph` p, `eval/accuracy` s | `graph/spectral.ts`: `dirichletEnergy`; `graph/matrices.ts`: `laplacian`; `kernels/kernel.ts`: `gram`; `numerics/linalg/cholesky.ts`: `choSolve`; TODO the penalised solve in `lib/learn/semi-supervised.ts` (a few lines) | |

### fit/classify/semi-supervised/generative/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `em-with-labels` | A Gaussian mixture whose labelled rows have their responsibilities pinned, one EM step per step: the components drawn, the labelled points holding them in place, the weight λ of the unlabelled rows as the knob (Nigam's); the cluster assumption broken on purpose (classes that do not match clusters) as the failure, where the unlabelled rows pull the components away from the classes. | `solver/logLikelihood` s, `solver/responsibilities` t!, `solver/means` t!, `model/density` p, `model/centres` p, `eval/accuracy` s | `mixture/gaussian.ts`: `gaussianEmAlgorithm`, `gaussianMStep`, `mixturePosterior`, `initialMixture`; TODO pinned responsibilities as an option of `gaussianEmAlgorithm` | |

### fit/classify/semi-supervised/margin/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `transductive-svm` | The unlabelled rows given labels and the margin pushed into low density: labels switched pair by pair by local search (one switch per step, the objective falling), C for the unlabelled rows annealed up as the knob; the boundary against the supervised SVM's on the same kernel; the balance constraint on the labels. | `solver/objective` s, `solver/switch` m, `solver/unlabelledC` s, `model/margin` p (a `marginRecord`), `eval/accuracy` s | `optim/smo.ts`: `smoAlgorithm`, `svcDual`; `exposition/margins.ts`: `marginRecord`; TODO `lib/learn/semi-supervised.ts`: `s3vmAlgorithm` | |
| `entropy-minimisation` | A probabilistic classifier with the entropy of its predictions on the unlabelled rows as a penalty (λ as the knob), trained by any gradient mode: the boundary moving out of the data as λ grows, the entropy on the tape; confident and wrong when the cluster assumption fails. | `solver/objective` s, `solver/entropy` s, `model/probabilities` p, `eval/accuracy` s | `losses/regularisers.ts`: `entropyPenalty`; `losses/classification.ts`: `crossEntropy`; `methods/shared/options/solvers.ts`: `smoothSolvers` | |

### train/networks/consistency/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `consistency-regularisation` | A small network on two moons or small images with the method as the mode: the Π-model (two stochastic passes made to agree), mean teacher (an exponential-moving-average teacher as the target), pseudo-label, FixMatch (a weak and a strong augmentation, a confidence threshold) and MixMatch; the supervised and the consistency losses on the tape, the decision surface smoothing between the clusters epoch by epoch, the label fraction and the augmentation strength as knobs; the ramp-up of the consistency weight as the schedule. | `training/supervisedLoss` s, `training/consistencyLoss` s, `training/confident` s (FixMatch), `training/teacherAccuracy` s, `model/params` p, `model/decision-surface` p, `eval/accuracy` s | `nn/training/fit.ts`: `fitAlgorithm`; `nn/module/containers.ts`: `mlp`, `Sequential`; `nn/functional/regularisation.ts`: `dropout`, `gaussianNoise`, `cutout`; `optim/first-order.ts`: `ema` (the teacher); `losses/classification.ts`: `crossEntropy`; `losses/regression.ts`: `mse`; `losses/divergence.ts`: `klDivergence`; `signal/image/geometry.ts`: `shift`, `rotate`, `zoom` (the augmentations); `data/batching.ts`: `epochBatches`; TODO the two-stream training loop and the augmentation slot in `shared/` | |

### data/labels/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `label-hiding` | A probe-only data method: the modifier's knobs (a fraction or a count, stratified by class or not), the rows keeping their labels marked, the hidden truth beside them, the generator's Bayes rate as the ceiling every method is measured against; the same slot that `weak-supervision` and `active-learning` start from. | `data/labelled` s, `data/truth` p, `data/bayesRate` s | `data/modifiers/shift.ts`: `semiSupervisedSplit`; `data/truth.ts`: `bayesPredict`, `bayesErrorEstimate` | |

### study/learning-curve/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `label-budget-curve` | Every method above against the supervised fit on the labelled rows alone, the label fraction as the x-axis, several seeds as the band: where the unlabelled rows help, where they hurt (the broken cluster assumption), and the fraction past which they no longer matter. | `trial/labelFraction` s, `trial/accuracy` s, `study/curve` proj (a `learningCurve` per method) | `learn/model-selection/cross-validation.ts`: `learningCurve`; `exposition/searches.ts`: `learningCurve` | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `labels.ts` | The partially labelled data slot: generators with their labels hidden by the modifier and the truth kept (the unlabelled marker, the Bayes rate, the cluster-class correspondence or its planted breakage), with the label fraction as the shared knob; shared with `weak-supervision` and `active-learning` | every group |
| `graphs.ts` | The neighbour graph builder (k or radius, an RBF affinity with its width) as a `Structure` with label mass per node at the viewed step; shared with `unsupervised-learning`'s graph methods | `graph/`, `margin/` |
| `augmentations.ts` | The augmentation slot for the consistency trainers: noise and feature dropout for points, shifts, rotations, crops and cutout for images, with a weak and a strong setting; shared with `representation-learning` | `train/networks/consistency/` |
| `records.ts` | The pseudo-label record (rows added per round with their correctness), the labelled-against-unlabelled marker style on the rows, the graph with label mass | `wrappers/`, `graph/` |

## Examples to make

- Two moons with four labels: label propagation step by step, against a supervised fit on the four alone.
- Self-training's pseudo-labels appearing round by round, with a planted error spreading.
- EM with labels on a mixture: the labelled points pinning the components, then the cluster assumption broken.
- Mean teacher on two moons: the decision surface smoothing between the clusters as the consistency weight ramps up.

## Rendering

- Rows with assignments and confidence as colour, decision surfaces, the neighbour graph with label mass as colour (structure view), curves.
- Missing: a labelled-against-unlabelled marker style on the rows panel; the graph overlay on points; a pseudo-label timeline (rows added per round, coloured by correctness).

## Problems

- The library gap is one file, `lib/learn/semi-supervised.ts` (self-training, co-training, label propagation and spreading, the S3VM), plus pinned responsibilities on the mixture's EM; the consistency trainers wait on the worker producer for anything beyond toy size.
- The three label chapters (`semi-supervised-learning`, `weak-supervision`, `active-learning`) share one data slot (labels hidden, labels weak, labels bought) and one truth; `data/labels/` serves all three, and the chapters could be read as one "learning with fewer labels" part. Recorded, not proposed: their methods differ.
- `graph/community.ts: labelPropagationAlgorithm` is community detection and takes the name; the semi-supervised one needs another (`labelPropagationAlgorithm` in `lib/learn/semi-supervised.ts` would collide on the barrel's flat export).
- `train/networks/consistency/` is the one folder of this chapter in the torch world; whether it belongs beside `feedforward-networks`' trainers or here is a question of the page (the page is about the labels, so here).
