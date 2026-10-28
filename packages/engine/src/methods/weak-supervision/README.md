# Weak supervision (part 5, learning paradigms)

Labels that are indirect: label models over several noisy sources (majority vote, Dawid-Skene by EM over crowd workers, the data-programming label model over labelling functions), noisy labels (the noise transition estimated, confident learning, loss correction), positive-unlabelled learning, learning from label proportions and multiple-instance learning, and the downstream model trained on weak labels against clean ones, as sessions over `learn`, `pgm` (the label models are latent-class networks), `losses`, `optim/smo` and the models of `supervised-learning`, with the planted sources of `datasets/weak-labels.ts` as the truth.

## Structure

A label model turns several weak sources into one probabilistic label before any model is fitted, so the label models and the noise estimators are data preparation and live under `data/labels/` (section 3 of `methods/README.md`); the learners that take weak labels as their data slot are classifiers and live under `fit/classify/weak/`; the comparison of the downstream model on weak against clean labels is a study.

```
data/labels/
  label-models/       majority vote, Dawid-Skene, the data-programming label model, worker and function diagnostics
  noisy-labels/       the noise transition estimated from anchor points, confident learning (which labels are wrong)
fit/classify/weak/
  noisy/              forward and backward loss correction, robust losses, co-teaching on a small network
  positive-unlabelled/ the class prior estimated, the unbiased and non-negative PU risks
  proportions/        learning from label proportions
  multiple-instance/  mi-SVM and MIL pooling (max, mean, attention) on bags
study/weak-labels/    the downstream model on weak labels against clean labels, by the number and accuracy of sources
shared/               the planted sources and bags, the votes board, the shared label slot (below)
```

Nothing is built; the old plan's `weak-supervision/` folder (label models, noisy labels, positive-unlabelled, proportions, multiple-instance) maps onto the folders above.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The classifiers log the standard `fit/iteration/end` e, `model/params` p, `model/decision-surface` p and `model/predictions` p; the label models yield a label per row (`model/labels` p) and the quality of that label against the clean truth (`eval/labelAccuracy` s). The status is `built <path>`, `partly` or empty; library readiness is in the needs column (a function named there exists; `TODO:` names what is missing and its proposed home).

### data/labels/label-models/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `majority-vote` | The baseline over labelling functions or workers that vote or abstain: ties, coverage (rows no source labels), the weighted vote with known accuracies as the mode; the accuracy of the vote against the number of sources and their planted accuracies (knobs), and its ceiling when the sources are correlated. | `sources/count` s, `sources/coverage` s, `model/labels` p, `model/votesBoard` p (sources against rows), `eval/labelAccuracy` s | `datasets/weak-labels.ts`: `labellingFunctions`, `crowdLabels`; TODO `lib/learn/weak-supervision.ts`: `majorityVote` (a few lines) | |
| `dawid-skene` | EM one step per step: each worker's confusion matrix and the class priors re-estimated from the current posterior, the posterior over true labels sharpening, the log likelihood rising; one bad annotator's confusion matrix found (the planted matrix beside the estimate), a spammer and an adversarial worker as knobs; the one-coin model as the simpler mode. The model is a Bayesian network with a latent class and one child per worker, so it draws and its EM is the library's. | `solver/logLikelihood` s, `solver/confusions` t!, `solver/priors` t!, `solver/posterior` t!, `model/labels` p, `model/confusions` p (per worker against the truth), `model/structure` p, `eval/labelAccuracy` s | `datasets/weak-labels.ts`: `crowdLabels`; `pgm/models/bayes-net.ts`: `bayesNet`; `pgm/learning/parameters.ts`: `emAlgorithm` (the latent-class network); `metrics/classification.ts`: `confusionMatrix`; TODO the wrapper `dawidSkene` in `lib/learn/weak-supervision.ts` with the one-coin and GLAD modes | |
| `label-model` | The data-programming label model: the accuracies and the correlations of labelling functions learnt from their agreements alone (no ground truth; the covariance of votes with the planted dependency structure as the knob), the probabilistic labels it emits; the matrix of functions against rows as a board (votes, abstentions, conflicts); against majority vote as the functions' accuracies, coverage and overlaps vary. | `solver/objective` s, `solver/accuracies` t!, `solver/correlations` t, `model/labels` p (probabilistic), `model/votesBoard` p, `eval/labelAccuracy` s | `datasets/weak-labels.ts`: `labellingFunctions`; `stats/covariance.ts`: `empiricalCovariance`; TODO `lib/learn/weak-supervision.ts`: `labelModel` (the agreement-based estimate), or a `probprog` program with `probprog/engines/svi.ts` as the second mode | |

### data/labels/noisy-labels/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `noise-transition` | The transition matrix (which class is mislabelled as which) estimated from a classifier's confident predictions (anchor points), against the planted matrix of the label-noise modifier; symmetric against class-dependent noise as the knob, the estimate improving with the classifier. | `estimate/transition` t!, `estimate/error` s, `model/anchors` p | `data/modifiers/labels.ts`: `withLabelNoise`, `noiseTransition`; `learn/model-selection/cross-validation.ts`: `crossValPredict`; TODO `estimateNoiseTransition` in `lib/learn/weak-supervision.ts` (a few lines) | |
| `confident-learning` | The joint distribution of given and true labels from out-of-sample probabilities, the rows ranked by the chance their label is wrong: the planted flips found (precision at k as k grows), pruning the suspects and refitting as the mode; shared with `data-centric-learning`. | `rank/k` s, `rank/precision` s, `model/suspects` p (rows ranked), `model/joint` p, `eval/accuracy` s (after pruning) | `learn/model-selection/cross-validation.ts`: `crossValPredict`; `data/modifiers/labels.ts`: `withLabelNoise`; TODO `confidentLearning` in `lib/learn/weak-supervision.ts` | |

### fit/classify/weak/noisy/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `loss-correction` | Training under label noise with the correction as the mode: forward and backward correction with the transition matrix (true or estimated above), a robust loss (mean absolute error, generalised cross-entropy, symmetric cross-entropy), co-teaching (two networks exchanging their small-loss rows); the noise rate as the knob, the test accuracy against the unaware model, the memorisation of noisy labels visible as the training loss keeps falling after the test accuracy peaks. | `training/loss` s, `training/cleanLoss` s, `training/noisyLoss` s, `eval/accuracy` s, `model/decision-surface` p | `losses/classification.ts`: `crossEntropy`; `losses/regression.ts`: `mae`; `data/modifiers/labels.ts`: `noiseTransition`, `withLabelNoise`; `nn/training/fit.ts`: `fitAlgorithm`; `methods/shared/options/solvers.ts`: `smoothSolvers` (the linear case); TODO the corrected and robust losses (a few lines over `losses/classification.ts`) and the co-teaching loop in `shared/` | |

### fit/classify/weak/positive-unlabelled/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `positive-unlabelled` | Only positives labelled: the class prior π as the knob, or estimated (Elkan-Noto from the classifier's scores on the labelled positives; a mixture-proportion estimate as the mode), the unbiased PU risk and its non-negative correction (nnPU) as training modes by any gradient solver: the negative risk going below zero and the boundary overfitting without the correction, two moons with one class labelled; the selection bias of which positives get labelled (SCAR against biased) as the data mode. | `solver/objective` s, `solver/negativeRisk` s, `solver/prior` s, `model/decision-surface` p, `eval/accuracy` s | `datasets/weak-labels.ts`: `positiveUnlabelled`; `losses/classification.ts`: `binaryCrossEntropyWithLogits`; `methods/shared/options/solvers.ts`: `smoothSolvers`; TODO `lib/learn/weak-supervision.ts`: `elkanNotoPrior`, `puRisk`, `nnPuRisk` | |

### fit/classify/weak/proportions/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `label-proportions` | Bags with only their class proportions observed, the bag size as the knob: a probabilistic classifier trained to match the proportions (the proportion loss), the per-row labels recovered as the bags get smaller and lost as they grow; the bags drawn with their proportions. | `solver/proportionLoss` s, `model/bags` p (with their proportions), `model/decision-surface` p, `eval/accuracy` s | `datasets/weak-labels.ts`: `labelProportionBags`; `methods/shared/options/solvers.ts`: `smoothSolvers`; TODO the proportion loss in `lib/learn/weak-supervision.ts` (a few lines) | |

### fit/classify/weak/multiple-instance/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `multiple-instance` | Bags positive when any instance is: mi-SVM (the instance labels inside positive bags switched one iteration per step, an SVM refitted) and MIL pooling on a small network (max, mean and attention pooling as modes; the attention weights showing which instance made the bag positive, against the planted witness) as the two families; the bag size and the witness rate as knobs. | `solver/objective` s, `solver/switched` s, `model/bags` p, `model/witnesses` p (attention or the instance labels against the planted witness), `eval/bagAccuracy` s | `optim/smo.ts`: `smoAlgorithm`, `svcDual`; `nn/module/attention.ts`: `MultiheadAttention`; `nn/functional/pooling.ts`: `globalMaxPool`, `globalAvgPool`; `nn/training/fit.ts`: `fitAlgorithm`; TODO a bag generator with witness instances in `shared/bags.ts` and `miSvmAlgorithm` in `lib/learn/weak-supervision.ts` | |

### study/weak-labels/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `weak-against-clean` | The downstream classifier trained on a label model's labels against the same classifier on clean labels, as the number of sources, their accuracy and their coverage vary (a sweep): the gap closing with sources, the point at which more weak labels beat fewer clean ones (the label budget as the other axis). | `trial/sources` s, `trial/accuracy` s, `trial/weakAccuracy` s, `trial/cleanAccuracy` s, `study/curves` proj | a `study/sweep` over `data/labels/label-models/` and any classifier of `fit/classify/`; `learn/model-selection/cross-validation.ts`: `learningCurve` | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `sources.ts` | The planted sources with their truths: labelling functions with accuracy, coverage and correlation knobs, crowd workers with their confusion matrices (good, spammer, adversarial), from `datasets/weak-labels.ts`; shared with `data-centric-learning` | `label-models/`, `study/` |
| `bags.ts` | Bags for multiple-instance and proportion learning with their witness instances and proportions; the bag size and the witness rate as knobs (the proportion bags exist in `datasets/weak-labels.ts`, the MIL bags are to add) | `proportions/`, `multiple-instance/` |
| `labels.ts` | The shared label slot of the three label chapters: the clean truth kept beside the weak, noisy or hidden labels; from `data/modifiers/labels.ts` and `data/modifiers/shift.ts` | every group |
| `records.ts` | The votes board (sources against rows: votes, abstentions, conflicts, with the agreement), a confusion matrix per worker against the truth, the suspect ranking | `label-models/`, `noisy-labels/` |

## Examples to make

- Labelling functions with planted accuracies: the label model's estimates converging, the weak labels against the truth, the votes board.
- Dawid-Skene with one bad annotator: its confusion matrix found, the posterior sharpening.
- Positive-unlabelled on two moons: the prior as the knob, the boundary moving, the negative risk with and without the correction.
- Confident learning: the planted flips ranked to the top.

## Rendering

- Rows with weak labels as colour and disagreement as a marker, confusion matrices per annotator, curves of accuracy, the structure view for the latent-class network.
- Missing: a labelling-function board panel (functions against rows, votes as cells, conflicts marked); a bag panel (instances grouped, the witness marked).

## Problems

- Keeps its name. The label models are one file away (`lib/learn/weak-supervision.ts`), and Dawid-Skene needs no new algorithm: it is `pgm/learning/parameters.ts: emAlgorithm` on a latent-class Bayesian network, which also means it draws.
- The label models are data preparation (`data/labels/`), not models, and `confident-learning` is claimed by `data-centric-learning` as well; it lives here with the noise it diagnoses and that chapter carries a `see` row.
- The three label chapters share one data slot (hidden, weak, bought labels); `data/labels/` serves all three (see `semi-supervised-learning`'s Problems).
