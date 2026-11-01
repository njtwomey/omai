# Loss functions (part 4, principles of learning)

Losses looked at and trained with: regression losses (squared, absolute, Huber, log-cosh, quantile, expectile), classification losses (logistic, hinge, exponential, modified Huber, perceptron, focal, label smoothing), penalties (ridge, lasso, elastic net), likelihood losses, approximate softmax, ranking losses, contrastive and metric-learning losses, as sessions over `losses` with a model and solver slot so a loss is seen both as a curve and as a training outcome. This chapter is the showcase of three shared slots: `regressionLosses` and `classificationLosses` (`methods/shared/options/objectives.ts`, each loss with its knobs and its properties: convex, smooth, a likelihood) and `penalties` (the same file: strength, value, prox, sparse). A method's objective slot is a loss and a penalty from these tables, and its validity rules read their properties.

## Structure

A loss drawn as a function of the residual or margin is a procedure with no data, so it lives under `simulate/numerics/losses/` (section 3 of `methods/README.md`). A loss as a training outcome is the objective slot of a task method: the comparison of losses on one model, solver and dataset is a sweep over that slot and lives in `study/sweep/`, taking the task method as its inner slot. The families that only make sense inside a larger model (approximate softmax in a language model, ranking losses in a ranker, contrastive losses in a representation learner) are rows here with the loss as the knob and `see` rows pointing at the chapter whose model it is.

```
simulate/numerics/losses/
  curves/             a loss and its derivative over the residual or the margin, its properties read off; the prox of a penalty drawn
study/sweep/objective/
  regression/         `regressionLosses` × `penalties` on one regressor and solver: robustness to outliers, quantiles, expectiles, heteroscedasticity
  classification/     `classificationLosses` × `penalties` on one classifier and solver: boundaries compared, imbalance, focal and label smoothing, class weights
  penalties/          the penalty as the knob: ridge against lasso against elastic net at one strength, the path over the strength (see `supervised-learning`)
  likelihood/         losses that are negative log-likelihoods: Poisson, Gaussian with a learned variance, Tweedie, censored; the family as the knob
train/
  representation/     contrastive losses watched on 2-D embeddings (see `representation-learning`)
  sequence/           approximate softmax against the full one (see `large-language-models`)
fit/rank/             pairwise and listwise ranking losses on a small ranking (see `recommendation-and-retrieval`)
shared/               the residual and margin grids, the planted-outlier and imbalance modifiers, the per-loss summary record (below)
```

Nothing of this chapter is built; its exemplar is the loss slot of logistic regression (section 8 of `methods/README.md`), which is also `supervised-learning`'s first task method.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. For a curve method the step is one knob value of the loss (δ, τ, γ) and the picture is a probe; for a sweep the step is one entry of the slot, with the inner fit's tape kept.

### curves/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `regression-loss-curve` | Every entry of `regressionLosses` as a function of the residual with its derivative (the influence function): squared grows without bound, absolute is flat (robust, nonsmooth at zero), Huber turns from one to the other at δ (the step), log-cosh is Huber smoothed, pinball tilts with τ, expectile tilts the square; the table's properties (`convex`, `smooth`, `likelihood`) read off the picture. | `curve/knob` s, `curve/loss` p (the loss and its derivative over the residual grid), `curve/properties` m | `losses/regression.ts`: `mse`, `mae`, `huber`, `logCosh`, `pinball`, `expectile`, `smoothL1`; `methods/shared/options/objectives.ts`: `regressionLosses` (the slot, with each entry's `space` and `properties`); `autodiff`: `grad` for the derivative | |
| `classification-loss-curve` | Every entry of `classificationLosses` over the margin y·f(x): zero-one as the target, logistic and exponential as smooth upper bounds, hinge with its margin knob (zero beyond it, so support vectors), squared hinge, modified Huber, perceptron (zero for any correct sign); focal with γ flattening the easy examples and label smoothing capping the confident ones; the multiclass hinges (Crammer-Singer, Weston-Watkins) beside the cross-entropy. | `curve/knob` s, `curve/loss` p (over the margin), `curve/properties` m | `losses/margin.ts`: `logistic`, `hinge`, `squaredHinge`, `exponential`, `modifiedHuber`, `perceptron`, `crammerSinger`, `westonWatkins`; `losses/classification.ts`: `binaryCrossEntropyWithLogits` (`labelSmoothing`), `crossEntropy` (`labelSmoothing`, `classWeight`), `focal`, `categoricalFocal`; `methods/shared/options/objectives.ts`: `classificationLosses` (focal, label smoothing and class weights TODO as entries or knobs of the table) | |
| `penalty-curve` | Every entry of `penalties` on a coefficient: the value (the l1 corner at zero against the l2 bowl), the constraint set drawn in 2-D (the diamond, the disc, the elastic net's rounded diamond) with the loss's contours touching it, the prox as a map of the coefficient (soft thresholding's dead zone, shrinkage's slope), the strength as the step. | `curve/strength` s, `curve/penalty` p, `curve/prox` p, `curve/constraintSet` p | `losses/regularisers.ts`: `l1Penalty`, `l2Penalty`, `elasticNet`, `groupLasso`, `nuclearNorm`; `optim/proximal.ts`: `proxL1`, `proxSquaredL2`, `proxElasticNet`, `proxGroupLasso`, `proxNuclear`; `methods/shared/options/objectives.ts`: `penalties` (group lasso and nuclear norm TODO as entries) | |
| `divergence-curve` | The losses between distributions: KL in both directions on two Gaussians (mode seeking against mass covering as the means separate, the step), Jensen-Shannon bounded, MMD with its bandwidth, the Sinkhorn divergence with ε; the pictures that explain the generative chapters' choices. | `curve/separation` s, `curve/divergences` t, `curve/densities` p | `losses/divergence.ts`: `klDivergence`, `jensenShannon`, `mmd`, `sinkhorn`, `distillation`; `dists/divergences.ts`: `klDivergence`, `hellinger`, `wasserstein2Gaussian` | |

### study/sweep/objective/regression/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `robust-regression` | The `regressionLosses` slot swept on one linear regressor and one solver with outliers planted (their share and size as knobs): squared chases them, absolute and Huber do not, the fitted line per loss overlaid on the data, the coefficients' error to the clean truth; δ of Huber as a second sweep. The solver follows the loss (IRLS for absolute and Huber, closed form for squared, a `smoothSolvers` mode for any smooth entry) through the validity rules. | `sweep/loss` m (the entry and its knobs), `sweep/coefficientError` s, `sweep/fit` t!, `fit/line` p (per entry), `solver/objective` s (the inner tape) | `methods/shared/options/objectives.ts`: `regressionLosses`; `linear/solvers.ts`: `irlsAlgorithm`, `leastSquaresSolve`, `quasiNewtonSolve`; `methods/shared/options/solvers.ts`: `smoothSolvers`; `datasets/outliers.ts`: `linearOutliers`; `stats/robust.ts`: `huber`, `ransacAlgorithm` for the comparison; TODO `methods/shared/sweeps.ts` (the sweep base over an inner method, section 4 of `methods/README.md`) | |
| `quantile-and-expectile` | Pinball at several τ on heteroscedastic data (the step is τ): the fitted quantile lines fanning out, crossing where the model is misspecified (an event); expectiles beside them, smoother but harder to read; the coverage of the τ-band against its nominal value. | `sweep/tau` s, `sweep/coverage` s, `sweep/crossing` e, `fit/lines` p (every quantile so far), `solver/objective` s | `losses/regression.ts`: `pinball`, `expectile`; `datasets/regression-curves.ts`: `heteroscedastic`, `quantileCurves` (the true quantiles); `linear/solvers.ts`: `irlsAlgorithm` | |
| `likelihood-losses` | The loss as a negative log-likelihood with the family as the mode: Poisson on counts, Gaussian with a learned variance (the mean and the spread fitted together, the spread tracking the data's), Tweedie with its power, a censored likelihood on truncated targets: the fitted mean and its band over the data, the deviance residuals. | `sweep/family` m, `fit/mean` p, `fit/band` p, `fit/deviance` s, `solver/objective` s | `losses/regression.ts`: `poissonNll`, `gaussianNll`, `tweedieDeviance`; `losses/density.ts`: `negativeLogLikelihood`, `heteroscedasticGaussianNll`, `censoredNll`, `mixtureDensityNll`; `dists/families/dispersion.ts`: `poisson`, `gamma`, `tweedie`; `metrics/deviance.ts`: `meanPoissonDeviance`, `meanTweedieDeviance` | |

### study/sweep/objective/classification/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `loss-slot-of-logistic-regression` | The exemplar: `classificationLosses` swept on one linear classifier, one solver and one 2-D dataset, the same key throughout: the boundaries overlaid (logistic and exponential pulled by far points, hinge fixed by the margin's support vectors, perceptron stopping at any separator), the margin distribution per loss, the held-out log loss and accuracy; which losses give probabilities (the `likelihood` property) and which only scores. | `sweep/loss` m, `sweep/accuracy` s, `sweep/logLoss` s, `sweep/margins` t!, `model/decision-surface` p (per entry), `solver/objective` s | `methods/shared/options/objectives.ts`: `classificationLosses`; `methods/shared/options/solvers.ts`: `smoothSolvers` (hinge and perceptron are nonsmooth: subgradient or SMO modes TODO in the validity table); `datasets/classification.ts`: `moons`, `blobs`, `gaussianQuantiles`; `learn/bases/kinds.ts`: `kindProbes` (the decision surface); the sweep base (TODO) | |
| `imbalance` | The class ratio as the knob with the loss as the mode: plain cross-entropy drifts to the majority, class weights and focal's γ pull the boundary back; the per-class recall per step, the probability histogram per class (focal flattening the easy ones), the calibration cost of focal (see `decision-theory-and-calibration`). | `sweep/ratio` s, `sweep/loss` m, `sweep/recallPerClass` t, `sweep/probabilities` p, `model/decision-surface` p | `losses/classification.ts`: `focal`, `crossEntropy` (`classWeight`); `learn/preprocessing/resampling.ts`: `randomOverSampler`, `smote` for the data-side comparison; `metrics/classification.ts`: `recallScore`, `balancedAccuracyScore` | |
| `label-smoothing` | ε as the step on a small network or a logistic model: the logits' norm shrinking, the maximum probability capped at 1 − ε + ε/K, the penultimate features clustering tighter per class (a 2-D projection), the calibration before and after. | `sweep/epsilon` s, `sweep/logitNorm` s, `sweep/maxProbability` s, `sweep/ece` s, `model/embedding` p | `losses/classification.ts`: `crossEntropy` (`labelSmoothing`); `metrics/probabilistic.ts`: `expectedCalibrationError`; `nn/training/fit.ts`: `fitAlgorithm` | |
| `multiclass-and-multilabel` | Softmax cross-entropy against one-vs-rest logistic against the multiclass hinges on a three-class dataset: the boundaries, the rejected regions, the per-class losses; multilabel soft margin with the label cardinality as the knob. | `sweep/loss` m, `sweep/accuracy` s, `model/decision-surface` p, `sweep/perClassLoss` t | `losses/classification.ts`: `crossEntropy`, `nllLoss`, `multiLabelSoftMargin`; `losses/margin.ts`: `crammerSinger`, `westonWatkins`; `datasets/make-classification.ts`: `makeClassification`, `makeMultilabelClassification` | |

### study/sweep/objective/penalties/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `penalty-slot` | The `penalties` slot swept at one strength on one model and loss: ridge shrinking every coefficient, lasso zeroing some (the active set), elastic net between them (the l1 share as a second knob); the coefficients as bars per entry, the held-out error, the solver following the penalty (closed form or a `smoothSolvers` mode for l2, proximal gradient or coordinate descent for l1 and elastic net, through the `hasProx` and `smooth` properties). | `sweep/penalty` m, `sweep/coefficients` t!, `sweep/nonzeros` s, `sweep/heldOutError` s, `solver/objective` s | `methods/shared/options/objectives.ts`: `penalties`; `optim/splitting.ts`: `proximalGradientAlgorithm`; `optim/sparse.ts`: `lassoCoordinateDescentAlgorithm`; `linear/solvers.ts`: `svdRidge`, `coordinateDescentSolve` | |
| `regularisation-path` | see `supervised-learning` (the lasso path over the strength, built as `methods/linear/lassoPath.ts`, with the cross-validated choice in `model-selection`) | | | |
| `structured-penalties` | Group lasso selecting whole groups (the groups' norms per step), the nuclear norm on a matrix (its singular values thresholded, the rank falling), an orthogonality penalty on a weight matrix, total variation on a signal (the staircase): the penalty as the mode, the strength as the step. | `sweep/strength` s, `sweep/groupNorms` t, `sweep/singularValues` t, `sweep/rank` s, `fit/signal` p | `losses/regularisers.ts`: `groupLasso`, `nuclearNorm`, `orthogonality`, `entropyPenalty`; `optim/proximal.ts`: `proxGroupLasso`, `proxNuclear`; `optim/total-variation.ts`: `proxTv1d`, `tv2dAlgorithm` | |

### specialised families

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `approximate-softmax` | A large vocabulary with the output loss as the mode: the full softmax, sampled softmax (the negatives per step, the bias of the estimate against the full gradient), hierarchical softmax over a Huffman tree (the path's length per word), noise-contrastive estimation; the gradient's cosine to the full one and the cost per step on the tape. The model is `large-language-models`'. | `loss/value` s, `loss/fullValue` s, `loss/gradientCosine` s, `loss/cost` s, `loss/negatives` t, `tree/path` p | `losses/large-output.ts`: `sampledSoftmax`, `sampleClasses`, `hierarchicalSoftmax`, `huffmanCoding`; TODO `noiseContrastive` in the same file; `datasets/ranking.ts`: `zipfianCatalogue` for the frequencies | |
| `ranking-losses` | A small ranking with the loss as the mode: pairwise (margin ranking, RankNet's probability of the right order, BPR) and listwise (ListNet, ListMLE, LambdaRank's NDCG-weighted gradients, ApproxNDCG): the scores per item moving per step, the induced order, NDCG on the tape, which pairs the loss is pushing (a message). The ranker is `recommendation-and-retrieval`'s. | `scores/items` t!, `order/ranking` t!, `eval/ndcg` s, `loss/value` s, `loss/pairPushed` m | `losses/ranking.ts`: `marginRanking`, `rankNet`, `pairwiseHinge`, `bpr`, `lambdaRank`, `listNet`, `listMle`, `approxNdcg`; `metrics/ranking.ts`: `ndcgScore` | |
| `contrastive-losses` | 2-D embeddings under the loss as the mode (contrastive pairs, triplet with its margin, InfoNCE and NT-Xent with the temperature as the knob, supervised contrastive, ArcFace and CosFace's angular margins): the points pulled and pushed step by step, the positive and negative distances' histograms separating, the temperature's effect on the hardness weighting. The learner is `representation-learning`'s. | `embedding/points` t!, `embedding/positiveDistance` s, `embedding/negativeDistance` s, `loss/value` s, `loss/temperature` s, `model/embedding` p | `losses/contrastive.ts`: `contrastive`, `tripletMargin`, `infoNce`, `ntXent`, `supervisedContrastive`, `arcFace`, `cosFace`, `cosineEmbedding`; `exposition/embeddings.ts`: `embeddingSteps` | |
| `segmentation-losses` | A binary mask with the loss as the mode: cross-entropy against Dice and Tversky (α, β trading false positives for negatives), focal Tversky, the Lovász hinge optimising IoU directly; the predicted mask per step as a picture, IoU and Dice on the tape, the behaviour on a tiny foreground (where cross-entropy ignores it). The model is `computer-vision`'s. | `mask/prediction` t!, `eval/iou` s, `eval/dice` s, `loss/value` s, `mask/picture` p | `losses/segmentation.ts`: `dice`, `tversky`, `focalTversky`, `lovaszHinge`, `lovaszSoftmax`; `metrics/segmentation.ts`: `maskIou`, `diceCoefficient`; `datasets/images.ts`: `shapes` | |
| `generative-losses` | see `generative-models` (the adversarial losses and gradient penalties, the ELBO and its KL, score matching, diffusion and flow-matching targets, each a loss of `losses/generative.ts` watched on its model) | | | |
| `physics-losses` | see `scientific-machine-learning` (the PDE residual, boundary and initial losses with their gradient-norm and NTK weightings) | | | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `grids.ts` | The residual and margin grids a loss is drawn over, the derivative by `autodiff`, and the curve record (loss, derivative, the zero-one or squared reference) | `curves/` |
| `modifiers.ts` | Data modifiers with their truths: planted outliers (share, size, leverage), class imbalance by ratio, heteroscedastic noise, truncation for the censored likelihood; over `data/recipes.ts`' modifiers where they exist | `study/sweep/objective/*` |
| `summary.ts` | The per-entry summary record of a sweep over an objective slot: the entry and its knobs, the fitted parameters, the held-out metrics, the margin or residual distribution, for the comparison panel | `study/sweep/objective/*` |
| `validity.ts` | The textbook pairing of losses and solvers (squared: closed form; absolute, Huber, pinball: IRLS; smooth: any `smoothSolvers` mode; hinge, perceptron: subgradient or SMO) as `needs` entries, until the task methods carry them (section 7.1 of `methods/README.md`) | `study/sweep/objective/*`, `supervised-learning` |

## Examples to make

- The loss slot of logistic regression: logistic, hinge, exponential with the same solver on the same data, the boundaries compared and the margins' histograms beneath.
- Quantile regression at several quantiles on heteroscedastic data, the band's coverage against its nominal value.
- Huber's δ as the knob on data with outliers, the fitted line and the influence function side by side.
- Contrastive training in 2-D: the embedding points pulled and pushed step by step, the temperature as the knob.
- Every regression loss and its derivative on one axis, the table's properties beside each.

## Rendering

- Curves of the loss and its derivative (`curves`), decision boundaries compared (`decisionSurface`), embeddings moving (`embeddingSteps`), masks as images.
- Missing: the comparison panel over an objective slot (one panel per entry under one cursor, or overlaid boundaries with a legend); a constraint-set panel (the penalty's ball with the loss's contours touching it).

## Problems

- New chapter. The option tables exist (`methods/shared/options/objectives.ts`); what is missing is the sweep base that drives a task method over a slot (`methods/shared/sweeps.ts`) and the task methods themselves (`supervised-learning`'s). Focal, label smoothing, class weights, group lasso and the nuclear norm are in `losses` but not yet entries or knobs of the tables.
- The nonsmooth classification losses (hinge, perceptron) have no mode in `smoothSolvers`; the validity table needs a subgradient or SMO mode before the loss slot of logistic regression covers its whole table.
- The specialised families belong to their models' chapters; the rows here are the loss as the knob with everything else fixed, and should stay thin.
