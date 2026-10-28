# Supervised learning (part 5, learning paradigms)

The supervised models, each generic in its slots: regression (least squares, ridge, lasso, elastic net, robust, quantile, polynomial and spline bases), classification (logistic, linear and quadratic discriminants, perceptron), generalised linear models, generative classifiers (naive Bayes), nearest neighbours, kernel methods and support vector machines, Gaussian processes, trees and ensembles, generalised additive models, multilabel and multi-output wrappers, structured prediction, ordinal regression and survival learning, as sessions over `linear`, `kernels`, `nn` (for gradient solvers), `learn`, `losses`, `optim` and the tree and ensemble algorithms moving to `lib`. Sections as subfolders: `regression/`, `classification/`, `generalised-linear-models/`, `generative-classifiers/`, `nearest-neighbours/`, `kernel-methods/`, `gaussian-processes/`, `trees-and-ensembles/`, `generalised-additive-models/`, `multilabel-and-multioutput/`, `structured-prediction/`, `ordinal-regression/`, `survival/`.

## Capabilities

- Linear models with every slot open: objective (any regression or classification loss) × penalty (none, l2, l1, elastic net, group) × solver (closed form, IRLS, Newton, coordinate descent, proximal, any `optim` optimiser, SGD over batches) × basis (raw, polynomial, splines, random features); the validity table says which go together; log the coefficients, the objective, the step; probes: the fit and its band, the decision surface, the residuals.
- Regularisation paths and their CV curves (exist for the lasso; generalise to every penalty).
- Generalised linear models: every exponential family with its link, IRLS stepped.
- Naive Bayes families on counts, booleans and reals; the class-conditional densities as probes.
- Nearest neighbours: k, the metric and the weighting as knobs, the neighbourhood drawn for a query; radius neighbours and nearest centroids.
- Kernel methods: kernel ridge, kernel PCA (with `unsupervised-learning`), SVMs by SMO stepped with the support vectors appearing; the kernel slot (every `kernels` kernel with its hyperparameters).
- Gaussian processes: the prior, the posterior as points arrive, marginal-likelihood fitting of the kernel's hyperparameters stepped, sparse approximations, classification by Laplace or EP.
- Trees: CART, ID3, C4.5 grown split by split with every candidate split evaluated; pruning paths; ensembles: bagging, random forests, extra trees, AdaBoost (exists for Haar stumps), gradient boosting with its losses, stacking.
- GAMs: smooth terms by penalised splines, backfitting stepped, smoothness selection.
- Wrappers as slots: one-vs-rest, one-vs-one, output codes, classifier chains over any base.
- Structured prediction: CRFs for sequence labelling, structured perceptron; ordinal regression by thresholds; Cox and parametric survival models (with `statistics`).

## Examples to make

- Logistic regression's solver race: IRLS, Newton, gradient descent and SGD from the same start on the same data, the boundary and the objective.
- The penalty slot: ridge, lasso and elastic-net paths on one dataset, coefficients against the penalty.
- A tree growing: the split just chosen with every candidate's impurity, the leaf regions on the plane.
- Gradient boosting: residuals shrinking round by round, the staged prediction.
- A Gaussian process posterior updating point by point, the kernel as a knob with its hyperparameters.
- SMO: the pair chosen each step, the margin and support vectors.
- k nearest neighbours: k as the knob, the decision surface smoothing.

## Rendering

- Decision surfaces, fits with bands, regularisation paths, trees (`tree`, `pruningPath`), ensemble stages, CV curves, coefficient tables, neighbours (`neighbours`), margins (`margin`).
- Missing: the solver and penalty comparison panel; a candidate-splits panel for trees (every candidate's score at the viewed step); a kernel-matrix heatmap probe.

## Shared

- `shared/`: the linear model's slot tables (objective, penalty, solver, basis) and validity table, shared verbs (`linear/_verbs.ts` exists), the kernel slot, the base-estimator slot for wrappers and ensembles, the standard probes of classifiers and regressors (`methods/shared/kinds`).
- Trees, ensembles, SMO, GP fitting and GAM backfitting are not all in `lib` yet: `lib/trees`, `lib/ensembles`, `lib/svm` (or `kernels`), `lib/gp` first.

## Problems

- `linear/`, `gam/`, `svm/`, `kernel-methods/`, `gp/`, `naive-bayes/`, `discriminant/`, `neighbours/`, `trees/`, `ensemble/`, `multiclass/` move here as sections.
- The biggest chapter: the slot design is proved here first (logistic regression with every slot) before the rest follows.
- Estimators against methods: these are `learn` estimators today, made into methods by the host; the fold of the two contracts (problem 3 of the top README) is decided on this chapter.
