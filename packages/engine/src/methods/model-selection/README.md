# Model selection (part 4, principles of learning)

Choosing among models: cross-validation and its variants, selection bias and nested cross-validation, information criteria, Bayesian model selection, regularisation, hyperparameter optimisation (grid, random, Bayesian, successive halving, population-based) and the comparison of learning algorithms, as sessions over `tune`, `learn` and `metrics` driving the models of other chapters.

## Capabilities

- Cross-validation: k-fold, stratified, grouped, time-series splits drawn and scored fold by fold; the CV curve over a knob; nested CV against the optimistic single loop.
- Information criteria: AIC and BIC against CV on a model sequence; Bayes factors and the evidence (with `probabilistic-inference`).
- Regularisation: a regularisation path (`supervised-learning` lasso path exists) with its CV curve and one-standard-error rule.
- Hyperparameter optimisation: grid and random search, Bayesian optimisation (`tune/bayesianOptimisation` exists), Hyperband and successive halving, population-based training; options: the inner method, its space, the sampler, the budget.
- Comparing algorithms: paired tests over folds, critical difference diagrams, many datasets.

## Examples to make

- Nested CV against naive CV on noise features: the optimistic score exposed.
- The CV curve with the one-standard-error rule picking the simpler model.
- Successive halving: configurations dropped round by round, the survivors' curves.
- Bayesian optimisation's surrogate and acquisition over one or two knobs (exists), with the inner method pluggable.

## Rendering

- CV curves with bands (`crossValidationCurve`), search trials (`searchTrials`), hyperparameter surfaces, fold diagrams.
- Missing: the sweep panel (a knob on x, every metric, click a point to open that run); a bracket diagram for halving.

## Shared

- `shared/`: the sweep base lives here or in `methods/shared/sweeps`: a grid or sampler over an inner method with the inner tapes kept or summarised; the scoring harness.

## Problems

- `tune/` moves here. Hyperparameter optimisation over any method means a method can be the data of another: the nested-session problem in full.
