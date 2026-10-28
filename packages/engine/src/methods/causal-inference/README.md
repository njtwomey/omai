# Causal inference (a methods chapter; the encyclopedia splits it between experimental design, causal discovery and the trustworthy part)

Causes estimated: identification on a causal graph (back-door, front-door, instruments), estimators of treatment effects (regression adjustment, matching, propensity scores and inverse probability weighting, doubly robust and double machine learning, difference in differences, regression discontinuity), heterogeneous effects and uplift models, and sensitivity analysis, as sessions over `pgm` (causal graphs), `stats` and the models of `supervised-learning`.

## Capabilities

- Data from a known structural causal model with the true effect, the confounding strength as a knob.
- Identification: the adjustment set found on the graph, d-separation queried.
- Estimators: each estimator's estimate converging as n grows with its interval, against the truth; propensity scores' overlap as a probe; the bias of naive comparison.
- Heterogeneous effects: meta-learners (S, T, X) and causal forests' conditional effects over the covariate plane.
- Sensitivity: the estimate against an unobserved confounder's strength.

## Examples to make

- Simpson's paradox as a structural model: the naive and adjusted estimates with the confounder's strength as the knob.
- Propensity-score overlap: the two groups' score histograms and the weights.
- Double machine learning's cross-fitting stepped, the estimate's interval against the truth.
- Uplift over the plane from a planted heterogeneous effect.

## Rendering

- Graphs with the adjustment set highlighted (structure view), estimates with intervals converging, surfaces of effect over the plane, histograms of overlap.
- Missing: a graph query overlay (paths blocked and open).

## Shared

- `shared/`: the structural causal model slot with truth, shared with `exploration-and-discovery` (discovery) and `statistics` (experiments).

## Problems

- `causal/` moves here (effects); discovery goes to `exploration-and-discovery`. Decide whether one chapter should hold both.
