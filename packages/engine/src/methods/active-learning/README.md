# Active learning (part 5, learning paradigms; a chapter the encyclopedia folds into online learning and data-centric learning)

Choosing what to label: query strategies (uncertainty, margin, entropy, query by committee, expected error and variance reduction, density weighting, core sets, BALD) in pool and stream settings, and the learning curve by label budget, as sessions over the models of `supervised-learning` with a labelled set that grows step by step.

## Capabilities

- The query loop as the algorithm: fit, score the pool, pick, label, repeat; options: the model, the strategy, the batch size, the budget; log the chosen points, the model's accuracy on held-out data and the learning curve against random sampling.
- Strategies as a slot shared by every model that gives scores or probabilities; committees over any base.
- Stream setting: a point arrives, the strategy decides to ask or not, with a budget.
- Failure cases: sampling bias, outliers chosen by uncertainty, the cold start.

## Examples to make

- Uncertainty sampling against random on two moons: the chosen points marked in order, the two learning curves.
- Query by committee: the committee's disagreement surface and the point chosen.
- Density-weighted uncertainty avoiding the outliers that plain uncertainty chases.

## Rendering

- Rows with the labelled set growing (order as colour), decision surfaces, learning curves (`learningCurve`), the strategy's score surface as a probe.
- Missing: a query-order overlay on the rows panel.

## Shared

- `shared/`: the query loop base, the strategy slot table, the pool and stream data slots (with `semi-supervised-learning`'s partial-label modifier).

## Problems

- Keeps its name. Each query step refits a model: budget the inner fit or warm-start it.
