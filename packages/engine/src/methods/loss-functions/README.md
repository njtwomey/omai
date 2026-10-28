# Loss functions (part 4, principles of learning)

Losses looked at and trained with: regression losses (squared, absolute, Huber, quantile, log-cosh), classification losses (logistic, hinge, exponential, focal, label smoothing), approximate softmax, ranking losses and contrastive and metric-learning losses, as sessions over `losses` with a model and solver slot so a loss is seen both as a curve and as a training outcome.

## Capabilities

- The loss as a function: its curve and gradient against the residual or margin, its properties (convexity, robustness, prox).
- The loss as a training outcome: the same model and solver with the loss as the knob; outliers planted to show robustness; class imbalance for focal and weighted losses.
- Approximate softmax: sampled softmax and noise-contrastive estimation against the full softmax on a large vocabulary.
- Ranking: pairwise and listwise losses on a small ranking.
- Contrastive: InfoNCE and triplet losses on 2-D embeddings moving under training.

## Examples to make

- The loss slot of logistic regression: logistic, hinge, exponential, with the same solver, on the same data, the boundaries compared.
- Quantile regression at several quantiles on heteroscedastic data.
- Huber's delta as the knob on data with outliers.
- Contrastive training in 2-D: the embedding points pulled and pushed step by step.

## Rendering

- Curves of the loss and its derivative, decision boundaries compared, embeddings moving (`embeddingSteps`).
- Missing: the comparison panel over the loss slot.

## Shared

- `shared/`: the objective slot table for the whole library (every `losses` loss with its properties), used by every model's objective slot.

## Problems

- New chapter. The objective slot definition belongs in `methods/shared/slots/objectives.ts`; this chapter is its showcase.
