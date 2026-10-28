# Feature engineering and selection (part 3, data)

Features built and chosen: construction, temporal and relational aggregates, automated synthesis, and selection by filter, wrapper and embedded methods with its stability, as sessions over `learn`, `metrics` and the model slots of `supervised-learning`.

## Capabilities

- Construction: interactions, polynomial terms, aggregates, cyclical encodings, with a model's score as each is added.
- Automated synthesis: a search over transformations scored by a model, stepped.
- Filter selection: univariate scores, mutual information, relevance and redundancy (mRMR) ranked.
- Wrapper selection: forward, backward and recursive feature elimination stepped, the score at each size.
- Embedded: lasso paths (`supervised-learning`), tree importances, permutation and drop-column importance.
- Stability and inference: stability selection over subsamples, knockoffs, post-selection inference.

## Examples to make

- Recursive feature elimination one feature at a time, the score and the surviving set.
- Importance measures disagreeing on correlated features (impurity against permutation against drop-column).
- Stability selection: selection frequencies rising with the penalty.
- Knockoffs: the knockoff statistics and the threshold at a chosen false discovery rate.

## Rendering

- Bars for importances at the viewed step, a path of selection frequencies, the score against the set size.
- Missing: a feature-set panel (which features are in, over steps, as a strip).

## Shared

- `shared/`: the importance record (several measures, one table), the selection study base (a loop of fits with a changing feature set).

## Problems

- New chapter. Selection studies nest model fits; the sweep base must summarise the inner tapes.
