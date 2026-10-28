# Exploration and discovery (part 3, data)

A dataset explored: profiling, visual exploration, dependence discovery, pattern mining and causal discovery, as sessions over `stats`, `information`, `pgm` (structure learning) and pattern-mining algorithms (moving to `lib`).

## Capabilities

- Profiling: per-column summaries, missingness maps, distinct counts, as a table with a dataset slot.
- Visual exploration: pair plots, embeddings for overview (`unsupervised-learning`), slices.
- Dependence: correlation, mutual information, distance correlation and kernel dependence (HSIC) between every pair, estimated as samples arrive; options: the measure, the dataset.
- Pattern mining: Apriori level by level, FP-growth's tree, sequential patterns, subgroup discovery with its quality measure; options: support, confidence, the data.
- Causal discovery: PC's skeleton and orientation phases, GES's score climbing, LiNGAM, on data from a known graph; options: the graph, the test, the sample size.

## Examples to make

- Apriori: candidate itemsets generated and pruned level by level, support as the knob.
- PC on data from a planted DAG: edges removed test by test, then oriented, against the truth.
- Dependence measures on nonlinear relations: correlation misses what mutual information finds.
- Subgroup discovery: the best subgroups and their quality as the beam advances.

## Rendering

- Tables, graphs with edges appearing and orienting (structure view), itemset lattices (`tree`/`diagram`), heatmaps of dependence.
- Missing: a lattice or level-wise panel for itemsets; a graph-against-truth overlay (true edges, found edges, wrong edges).

## Shared

- `shared/`: planted structure (a DAG with its data, a transaction generator with planted patterns) with truth, shared with `causal-inference`.

## Problems

- `patterns/` and the discovery half of `causal/` move here. Pattern-mining algorithms are not in `lib` yet: they go to a `lib/patterns` domain first.
