# Recommendation and retrieval (part 7, application domains)

Recommenders and retrieval models: collaborative filtering (neighbourhood methods, matrix factorisation by ALS and SGD, implicit feedback), feature-interaction models (factorisation machines), deep, sequential, graph and generative recommenders at toy scale, cold start and context, the loss families (pointwise, pairwise, listwise) and learning to rank from clicks, debiasing (position bias, exposure), evaluation and beyond-accuracy objectives (diversity, novelty), online learning and experimentation for recommenders, and retrieval models (TF-IDF and BM25, probabilistic retrieval, dense retrieval with approximate nearest neighbours), as sessions over `text`, `numerics`, `nn`, `losses`, `metrics` and the MovieLens samples in `datasets`.

## Capabilities

- Matrix factorisation stepped: the factors moving, the reconstruction error and the held-out metrics on the tape; options: the solver (ALS, SGD), the rank, the regularisation, implicit against explicit feedback.
- Neighbourhood methods: a user's neighbours and the predicted ratings as probes.
- Ranking losses as a slot on one model: pointwise against pairwise (BPR) against listwise, the ranking metrics on the tape.
- Position bias: clicks simulated under a bias model, the unbiased estimator recovering relevance.
- Retrieval: an inverted index built document by document, BM25's scores for a query as a probe; a dense retriever's embedding space and the approximate-nearest-neighbour index (HNSW's graph) growing.
- Evaluation: precision at k, NDCG, coverage and diversity as the model trains.

## Examples to make

- ALS against SGD on a small ratings matrix: the factors and the error, the rank as the knob.
- BPR's pairwise updates moving an item up a user's ranking.
- Position bias: the biased against the debiased relevance estimates converging.
- HNSW's layers growing as points are inserted, a query's search path.

## Rendering

- Heatmaps of ratings and factors, ranking strips, embedding steps, graph views for HNSW, curves of metrics.
- Missing: a ranking-strip panel that follows the cursor; a ratings-matrix panel with predicted cells.

## Shared

- `shared/`: the interaction data slot (sampled MovieLens, synthetic users with planted preferences), the ranking probes and metrics harness.

## Problems

- `retrieval/` moves here. Matrix factorisation solvers and BM25 are partly in `lib` (`numerics`, `text`); the rest first.
