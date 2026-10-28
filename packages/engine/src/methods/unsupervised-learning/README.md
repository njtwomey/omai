# Unsupervised learning (part 5, learning paradigms)

Structure without labels: clustering (k-means and its variants, hierarchical, DBSCAN, OPTICS, mean shift, spectral, affinity propagation, Gaussian mixtures by EM and variational inference), dimensionality reduction (PCA and its variants, factor analysis, ICA, NMF, dictionary learning, PLS and CCA, random projections), manifold learning (MDS, Isomap, LLE, spectral embedding, t-SNE, UMAP, self-organising maps), topic models, density estimation (histograms, kernels, mixtures, flows), density-ratio estimation and covariance estimation (shrinkage, robust, sparse inverse), as sessions over `cluster`, `mixture`, `numerics`, `kernels`, `graph` and `stats`. Sections as subfolders: `clustering/`, `dimensionality-reduction/`, `manifold-learning/`, `topic-models/`, `density-estimation/`, `density-ratio-estimation/`, `covariance-estimation/`.

## Capabilities

- Clustering with open slots: the initialisation (random, k-means++, from another method), the distance or kernel, the assignment rule (hard, soft, fuzzy), k or the bandwidth; log assignments and centres each step; probes: assignments, centres, the dendrogram, the reachability plot.
- Mixtures: EM and variational inference (exist) with the covariance type as a knob, the responsibilities as a probe, the component count selected by BIC or the variational weights.
- Dimensionality reduction: PCA by the power method or SVD stepped, ICA by FastICA's iterations, NMF by multiplicative updates, dictionary learning alternating; the components and the reconstruction as probes.
- Manifold learning: t-SNE and UMAP's embeddings moving step by step (`embeddingSteps`), the perplexity or neighbours as knobs; Isomap's geodesic graph; SOM's grid folding onto the data.
- Topic models: LDA by Gibbs sampling and by variational inference (with `probabilistic-inference`), the topics' top words at the viewed step.
- Density estimation: kernel density with bandwidth selection, Gaussian mixtures, normalising flows (with `generative-models`); density ratios by classifiers and KLIEP.
- Covariance: shrinkage's intensity chosen, the graphical lasso's sparsity pattern appearing along its path.

## Examples to make

- k-means with k-means++ against random starts, several runs overlaid, the inertia distribution.
- DBSCAN's epsilon as the knob with the core, border and noise points coloured as it expands.
- Agglomerative clustering merging pair by pair, the dendrogram growing.
- t-SNE's early exaggeration phase then settling, perplexity as the knob.
- The graphical lasso's precision matrix sparsifying along the path, against the planted graph.
- LDA's topics converging on a small corpus, the word-topic matrix as a heatmap.

## Rendering

- Clustering steps, mixture steps, embedding steps, components, dendrograms (`tree`), reachability plots, heatmaps, density surfaces.
- Missing: a dendrogram panel that follows the cursor; a word-list probe (`tokens` exists) ranked per topic.

## Shared

- `shared/`: the clusterer and transformer standard probes (`methods/shared/kinds`), the initialisation slot, the distance and kernel slots shared with `supervised-learning`, the planted-cluster data with truth.
- Hierarchical, density-based, spectral and affinity clustering, ICA, NMF, dictionary learning, the manifold methods and the graphical lasso are not all in `lib`: `lib/cluster` grows, `lib/decomposition` and `lib/manifold` are new.

## Problems

- `cluster/`, `mixture/`, `decomposition/`, `manifold/`, `covariance/` move here as sections; `embedding/` goes to `representation-learning`.
