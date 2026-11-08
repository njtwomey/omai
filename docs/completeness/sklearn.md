# scikit-learn and imbalanced-learn

Part of the [completeness survey](README.md). scikit-learn is the reference for the estimator API, preprocessing, model selection, metrics, clustering, decomposition and the classical learners. omai does not copy the class-per-model layout: a family is one configurable estimator (`linearModel` covers LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, Poisson, Gamma, Tweedie and SGD). Matching is therefore done by concept, with the nearest omai name written down in `scripts/completeness-survey/verdicts/sklearn.txt`.

Counts are of names: "covered" means an omai export or a documented configuration does what the reference name does, "partial" means a different shape or fewer options, "absent" means nothing found by name or by concept, and "out of scope" is plumbing, GPU or production features, downloads and the like that omai should not copy. Semantics were not compared.

## Summary

| library      | source               | area                     | covered | partial | absent | out of scope |
| ------------ | -------------------- | ------------------------ | ------: | ------: | -----: | -----------: |
|              |                      | base                     |       0 |       0 |      0 |           17 |
|              |                      | calibration              |       3 |       0 |      0 |            0 |
|              |                      | cluster                  |      20 |       3 |      4 |            0 |
|              |                      | compose                  |       4 |       0 |      0 |            0 |
|              |                      | covariance               |      16 |       0 |      0 |            0 |
|              |                      | cross_decomposition      |       4 |       0 |      0 |            0 |
|              |                      | datasets                 |      27 |       0 |      0 |           19 |
|              |                      | decomposition            |      11 |       2 |      7 |            0 |
|              |                      | discriminant_analysis    |       2 |       0 |      0 |            0 |
|              |                      | dummy                    |       0 |       0 |      2 |            0 |
|              |                      | ensemble                 |      18 |       0 |      0 |            1 |
|              |                      | feature_extraction       |       0 |       0 |      2 |            0 |
|              |                      | feature_extraction.image |       1 |       0 |      4 |            0 |
|              |                      | feature_extraction.text  |       6 |       0 |      0 |            1 |
|              |                      | feature_selection        |      18 |       0 |      0 |            1 |
|              |                      | frozen                   |       0 |       0 |      0 |            1 |
|              |                      | gaussian_process         |       2 |       0 |      0 |            0 |
|              |                      | gaussian_process.kernels |      15 |       0 |      0 |            3 |
|              |                      | impute                   |       4 |       0 |      0 |            0 |
|              |                      | inspection               |       4 |       0 |      0 |            0 |
|              |                      | isotonic                 |       2 |       0 |      0 |            1 |
|              |                      | kernel_approximation     |       5 |       0 |      0 |            0 |
|              |                      | kernel_ridge             |       1 |       0 |      0 |            0 |
|              |                      | linear_model             |      42 |       2 |      0 |            0 |
|              |                      | manifold                 |      10 |       0 |      0 |            0 |
|              |                      | metrics                  |      73 |       0 |      2 |            3 |
|              |                      | metrics.pairwise         |      24 |       0 |      0 |            2 |
|              |                      | mixture                  |       0 |       2 |      0 |            0 |
|              |                      | model_selection          |      30 |       0 |      2 |            3 |
|              |                      | multiclass               |       3 |       0 |      0 |            0 |
|              |                      | multioutput              |       4 |       0 |      0 |            0 |
|              |                      | naive_bayes              |       5 |       0 |      0 |            0 |
|              |                      | neighbors                |      16 |       0 |      0 |            0 |
|              |                      | neural_network           |       3 |       0 |      0 |            0 |
|              |                      | pipeline                 |       4 |       0 |      0 |            0 |
|              |                      | preprocessing            |      28 |       0 |      1 |            0 |
|              |                      | random_projection        |       3 |       0 |      0 |            0 |
|              |                      | semi_supervised          |       3 |       0 |      0 |            0 |
|              |                      | svm                      |       6 |       2 |      0 |            0 |
|              |                      | tree                     |       7 |       0 |      0 |            1 |
|              |                      | utils                    |       6 |       0 |      0 |            0 |
| **sklearn**  | introspection 1.9.1  | **all**                  | **430** |  **11** | **24** |       **53** |
|              |                      | combine                  |       0 |       0 |      2 |            0 |
|              |                      | ensemble                 |       0 |       0 |      4 |            0 |
|              |                      | metrics                  |       7 |       0 |      0 |            0 |
|              |                      | over_sampling            |       4 |       0 |      4 |            0 |
|              |                      | pipeline                 |       2 |       0 |      0 |            0 |
|              |                      | under_sampling           |       3 |       0 |      8 |            0 |
| **imblearn** | introspection 0.14.2 | **all**                  |  **16** |   **0** | **18** |        **0** |

## Where omai is already strong

- Linear and generalised linear models, path algorithms (LARS, coordinate descent), robust and quantile regression, with the solver as a slot.
- Trees, forests, boosting, stacking, voting; kernels (including approximations); SVMs; Gaussian processes with exact marginal likelihood.
- Model selection: nearly every splitter of scikit-learn plus nested CV, learning and validation curves, permutation tests; calibration and conformal prediction well beyond scikit-learn.
- Clustering (k-means variants, DBSCAN, OPTICS, HDBSCAN, spectral, affinity propagation, BIRCH), manifold learning (t-SNE, UMAP, PaCMAP, TriMap), covariance estimation.
- Every iterative routine is stepped and traced; scikit-learn offers `partial_fit` at best.

## Absent and partial items

Priority is high, medium or low by how many learners would expect the thing, how many pages it unlocks and how well it shows off a stepped, traced run; not by recency. Sizes: S a day or less, M a few days, L a week or more. Homes follow `docs/structure.md`. "Nearest in omai" is what a user would reach for today. Items are grouped; the full list of names per group is in `data/match-<lib>.json`.

### sklearn

| reference (area)                                                                                                                                                       | what it is and why it matters                                                                                                                       | nearest in omai                                          | proposed home                        | size | priority | status  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------ | ---- | -------- | ------- |
| **estimator wrappers over existing algorithms**: `KMeans`, `MiniBatchKMeans`, `k_means`, `BayesianGaussianMixture`, `GaussianMixture` (cluster, mixture)               | the stepped algorithms exist; there is no fit/predict estimator wrapper for plain k-means                                                           | cluster#lloydAlgorithm, cluster#miniBatchKMeansAlgorithm | cluster/kmeans.ts                    | S    | high     | partial |
| **baselines**: `DummyClassifier`, `DummyRegressor` (dummy)                                                                                                             | baselines (most frequent, prior, mean, median, constant) are the first thing a learner compares against                                             | nothing close                                            | learn/dummy.ts                       | S    | high     | absent  |
| **topic models**: `LatentDirichletAllocation` (decomposition)                                                                                                          | a topic corpus generator and coherence measure exist, but not the model (variational Bayes or collapsed Gibbs); a probprog program could express it | text#topicCoherence, datasets#topicCorpus                | decomposition/lda.ts (or probprog)   | M    | high     | absent  |
| **dictionary learning and sparse coding**: `DictionaryLearning`, `MiniBatchDictionaryLearning`, `SparseCoder`, `dict_learning`, `dict_learning_online` (decomposition) | only the convolutional version exists; plain sparse coding and dictionary learning (K-SVD, online) are absent                                       | decomposition#convolutionalDictionaryMethod              | decomposition/dictionary-learning.ts | M    | medium   | absent  |
| **threshold tuning**: `FixedThresholdClassifier`, `TunedThresholdClassifierCV` (model_selection)                                                                       | decision-threshold tuning on a probabilistic classifier                                                                                             | nothing close                                            | learn/model-selection/threshold.ts   | S    | medium   | absent  |
| **decomposition wrappers**: `TruncatedSVD` (decomposition)                                                                                                             | the SVD and LSA exist; no estimator that is PCA without centring on sparse input                                                                    | numerics/linalg#svd, text#lsa                            | decomposition/truncated-svd.ts       | S    | medium   | partial |
| **image patches**: `PatchExtractor`, `extract_patches_2d`, `img_to_graph`, `reconstruct_from_patches_2d` (feature_extraction.image)                                    | image patches and pixel-grid graphs for clustering                                                                                                  | graph#gridGraph                                          | vision/patches.ts                    | S    | low      | absent  |
| **biclustering**: `SpectralBiclustering`, `SpectralCoclustering`, `consensus_score` (cluster, metrics)                                                                 | only the generator of biclusters exists, not the algorithms                                                                                         | datasets#makeBiclusters                                  | cluster/biclustering.ts              | M    | low      | absent  |
| `get_bin_seeds`, `kmeans_plusplus` (cluster)                                                                                                                           | k-means++ seeding is an option of initialCentres (KMeansInit); no standalone function                                                               | cluster#initialCentres                                   | to be placed                         | -    | low      | absent  |
| `DictVectorizer`, `FeatureHasher` (feature_extraction)                                                                                                                 | hashing of strings exists for text; no dict-of-features vectoriser                                                                                  | text#hashingVectorize                                    | text/vectorisers                     | S    | low      | absent  |
| `LinearSVC`, `LinearSVR` (svm)                                                                                                                                         | kernel SVM with a linear kernel; no liblinear-style primal solver                                                                                   | svm#svc, svm#svr                                         | to be placed                         | -    | low      | partial |
| `MiniBatchNMF` (decomposition)                                                                                                                                         | NMF exists; mini-batch form does not                                                                                                                | decomposition#nmf                                        | to be placed                         | S    | low      | absent  |
| `MiniBatchSparsePCA` (decomposition)                                                                                                                                   | sparse PCA exists; mini-batch variant does not                                                                                                      | decomposition#sparsePcaAlgorithm                         | to be placed                         | S    | low      | partial |
| `ARDRegression` (linear_model)                                                                                                                                         | automatic relevance determination exists for kernel machines (RVM); not as a linear-model regressor                                                 | gp#relevanceVectorMachine, kernels#ard                   | linear/ard.ts                        | S    | low      | partial |
| `LassoLarsIC` (linear_model)                                                                                                                                           | lars path exists; the AIC/BIC choice along it is not a single call                                                                                  | linear#lars                                              | linear/lars.ts                       | S    | low      | partial |
| `class_likelihood_ratios` (metrics)                                                                                                                                    | positive and negative likelihood ratio (diagnostic testing); easy                                                                                   | metrics                                                  | metrics/classification.ts            | S    | low      | absent  |
| `add_dummy_feature` (preprocessing)                                                                                                                                    | only the nearest thing in the next column                                                                                                           | linear#withIntercept                                     | to be placed                         | -    | low      | absent  |

### imblearn

| reference (area)                                                                                                                                                                                                                                                                           | what it is and why it matters                                                                                                                                                                                                         | nearest in omai                                                              | proposed home                     | size | priority | status |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------- | ---- | -------- | ------ |
| **imbalanced-learn resampling**: `SMOTEENN`, `SMOTETomek`, `BalancedBaggingClassifier`, `BalancedRandomForestClassifier`, `EasyEnsembleClassifier`, `RUSBoostClassifier`, `KMeansSMOTE`, `SMOTEN`, `SMOTENC`, `SVMSMOTE`, `AllKNN`, `ClusterCentroids` and 6 more (combine, ensemble, ...) | SMOTE, borderline-SMOTE, ADASYN, Tomek links, NearMiss and random sampling exist; the categorical SMOTE variants (SMOTEN, SMOTENC), the combined samplers (SMOTE+ENN, SMOTE+Tomek) and the edited-nearest-neighbour family are absent | learn/preprocessing#smote, #borderlineSmote, #adasyn, #tomekLinks, #nearMiss | learn/preprocessing/resampling.ts | S    | medium   | absent |
