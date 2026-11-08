# Other domains: causal, fairness, explainability, survival, anomaly, vision, text and tuning

Part of the [completeness survey](README.md). The nearest libraries for the domains of omai that the main families do not cover: DoWhy and EconML (causal), fairlearn (fairness), SHAP and Captum (explainability), lifelines and scikit-survival (survival), PyOD (anomaly detection), scikit-image (vision), NLTK and gensim (text), Optuna (tuning). Introspected: fairlearn, shap, captum, scikit-image, nltk, gensim, optuna. From memory: DoWhy, EconML, lifelines, scikit-survival, PyOD. Kornia, spaCy, Hugging Face and similar were skipped: omai has no pretrained-model domain by design.

Counts are of names: "covered" means an omai export or a documented configuration does what the reference name does, "partial" means a different shape or fewer options, "absent" means nothing found by name or by concept, and "out of scope" is plumbing, GPU or production features, downloads and the like that omai should not copy. Semantics were not compared.

## Summary

| library             | source               | area                  | covered | partial | absent | out of scope |
| ------------------- | -------------------- | --------------------- | ------: | ------: | -----: | -----------: |
|                     |                      | api                   |      10 |       0 |      6 |            0 |
|                     |                      | estimators            |       7 |       0 |      0 |            0 |
|                     |                      | refuters              |       6 |       0 |      0 |            0 |
| **dowhy**           | memory               | **all**               |  **23** |   **0** |  **6** |        **0** |
|                     |                      | dml                   |       4 |       2 |      0 |            0 |
|                     |                      | drlearner             |       1 |       3 |      0 |            0 |
|                     |                      | forest                |       2 |       0 |      1 |            0 |
|                     |                      | inference             |       2 |       0 |      0 |            0 |
|                     |                      | iv                    |       0 |       7 |      1 |            0 |
|                     |                      | metalearners          |       3 |       1 |      0 |            0 |
|                     |                      | policy                |       0 |       0 |      3 |            0 |
|                     |                      | validate              |       0 |       0 |      2 |            0 |
| **econml**          | memory               | **all**               |  **12** |  **13** |  **7** |        **0** |
|                     |                      | adversarial           |       2 |       0 |      0 |            0 |
|                     |                      | datasets              |       1 |       0 |      4 |            1 |
|                     |                      | metrics               |      42 |       0 |      0 |            0 |
|                     |                      | postprocessing        |       2 |       0 |      0 |            0 |
|                     |                      | preprocessing         |       2 |       0 |      0 |            0 |
|                     |                      | reductions            |      17 |       0 |      0 |            0 |
| **fairlearn**       | introspection 0.15.0 | **all**               |  **66** |   **0** |  **4** |        **1** |
|                     |                      | shap                  |      23 |       0 |      0 |            5 |
|                     |                      | explainers            |      22 |       0 |      0 |            0 |
|                     |                      | maskers               |       9 |       0 |      0 |            0 |
| **shap**            | introspection 0.54.1 | **all**               |  **54** |   **0** |  **0** |        **5** |
|                     |                      | attr                  |      56 |       0 |     17 |            0 |
|                     |                      | concept               |       6 |       0 |      0 |            0 |
|                     |                      | metrics               |       3 |       0 |      0 |            0 |
|                     |                      | robust                |       5 |       0 |      0 |            0 |
| **captum**          | introspection 0.9.0  | **all**               |  **70** |   **0** | **17** |        **0** |
|                     |                      | datasets              |       0 |       0 |      4 |            0 |
|                     |                      | fitters               |      11 |       0 |      6 |            0 |
|                     |                      | statistics            |       3 |       0 |      2 |            0 |
|                     |                      | utils                 |       4 |       0 |      2 |            0 |
| **lifelines**       | memory               | **all**               |  **18** |   **0** | **14** |        **0** |
|                     |                      | metrics               |       4 |       2 |      0 |            0 |
|                     |                      | models                |       2 |       1 |      7 |            0 |
|                     |                      | nonparametric         |       2 |       1 |      1 |            0 |
|                     |                      | utils                 |       1 |       0 |      0 |            0 |
| **scikit_survival** | memory               | **all**               |   **9** |   **4** |  **8** |        **0** |
|                     |                      | models                |      11 |       0 |     24 |            0 |
|                     |                      | utils                 |       4 |       0 |      0 |            0 |
| **pyod**            | memory               | **all**               |  **15** |   **0** | **24** |        **0** |
|                     |                      | color                 |      41 |       0 |      0 |            0 |
|                     |                      | data                  |       0 |       0 |      1 |           39 |
|                     |                      | draw                  |      17 |       0 |      0 |            0 |
|                     |                      | exposure              |      10 |       0 |      0 |            0 |
|                     |                      | feature               |      41 |       0 |      0 |            0 |
|                     |                      | filters               |      46 |       0 |      0 |            0 |
|                     |                      | graph                 |       8 |       0 |      7 |            0 |
|                     |                      | measure               |      10 |      24 |      0 |            0 |
|                     |                      | metrics               |      10 |       0 |      0 |            0 |
|                     |                      | morphology            |      41 |       0 |      0 |            0 |
|                     |                      | registration          |       3 |       0 |      0 |            0 |
|                     |                      | restoration           |       9 |       0 |      8 |            0 |
|                     |                      | segmentation          |      12 |       0 |      8 |            0 |
|                     |                      | transform             |      33 |       0 |      5 |            0 |
|                     |                      | util                  |       0 |       0 |      0 |           25 |
| **skimage**         | introspection 0.26.0 | **all**               | **281** |  **24** | **29** |       **64** |
|                     |                      | chunk                 |       0 |       0 |      0 |           14 |
|                     |                      | classify              |      11 |       1 |      9 |            0 |
|                     |                      | cluster               |       7 |       0 |      0 |            0 |
|                     |                      | lm                    |       9 |       0 |      0 |            0 |
|                     |                      | metrics               |      28 |       0 |      0 |            0 |
|                     |                      | parse                 |       2 |      41 |      0 |            4 |
|                     |                      | probability           |      24 |       0 |      0 |            0 |
|                     |                      | sentiment             |       0 |       0 |      0 |            2 |
|                     |                      | stem                  |      11 |       0 |      0 |            0 |
|                     |                      | tag                   |       0 |      26 |      7 |            0 |
|                     |                      | tokenize              |      31 |       0 |      1 |            0 |
|                     |                      | translate             |      10 |       0 |     12 |            0 |
| **nltk**            | introspection 3.10.4 | **all**               | **133** |  **68** | **29** |       **20** |
|                     |                      | corpora               |       1 |       0 |      0 |           12 |
|                     |                      | matutils              |       0 |       0 |      0 |            5 |
|                     |                      | models                |      13 |       1 |     10 |            0 |
|                     |                      | parsing.preprocessing |       1 |       0 |      0 |            0 |
|                     |                      | similarities          |       4 |       0 |      6 |            0 |
| **gensim**          | introspection 4.4.0  | **all**               |  **19** |   **1** | **16** |       **17** |
|                     |                      | importance            |       5 |       0 |      0 |            0 |
|                     |                      | pruners               |       9 |       0 |      0 |            0 |
|                     |                      | samplers              |      12 |       0 |      0 |            0 |
|                     |                      | storages              |       0 |       0 |      0 |           15 |
|                     |                      | terminator            |      12 |       0 |      0 |            0 |
|                     |                      | visualization         |      13 |       0 |      0 |            0 |
| **optuna**          | introspection 5.0.0  | **all**               |  **51** |   **0** |  **0** |       **15** |

## Where omai is already strong

- Explainability: exact, kernel, sampling and tree SHAP, LIME, anchors, Integrated Gradients and the Captum attribution set, counterfactuals and recourse, TCAV, influence functions, data Shapley, mechanistic-interpretability tools (activation patching, logit lens, sparse autoencoders).
- Fairness: group, individual and intersectional measures, pre-, in- and post-processing methods, fairness under feedback, with `metricFrame`.
- Causal: regression adjustment, IPW, matching, AIPW, TMLE, DML, meta-learners, causal forests, instruments, RDD, synthetic control, difference-in-differences, sensitivity analysis.
- Survival: Kaplan-Meier, Nelson-Aalen, Cox (Breslow/Efron), AFT, log-rank family, Brier and dynamic AUC.
- Vision: OpenCV-shaped classical pipeline from filters through features, geometry, stereo and bundle adjustment.
- Text: tokenisers, stemmers, subword models, n-gram language models with smoothing, TF-IDF/BM25, Earley parsing and PCFG training, word2vec, fastText.

## Absent and partial items

Priority is high, medium or low by how many learners would expect the thing, how many pages it unlocks and how well it shows off a stepped, traced run; not by recency. Sizes: S a day or less, M a few days, L a week or more. Homes follow `docs/structure.md`. "Nearest in omai" is what a user would reach for today. Items are grouped; the full list of names per group is in `data/match-<lib>.json`.

### dowhy

| reference (area)                                                                                                                                                                                            | what it is and why it matters                                                                                                                                                                                         | nearest in omai                       | proposed home     | size | priority | status |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ----------------- | ---- | -------- | ------ |
| **graph-based causal attribution**: `gcm.arrow_strength`, `gcm.intrinsic_causal_influence`, `gcm.distribution_change`, `gcm.anomaly_scores`, `gcm.auto.assign_causal_mechanisms`, `gcm.falsify_graph` (api) | graphical causal model tools: strength of each edge, attribution of a node's variation to its ancestors (Shapley over a causal graph), root-cause analysis of an anomaly or a distribution change, falsifying a graph | pgm/causal#scm, explain#shapleyValues | pgm/causal/gcm.ts | M    | medium   | absent |

### econml

| reference (area)                                                                                                                                                                                                                                                     | what it is and why it matters                                                                                                          | nearest in omai                                                  | proposed home          | size | priority | status  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------- | ---- | -------- | ------- |
| **policy learning and CATE validation**: `DeepIV`, `CausalIVForest`, `DRPolicyTree`, `DRPolicyForest`, `PolicyTree`, `DRTester`, `BLPEvaluator` (iv, forest, ...)                                                                                                    | policy learning from observational data (a tree that assigns treatments) and CATE validation (best linear predictor test, calibration) | causal#upliftTree, #causalForest                                 | causal/policy.ts       | M    | medium   | absent  |
| **causal ML variants**: `SparseLinearDML`, `KernelDML`, `DomainAdaptationLearner`, `LinearDRLearner`, `SparseLinearDRLearner`, `ForestDRLearner`, `DMLIV`, `NonParamDMLIV`, `OrthoIV`, `DRIV`, `LinearDRIV`, `IntentToTreatDRIV` and 1 more (dml, metalearners, ...) | final-stage variants (sparse, kernel, forest) and DML with instruments; the building blocks exist                                      | causal#doubleMachineLearning, #drLearner, #instrumentalVariables | causal/dml-variants.ts | M    | low      | partial |

### fairlearn

| reference (area)                                                                                                                            | what it is and why it matters                                                                   | nearest in omai      | proposed home        | size | priority | status |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------- | -------------------- | ---- | -------- | ------ |
| **real fairness benchmark datasets**: `fetch_acs_income`, `fetch_bank_marketing`, `fetch_credit_card`, `fetch_diabetes_hospital` (datasets) | further real fairness benchmarks (COMPAS is not in fairlearn either); small vendored subsamples | datasets#adultIncome | datasets/fairness.ts | S    | low      | absent |

### captum

| reference (area)                                                                                                                                                                                                                                      | what it is and why it matters                | nearest in omai | proposed home | size | priority | status |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | --------------- | ------------- | ---- | -------- | ------ |
| `ClassSummarizer`, `CommonStats`, `Count`, `ImageMaskInput`, `InterpretableEmbeddingBase`, `InterpretableInput`, `LLMAttributionResult`, `RemoteLLMProvider`, `StdDev`, `Summarizer`, `SummarizerSingleTensor`, `TextTemplateInput` and 5 more (attr) | LLM attribution wrappers and summary helpers | explain         | to be placed  | -    | low      | absent |

### lifelines

| reference (area)                                                                                                                                                                                                                                                                                                                                     | what it is and why it matters                                                                                                      | nearest in omai       | proposed home             | size | priority | status |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------- | ---- | -------- | ------ |
| **competing risks and time-varying Cox**: `CoxTimeVaryingFitter`, `AalenAdditiveFitter`, `PiecewiseExponentialFitter`, `SplineFitter`, `AalenJohansenFitter`, `BreslowFlemingHarringtonFitter`, `pairwise_logrank_test`, `survival_difference_at_fixed_point_in_time_test`, `add_covariate_to_timeline`, `to_long_format` (fitters, statistics, ...) | time-varying covariates in Cox, the Aalen-Johansen estimator for competing risks, piecewise exponential and spline survival models | survival#coxPh        | survival/                 | M    | medium   | absent |
| **real survival datasets**: `load_rossi`, `load_waltons`, `load_lung`, `load_regression_dataset` (datasets)                                                                                                                                                                                                                                          | real survival datasets (Rossi recidivism, lung, Waltons); omai has a generator with a known hazard                                 | datasets#survivalData | datasets/survival-real.ts | S    | medium   | absent |

### scikit_survival

| reference (area)                                                                                                                                                                 | what it is and why it matters                                                                                             | nearest in omai                    | proposed home               | size | priority | status  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | --------------------------- | ---- | -------- | ------- |
| **survival forests**: `RandomSurvivalForest`, `GradientBoostingSurvivalAnalysis`, `ComponentwiseGradientBoostingSurvivalAnalysis`, `SurvivalTree`, `ExtraSurvivalTrees` (models) | tree ensembles for censored data (log-rank splitting criterion, cumulative hazard leaves)                                 | trees#growTree, ensembles          | survival/forest.ts          | M    | medium   | absent  |
| **competing risks and time-varying Cox**: `cumulative_incidence_competing_risks` (nonparametric)                                                                                 | cumulative incidence under competing risks (Aalen-Johansen)                                                               | survival#kaplanMeier               | survival/competing-risks.ts | S    | medium   | absent  |
| `FastSurvivalSVM`, `FastKernelSurvivalSVM` (models)                                                                                                                              | ranking-based survival SVMs                                                                                               | svm                                | survival/svm.ts             | M    | low      | absent  |
| `concordance_index_ipcw`, `as_concordance_index_ipcw_scorer` (metrics)                                                                                                           | penalised Cox: the solver slot and penalty are available through the smooth-problem design; IPCW concordance is not named | survival#coxPh, #censoringSurvival | survival/cox-penalised.ts   | S    | low      | partial |
| `CoxnetSurvivalAnalysis` (models)                                                                                                                                                | penalised Cox: the solver slot and penalty are available through the smooth-problem design; IPCW concordance is not named | survival#coxPh, #censoringSurvival | survival/cox-penalised.ts   | S    | low      | partial |
| `ipcw_estimator` (nonparametric)                                                                                                                                                 | penalised Cox: the solver slot and penalty are available through the smooth-problem design; IPCW concordance is not named | survival#coxPh, #censoringSurvival | survival/cox-penalised.ts   | S    | low      | partial |

### pyod

| reference (area)                                                                                                                                                | what it is and why it matters                                                                                                                                                                                                                   | nearest in omai                   | proposed home | size | priority | status |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------- | ---- | -------- | ------ |
| **more anomaly detectors**: `HBOS`, `COPOD`, `ECOD`, `ABOD`, `CBLOF`, `SOD`, `LODA`, `SOS`, `LSCP`, `FeatureBagging`, `AutoEncoder`, `VAE` and 12 more (models) | fast and parameter-free detectors a practitioner meets first: histogram-based (HBOS), copula-based (COPOD, ECOD), angle-based (ABOD), cluster-based (CBLOF), LODA, plus the feature-bagging and score-combination wrappers (combination exists) | anomaly#combineScores, #aomScores | anomaly/      | M    | high     | absent |

### skimage

| reference (area)                                                                                                                                                                                                                                                                               | what it is and why it matters                                                                         | nearest in omai                                                                                  | proposed home               | size | priority | status  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------- | ---- | -------- | ------- |
| **region properties**: `CircleModel`, `EllipseModel`, `LineModelND`, `RansacModelProtocol`, `approximate_polygon`, `block_reduce`, `blur_effect`, `euler_number`, `grid_points_in_poly`, `inertia_tensor`, `inertia_tensor_eigvals`, `intersection_coeff` and 12 more (measure)                | per-region properties are spread over several vision functions; regionprops as one table is not there | vision (connectedComponentsWithStats, moments, huMoments, fitEllipse, contourArea, approxPolyDP) | vision/regionprops.ts       | M    | medium   | partial |
| **tomography**: `shepp_logan_phantom`, `frt2`, `ifrt2`, `iradon`, `iradon_sart`, `radon` (data, transform)                                                                                                                                                                                     | the standard CT phantom, needed for the Radon transform pages                                         | signal/image                                                                                     | datasets/images.ts          | M    | medium   | absent  |
| **classical vision long tail**: `RAG`, `cut_normalized`, `cut_threshold`, `merge_hierarchical`, `rag_boundary`, `rag_mean_color`, `show_rag`, `chan_vese`, `checkerboard_level_set`, `disk_level_set`, `inverse_gaussian_gradient`, `morphological_chan_vese` and 3 more (graph, segmentation) | region adjacency graphs and normalised cuts                                                           | vision#graphCut                                                                                  | vision/rag.ts               | M    | low      | absent  |
| **image restoration**: `ball_kernel`, `calibrate_denoiser`, `cycle_spin`, `denoise_invariant`, `ellipsoid_kernel`, `richardson_lucy`, `rolling_ball`, `unsupervised_wiener` (restoration)                                                                                                      | image deconvolution (Richardson-Lucy, Wiener), wavelet denoising, phase unwrapping                    | signal/image                                                                                     | signal/image/restoration.ts | M    | low      | absent  |

### nltk

| reference (area)                                                                                                                                                                                                                                                                                                                                                                                                        | what it is and why it matters                                                                                                                                                                                | nearest in omai                            | proposed home   | size | priority | status  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | --------------- | ---- | -------- | ------- |
| **syntactic parsing**: `BottomUpChartParser`, `BottomUpLeftCornerChartParser`, `BottomUpProbabilisticChartParser`, `ChartParser`, `DependencyEvaluator`, `DependencyGraph`, `FeatureBottomUpChartParser`, `FeatureBottomUpLeftCornerChartParser`, `FeatureChartParser`, `FeatureIncrementalBottomUpChartParser`, `FeatureIncrementalBottomUpLeftCornerChartParser`, `FeatureIncrementalChartParser` and 29 more (parse) | Earley and PCFG training exist; the steppable chart-parser family (top-down, bottom-up, left-corner, shift-reduce, recursive descent), CKY, and dependency parsers (projective, transition-based) are absent | text#earley, #trainPcfg, logic             | text/parsing.ts | M    | medium   | partial |
| **classical NLP**: `accuracy`, `AffixTagger`, `BigramTagger`, `BrillTagger`, `BrillTaggerTrainer`, `CRFTagger`, `ClassifierBasedPOSTagger`, `ClassifierBasedTagger`, `ContextTagger`, `DefaultTagger`, `HiddenMarkovModelTagger`, `HiddenMarkovModelTrainer` and 15 more (classify, tag)                                                                                                                                | sequence taggers exist as HMM and CRF models in pgm; there is no n-gram/backoff or perceptron tagger and no POS-tagging wrapper (no pretrained models)                                                       | pgm#linearChainCrf, pgm/learning#baumWelch | text/tagging.ts | S    | low      | partial |
| **classical NLP**: `Senna`, `HunposTagger`, `SennaChunkTagger`, `SennaNERTagger`, `SennaTagger`, `StanfordNERTagger`, `StanfordPOSTagger`, `StanfordTagger`, `StanfordSegmenter`, `AlignedSent`, `IBMModel`, `IBMModel1` and 9 more (classify, tag, ...)                                                                                                                                                                | Brill transformation-based tagging                                                                                                                                                                           | text                                       | text/tagging.ts | M    | low      | absent  |
| `RTEFeatureExtractor`, `TextCat`, `WekaClassifier`, `call_megam`, `config_megam`, `config_weka`, `rte_classifier`, `rte_features` (classify)                                                                                                                                                                                                                                                                            | recognising textual entailment, language identification (TextCat), external tool bridges                                                                                                                     | nothing close                              | to be placed    | -    | low      | absent  |

### gensim

| reference (area)                                                                                                                                                                | what it is and why it matters                                                                                                                      | nearest in omai                           | proposed home        | size | priority | status  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------- | ---- | -------- | ------- |
| **topic models**: `AuthorTopicModel`, `EnsembleLda`, `HdpModel`, `LdaModel`, `LdaMulticore`, `LdaSeqModel` (models)                                                             | topic models (LDA, HDP, dynamic LDA): the standard text-mining model; the coherence measure and a topic corpus generator exist, the model does not | text#topicCoherence, datasets#topicCorpus | decomposition/lda.ts | M    | high     | absent  |
| `LevenshteinSimilarityIndex`, `SparseTermSimilarityMatrix`, `TermSimilarityIndex`, `UniformTermSimilarityIndex`, `WmdSimilarity`, `WordEmbeddingSimilarityIndex` (similarities) | embedding translation, word mover's distance similarity                                                                                            | embedding                                 | embedding/wmd.ts     | S    | low      | absent  |
| `BackMappingTranslationMatrix`, `TranslationMatrix`, `VocabTransform` (models)                                                                                                  | embedding translation, word mover's distance similarity                                                                                            | embedding                                 | embedding/wmd.ts     | S    | low      | absent  |
| **document embeddings**: `Doc2Vec` (models)                                                                                                                                     | paragraph vectors (PV-DM, PV-DBOW)                                                                                                                 | embedding (word2vec)                      | embedding/doc2vec.ts | S    | low      | absent  |
| `Phrases` (models)                                                                                                                                                              | collocation scores exist; the phrase-merging transformer does not                                                                                  | text#collocationCounts, #pmi              | text/phrases.ts      | S    | low      | partial |
