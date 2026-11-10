/**
 * Write the family files of the completeness survey from `docs/completeness/data/match-<lib>.json`.
 *
 *   node scripts/completeness-survey/report.ts
 *
 * Output: `docs/completeness/<family>.md` (first pass: sklearn, scipy-statsmodels, probabilistic, graphical-models, rl, torch,
 * graph-neural-networks, other; second pass: metric-learning, self-supervised, training-and-paradigms,
 * bayesian-optimisation-and-gp, tabular-and-interpretable, torch-ecosystem, structured-prediction, adjacent; third pass: covariance) and
 * `docs/completeness/data/counts.json`. `README.md` and `repertoire.md` are written
 * by hand. Run `match.ts` first. The prose of each family (what it covers, where omai is strong) is in this file.
 */
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const ROOT = resolve(import.meta.dirname, '..', '..')
const DIR = join(ROOT, 'docs', 'completeness')
const DATA = join(DIR, 'data')
const HERE = import.meta.dirname

interface Row {
  lib: string
  area: string
  name: string
  status: 'full' | 'partial' | 'absent' | 'skip'
  how: string
  omai?: string
  note?: string
  home?: string
  size?: string
  priority?: string
  group?: string
}

interface Family {
  file: string
  title: string
  libs: string[]
  intro: string
  strong: string[]
  /** Libraries whose every entry is listed (covered ones too), for the families that were checked name by name. */
  fullTables?: string[]
  /** Libraries judged by area, not by name: shown in the summary but left out of the family totals. */
  areaLevel?: string[]
  /** Extra prose after the strengths (what the comparison cannot see, how the family was checked). */
  after?: string[]
  /** Libraries whose names repeat entries counted in an earlier family: listed in the tables, left out of the totals. */
  relisted?: string[]
  /** Hand-written Markdown fragments (files under `scripts/completeness-survey/prose/`): `before` after the intro, `closing` at the end. */
  before?: string[]
  closing?: string[]
}

const FAMILIES: Family[] = [
  {
    file: 'sklearn',
    title: 'scikit-learn and imbalanced-learn',
    libs: ['sklearn', 'imblearn'],
    intro:
      'scikit-learn is the reference for the estimator API, preprocessing, model selection, metrics, clustering, ' +
      'decomposition and the classical learners. omai does not copy the class-per-model layout: a family is one ' +
      'configurable estimator (`linearModel` covers LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, ' +
      'Poisson, Gamma, Tweedie and SGD). Matching is therefore done by concept, with the nearest omai name written ' +
      'down in `scripts/completeness-survey/verdicts/sklearn.txt`.',
    strong: [
      'Linear and generalised linear models, path algorithms (LARS, coordinate descent), robust and quantile regression, with the solver as a slot.',
      'Trees, forests, boosting, stacking, voting; kernels (including approximations); SVMs; Gaussian processes with exact marginal likelihood.',
      'Model selection: nearly every splitter of scikit-learn plus nested CV, learning and validation curves, permutation tests; calibration and conformal prediction well beyond scikit-learn.',
      'Clustering (k-means variants, DBSCAN, OPTICS, HDBSCAN, spectral, affinity propagation, BIRCH), manifold learning (t-SNE, UMAP, PaCMAP, TriMap), covariance estimation (the estimators; the area around them is surveyed in `covariance.md`).',
      'Every iterative routine is stepped and traced; scikit-learn offers `partial_fit` at best.',
    ],
  },
  {
    file: 'scipy-statsmodels',
    title: 'scipy, statsmodels and time-series libraries',
    libs: ['scipy', 'statsmodels', 'sktime', 'tslearn', 'prophet'],
    intro:
      'scipy supplies the numerical and statistical base (stats, optimize, linalg, signal, interpolate, integrate, ' +
      'spatial, special, sparse, fft, cluster); statsmodels the regression, GLM, test and time-series classics; ' +
      'sktime, tslearn and prophet the forecasting and time-series-learning layer. scipy and statsmodels were ' +
      'introspected; sktime, tslearn and prophet are from memory.',
    strong: [
      'Statistics: about 170 test exports, power analysis, effect sizes, resampling, copulas, extreme-value fits, smoothing; the distribution families of `dists` carry log-densities, sampling and bijectors under one contract.',
      'Optimisation: all the scipy minimisers plus optax-style optimisers, trust regions, LP/QP/MILP, global optimisers, and every one runs as a traced Algorithm.',
      'Time series: ARMA/ARIMA, exponential smoothing and ETS, STL and MSTL, VAR and VECM, Johansen, GARCH family, Kalman/EKF/UKF/particle filters, matrix profile, change-point detection, shapelets.',
      'Numerics: dense and sparse linear algebra, Krylov solvers, quadrature, interpolation, FFT/NUFFT, polynomials, inverse problems, ODE/PDE solvers with dense output and events.',
      'Signal processing and LTI systems cover scipy.signal at its full width, plus audio features.',
    ],
  },
  {
    file: 'probabilistic',
    title: 'Probabilistic programming and Bayesian workflow',
    libs: ['numpyro', 'arviz', 'pymc', 'tensorflow_probability'],
    intro:
      'NumPyro, PyMC, TensorFlow Probability and ArviZ cover probabilistic programs, inference engines, ' +
      'distributions and bijectors, and the Bayesian diagnostic workflow. torch.distributions is in torch.md. ' +
      'NumPyro and ArviZ were introspected; PyMC and TFP are from memory.',
    strong: [
      'Programs with `sample`, `observe`, plates and effect handlers; MCMC (HMC, NUTS, Gibbs, slice, MALA, SGLD, SMC, tempered SMC), variational inference with auto-guides, Laplace, ABC, exact enumeration and conjugate engines behind one `infer`.',
      'Diagnostics as numbers: R-hat, ESS, MCSE, HDI, divergences, BFMI, PSIS-LOO, WAIC, model comparison and stacking weights.',
      'Distributions: dozens of families, compositions (mixture, truncated, censored, transformed), copulas, GAMLSS families; about 50 bijector exports including coupling and autoregressive flows.',
      'Every sampler is a stepped algorithm with the chain on the tape, so a posterior can be watched filling in.',
    ],
  },
  {
    file: 'graphical-models',
    title: 'Graphical models',
    libs: ['pgmpy', 'hmmlearn'],
    intro:
      "pgmpy is the closest library to omai's `pgm` domain: factors, Bayesian and Markov networks, structure and " +
      'parameter learning, exact and approximate inference, causal inference. hmmlearn covers HMMs. Both were ' +
      'introspected. pyAgrum and libpgm were not installed and are not listed separately; their scope is the same.',
    strong: [
      'Exact inference: variable elimination with elimination-order heuristics, junction trees, belief propagation, max-product and MAP, MPLP, treewidth; approximate: Gibbs, likelihood weighting, rejection, mean field, variational message passing, EP, tree-reweighted BP.',
      'Models beyond pgmpy: CRFs (linear-chain and general), Gaussian MRFs, influence diagrams, plate models, factor graphs, Ising and Potts samplers (Swendsen-Wang, Wolff), HMM variants (HSMM, IO-HMM, factorial, GMM-HMM), linear dynamical systems.',
      'Structure learning (PC, GES, hill climbing, Chow-Liu, NOTEARS, DirectLiNGAM, FCI), identification (backdoor, front-door, instruments), counterfactuals and twin networks.',
      'Inference algorithms run stepwise on the tape (messages, eliminations, sweeps).',
    ],
  },
  {
    file: 'rl',
    title: 'Reinforcement learning',
    libs: ['gymnasium', 'stable_baselines3', 'cleanrl', 'pettingzoo'],
    intro:
      'gymnasium for environments, wrappers, spaces and vector environments; Stable-Baselines3 and CleanRL for the ' +
      'algorithms and their supporting pieces (policies, buffers, noise, callbacks); PettingZoo for multi-agent ' +
      'environments. gymnasium was introspected; the others are from memory.',
    strong: [
      'Tabular and function-approximation methods of Sutton and Barto (DP, MC, TD, n-step, eligibility traces, tile coding, Dyna, prioritised sweeping) plus bandits with regret bounds and best-arm identification.',
      'Deep RL: DQN family (double, dueling, C51, QR-DQN, IQN, Rainbow), PPO, TRPO, A2C, DDPG, TD3, SAC, HER, model-based (PETS, MBPO, Dreamer, MuZero), offline (BCQ, CQL, IQL, TD3+BC, Decision Transformer), imitation (DAgger, GAIL, MaxEnt IRL), multi-agent (VDN, QMIX, MADDPG), safe RL (shields, Lagrangian PPO), curiosity.',
      'Infrastructure: wrappers, vector environments, replay (uniform, prioritised, n-step), returns and advantage estimators including V-trace and Retrace, evaluation statistics with bootstrap intervals, AEC and parallel multi-agent APIs.',
      'Environments: classic control, toy text, bandits, MinAtar, grid worlds, several continuous-control tasks with analytic models, board games (tic-tac-toe, Connect Four, Go, Nim), Kuhn and Leduc poker, MPE-style particle worlds.',
    ],
  },
  {
    file: 'torch',
    title: 'PyTorch',
    libs: ['torch', 'torchvision', 'torchmetrics'],
    intro:
      'torch.nn (layers and containers), torch.nn.functional, torch.nn.init, torch.optim and its schedulers, ' +
      'torch.distributions, torch.autograd and torch.func, torch.utils.data, plus torchvision transforms and datasets ' +
      'and torchmetrics. All introspected from the installed versions.',
    strong: [
      'Layers: nearly all the convolution, pooling, normalisation, recurrent and attention layers of torch.nn, plus Mamba, RetNet, KAN, neural ODEs/CDEs/SDEs, Fourier neural operators, EGNN, mixture of experts, diffusion U-Net pieces, Hopfield layers.',
      'Optimisers: SGD with momentum, Adam family, Lion, LAMB, Adafactor, Shampoo, L-BFGS and 15 schedules as one slot; gradient clipping; EMA, SWAG.',
      'Autodiff: JAX-style `grad`, `vjp`, `jvp`, `vmap`, Hessians, Taylor mode, checkpointing, `checkPrimitive` for new primitives; differentiation is opt-in so there is no `no_grad`.',
      'Losses: about 190 exports, including contrastive, ranking, preference (DPO, KTO, ORPO), score matching and diffusion objectives.',
      'Interpretability and compression built in: LoRA, pruning, quantisation (GPTQ, AWQ), adversarial attacks and certification.',
    ],
  },
  {
    file: 'graph-neural-networks',
    title: 'Graph neural networks and graph libraries',
    libs: ['torch_geometric', 'networkx'],
    intro:
      'PyTorch Geometric (and DGL, which has the same scope) for GNN layers, pooling, batching, samplers, transforms ' +
      "and datasets; NetworkX for classical graph algorithms, since omai's `graph` domain is NetworkX-shaped. " +
      'NetworkX was introspected; PyG is from memory (it is not installed).',
    strong: [
      'Layers: GCN, GraphSAGE, GAT, GIN, relational GCN, EGNN and a graph transformer, on a general message-passing function; spectral filters (Chebyshev, heat kernel, wavelets).',
      'Sampling: neighbour sampling with variance estimates, layer-wise importance sampling, random-walk subgraphs, node2vec walks; positional encodings (Laplacian, random walk); Weisfeiler-Leman.',
      'Classical algorithms: centralities, community detection (Louvain, Leiden, Girvan-Newman, label propagation), flows and cuts, matching, shortest paths, spanning trees, isomorphism (VF2), percolation, epidemics on networks.',
      'Everything above that is iterative runs as a traced algorithm.',
    ],
  },
  {
    file: 'other',
    title: 'Other domains: causal, fairness, explainability, survival, anomaly, vision, text and tuning',
    libs: [
      'dowhy',
      'econml',
      'fairlearn',
      'shap',
      'captum',
      'lifelines',
      'scikit_survival',
      'pyod',
      'skimage',
      'nltk',
      'gensim',
      'optuna',
    ],
    intro:
      'The nearest libraries for the domains of omai that the main families do not cover: DoWhy and EconML (causal), ' +
      'fairlearn (fairness), SHAP and Captum (explainability), lifelines and scikit-survival (survival), PyOD ' +
      '(anomaly detection), scikit-image (vision), NLTK and gensim (text), Optuna (tuning). Introspected: fairlearn, ' +
      'shap, captum, scikit-image, nltk, gensim, optuna. From memory: DoWhy, EconML, lifelines, scikit-survival, ' +
      'PyOD. Kornia, spaCy, Hugging Face and similar were skipped: omai has no pretrained-model domain by design.',
    strong: [
      'Explainability: exact, kernel, sampling and tree SHAP, LIME, anchors, Integrated Gradients and the Captum attribution set, counterfactuals and recourse, TCAV, influence functions, data Shapley, mechanistic-interpretability tools (activation patching, logit lens, sparse autoencoders).',
      'Fairness: group, individual and intersectional measures, pre-, in- and post-processing methods, fairness under feedback, with `metricFrame`.',
      'Causal: regression adjustment, IPW, matching, AIPW, TMLE, DML, meta-learners, causal forests, instruments, RDD, synthetic control, difference-in-differences, sensitivity analysis.',
      'Survival: Kaplan-Meier, Nelson-Aalen, Cox (Breslow/Efron), AFT, log-rank family, Brier and dynamic AUC.',
      'Vision: OpenCV-shaped classical pipeline from filters through features, geometry, stereo and bundle adjustment.',
      'Text: tokenisers, stemmers, subword models, n-gram language models with smoothing, TF-IDF/BM25, Earley parsing and PCFG training, word2vec, fastText.',
    ],
  },

  {
    file: 'covariance',
    title: 'Covariance, precision and correlation estimation',
    libs: [
      'sklearn_covariance',
      'statsmodels_cov',
      'scipy_cov',
      'pyriemann',
      'skfolio',
      'pypfopt',
      'riskfolio',
      'skggm',
      'gglasso',
      'regain',
      'robpy',
      'rmt_denoise',
      'geomstats',
      'pymanopt',
      'arch_cov',
      'nilearn_connectome',
      'mne_cov',
      'r_covtools',
      'r_rrcov',
      'r_robustbase',
      'r_cellwise',
      'r_corpcor',
      'r_huge',
      'r_nlshrink',
      'r_poet',
      'r_rags2ridges',
      'r_bdgraph',
      'r_other_graphical',
      'pandas_cov',
      'river_cov',
      'js_peers',
      'pymc_priors',
      'timevarying_and_high_frequency',
      'sparse_and_structured',
      'graph_structure_methods',
      'covariance_testing',
      'random_matrix_theory',
      'missing_streaming_weighted',
      'spd_geometry_concepts',
    ],
    relisted: ['sklearn_covariance', 'statsmodels_cov', 'scipy_cov'],
    before: ['covariance-inventory.md'],
    closing: ['covariance-peers.md', 'covariance-add.md'],
    intro:
      'A survey of covariance, precision and correlation estimation and what sits next to it, added in a third pass because the first two ' +
      'saw no gap: `sklearn.covariance` is fully covered (all 16 names), and that is the only lens the first pass had on the subject. ' +
      'This file widens it to the libraries where the subject lives: scikit-learn and statsmodels again (a slice, re-listed), pyRiemann, ' +
      'geomstats and PyManopt (SPD matrices), skfolio, PyPortfolioOpt and Riskfolio-Lib (risk models), skggm, gglasso and regain ' +
      '(graphical models), robpy (robust scatter), rmt_denoise, arch (long-run covariance), nilearn and MNE (connectomes and noise ' +
      'covariance), the R packages that have no Python peer (the CRAN reference-manual indexes of CovTools, rrcov, robustbase, cellWise, ' +
      'corpcor, huge, nlshrink, POET, rags2ridges and BDgraph) and concept lists for the parts of the literature no package covers whole ' +
      '(sparse and structured estimators, graph-structure methods, covariance tests, random matrix theory, missing data and streaming, ' +
      'time-varying and high-frequency covariance, SPD geometry). Readings: sklearn, statsmodels and scipy were introspected from the ' +
      'installed packages; pyriemann, skfolio, PyPortfolioOpt, Riskfolio-Lib, skggm, gglasso, regain, robpy, rmt_denoise, geomstats, ' +
      'pymanopt, arch, nilearn and MNE were not installed and not imported: each was downloaded as a wheel (an sdist for skggm) into a ' +
      'scratch directory and read with `ast` by `enumerate.py` (the `OMAI_SURVEY_SRC` route), so the names are the release ones ' +
      '(`ref-*.json` records the version). The CRAN lists are the index pages of the reference manuals; the pandas, river, JavaScript, ' +
      'PyMC and the concept lists are from memory and are checklists, not inventories. `skggm` has no wheel and its sdist was read; ' +
      '`arch` has no multivariate GARCH, so only its kernel long-run covariance module is listed.',
    strong: [
      'The sklearn.covariance estimators: empirical, shrunk, Ledoit-Wolf, OAS, FastMCD (with the concentration steps traced) and the elliptic envelope, and the graphical lasso by coordinate descent with the primal-dual gap, the objective and the edge count on the tape and the path of partial correlations as an exposition.',
      'The pieces around them that a covariance estimator is built from: weighted `cov` and `corrcoef`, partial correlation, distance correlation and its test, a Kendall tau matrix, factor analysis, probabilistic PCA and sparse PCA (the low-rank-plus-diagonal covariance), the Wishart, inverse Wishart, LKJ and matrix-normal distributions with the inverse-Wishart conjugate update, Box M, Hotelling, MANOVA, Mardia and Little tests, `sqrtm`, `logm`, `expm`, `cholesky` with rank-one updates, Toeplitz and circulant matrices, shrinkage discriminant analysis and the Gaussian mixture covariance types.',
      "Mahalanobis in every place it is used (distances registry, k-NN, matching, elliptic envelope), covariance ellipses as data, a correlation chart in the renderer and a streaming regularised covariance inside the sampler's mass-matrix adaptation.",
      'Every iterative estimator already runs as a stepped algorithm; the survey finds nothing that is a one-shot function where a library would have an iteration.',
    ],
    after: [
      'Reading the counts: the re-listed slice (sklearn.covariance, a slice of statsmodels and of scipy) repeats names the first pass already counted and is left out of the ' +
        'family totals in the README. The remaining libraries add the names below. Concept lists (the last seven tables) are lists of methods, not of API names, ' +
        'so their absent counts say how much of the literature is not in omai, not how many functions are missing. ' +
        'Two first-pass verdicts were corrected here and in their own files: `GraphicalLassoCV` is partial (omai has no estimator that chooses the penalty by cross-validation, ' +
        'EBIC or stability, only the path as an exposition), and `statsmodels.stats.api.test_cov*` were counted partial by a group rule and are absent (the one-sample tests of a covariance matrix) ' +
        'except `test_cov_oneway`, which is Box M.',
    ],
  },
  {
    file: 'metric-learning',
    title: 'Metric and similarity learning',
    libs: ['pytorch_metric_learning', 'metric_learn', 'faiss'],
    intro:
      'pytorch-metric-learning (losses, miners, distances, reducers, regularizers, samplers, testers and the ' +
      'AccuracyCalculator), metric-learn (the scikit-learn-contrib learners of a Mahalanobis metric) and FAISS ' +
      '(the vector-index concepts: flat, IVF, product quantisation, HNSW, LSH). None is installed, so all three ' +
      'lists are from memory; every name was then looked up in the export list and the source of omai.',
    strong: [
      'Losses: contrastive, triplet margin, InfoNCE and NT-Xent, supervised contrastive, ArcFace and CosFace, VICReg and Barlow Twins, alignment and uniformity, CLIP, plus the learning-to-rank losses (RankNet, LambdaRank, ListNet, ApproxNDCG) that a retrieval page also needs.',
      'Learned metrics: NCA and LMNN as stepped algorithms with the solver as a slot, so the embedding at step t is on the tape (`fittedAt(t)`); metric-learn ships them as one-shot fits.',
      'Retrieval and indexing: exact, k-d tree and ball-tree search under any registered distance, HNSW, IVF and LSH indexes with the graph and the buckets as traced runs, precision@k, recall@k, hit rate, MRR, NDCG, MAP and R-precision, trustworthiness and continuity.',
      'omai ahead: a metric learner that can be watched converge, an HNSW build that can be replayed, and distances that are a registry rather than a flag.',
    ],
    fullTables: ['pytorch_metric_learning', 'metric_learn', 'faiss'],
    after: [
      'What is missing is concentrated in one place: everything that works on a labelled batch. pytorch-metric-learning is built around ' +
        '"a loss over all pairs or triplets of a batch, a miner that selects the informative ones, a sampler that makes sure the batch ' +
        'has positives". omai has the losses for explicit pairs and triplets and for in-batch positives, but no miners, no m-per-class sampler, ' +
        'and none of the multi-similarity, lifted-structure, circle, N-pairs or proxy-based losses. The catalogue plans mining ' +
        '(`representation-learning/triplet-and-margin`, "the mining as the mode") and the `learn/metric-learning` chapter, and neither is built. ' +
        'A single file of mining and a second of batch-label losses would close most of it; the proxy losses build on the class weight ' +
        'vectors that `arcFace` already has.',
      'On the metric-learn side, omai has the two learners that sklearn itself ships (NCA, LMNN). The pair-constraint learners ' +
        '(ITML above all, then LSML, SDML, MMC) and the local Fisher and kernel-regression variants (LFDA, MLKR) are absent, as is the data shape they ' +
        'need (pairs, triplets and quadruplets of rows, built from labels).',
    ],
  },
  {
    file: 'self-supervised',
    title: 'Self-supervised and representation learning',
    libs: ['lightly', 'solo_learn', 'sentence_transformers', 'representation_learning'],
    intro:
      'lightly and solo-learn (SimCLR, MoCo, BYOL, SimSiam, DINO, SwAV, Barlow Twins, VICReg, MAE and their heads, ' +
      'memory banks and augmentation recipes), sentence-transformers (bi-encoders trained with in-batch negatives, ' +
      'evaluators), and the autoencoder, contrastive-predictive and disentanglement vocabulary of the literature. ' +
      'All from memory; nothing is installed.',
    strong: [
      'Objectives: InfoNCE, NT-Xent, supervised contrastive, CLIP, Barlow Twins, VICReg, alignment and uniformity, distillation, Sinkhorn, vector quantisation with its EMA codebook, the ELBO with a beta weight, the importance-weighted bound and score-matching losses.',
      'Representation diagnostics: linear probes, CKA, effective rank, alignment and uniformity, sparse autoencoders for activations, neighbour purity and analogy tests for word vectors.',
      'The catalogue plans the trainers (`representation-learning`, `generative-models`); what exists is the loss side.',
    ],
    after: [
      'The gap is the trainers: omai has the losses a self-supervised method needs and none of the assembled methods. A SimCLR or BYOL ' +
        'page needs a two-view augmentation, a projection head, the loss, an EMA target for BYOL and a linear probe as the evaluation; ' +
        'with the stepped tape, "why BYOL does not collapse" (the embedding rank at each step) is a better demonstration than the paper has. ' +
        'The autoencoder family (plain, denoising, sparse, VAE, VQ-VAE) is the same: losses and the quantiser exist, the model and the ' +
        'latent-traversal view do not.',
    ],
  },
  {
    file: 'training-and-paradigms',
    title: 'Training loops and learning paradigms',
    libs: [
      'skorch',
      'pytorch_lightning',
      'ignite',
      'learn2learn',
      'higher',
      'avalanche',
      'river',
      'modal',
      'snorkel',
      'cleanlab',
      'small_text',
      'opacus',
      'flower',
    ],
    intro:
      'The libraries that structure training and the paradigms beyond one dataset and one loss: skorch, PyTorch Lightning and Ignite ' +
      '(callbacks, schedulers, accumulation, early stopping), learn2learn and higher (meta-learning), Avalanche (continual ' +
      'learning), river (online learning and drift), Opacus (differential privacy), Flower (federated learning), modAL and small-text ' +
      '(active learning), Snorkel (weak supervision) and cleanlab (noisy labels, data audits). Introspected: skorch, modAL, Opacus. The rest are from memory.',
    strong: [
      'Training loop: early stopping, best checkpoint, learning-rate monitors, plateau schedules, gradient accumulation and clipping, freezing, EMA and SWAG, all as pieces of one traced run (`nn/training`); the "callbacks" of Lightning and Ignite are the tape and its probes.',
      'Active learning: every query strategy of modAL and small-text (uncertainty, margin, entropy, query-by-committee, BALD, BatchBALD, expected error and model change, core-set, BADGE, density weighting, stream-based), with the loop as an algorithm.',
      'Privacy: DP-SGD with per-example clipping, RDP accounting, Gaussian, Laplace and exponential mechanisms, randomised response, membership, attribute and model-inversion attacks.',
      'Weak supervision and noisy labels: label models (majority, Dawid-Skene, a Snorkel-style model), noise transition estimation, confident learning (`findLabelIssues`), co-teaching, forward and backward correction, PU learning.',
      'Online learning: Hoeffding trees, the passive-aggressive and perceptron learners, online and kernel regressors, FTRL, bandits, drift detectors (ADWIN, KSWIN, Page-Hinkley, DDM, EDDM), streaming metrics, stream generators with concept drift.',
    ],
    after: [
      'Three paradigms with a catalogue chapter have no code at all: meta-learning (MAML, Reptile, prototypical networks and an episode sampler), ' +
        'continual learning (EWC, replay, LwF, a stream of tasks and the forgetting matrix) and federated learning (FedAvg, FedProx). Together with ' +
        'prequential evaluation, they are the clearest gaps of this family. All three are inner-loop-over-trainers programs, and `autodiff`, ' +
        '`rl#replayBuffer` and `optim#empiricalFisher` are what they would be built from.',
      '`higher` is marked out of scope: its job (differentiating through an inner optimisation loop) is what `grad` of a function of functional ' +
        'optimiser steps already does.',
    ],
  },
  {
    file: 'bayesian-optimisation-and-gp',
    title: 'Bayesian optimisation, Gaussian processes, Pyro and BlackJAX',
    libs: ['botorch', 'gpytorch', 'gpflow', 'skopt', 'pyro', 'blackjax'],
    intro:
      'BoTorch, scikit-optimize and the GP stacks GPyTorch and GPflow (acquisition functions, multi-output, variational and sparse GPs, ' +
      'likelihoods, kernels), Pyro (guides, ELBO variants, the effect-handler vocabulary, contrib GPs and experimental design) and ' +
      'BlackJAX (the sampler zoo). Introspected: GPyTorch and BlackJAX. BoTorch, GPflow, scikit-optimize and Pyro are from memory.',
    strong: [
      'Gaussian processes: exact regression with any kernel and the marginal likelihood, sparse GPs with SoR, DTC, FITC and VFE, Laplace and EP classification, relevance vector machines, kernels including ARD, spectral mixture, periodic and compositions.',
      'Bayesian optimisation: EI, log-EI, PI, UCB with a GP surrogate, TPE, CMA-ES, quasi-random samplers, Hyperband, successive halving, pruners, importance by fANOVA.',
      'Pyro vocabulary: SVI, Trace, mean-field and Renyi ELBOs, auto-guides (normal, delta, multivariate, low-rank, Laplace, IAF), the handlers (trace, replay, block, condition, do, mask, scale, seed, plate, markov, enumerate), `infer_discrete`, SMC and SVGD.',
      'Samplers: HMC, NUTS, Riemannian HMC, MALA, SGLD, slice, elliptical slice, SMC, tempered SMC, annealed importance sampling, with the whole chain on the tape.',
    ],
    after: [
      'The batch and information-theoretic acquisition functions of BoTorch, multi-output and variational GPs, and a few Pyro features ' +
        '(reparametrisation handlers, enumerated ELBO, optimal experimental design) are the visible gaps. Nested sampling and the stochastic-gradient ' +
        'HMC family (SGHMC, SGNHT) are the BlackJAX samplers a Bayesian course would meet that omai lacks.',
    ],
  },
  {
    file: 'tabular-and-interpretable',
    title: 'Tabular, interpretable and feature libraries; diagnostic charts',
    libs: [
      'interpret',
      'imodels',
      'category_encoders',
      'feature_engine',
      'mlxtend',
      'skmultilearn',
      'sklearn_extra',
      'umap',
      'hdbscan',
      'trimap',
      'pacmap',
      'yellowbrick',
      'scikitplot',
      'lime',
      'xgboost_lightgbm_catboost',
    ],
    intro:
      'interpret and imodels (explainable boosting, rule lists, optimal trees), category_encoders and feature-engine (encoders and feature ' +
      'creation), mlxtend, scikit-multilearn, scikit-learn-extra, umap-learn, hdbscan, TriMap and PaCMAP, yellowbrick and scikit-plot ' +
      '(diagnostic charts), LIME, and the feature set of XGBoost, LightGBM and CatBoost. Introspected: LIME. Everything else is from memory.',
    strong: [
      'Glass-box models: explainable boosting machines (cyclic boosting with pairwise terms), rule lists (CORELS, Bayesian rule lists, CN2, PRISM, RIPPER, OneR), RuleFit, decision tables, and `rules` as a folder of its own.',
      'Dimension reduction and clustering: t-SNE, UMAP, TriMap, PaCMAP, HDBSCAN with the condensed tree, OPTICS reachability, k-medoids and CLARA.',
      'Diagnostics: ROC, PR, calibration, confusion matrix, learning curve, residuals, decision regions, scree and biplot, importance, partial dependence and ICE, regularisation paths, threshold sweeps, reliability and conformal views.',
      'Model comparison: McNemar, Cochran Q, the 5x2 cv F test, bootstrap .632, bias-variance decomposition; stacking and voting; sequential feature selection, RFE, mRMR, stability selection, knockoffs.',
    ],
    after: [
      'The recurring gaps are small: categorical encoders (WoE, CatBoost-style ordered statistics, Helmert and sum contrasts), feature filters (constant, duplicate and ' +
        'correlated columns, probe features), calendar features, monotone constraints and DART or GOSS in the boosters, FIGS and hierarchical shrinkage for interpretable trees, ' +
        'multi-label learners (ML-kNN, RAkEL) and a handful of charts (silhouette, jointplot, cumulative gain and lift).',
    ],
  },
  {
    file: 'torch-ecosystem',
    title: 'The torch ecosystem beyond torch: architectures, audio, text, augmentation and inspection',
    libs: [
      'torchvision_models',
      'timm',
      'torchaudio',
      'librosa',
      'torchtext',
      'kornia',
      'albumentations',
      'torchinfo_fx_onnx',
    ],
    intro:
      'torchvision.models and timm (the architecture families a learner expects), torchaudio and librosa (audio transforms and features), ' +
      'torchtext (vocabularies and tokenisers), kornia and albumentations (augmentation and classical vision), and the inspection side of the ecosystem ' +
      '(torchinfo, torch.fx, ONNX). Introspected: torchvision.models and librosa. The others are from memory. Pretrained weights, detection and video model zoos, ' +
      'GPU kernels and serving are out of scope on purpose, and are marked so.',
    strong: [
      'Blocks: ResNet basic and bottleneck, squeeze-and-excitation, ConvNeXt, MBConv, U-Net stages, feature pyramids, transformer blocks and patch embedding, mixers, state-space layers, Mamba, RetNet, RWKV, KAN, neural operators.',
      'Audio: spectrograms, mel and MFCC, chroma, pitch (YIN, pYIN), onset and beat tracking, tempograms, SpecAugment, time stretching and pitch shifting, gammatone and ERB scales, masking and source separation baselines; `signal` at scipy.signal width.',
      'Vision (classical, OpenCV-shaped): filters, morphology, features (Harris, FAST, ORB, BRIEF, SIFT), matching, RANSAC, homography and essential matrices, stereo, calibration, bundle adjustment, optical flow, segmentation (SLIC, GrabCut, mean shift), tracking.',
      'Text: tokenisers (regex, treebank, tweet, BPE, WordPiece, unigram), vocabularies, n-gram models, TF-IDF, BM25, word2vec, GloVe, fastText, parsers.',
      'Inspection: `describe`, `summary`, layer tables, FLOP and memory estimates, receptive fields, and a scene layer that draws structures.',
    ],
    after: [
      'omai has the parts of the architectures and none of the assembled, named networks. A learner who wants "ResNet-18 at a small scale, trained on 28x28 digits and ' +
        'then described" has to wire it by hand; VGG-style plain stacks (the baseline for residual connections), a vision transformer and a small U-Net as builders with depth and width ' +
        'as knobs would close that. The augmentation side is the same: crop, flip, cutout, mixup and cutmix exist, the colour and blur modifiers, the policies (RandAugment) and the ' +
        'label-carrying geometric ones (boxes, keypoints, masks) do not.',
    ],
  },
  {
    file: 'structured-prediction',
    title: 'Structured prediction and sequence labelling (CRFs and their neighbours)',
    libs: ['crf_libraries', 'crf_tools'],
    intro:
      'sklearn-crfsuite, python-crfsuite (CRFsuite) and pytorch-crf (introspected from the installed packages), and from memory CRF++, AllenNLP and flair ' +
      '(BiLSTM-CRF taggers), pystruct (structured SVMs and the structured perceptron), seqeval (entity-level scores) and the structured-prediction ' +
      'literature (n-best and constrained decoding, semi-Markov, skip-chain and higher-order CRFs, latent and partial-label CRFs, CRF layers on neural encoders, ' +
      'CRF-as-RNN, MEMMs, parsers). The source of omai was read for what each item corresponds to.',
    strong: [
      'Linear-chain CRFs in both the CRFsuite and CRF++ forms: score, log-likelihood, marginals, Viterbi, posterior decoding, feature counts and expected counts, template-driven features from all three template languages (CRF++, Wapiti, CRFsuite) through one extractor, L1 and L2 penalties.',
      'Training through one solver slot over a small contract (`StructuredOracle`): L-BFGS, every generic smooth solver, OWL-QN for L1, stochastic gradient, the perceptron, averaged perceptron, passive-aggressive and AROW, and the max-margin learners (subgradient, Frank-Wolfe, one-slack and n-slack cutting planes), on the same tape, which neither CRFsuite nor pystruct offer side by side.',
      'CRFs beyond the chain, all in `pgm/structured`: trees, grids, skip chains, order-k chains and arbitrary graphs with tied weights, semi-Markov CRFs, latent and partial-label CRFs, the MEMM as the label-bias foil, with exact inference where the graph allows it and loopy or tree-reweighted BP where not; MAP by junction trees, MPLP, graph cuts, alpha-expansion and AD3.',
      'Constrained, n-best and sampled decoding; the CRF layer over a neural encoder (checked against pytorch-crf) and the BiLSTM-CRF tagger; seqeval-style entity metrics (BIO, IOB1, BIOES, BILOU; strict and lenient; micro, macro and per type), scheme-violation counts and `sequenceAccuracyScore`; CKY, Earley, PCFG training, arc-standard and Eisner dependency parsing.',
    ],
    fullTables: ['crf_libraries'],
    after: [
      'As of 0.1.12 the ring around the CRF is built (decoding, neural layer and tagger, structured SVMs, MEMM, semi-Markov, latent and partial-label CRFs, graph-cut and AD3 MAP) and ' +
        'everything lives in `pgm/structured` and `pgm/inference`. What is still missing: dense CRFs with Gaussian-kernel potentials and mean-field layers (CRF-as-RNN), ' +
        'latent and partial-label models on graphs (they exist for chains), a latent SSVM, user-defined segment features for the semi-Markov CRF, a biaffine dependency parser, ' +
        'a pretrained-embedding or character encoder for the tagger, and full QPBO (probing and improvement).',
    ],
  },
  {
    file: 'adjacent',
    title:
      'Adjacent libraries: offline RL, control, change points, optimisation, geometry, algebra, causal discovery, uncertainty, forecasting, recommenders, data checks, games',
    libs: [
      'd3rlpy',
      'imitation',
      'control',
      'filterpy',
      'pykalman',
      'ruptures',
      'stumpy',
      'pysindy',
      'cma',
      'pymoo',
      'nevergrad',
      'optax',
      'cvxpy',
      'pot',
      'shapely',
      'trimesh',
      'sympy',
      'problog',
      'causallearn',
      'gcastle',
      'mapie',
      'uncertainty_toolbox',
      'neuralforecast',
      'darts',
      'gluonts',
      'surprise',
      'implicit',
      'lightfm',
      'recbole',
      'deepchecks',
      'evidently',
      'nashpy',
      'openspiel',
      'datasketch',
      'pysubgroup',
      'pingouin',
      'pyitlib',
      'npeet',
      'sdeint',
      'numpy',
    ],
    areaLevel: ['sympy', 'pot'],
    intro:
      'Libraries that sit next to the other domains of omai and were not in the first pass. Introspected from the installed environment: ' +
      'control, filterpy, ruptures, stumpy, pysindy, cma, pymoo, cvxpy, POT, optax, shapely, SymPy, causal-learn, MAPIE, nashpy, OpenSpiel (game names and ' +
      'algorithm modules), datasketch, pysubgroup, pingouin, pyitlib, npeet, sdeint, numpy (the array namespace, by group). From memory: d3rlpy, imitation, pykalman, nevergrad, trimesh, problog (found to be covered), gcastle, uncertainty-toolbox, ' +
      'neuralforecast, darts, GluonTS, surprise, implicit, LightFM, RecBole, deepchecks, evidently. SymPy and POT have hundreds of names and were judged by ' +
      'group, so they are in the summary and the tables but left out of the headline totals.',
    strong: [
      'Control and filtering: the LTI toolbox of python-control to the last function, LQR, LQG, pole placement, H-infinity, MPC, iLQR, PID tuning, and Kalman, extended, unscented, ensemble and particle filters with smoothers.',
      'stumpy to the last function (matrix profile in six algorithms, motifs, discords, chains, snippets, FLUSS and FLOSS, MPdist) as stepped algorithms.',
      'Game theory: OpenSpiel-style CFR, CFR+ and Monte Carlo CFR, sequence-form LPs, exploitability, support and vertex enumeration, Lemke-Howson, fictitious play, replicator dynamics, regret matching, coalition games, auctions and voting.',
      'Conformal prediction in the MAPIE style: split, cross, jackknife+, CV+, CQR, Mondrian, adaptive, weighted, APS and RAPS.',
      "Optimisation: optax at its full width (283 of 306 names), POT's exact, entropic, unbalanced and Gromov-Wasserstein solvers, NSGA-II, CMA-ES, differential evolution and swarm methods.",
      'Recommenders: implicit-feedback ALS, matrix completion, factorisation machines, DeepFM, DLRM, two-tower, BPR and ranking losses, click models, MovieLens.',
    ],
    after: [
      'The recurring gaps in this group are the evaluation and reporting layers (a data-audit report like deepchecks and Evidently, conformal risk control and ' +
        'exchangeability tests from MAPIE), forecasting as a pipeline (windows, transformers, backtests, baselines), neighbourhood and graph collaborative filtering, ' +
        'the classic board-game shelf (chess, Othello, checkers, Hex), multi-objective evolutionary algorithms beyond NSGA-II with the ZDT and DTLZ problems, and ' +
        'optimal-transport domain adaptation.',
    ],
  },
]

const PRI: Record<string, number> = { high: 3, medium: 2, low: 1 }
const SIZE: Record<string, number> = { L: 3, M: 2, S: 1 }

function cell(s: string | undefined): string {
  return (s ?? '').replace(/\|/g, '/').replace(/\s+/g, ' ').trim()
}

function shortArea(lib: string, area: string): string {
  return area.startsWith(`${lib}.`) ? area.slice(lib.length + 1) : area
}

function names(rows: Row[], max = 12): string {
  const n = rows.map((r) => r.name)
  const shown = n.slice(0, max).map((x) => `\`${x}\``)
  return n.length > max ? `${shown.join(', ')} and ${n.length - max} more` : shown.join(', ')
}

function main(): void {
  const rows: Row[] = readdirSync(DATA)
    .filter((f) => f.startsWith('match-') && f.endsWith('.json'))
    .sort()
    .flatMap((f) => JSON.parse(readFileSync(join(DATA, f), 'utf8')) as Row[])
  const versions: Record<string, { version: string; source: string }> = {}
  for (const r of rows) versions[r.lib] ??= { version: '', source: '' }
  const refs = (lib: string): { version: string; source: string } => {
    try {
      const d = JSON.parse(readFileSync(join(DATA, `ref-${lib}.json`), 'utf8')) as { version: string }
      return { version: d.version, source: 'introspection' }
    } catch {
      return { version: '', source: 'memory' }
    }
  }
  const counts: Record<string, unknown> = {}
  for (const fam of FAMILIES) {
    const out: string[] = []
    const famRows = rows.filter((r) => fam.libs.includes(r.lib))
    const tot = { full: 0, partial: 0, absent: 0, skip: 0, notLow: 0 }
    const excluded = [...(fam.areaLevel ?? []), ...(fam.relisted ?? [])]
    const rel = { full: 0, partial: 0, absent: 0, skip: 0 }
    for (const r of famRows.filter((x) => (fam.relisted ?? []).includes(x.lib))) rel[r.status]++
    for (const r of famRows.filter((x) => !excluded.includes(x.lib))) {
      tot[r.status]++
      if ((r.status === 'absent' || r.status === 'partial') && (r.priority === 'high' || r.priority === 'medium'))
        tot.notLow++
    }
    counts[fam.file] = { ...tot, ...(fam.relisted ? { relisted: rel } : {}), libs: {} as Record<string, unknown> }
    out.push(`# ${fam.title}`, '')
    out.push(`Part of the [completeness survey](README.md). ${fam.intro}`, '')
    for (const f of fam.before ?? []) out.push(readFileSync(join(HERE, 'prose', f), 'utf8').trimEnd(), '')
    out.push(
      'Counts are of names: "covered" means an omai export or a documented configuration does what the reference name does, ' +
        '"partial" means a different shape or fewer options, "absent" means nothing found by name or by concept, and ' +
        '"out of scope" is plumbing, GPU or production features, downloads and the like that omai should not copy. ' +
        'Semantics were not compared.',
      '',
    )
    out.push('## Summary', '')
    out.push('| library | source | area | covered | partial | absent | out of scope |')
    out.push('| --- | --- | --- | ---: | ---: | ---: | ---: |')
    for (const lib of fam.libs) {
      const lr = famRows.filter((r) => r.lib === lib)
      if (lr.length === 0) continue
      const v = refs(lib)
      const areas = [...new Set(lr.map((r) => r.area))].sort()
      const lt = { full: 0, partial: 0, absent: 0, skip: 0, notLow: 0 }
      for (const a of areas) {
        const c = { full: 0, partial: 0, absent: 0, skip: 0 }
        for (const r of lr.filter((x) => x.area === a)) {
          c[r.status]++
          if ((r.status === 'absent' || r.status === 'partial') && (r.priority === 'high' || r.priority === 'medium'))
            lt.notLow++
        }
        for (const k of ['full', 'partial', 'absent', 'skip'] as const) lt[k] += c[k]
        out.push(`|  |  | ${shortArea(lib, a)} | ${c.full} | ${c.partial} | ${c.absent} | ${c.skip} |`)
      }
      out.push(
        `| **${lib}** | ${v.source}${v.version ? ` ${v.version}` : ''} | **all** | **${lt.full}** | **${lt.partial}** | **${lt.absent}** | **${lt.skip}** |`,
      )
      ;(counts[fam.file] as { libs: Record<string, unknown> }).libs[lib] = { ...lt, ...v }
    }
    out.push('')
    out.push('## Where omai is already strong', '')
    for (const s of fam.strong) out.push(`- ${s}`)
    out.push('')
    for (const a of fam.after ?? []) out.push(a, '')
    if (fam.relisted?.length) {
      out.push(
        `${fam.relisted.map((l) => `\`${l}\``).join(', ')} repeat${fam.relisted.length > 1 ? '' : 's'} names the first pass already counted (in sklearn.md, scipy-statsmodels.md); ` +
          'their rows are in the tables but not in the totals of `counts.json` (they are under `relisted`) and the README.',
        '',
      )
    }
    if (fam.areaLevel?.length) {
      out.push(
        `${fam.areaLevel.map((l) => `\`${l}\``).join(' and ')} ${fam.areaLevel.length > 1 ? 'were' : 'was'} judged by area, not by name; ` +
          'their rows are in the tables but not in the totals of `counts.json` and the README.',
        '',
      )
    }
    out.push('## Absent and partial items', '')
    out.push(
      'Priority is high, medium or low by how many learners would expect the thing, how many pages it unlocks and how well ' +
        'it shows off a stepped, traced run; not by recency. Sizes: S a day or less, M a few days, L a week or more. ' +
        'Homes follow `docs/structure.md`. "Nearest in omai" is what a user would reach for today. Items are grouped; ' +
        'the full list of names per group is in `data/match-<lib>.json`.',
      '',
    )
    for (const lib of fam.libs) {
      const lr = famRows.filter((r) => r.lib === lib && (r.status === 'absent' || r.status === 'partial'))
      if (lr.length === 0) continue
      out.push(`### ${lib}`, '')
      const areas = [...new Set(lr.map((r) => r.area))].sort()
      // group rows: by explicit group, else by (area, status, note)
      const groups = new Map<string, Row[]>()
      for (const r of lr) {
        const key = r.group ? `g|${r.group}|${r.status}` : `a|${r.area}|${r.status}|${r.note ?? ''}|${r.omai ?? ''}`
        groups.set(key, [...(groups.get(key) ?? []), r])
      }
      const list = [...groups.values()].sort((a, b) => {
        const pa = Math.max(...a.map((r) => PRI[r.priority ?? 'low'] ?? 1))
        const pb = Math.max(...b.map((r) => PRI[r.priority ?? 'low'] ?? 1))
        return pb - pa || b.length - a.length
      })
      void areas
      out.push(
        '| reference (area) | what it is and why it matters | nearest in omai | proposed home | size | priority | status |',
      )
      out.push('| --- | --- | --- | --- | --- | --- | --- |')
      for (const g of list) {
        const first = g[0]
        if (!first) continue
        const byPriority = [...g].sort((a, b) => (PRI[b.priority ?? 'low'] ?? 1) - (PRI[a.priority ?? 'low'] ?? 1))
        const withNote = byPriority.find((r) => r.note) ?? first
        const pri = g.reduce(
          (m, r) => ((PRI[r.priority ?? 'low'] ?? 1) > (PRI[m] ?? 1) ? (r.priority ?? 'low') : m),
          'low',
        )
        const size = g.reduce((m, r) => ((SIZE[r.size ?? ''] ?? 0) > (SIZE[m] ?? 0) ? (r.size ?? '') : m), '')
        const home = g.find((r) => r.home)?.home
        const areaSet = [...new Set(g.map((r) => shortArea(lib, r.area)))]
        const label = first.group ? `**${first.group}**: ${names(g)}` : `${names(g)}`
        const what =
          cell(withNote.note) ||
          (first.omai ? 'only the nearest thing in the next column' : 'no counterpart found by name or concept')
        const status = [...new Set(g.map((r) => r.status))].join('/')
        out.push(
          `| ${cell(label)} (${cell(areaSet.slice(0, 2).join(', '))}${areaSet.length > 2 ? ', ...' : ''}) | ${what} | ${cell(first.omai) || 'nothing close'} | ${cell(home) || 'to be placed'} | ${size || '-'} | ${pri} | ${status} |`,
        )
      }
      out.push('')
    }
    for (const lib of fam.fullTables ?? []) {
      const lr = famRows.filter((r) => r.lib === lib)
      out.push(`## Every name checked: ${lib}`, '')
      out.push(
        'Each name of the reference list with the verdict and the nearest thing in omai (or what is missing). "covered" means the same method under omai\'s name.',
        '',
      )
      out.push('| reference | area | verdict | nearest in omai, or what it is |')
      out.push('| --- | --- | --- | --- |')
      for (const r of lr) {
        const status = r.status === 'skip' ? 'out of scope' : r.status === 'full' ? 'covered' : r.status
        out.push(
          `| \`${r.name}\` | ${shortArea(lib, r.area)} | ${status} | ${cell(r.omai) ? `${cell(r.omai)}${r.note ? `: ${cell(r.note)}` : ''}` : cell(r.note) || '-'} |`,
        )
      }
      out.push('')
    }
    for (const f of fam.closing ?? []) out.push(readFileSync(join(HERE, 'prose', f), 'utf8').trimEnd(), '')
    writeFileSync(join(DIR, `${fam.file}.md`), `${out.join('\n')}\n`)
  }
  writeFileSync(join(DATA, 'counts.json'), `${JSON.stringify(counts, null, 1)}\n`)
  console.log(JSON.stringify(counts, null, 0))
}

main()
