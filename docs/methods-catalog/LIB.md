# The library audit: what the methods ask of `lib/`, and what belongs there

A working document, 2026-10-28, composed from the 52 chapter READMEs (1,284 method rows, 457 pieces marked TODO).
It answers three questions: where the line between `lib/` and `methods/shared/` runs; what the library lacks, grouped
and ordered by how much waits on it; and what the migration of the pieces built this week looks like. The render
side is `RENDER.md`.

## 1. The line: `lib/` against `methods/shared/`

The test for a piece of shared code is: **would it make sense to someone who never reads a tape?** If a function is an
algorithm, a data structure, a registry of library things, a loop that runs other things (a trainer, a sweep, a
stream), a table binding library functions to knobs, or a generator with a truth, the answer is yes and it is library
code: it gets the full standard (TSDoc with checked examples, oracle goldens where a reference exists, property tests),
it is importable by anyone, and it is tested once. A piece belongs in `methods/shared/` only when it is about methods
as explanation objects and about nothing else: something that composes several *methods* (not several library
functions) and would be meaningless without the method layer. Looking for such a thing across the fifty-two chapters
finds none. Everything the READMEs put in a chapter's `shared/` or in `methods/shared/` has a library home:

| what the READMEs put in a `shared/`                                          | its library home                                                                                                                              |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| planted problems and generators with a truth (bags, classics, sources, pairs, superposition, cluster, corpora, two-view, mixed sources, point clouds) | `lib/datasets/` (generators register with tags, as every dataset does) |
| loops: the trainer, the sweep and repeated-fit base, the race, the stream, the online loop, the two-stream loop, the replay buffer, reflow | `lib/learn/bases/` (beside `iterative.ts`) and `lib/data/streams.ts`, `lib/rl/buffers.ts` |
| measures: alignment and uniformity, validity indices, decision thresholds, risk-coverage, baselines                 | `lib/metrics/`                                                                                        |
| slices of a loss landscape, the LR range test                                 | `lib/optim/` (`landscape.ts`; the range test is a sweep over `schedules`)                                                                    |
| augmentations and miners                                                     | `lib/data/modifiers/`, `lib/losses/`                                                                                                         |
| small programs (LDA, probabilistic PCA) and small models                     | `lib/probprog/models/`, `lib/pgm/models/`                                                                                                     |
| the option sets built this week: `modes.ts`, `options/{table,solvers,objectives,schedules}.ts` | `lib/trace/modes.ts` (a table of algorithm factories over a `union`; trace already depends on space), `lib/space/tables.ts` (`optionTable`), `lib/optim/solvers.ts`, `lib/losses/tables.ts`, `lib/optim/schedule-table.ts` |
| `_scene.ts`, `_picture.ts` of the vision methods                             | `lib/datasets/` (the scenes) and `lib/exposition/` (a picture is a surface record)                                                           |

So the recommendation is to **dissolve `methods/shared/`**: move its five files to the homes above (a `git mv`, the
imports, the barrels), and let the layering rule stay as it is (methods import the core, never the other way).
What remains under `methods/` is then exactly what section 1 of the design says a method is: a few dozen lines naming
slots, a contract and probes, and nothing an outside project could want without the tape. The registries already
went this way (the agents put them in `lib/base`, `lib/datasets`, `lib/environments`), and `lib/distances` did too.
The one cost is that `lib/trace` gains a dependency on `lib/space` for `modes.ts`; it has that dependency already
(`session.ts` reads a `DictSpace`).

Two rules follow for the library, both already true of the best of it and worth making explicit in the conventions:

- **Every iterative routine ships as an `Algorithm`**, with the whole-process function as its runner. Many TODOs in
  the tables are not missing capability but a missing stepped form of something that exists: Lanczos and Arnoldi,
  one-sided Jacobi SVD, CKY and Earley, STL, the iterative imputer, recursive feature elimination, stability
  selection, permutation importance, data Shapley, Dawid-Skene (it is `pgm/learning/parameters.ts: emAlgorithm` on a
  latent-class net), the Hungarian and auction algorithms' inner loops, bundle adjustment's Levenberg-Marquardt
  iterations, calibration's iterations, Huffman's merges, arithmetic coding's narrowing, the perceptron. A routine
  that is already an `Algorithm` inside a wrapper only needs exporting.
- **Every generator carries its truth and registers with tags.** The data slot of a method compares with the truth;
  the chapters ask for about thirty new planted problems (leaks, duplicates, shifts, panels with adoption, discontinuity
  designs, transactions with patterns, tagged corpora, captioned scenes, in-context regression tasks, coupled series,
  a spoken-digit set). They are datasets, not helpers.

## 2. What the library lacks, by weight

The weight of a piece is the number of method rows waiting on it, with the rows' own importance read from the
"why" cells. Sizes are rough: small is a file of a few functions, medium a folder or a few hundred lines, large a
domain with an oracle group.

### 2.1 The bases: four pieces that most rows wait on

| piece                                     | home                                 | waits on it                                                                                                      | size   |
| ----------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ------ |
| the trainer base (modules, optimiser, batches, evaluation at a schedule, parameter checkpoints for probes) over `nn/training/fit.ts: fitAlgorithm` | `lib/learn/bases/trainer.ts` | about 70% of the rows of the eight neural chapters, representation learning, the neural rows of transfer, meta, multi-task, multimodal, time series, recommendation | medium |
| the `gradient` mode of the smooth solvers: an `optim/transform.ts` chain with batches and a `schedules` entry, as section 7.1 of the design describes | `lib/optim/solvers.ts` | optimisation's race and schedule rows, loss-functions, every linear model's stochastic modes, the online learners | small |
| the sweep and repeated-fit base: a grid or sampler over one knob of an inner method, with the inner tapes kept or summarised; the race as a sweep over a slot | `lib/learn/bases/sweep.ts` | model-selection (cross-validation, learning curves, cost studies), learning-theory and statistics (repeated samples), optimisation (races), loss-functions, features (selection studies), active learning (budget curves), data-centric (removal curves): written as four stopgaps in the READMEs | medium |
| the stream base: data arriving one point or batch per step with planted drift and recurrence          | `lib/data/streams.ts`                | online learning, shift and drift, time series, online experimentation, signal processing's detection, anomaly's series rows | small  |

The agent base (an environment, an agent, a schedule, the standard probes) is the fifth; `lib/rl` has the rollouts
and `lib/agents` the learners, so it is a composition of existing pieces in `lib/learn/bases/agent.ts`.

### 2.2 New domains

Ordered by what they unblock. Each is a folder with an index summary, the full standard and an oracle case module
where a reference library exists.

| domain               | holds                                                                                                                                              | unblocks                                                                                                                                 | reference for fixtures            |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `lib/trees`          | the tree structure, prediction, decision paths and regions (moved from `explain/shapley.ts` and `explain/surrogates.ts: surrogateTree`); criteria (Gini, entropy, log loss, squared, absolute, Poisson); the grower as an `Algorithm` with candidate splits recorded; classic growers (ID3, C4.5, CHAID); pruning (weakest link, reduced error); the Hoeffding tree | supervised learning (7 rows), anomaly (isolation forests are tree-shaped but need no criterion), rules (RuleFit), causal forests, explain (tree SHAP, surrogates), model selection | scikit-learn `tree`, river        |
| `lib/ensembles`      | bagging with out-of-bag bookkeeping, forests and extra trees, random-trees embedding, the generic AdaBoost (the Haar instance of `vision/haar.ts` becomes a special case; SAMME.R, R2), gradient boosting with its losses, histogram boosting (binning, leaf-wise growth, missing values), stacking and voting; the cyclic booster of `explain/glassbox.ts` moves here | supervised learning, anomaly, trustworthy (surrogates, importances), learning to rank (LambdaMART) | scikit-learn `ensemble`, LightGBM |
| `lib/svm`            | the classifier and regressor over the duals of `optim/smo.ts` (the kernel expansion as the decision function, ν forms, one-versus-one), the one-class and SVDD duals | supervised learning, anomaly, semi-supervised (S3VM)                                                                                    | scikit-learn `svm`                |
| `lib/gp`             | regression over any kernel with the log marginal likelihood and its gradient, restarts, posterior samples; classification by Laplace and EP; sparse approximations; multi-output; the relevance vector machine | supervised learning, Bayesian optimisation (`tune/surrogate.ts` has a special case), scientific (multi-fidelity), decision theory (uncertainty) | scikit-learn `gaussian_process`, GPy |
| `lib/timeseries`     | ARMA and ARIMA (innovations, conditional sum of squares, exact likelihood through `dynamics/filtering/kalman.ts`), SARIMA, VAR with impulse responses and Granger tests, cointegration, volatility (ARCH, GARCH), exponential smoothing and ETS as one-observation-per-step algorithms, classical and STL decomposition, the matrix profile family (STOMP, STUMP, SCRIMP, STUMPI, motifs, discords, chains, FLUSS, MPdist, snippets, pan and multidimensional profiles) over the kernels of `signal/sliding.ts` and `signal/mass.ts`, online detectors (CUSUM, Page-Hinkley), shapelets, feature sets, hierarchical reconciliation | time series (16 of its 48 rows), anomaly's series rows, signal processing's series section, shift and drift's monitoring | statsmodels, stumpy, sktime       |
| `lib/anomaly`        | isolation forests (plain and extended), local and connectivity outlier factors, score normalisation and combination, the contamination-to-threshold rule | anomaly detection                                                                                                                        | PyOD, scikit-learn                |
| `lib/decomposition`  | PCA as a transformer (incremental, sparse, whitening, inverse), FastICA, NMF (multiplicative, HALS), CCA and PLS (NIPALS), random projections' assembly | unsupervised learning, representation evaluation, signal separation, recommendation (completion)                                       | scikit-learn `decomposition`      |
| `lib/manifold`       | classical and metric MDS (SMACOF), Isomap, LLE, t-SNE (affinities, gradient, Barnes-Hut), UMAP, PaCMAP, self-organising maps, embedding quality (trustworthiness, continuity, co-ranking) | unsupervised learning, representation learning's views                                                                                   | scikit-learn `manifold`, umap-learn |
| `lib/covariance`     | shrinkage, fast MCD, the graphical lasso                                                                                                           | unsupervised learning, anomaly (Mahalanobis), statistics                                                                                 | scikit-learn `covariance`         |
| `lib/survival`       | Kaplan-Meier, Nelson-Aalen and Greenwood as one-event-per-step algorithms, the log-rank test, Cox partial likelihood with ties and the baseline hazard, accelerated-failure-time families, concordance and time-dependent Brier and AUC | statistics and supervised learning (both ask; one home)                                                                                  | lifelines                         |
| `lib/gam`            | smooth terms and the penalised design, backfitting and penalised IRLS as algorithms, REML and UBRE selection, effective degrees of freedom per term, GAMLSS | supervised learning                                                                                                                      | pyGAM, mgcv (by fixtures)         |
| `lib/kernel-methods` | kernel ridge, online kernel recursive least squares                                                                                                | supervised, online learning                                                                                                              | scikit-learn                      |
| `lib/rules`          | sequential covering (CN2, PRISM), RIPPER and IREP, rule lists (Bayesian, CORELS), rule sets (OneR, decision tables), all on the `learn` contract | logic and rule learning, trustworthy (explanation by rules)                                                                               | Orange, imodels                   |
| `lib/patterns`       | Apriori, FP-growth, Eclat, association rules and their measures, PrefixSpan and GSP, subgroup discovery with WRAcc                                 | exploration and discovery                                                                                                                | mlxtend, pysubgroup               |
| `lib/causal`         | estimation (regression adjustment, IPW, stratification, matching, AIPW, double machine learning as a traced algorithm), quasi-experimental designs (difference in differences, synthetic control, regression discontinuity, instruments), heterogeneous effects (S, T, X, R and DR learners; causal forests once trees exist), sensitivity (omitted-variable bias, E-values, Rosenbaum bounds, refutation); uplift metrics in `metrics/` | causal inference (11 of 14 rows), trustworthy, recommendation (debiasing)                                                                | DoWhy, EconML                     |
| `lib/sketches`       | Bloom, counting Bloom and cuckoo filters, count-min, HyperLogLog, MinHash and LSH, reservoir sampling, quantile sketches, top-k, skip lists          | data structures the user asked for (section 7.10 of the design); transfer learning's replay wants reservoir sampling                   | datasketch, exact counts          |
| `lib/embedding`      | word2vec (the pair sampler, subsampling, the loop as an algorithm), GloVe, knowledge-graph scorers with corruption sampling and filtered ranking    | representation learning, NLP, recommendation                                                                                              | gensim (by fixtures)              |

### 2.3 Growth of existing domains

The additions the tables name, by domain, with the chapters that wait; the oracle group is the domain's own.

- **`learn`** (the largest list): neighbours (the classifier and regressor over the spatial indexes, nearest centroid with shrinkage), discriminant analysis (pooled and per-class, kernel Fisher), naive Bayes (five families, class-conditional trees, AODE), multiclass reductions, multilabel and multi-output wrappers, metric learning (NCA, LMNN), semi-supervised (self-, co- and tri-training, label propagation and spreading, Laplacian regularisation, S3VM), weak supervision (majority vote, Dawid-Skene with one-coin and GLAD modes, the agreement label model, noise transition, confident learning, positive-unlabelled risks, label proportions, multiple-instance SVM), noisy labels (small-loss selection, co-teaching), drift detectors (DDM, EDDM, ADWIN, KSWIN, PSI) and shift estimation (black-box shift, EM prior adjustment, density ratios by classifier, KLIEP, uLSIF), uncertainty (deep ensembles, Monte Carlo dropout), coresets (k-centre greedy, sensitivity sampling, Bayesian coresets), knockoffs and selective inference, feature synthesis, filter selection (mRMR, Relief), click models, privacy (per-example clipping as a gradient transformation) and privacy attacks (loss threshold, shadow models, likelihood ratio, inversion, extraction), calibration additions (quantile calibration, PIT), conformal under shift, model selection additions (thresholdout, the ladder, the corrected resampled t-test, 5×2 CV), learning theory as functions of n and capacity (PAC, VC, Rademacher, margin, stability, PAC-Bayes, regret bounds, bias-variance, degrees of freedom), active learning strategies (expected model change, BALD, density weighting, core sets, BADGE).
- **`linear`**: the perceptron family as algorithms (plain, averaged, voted, kernel, pocket), passive-aggressive, LARS one knot per step, the group lasso penalty, ordinal regression, mixed models by REML, Theil-Sen, zero-inflated and hurdle GLMs, the multinomial family, the multi-task lasso path, regression diagnostics (hat matrix, leverage, Cook's distance, DFBETAS, VIFs, condition indices, Breusch-Pagan, White).
- **`cluster`**: DBSCAN, OPTICS and HDBSCAN, mean shift, agglomerative with the linkage matrix and cophenetic correlation, BIRCH, k-medoids (PAM, CLARA), fuzzy c-means, spectral assembly, affinity propagation, constrained k-means, Elkan's bounds and bisecting and kernel k-means, the gap statistic and stability.
- **`optim`**: the `gradient` mode (above), NMF updates, the one-class and SVDD duals in `smo.ts`, the expert learners (weighted majority, follow the leader and the perturbed leader, fixed share) in `online.ts`, landscape slices with filter normalisation, Sinkhorn is there already.
- **`nn`**: energy blocks (Hopfield, dense associative memory, RBM with contrastive divergence and the exact partition for small n), continuous blocks (CNF, Hamiltonian and Lagrangian), operators (DeepONet, spectral convolution and FNO), forecasting blocks (N-BEATS), recommender blocks (factorisation machines, field-aware, DeepFM, wide and deep), the graph transformer block, relational and equivariant graph layers, masked convolutions, LoRA and adapters and prefix tuning as parametrisations, receptive-field arithmetic, SWAG collection, adversarial attacks and certification (FGSM, PGD, DeepFool, Carlini-Wagner, boundary and square attacks, randomised smoothing, interval bound propagation).
- **`losses`**: CTC with its lattice, preference objectives (DPO, IPO, KTO), invariance penalties (IRM, V-REx), robust classification losses (symmetric and generalised cross-entropy, forward and backward correction, bootstrapping), focal and label smoothing and class weights as table entries, VQ and noise-contrastive and sliced score matching, Barlow Twins and VICReg, evidential negative log-likelihoods.
- **`dynamics`**: cellular automata (elementary, Life-like, Langton's ant, sandpile, forest fire), agent-based models (Schelling, Nagel-Schreckenberg, Wa-Tor, Sugarscape), lattice gases and lattice Boltzmann, the fractals (escape-time sets, chaos game and deterministic IFS with box counting, L-systems with a turtle, the named maps and attractors), celestial mechanics (Kepler's equation, orbits from elements, the n-body Hamiltonian, the restricted three-body field, null geodesics), named systems (Kuramoto, Rössler, Thomas, the double pendulum), control (lead-lag design, Pontryagin's Hamiltonian system, H∞ synthesis, adaptive control, recursive least squares for identification).
- **`graph`**: spreading (Gillespie epidemics, linear threshold and independent cascade, greedy influence), percolation, growth as stepped algorithms, effective resistance, neighbour sampling.
- **`stats`**: experiments (CUPED, post-stratification, regression adjustment, the delta method, cluster and switchback designs), design (randomised, blocked, matched pairs, factorial with aliases, two-way ANOVA), shrinkage (James-Stein), Little's MCAR test, concentration bounds.
- **`text`**: n-gram language models with smoothing modes, RAKE and TextRank graphs, the inverted index and probabilistic retrieval, record linkage (Fellegi-Sunter), spelling correction's channel model.
- **`pgm`**: rating systems (Elo, Glicko, TrueSkill over `gaussianEp`, Bradley-Terry), item response theory, HMM variants (mixture emissions, durations, input-dependent transitions), online CRF trainers (perceptron, passive-aggressive, MIRA, AROW), Swendsen-Wang.
- **`explain`**: the logit and tuned lens, activation patching, feature visualisation as an algorithm; the tree pieces move out.
- **`information`**: the information bottleneck, MDL code lengths; stepped Huffman, arithmetic and dictionary coders.
- **`sampling`**: variance reduction (control variates, antithetic pairs, stratification, Rao-Blackwell), gradient estimators (score function, pathwise, with baselines); `stochastic/diffusion-models.ts` (noise schedules, VP and VE SDEs, DDPM and DDIM samplers as algorithms, guided scores).
- **`numerics`**: matrix completion (SGD and ALS factorisation, implicit ALS, soft-impute), stationary iterations and two-grid, subspace iteration, Jacobi SVD steps, Lanczos stepped, float16 and bfloat16 emulation.
- **`datasets` and `data/modifiers`**: the planted problems of section 1; image modifiers (crops, flips, mixup, cutmix); leaks and duplicates; subpopulation shift and shortcuts.
- **`environments`, `agents`, `search`, `problems`, `logic`, `systems`, `vision`**: the 19-state random walk and Baird's counterexample, changing mazes, tile coding and the semi-gradient learners, goal-conditioned agents with hindsight relabelling, best-arm bandits and sliding-window UCB, off-policy Monte Carlo; particle swarms, NEAT, the classic puzzles and tours, SAT instances; production systems and Rete, uncertain reasoning, argumentation, description logic, Markov logic, ProbLog, differentiable logic; step-response measures and Padé delays; structure from motion over Levenberg-Marquardt, Canny's stages, Hough peaks, watershed and mean-shift segmentation as steps.

### 2.4 What already carries the most weight

The modules the rows cite most are the ones that must be solid before anything else, because every chapter leans on
them: `nn/training/fit.ts` (41 rows), `datasets/images.ts` (36), `numerics/spatial/brute.ts` (28), the classification
and regression losses (51), the `nn` module containers and layers (58), `datasets/corpus.ts` (22), `linear/solvers.ts`
(19), cross-validation (18), `optim/transform.ts` (17), the transformer blocks (15), `kernels/kernel.ts` (14),
`graph/spectral.ts` (14), the resampling and descriptive statistics (22), `dynamics/named-systems.ts` (13),
`autodiff/transforms.ts` (13). Of the option tables built this week, `objectives.ts` is cited by 12 rows and
`solvers.ts` by 11 before either has moved to its library home.

## 3. The order

1. **Dissolve `methods/shared/`** into the homes of section 1, and move the vision helpers. A day; it settles the
   question before anything else is built on the wrong side of the line.
2. **The bases**: the trainer, the `gradient` mode, the sweep, the stream, the agent base. Everything in section 8
   of the design's build order that is not done waits on these; the first exemplar (logistic regression with every
   mode) needs the `gradient` mode and the sweep (for the comparison page) and nothing else new.
3. **Trees and ensembles**, moving the two existing growers out of `explain`. They unblock the most chapters at
   once.
4. **The supervised core**: `svm`, `gp`, `kernel-methods`, `gam`, the `learn` and `linear` growth for classify and
   regress.
5. **Time series** with the matrix profile, and the stream-based pieces of `learn` (drift, shift).
6. **Unsupervised**: `cluster` growth, `decomposition`, `manifold`, `covariance`, `anomaly`.
7. **The paradigm pieces of `learn`**: semi-supervised, weak supervision, noisy labels, active strategies,
   uncertainty, coresets, privacy.
8. **Dynamics and graphs**: automata, agent-based models, fractals, celestial mechanics, spreading and percolation.
9. **The small new domains**: `survival`, `rules`, `patterns`, `causal`, `sketches`, `embedding`.
10. **The neural additions** and the long tail of section 2.3.

Each step is a batch of agent work on labelled worktrees with throttled tests, merged through one gate; the oracle
groups grow by one per reference library (`oracle-stats` for statsmodels and lifelines, `oracle-series` for stumpy
and sktime, `oracle-causal` for DoWhy and EconML, `oracle-sketch` for datasketch).
