# whitebox-js: structure (v2)

This document is the table of contents of the whole library as designed: every package, folder and file, with what
each file contains. It is written for review before implementation and does not describe existing code. Read it top to
bottom: after the conventions and the shared vocabulary (sections 2 and 3), it starts at the bottom of the dependency
graph (errors, tensors, random keys) and climbs one layer at a time to neural networks, reinforcement learning,
hyperparameter search, the named methods built on all of these, the render layer, the metadata layer and the two apps.

Nothing here has to be built in one go. The point is that every concept has a known home before it is written, so the
library can grow for years without reorganising, and so that everything in it speaks the same language.

Version 1 is kept at `docs/history/structure-v1.md`.

## 0. What changed in v2

1. **One differentiable operation set.** Tensor operations are primitives with derivative rules, and every numerical
   routine (solves, inverses, log-determinants, Cholesky, eigendecompositions, SVD, matrix functions, quadrature, root
   finding, ODE solves, sorting) is differentiable, either by composition or by a custom rule. There is no separate
   "autodiff copy" of the operations: a function written with tensors can be differentiated as it stands.
2. **A shared vocabulary** (section 3): thirteen contracts that every package uses, so that datasets, models, metrics,
   tuners, traces and figures fit together without adapters.
3. **Context and seeding.** A `Context` carries the seed, named key derivation, the tracer and trace policy; one
   sanctioned default context gives the "set the global seed once" experience without hidden state in the core.
4. **Trace contracts.** Algorithms declare what they can emit (`TraceSchema`); callers choose what to record
   (`TracePolicy`: channels, frequency, detail, gradients). Gradients, activations and states are first-class events.
5. **New core domains:** `symbolic`, `context`, `foundations`, `information`, `systems`, `search`, `stochastic`,
   `losses`, `metrics`, `games`, `tune`.
6. **Metrics** are their own domain, organised by the encyclopedia's sixteen evaluation categories.
7. **One parameter-selection system** (`tune`): studies, trials, samplers and pruners that work the same for
   estimators, neural networks, agents, probabilistic programs and arbitrary functions. Bayesian optimisation, TPE,
   Hyperband and population-based training are samplers and schedulers in that system.
8. **An inference planner.** `probprog` analyses a model and chooses an engine (exact, conjugate, Kalman,
   forward-backward, EM, Gibbs, NUTS, SVI, SMC, ABC) and explains why. Gaussian mixtures, HMMs, topic models, factor
   analysis and Bayesian regressions are programs solved by these engines rather than separate implementations.
9. **Linear models unified** on one penalised GLM core; GAMs deepened (the mgcv basis set, derivatives and curvature of
   smooths with simultaneous intervals, concurvity, basis-dimension checks).
10. **Reinforcement learning broadened** to the main families (value-based, policy-gradient, actor-critic, model-based,
    offline, imitation, multi-agent, exploration, hierarchical, safe), MinAtar-style and multi-agent environments, and
    a required `render` for every environment.
11. **Search-based discovery and GOFAI:** search, planning, constraint satisfaction, pattern mining, subgroup
    discovery, exceptional model mining, production systems, fuzzy logic, argumentation, evolutionary computation.
12. **Stochastic processes and stochastic calculus** (Ito integrals, SDEs, point processes, DPPs, Dirichlet processes).
13. **Mathematical foundations:** symbolic algebra, calculus, Taylor series and Taylor-mode differentiation,
    trigonometry, linear algebra with steps, curves and surfaces, number theory, combinatorics, approximation theory.
14. **Datasets as parameterised generators** with knobs declared as spaces and ground truth that can be queried
    (likelihoods, posteriors, Bayes error, true changepoints).
15. **Animation export** (GIF, MP4/WebM, animated SVG), 3-D projection in scenes, and maths typesetting.
16. **Approximate matching with shingles:** shingle indexes, MinHash-LSH candidate generation and similarity scoring for
    querying a corpus, a dense fastText mode on the same n-grams, and record linkage.
17. **New method areas:** scientific machine learning (neural ODEs, PINNs, neural operators, equation discovery),
    active learning, federated learning, fairness, privacy, adversarial robustness, experimentation, LLM wrappers with
    text fingerprinting and watermarking, audio, complex systems, foundations studies.
18. **A concept taxonomy** in `whitebox-meta`, so encyclopedia notes link to the functions that implement them.

## 0.1 Review notes, answered

| note                                                     | answer                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data is not in the tiers                                 | It is: `lib/data` (tier 5). The tier table in section 4 now gives every domain a one-line purpose.                                                                                                                                                                                                                                    |
| Model architectures                                      | `methods/neural/` is now organised by modality with named architectures (ResNet, U-Net, ViT, GPT, BERT, T5, Mamba, GNNs, ...); reusable blocks are in `lib/nn/blocks/`; any model draws itself through `lib/nn/describe.ts`.                                                                                                          |
| Feature unions, pipelines, scalers                       | `lib/learn/compose.ts` (pipeline, featureUnion, columnTransformer, transformedTarget) and `lib/learn/preprocessing/` (scalers with fitted parameters as data, encoders, imputers, feature maps).                                                                                                                                      |
| Dimensionality reduction (UMAP etc.)                     | `methods/manifold/` (UMAP, t-SNE, PaCMAP, Isomap, LLE, MDS, spectral embedding, diffusion maps, SOM) and `methods/decomposition/` (PCA family, ICA, NMF, factor analysis, CCA/PLS, random projections).                                                                                                                               |
| Calibration                                              | `lib/learn/calibration.ts`; calibration metrics in `lib/metrics/probabilistic.ts`.                                                                                                                                                                                                                                                    |
| GIF and MP4 of animations                                | `render/animation/` (timelines from traces, GIF, MP4/WebM, animated SVG).                                                                                                                                                                                                                                                             |
| Statistical tests with good wrappers                     | `lib/stats/tests/`: every test returns the same `TestResult`; added two-sample, independence, equivalence, Bayesian and sequential tests.                                                                                                                                                                                             |
| Probabilistic programming that chooses its own inference | `lib/probprog/analysis.ts`, `lib/probprog/infer.ts` (the planner) and `lib/probprog/engines/`.                                                                                                                                                                                                                                        |
| Gradients in traces; a contract for what is traced       | `lib/trace/schema.ts`, `lib/trace/policy.ts`, `lib/trace/taps.ts`; tensor events carry a role (param, grad, state, activation).                                                                                                                                                                                                       |
| Bayesian optimisation; one parameter-selection mechanism | `lib/tune/` for every kind of objective; BO, TPE, BOHB and PBT in `methods/tuning/` as samplers.                                                                                                                                                                                                                                      |
| Cross-validation variants and a single split             | `lib/learn/model-selection/splitters.ts` (holdout, train/validation/test, stratified, group, stratified group, time-series windows, purged and embargoed, spatial).                                                                                                                                                                   |
| All numerics differentiable (inverses, log-determinants) | Section 3.1; each `lib/numerics` file states its derivative rule.                                                                                                                                                                                                                                                                     |
| Strong splines for GAMs, curvature, `ns`/`ps`/`bs`       | `lib/stats/smoothing/` and `methods/gam/` (all mgcv bases, derivatives and curvature of smooths with simultaneous intervals).                                                                                                                                                                                                         |
| Spectrograms, windows, Lomb-Scargle, periodograms        | `lib/signal/spectral.ts`; estimator comparisons in `methods/signal/`.                                                                                                                                                                                                                                                                 |
| ODE and PDE solvers of various orders                    | `lib/dynamics/ode/` by family and order (explicit, adaptive, multistep, implicit, symplectic, exponential, Taylor) and `lib/dynamics/pde/` (finite difference, volume, element, spectral); order-verification studies.                                                                                                                |
| Everything traced                                        | Every iterative algorithm, plus step-recorded versions of direct methods (elimination, factorisations) and `trace/instrument.ts` for any function.                                                                                                                                                                                    |
| Parameterisable datasets with probe-able ground truth    | `lib/data/generator.ts`, `lib/data/truth.ts`; every generator declares its knobs as a `Space` and returns its generative model.                                                                                                                                                                                                       |
| Distributions parameterised by other components          | Parameters accept tensors and traced values with batch shapes; `lib/dists/compose/conditional.ts` makes distributions functions of inputs (used by GLMs, networks and GPs).                                                                                                                                                           |
| SMO                                                      | Generic in `lib/optim/smo.ts`; `methods/svm/` uses it.                                                                                                                                                                                                                                                                                |
| Gabor filters                                            | `lib/signal/image/filters.ts` and `lib/signal/image/features.ts`.                                                                                                                                                                                                                                                                     |
| DPPs and stochastic processes                            | New `lib/stochastic/`.                                                                                                                                                                                                                                                                                                                |
| Are trees reinvented?                                    | Yes, natively, like everything else: Python libraries are oracles for tests, never runtime dependencies.                                                                                                                                                                                                                              |
| Tree-augmented naive Bayes                               | `methods/naive-bayes/bayes-net-classifiers.ts` (TAN, AODE, k-dependence), built on `lib/pgm`.                                                                                                                                                                                                                                         |
| GMMs through probabilistic programming                   | `methods/mixture/` defines mixtures as programs solved by `probprog` engines; `gaussianMixture` is a familiar wrapper. Same for HMMs, LDA, factor analysis.                                                                                                                                                                           |
| PCA built on eigen/SVD                                   | Yes; every methods area has a "built from" line.                                                                                                                                                                                                                                                                                      |
| word2vec                                                 | `methods/embedding/word2vec.ts`.                                                                                                                                                                                                                                                                                                      |
| A fastText variant                                       | `methods/embedding/fasttext.ts` (subword vectors and the supervised classifier); it shares `lib/text/shingles.ts` with shingle matching, and powers the dense mode of `methods/retrieval/approximate-matching.ts`.                                                                                                                    |
| Active learning                                          | New `methods/active-learning/`.                                                                                                                                                                                                                                                                                                       |
| Elo and rating models                                    | `methods/models/rating.ts`.                                                                                                                                                                                                                                                                                                           |
| Classical vision (Hough)                                 | `lib/signal/image/features.ts` and `methods/vision/classical.ts`.                                                                                                                                                                                                                                                                     |
| Rendering every environment                              | `render(state)` is part of the environment contract; episodes re-render from traces by deterministic replay.                                                                                                                                                                                                                          |
| RL beyond the primitive                                  | `methods/rl/agents/` covers the main families; new environment suites.                                                                                                                                                                                                                                                                |
| NMF                                                      | `methods/decomposition/nmf.ts`.                                                                                                                                                                                                                                                                                                       |
| A more generic Boltzmann                                 | `methods/generative/energy-based/` (one energy-model protocol for Boltzmann machines, Hopfield networks, Ising models and neural EBMs).                                                                                                                                                                                               |
| Two-tower and sequential recommenders                    | `methods/retrieval/recommenders/two-tower.ts` and `sequential.ts`.                                                                                                                                                                                                                                                                    |
| Comprehensive metrics in my categories                   | New `lib/metrics/`, sixteen files in your order.                                                                                                                                                                                                                                                                                      |
| Information theory (MI, entropy, coding, zlib)           | New `lib/information/` (estimators, divergences, Huffman, arithmetic coding, LZ, DEFLATE/zlib, compression distance).                                                                                                                                                                                                                 |
| Learning theory is sparse                                | `methods/theory/` expanded (PAC, VC, Rademacher, PAC-Bayes, stability, NTK, implicit regularisation, benign overfitting, and more).                                                                                                                                                                                                   |
| Two-sample tests (MMD)                                   | `lib/stats/tests/two-sample.ts` and `lib/stats/tests/kernel.ts`.                                                                                                                                                                                                                                                                      |
| Text fingerprinting, LLM wrappers                        | `lib/text/fingerprinting.ts` and new `methods/llm/` (language-model protocol, adapters, watermarking, detection, stylometry).                                                                                                                                                                                                         |
| Mathematical foundations                                 | New `lib/symbolic/`, `lib/foundations/`, `lib/numerics/approximation/`, `lib/autodiff/taylor.ts`, `methods/foundations/`.                                                                                                                                                                                                             |
| Stochastic calculus for diffusion                        | `lib/stochastic/calculus.ts` and `lib/stochastic/sde.ts`; diffusion models build on them.                                                                                                                                                                                                                                             |
| Lazy loading                                             | Section 2: per-folder entry points, no side effects, a lazy module map, remote datasets, worker execution.                                                                                                                                                                                                                            |
| Everything speaking the same language                    | Section 3 and the "built from" lines.                                                                                                                                                                                                                                                                                                 |
| BOCPD with tunable changepoint data                      | `lib/dynamics/filtering/changepoint.ts`, `methods/datasets/changepoints.ts`, `methods/timeseries/changepoints.ts`.                                                                                                                                                                                                                    |
| A global seed                                            | `lib/context/`.                                                                                                                                                                                                                                                                                                                       |
| Subgroup discovery, exceptional model mining, GOFAI      | `lib/search/`, `methods/patterns/`, `methods/gofai/`, `methods/evolutionary/`.                                                                                                                                                                                                                                                        |
| Big files should be split                                | Large topics are folders; files over about 400 lines are split.                                                                                                                                                                                                                                                                       |
| Approximate search with shingles                         | `lib/text/shingles.ts`, `lib/text/approximate-search.ts`, `methods/retrieval/approximate-matching.ts`, `methods/retrieval/record-linkage.ts`.                                                                                                                                                                                         |
| PINNs, neural ODEs and their solvers                     | New area `methods/scientific/` (neural ODEs and relatives, PINNs, neural operators, equation discovery, surrogates), on `lib/dynamics/ode` solvers of every order with adjoint gradients (`lib/dynamics/ode/adjoint.ts`), PDE operators of networks (`lib/dynamics/pde/operators.ts`) and ODE blocks (`lib/nn/blocks/continuous.ts`). |

## 1. Shape of the repository

```
whitebox-js/
  packages/
    engine/      whitebox-engine   compute: from tensors to reinforcement learning, plus datasets       (published)
    render/      whitebox-render   headless figures: ECharts options, layouts, scenes, SVG, animation;  (published)
                                   React bindings in a separate entry point
    meta/        whitebox-meta     docs metadata: schema, extractor, example runner, worker, concepts   (published)
  apps/
    docs/        how to use whitebox-engine: how-tos, API reference, concepts, live examples, traces (private)
    gallery/     what whitebox-render can draw: charts, diagrams, scenes, animations, environments    (private)
  oracle/        Python ground truth: golden fixtures and dataset vendoring (uv, pydantic)
  scripts/       repository tooling in TypeScript
  docs/          structure.md (this file), decisions.md, extending.md, conventions.md, concepts.yaml, history/
  Makefile, mk/  the single interface for routine work
  CLAUDE.md, INIT.md, AUDIT.md, PORT.md
```

`docs` is the counterpart of `engine` and `gallery` is the counterpart of `render`. Outside projects (the encyclopedia,
the slide and blog tool) consume the three published packages as release tarballs.

## 2. Conventions every file follows

**Entry points.** Every folder is an entry point. `packages/engine/src/lib/numerics/linalg/` is imported as
`whitebox-engine/numerics/linalg`; its parent re-exports it as the namespace `linalg`. Index files are generated from
folder contents, so adding a file never edits another file.

**Core and applications.** `packages/engine/src/lib/` is the core: general machinery. `packages/engine/src/methods/` is
applications: named models, algorithms, environments, datasets and studies, each built from the core. The core never
imports methods.

**Layering.** Each core domain has a `domain.json` with a tier and may import only lower tiers. Each methods area has
an `area.json` naming the areas it may import. A generated dependency-cruiser configuration enforces both.

**One canonical home.** Every function has exactly one defining file. Another entry point may re-export it as an alias
(the docs record the canonical home), but nothing is implemented twice. Where two names would collide in meaning (a
differentiable training loss and an evaluation metric), they get distinct names: `losses.mse` versus
`metrics.meanSquaredError`.

**File size.** A file groups related exports around one idea: about 3 to 12 exports and 100 to 400 lines. Past 400
lines it is split by sub-idea; a topic with more than five files becomes a folder.

**Procedural core, thin classes.** Functions take explicit data and return new data. Randomness comes from keys or a
context. No module-level mutable state and no registration side effects, with one documented exception (the default
context, section 3.2). Familiar class APIs (`new PCA({ nComponents: 2 }).fit(X)`, `new nn.Linear(4, 8)`,
`env.step(action)`) wrap the functions with at most one level of inheritance, and both forms are documented.

**Native implementations.** Every algorithm is implemented here so that it can be traced and explained; this includes
trees, SVMs, decompositions and samplers. Python libraries are test oracles only. External JavaScript libraries are used
for charting (ECharts), graph layout (dagre), GIF and video encoding, code editing and maths typesetting.

**Names and verbs.** Familiar names where a field has a standard (numpy and scipy for numerics, scikit-learn for
estimators, PyTorch for layers, optax for optimiser transformations, gymnasium for environments, NetworkX for graphs,
NumPyro for programs, pgmpy for graphical models, mgcv for smooths), but one consistent set of verbs everywhere:

| verb                                             | meaning                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------- |
| `fit(data, options)`                             | learn parameters; returns a fitted value (never mutates)            |
| `predict`, `predictProba`, `predictDistribution` | point prediction, class probabilities, full predictive distribution |
| `transform`, `inverseTransform`                  | map data forwards and back                                          |
| `score(data)`                                    | the model's default metric on data                                  |
| `logProb`, `sample`                              | densities and draws, for distributions, models and generators       |
| `infer(model, data)`                             | posterior inference on a probabilistic program                      |
| `run`, `step`, `init`                            | drive an algorithm                                                  |
| `reset`, `step`, `render`                        | drive and draw an environment                                       |
| `describe`                                       | produce a structure or scene that renders the object                |
| `tune(objective, space)`                         | select parameters                                                   |
| `evaluate(model, data, metrics)`                 | compute metrics with confidence intervals                           |

Option names are camelCase versions of the familiar ones (`nComponents`, `maxIter`, `tol`, `learningRate`). Randomised
primitives take a `key` first (`normal(key, shape)`); higher-level functions take `{ key }` or `{ ctx }` in their
options and otherwise use the context rules of section 3.2.

**Inputs.** Every function that takes arrays accepts a `TensorLike` (a tensor, a typed array or nested arrays) and
converts it once with `asTensor`; outputs are tensors.

**Docstrings.** Every export has a TSDoc block: summary, `@param` per parameter, `@returns`, `@throws` where relevant,
and at least one `@example` with `// => value`, `// => figure`, `// => trace` or `// => throws Code` expectations that
the build runs. `@concept` tags link to the concept taxonomy (section 7), `@see` and `@reference` cite sources.

**Tests next to code.** For `foo.ts`: `foo.test.ts` (unit), `foo.property.test.ts` (fast-check properties: gradients
against finite differences, distribution identities, decomposition reconstruction, invariances), `foo.golden.test.ts`
(against oracle fixtures) and `foo.test-d.ts` (types). React code has `*.browser.test.tsx` run in Chromium.

**Python ground truth.** Every engine file with a Python counterpart has a case file at the mirrored path
`oracle/cases/<domain>/<file>.py`. Fixtures are committed JSON; tests never need Python. Each folder below names its
oracle libraries.

**Lazy loading.** Packages are `sideEffects: false` ESM with one entry point per folder, so bundlers load only what a
page imports. `whitebox-meta/modules` maps every entry point to a lazy `import()`; large datasets are release assets
fetched on demand; heavy work runs in a Web Worker through `whitebox-meta/worker`.

## 3. The common language

Everything in the library is written against these contracts. A reader who knows these thirteen ideas can read any file.

|   # | contract                                                        | defined in                                        | spoken by                                                                                     |
| --: | --------------------------------------------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------- |
|   1 | `Tensor`, `Value`, primitives with derivative rules             | `lib/tensor`, `lib/autodiff`                      | everything numeric                                                                            |
|   2 | `Key`, `Context`, named key derivation                          | `lib/random`, `lib/context`                       | everything random                                                                             |
|   3 | `Algorithm`, `TraceSchema`, `TracePolicy`, trace events         | `lib/trace`                                       | every iterative method                                                                        |
|   4 | `Space`                                                         | `lib/space`                                       | hyperparameters, dataset knobs, action and observation spaces, search spaces, figure controls |
|   5 | `Distribution`, `ConditionalDistribution`, `Bijector`           | `lib/dists`                                       | priors, likelihoods, every predictive output                                                  |
|   6 | `Model` (probabilistic program), `InferenceEngine`, `Posterior` | `lib/probprog`                                    | every probabilistic model and how it is solved                                                |
|   7 | `Estimator`, `Fitted`                                           | `lib/learn`                                       | every learner: linear, kernel, tree, neural, Bayesian                                         |
|   8 | data schemas, `DatasetGenerator`, `Truth`                       | `lib/data`                                        | datasets, estimators, splitters, metrics                                                      |
|   9 | `MetricSpec`, `MetricResult`, `TestResult`, `Estimate`          | `lib/base/results.ts`, `lib/metrics`, `lib/stats` | every metric and every statistical test                                                       |
|  10 | `Objective`, `Study`, `Trial`, `Sampler`                        | `lib/tune`                                        | every selection problem                                                                       |
|  11 | `Env`, `Agent`                                                  | `lib/rl`                                          | every sequential decision problem                                                             |
|  12 | `SearchProblem`                                                 | `lib/search`                                      | paths, planning, constraint satisfaction, games, pattern and rule search, ILP                 |
|  13 | `Scene`, `Structure`, `describe()`                              | `lib/scene`                                       | every picture and every graph-shaped thing                                                    |

### 3.1 Values and derivatives

A `Tensor` is a concrete array. A `Value` is a tensor or a traced value standing in for one inside a transformation
(`grad`, `jvp`, `vmap`). Every operation is a primitive (forward kernel, shape and dtype rules, `vjp`, `jvp`, batching
rule) or a composition of primitives, so every operation accepts values and is differentiable. Routines whose
derivative should not be obtained by differentiating their loops carry a custom rule: Cholesky, triangular solves,
`solve`, `inv`, `slogdet`, `eigh`, `svd`, `expm`, `sort`, `quad`, root finders and fixed points (implicit function
theorem), ODE solves (adjoint method), optimisation layers (`implicitArgmin`). Traces nest by level, so higher-order
derivatives (Hessians, Taylor coefficients, gradients of gradients for meta-learning) work. A transform takes the next
level for the duration of its call (a depth counter in `autodiff/_level.ts`, restored by `try`/`finally`), so a closure
that captures an outer transform's value is processed at the right level; this is internal bookkeeping that never
changes a result. Differentiation is opt-in per call, as in JAX: outside a transform a tensor is a plain array with no
graph attached.

### 3.2 Keys, seeds and the context

Randomness is explicit. `Key` values are immutable and derived, never advanced. A `Context` holds a seed and derives
keys by name: `ctx.key('init')`, `ctx.key('data', 3)`. The same names give the same numbers on every machine, in any
call order. A context also carries the tracer, the trace policy, the backend and default dtype. Every high-level
function resolves randomness with one rule: an explicit `key` option, else `ctx.key(<function name>)` from the
`ctx` option, else the default context. The default context (seed 0) can be replaced once at application start with
`setDefaultContext(createContext({ seed: 42 }))`. That is the only module-level state in the library, and the core
functions never read it; only the convenience layer (class wrappers, scripts) does. So "set the global seed" works,
and every reader of the encyclopedia sees the same numbers.

### 3.3 Algorithms and what they record

Every iterative method is an `Algorithm` (`init`, `step`, `done`, `observe`) with a `TraceSchema` declaring the
channels it can emit: metrics (name, description, unit, direction), tensors (name, role, shape), messages (kind,
payload schema) and episodes. `runAlgorithm` records what a `TracePolicy` asks for: which channels, how often (every
step, every n, log-spaced, on change, first and last), at what detail (summaries such as norms and histograms, or full
tensors), with or without gradients, within a size budget. Inner algorithms run as child runs, so traces are
hierarchical (an inner solver inside an outer loop, a trial inside a study). Each algorithm also runs untraced at full
speed: the policy decides, not the code. Direct methods (elimination, factorisations, decision-tree splits) offer step
records, and `instrument(fn, options)` traces any function call.

### 3.4 Spaces

A `Space` describes a set of values with metadata (label, description, unit, scale, default). The same type describes
a model's hyperparameters (every estimator declares `hyperSpace`), a dataset generator's knobs, an environment's
observations and actions, a tuning search space, and the controls of a figure. Because of this, any estimator can be
tuned without writing a search space, and any dataset can be explored with automatically generated controls.

### 3.5 Distributions everywhere

A `Distribution` is the universal uncertain value. Parameters are values (tensors or traced, with batch shapes), so
densities are differentiable and vectorised. A `ConditionalDistribution` is a function from inputs to a distribution;
GLM likelihoods, neural network heads, mixture density networks and GP predictives are all conditional distributions.
Every regressor and classifier offers `predictDistribution`, so log loss, CRPS, calibration and conformal prediction
apply uniformly, whatever the model.

### 3.6 Programs and inference

A probabilistic model is a function that calls `sample`, `observe`, `plate` and `deterministic`. `analyse(model, data)`
traces it once to find its sites, supports, plates, dependencies, conjugate pairs and discrete cardinalities.
`infer(model, data)` asks every available `InferenceEngine` whether it applies and at what cost, picks the best,
records the reason, and returns a `Posterior` (exact distributions, a variational approximation or chains of samples,
all queried the same way). Named models (Gaussian mixtures, HMMs, LDA, factor analysis, Bayesian regressions, IRT,
TrueSkill) are programs; their fast algorithms (EM, forward-backward, collapsed Gibbs, expectation propagation) are
engines. A new model is a new program; a new algorithm is a new engine. Neither edits the other.

### 3.7 Estimators

An estimator is created by a factory with hyperparameters and declares its `hyperSpace`; `est.fit(data, options)`
returns a `Fitted` value whose parameters are plain data (serialisable) and whose methods are the verbs above. Neural,
kernel, tree and Bayesian learners all follow it, so pipelines, cross-validation, calibration, conformal prediction,
explanation and tuning work with any of them.

### 3.8 Data, generators and truth

A dataset is one of a fixed set of schemas (regression, classification, tabular, ranking, corpus, sequence, time
series, events, comparisons, graph, image, trajectories). A `DatasetGenerator` declares its knobs as a `Space` and
returns `{ data, truth }`, where `truth` can be queried: the generating program (so `truth.logProb(x, y)` and
`truth.posterior(x)` work), the regression function, the Bayes error, true clusters, changepoints, latent coordinates
or the true causal effect. Every method can therefore be scored against the ground truth.

### 3.9 Results

Metrics and statistical tests return the same shapes from `lib/base/results.ts`: `Estimate` (value, standard error,
interval, method), `MetricResult` (an estimate plus per-class or per-sample breakdowns), `TestResult` (statistic,
p-value, degrees of freedom, effect size with interval, alternative, method, assumptions checked). A `MetricSpec`
declares a metric's name, inputs, direction, range and category, so tuners and reports know what "better" means.

### 3.10 Objectives and studies

Any selection problem is an `Objective` from a `Space` to one or more values (optionally reporting intermediate values
for pruning). Adapters turn estimators with a splitter and metric, training loops, agents in environments, programs
(evidence, ELBO, LOO) and black-box functions into objectives. A `Study` runs a `Sampler` (grid, random, Sobol, CMA-ES,
Bayesian optimisation, TPE) with optional pruning and multi-fidelity schedules, as a traced algorithm whose steps are
trials.

### 3.11 Environments and agents

Environments are functional (`reset(key)`, `step(state, action, key)`, `render(state)`) with gymnasium-style wrappers.
`render` is required and returns a `Scene`, so every environment can be drawn, animated and exported. Agents are
functional too (`init`, `act`, `learn`). Rollouts are traced algorithms, and any episode can be reconstructed from its
trace by deterministic replay.

### 3.12 Search problems

Paths in implicit graphs, puzzles, planning, constraint satisfaction, game trees, and the refinement searches behind
rule learning, subgroup discovery, exceptional model mining and ILP all use one `SearchProblem` contract and one family
of traced search algorithms.

### 3.13 Pictures and structures

A `Scene` is a 2-D drawing as data (with an optional 3-D layer projected by a camera). A `Structure` is any
graph-shaped thing as data: typed nodes, typed edges, groups (plates, repeated blocks), ports and annotations.
Graphical models, computation graphs, network architectures, decision trees, search trees, automata, pipelines and
graphs all describe themselves as structures; whitebox-render lays out and draws any structure the same way.

### 3.14 One worked path

A reader trains a classifier and sees every contract in turn:
1. `moons.generate(ctx.key('data'), { n: 400, noise: 0.2 })` returns `{ data, truth }`; its knobs are a `Space`.
2. `mlpClassifier({ hidden: [16, 16] })` is an `Estimator` with a `hyperSpace`.
3. `fit` runs a training `Algorithm`; the `TracePolicy` records loss every step and gradient norms every ten steps.
4. `predictDistribution` returns categorical distributions; `evaluate` returns `MetricResult`s with bootstrap intervals,
   and `truth.bayesError()` gives the floor.
5. `tune(estimatorObjective(est, data, { splitter: stratifiedKFold(5), metric: logLoss }), est.hyperSpace)` runs a
   study with Bayesian optimisation, each trial a child run.
6. `describe(fitted)` gives a `Structure` of the network; whitebox-render draws it, plots the trace, and exports the
   training animation as a GIF.
7. Every function used has a docstring example, checked by the build and shown live in the docs app.
## 4. whitebox-engine core, `packages/engine/src/lib/`

The core has 35 domains in 11 tiers. A domain may import only domains in lower tiers.

| tier | domain        | purpose                                                                                                                                                                                          |
| ---: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|    0 | `base`        | errors, JSON, shared result shapes, scalar helpers, version                                                                                                                                      |
|    1 | `scene`       | drawings (`Scene`), 3-D layers, rasters and graph-shaped descriptions (`Structure`) as data                                                                                                      |
|    1 | `tensor`      | arrays, dtypes, shapes, primitives with derivative rules, backends, operations, pytrees                                                                                                          |
|    2 | `random`      | immutable counter-based keys and samplers                                                                                                                                                        |
|    2 | `symbolic`    | symbolic expressions: simplification, differentiation, series, integration, LaTeX                                                                                                                |
|    3 | `autodiff`    | reverse, forward and Taylor-mode differentiation, vectorisation, gradient checks                                                                                                                 |
|    3 | `trace`       | algorithms, trace schemas and policies, events, recording, replay                                                                                                                                |
|    3 | `space`       | typed spaces for parameters, knobs, observations and actions                                                                                                                                     |
|    4 | `context`     | seeds, named keys, tracer and policy defaults                                                                                                                                                    |
|    5 | `numerics`    | special functions, linear algebra, sparse, polynomials, quadrature, roots, interpolation, Fourier, convolution, geometry, nearest neighbours, approximation, inverse problems                    |
|    5 | `data`        | dataset schemas, generators and truth, loading, batching, splits, modifiers                                                                                                                      |
|    6 | `foundations` | mathematical foundations: trigonometry, vectors, rotations, curves, surfaces, calculus, series, linear algebra with steps, number theory, combinatorics, complex numbers, elementary probability |
|    6 | `graph`       | graphs and graph algorithms, graph signal processing                                                                                                                                             |
|    6 | `dists`       | distributions, bijectors, exponential families, GLM families, copulas                                                                                                                            |
|    6 | `information` | entropy, divergences, mutual information estimators, source and channel coding, compression                                                                                                      |
|    6 | `systems`     | linear time-invariant systems: representations, responses, stability, discretisation                                                                                                             |
|    7 | `stats`       | descriptive statistics, tests, resampling, density estimation, smoothing bases                                                                                                                   |
|    7 | `optim`       | optimisers, minimisation, programming, sparse approximation, transport, implicit differentiation                                                                                                 |
|    7 | `kernels`     | kernel functions, compositions and explicit feature maps                                                                                                                                         |
|    7 | `signal`      | signal and image processing, spectral and time-frequency analysis, audio features                                                                                                                |
|    7 | `text`        | normalisation, tokenisation, subwords, shingles, vectorisers, similarity, parsing, fingerprinting                                                                                                |
|    7 | `search`      | state-space search, local and evolutionary search, adversarial search, constraint satisfaction, planning, refinement search                                                                      |
|    7 | `stochastic`  | stochastic processes and stochastic calculus, SDE solvers, point processes, DPPs, random measures                                                                                                |
|    7 | `sampling`    | Monte Carlo samplers and diagnostics                                                                                                                                                             |
|    8 | `logic`       | terms, unification, resolution, Datalog, SAT, first-order proving, ILP                                                                                                                           |
|    8 | `dynamics`    | ODE and PDE solvers, nonlinear dynamics, control, filtering, system identification                                                                                                               |
|    8 | `pgm`         | graphical models, their inference and learning, causal graphs                                                                                                                                    |
|    8 | `metrics`     | evaluation metrics in sixteen categories, with intervals and comparisons                                                                                                                         |
|    8 | `losses`      | differentiable training objectives and regularisers                                                                                                                                              |
|    8 | `rl`          | environment and agent contracts, wrappers, buffers, returns, planning, off-policy evaluation                                                                                                     |
|    8 | `games`       | normal-form, extensive-form and cooperative games, equilibria, learning in games, mechanisms                                                                                                     |
|    9 | `probprog`    | probabilistic programs, model analysis, inference engines and the planner, posteriors                                                                                                            |
|    9 | `learn`       | estimator protocol, preprocessing, composition, model selection, calibration, conformal, inspection, explanation, active learning, privacy                                                       |
|    9 | `nn`          | neural networks: functional layers, modules, blocks, description, training, decoding                                                                                                             |
|   10 | `tune`        | studies, objectives, samplers, pruners, multi-fidelity schedules                                                                                                                                 |

### 4.1 `base/` (tier 0)

- `errors.ts`: `WhiteboxError` (codes: shape, dtype, value, domain, schema, convergence, singular, not-implemented,
  not-differentiable), `isWhiteboxError`, `invariant`, `assertShape`, `assertFinite`.
- `json.ts`: `Json`, `JsonSchema`, `encodeNumber`/`decodeNumber`, `toJson`/`fromJson` (non-finite numbers, -0, typed
  arrays and bigints round-trip through tagged objects).
- `results.ts`: the shared result shapes: `Estimate` (value, standard error, interval, level, method), `TestResult`
  (statistic, p-value, degrees of freedom, effect size, alternative, method, assumptions), `MetricResult` (estimate,
  breakdowns, sample size), `MetricSpec` (name, inputs, direction, range, category, concept), `formatResult`.
- `numbers.ts`: `isClose`, `allClose`, `clamp`, `kahanSum` (Neumaier), `pairwiseSum`, `maxOf`, `minOf`, `argmaxOf`,
  `argminOf` (loops, never spreading arrays into `Math.max`), `range`, `linspaceNumbers`.
- `heap.ts`: `heap`, a binary heap with a comparator (the priority queue of neighbour searches, shortest paths and
  best-first search).
- `options.ts`: shared option shapes (`Tolerance`, `Reduction`, `Alternative`, `ConfidenceLevel`) and their defaults
  (`defaultTolerance`, `resolveTolerance`, `defaultReduction`, `defaultAlternative`, `defaultConfidenceLevel`,
  `resolveConfidenceLevel`).
- `version.ts`: `VERSION`, generated from `package.json`.

### 4.2 `scene/` (tier 1): pictures and structures as data

Tier 1 rather than 0 so that it can use `base` (errors, `clamp`, `maxOf`); nothing below it draws.
- `types.ts`: `Scene`, `SceneNode` (rect, circle, ellipse, line (an arrow is a line with a head), path (a polygon is a
  closed path), text, tex, image, group, `layer3d`: a 3-D scene and camera drawn into a box), `Style` (with the default
  fill and stroke rules every renderer follows), `ThemeToken` (`fg`, `bg`, `surface`, `muted`, `grid`, `accent`,
  `tab0`..`tab9`), `Paint` (a token, `none` or a literal colour), `Pixels`, `Transform` (translate, rotate, scale).
- `build.ts`: constructors for every node (`rect`, `circle`, `ellipse`, `line`, `arrow`, `path`, `polygon`, `text`,
  `tex` (a maths label, typeset by the renderer), `image` (from pixels, or a 2-D or RGB(A) tensor or nested arrays),
  `group`), `scene`, `toPixels` (imshow-style conversion), `isScene`.
- `geometry.ts`: `bounds`, `composeTransforms`, `applyTransform`, `invertTransform`, `translate`, `rotate`, `scaleTo`,
  `fit`, `grid` (filmstrips).
- `three-d.ts`: a small 3-D layer: `Scene3D` of points, segments, arrows, polylines, faces (triangles and polygons),
  surfaces from grids and labels (`scene3d`, `point3d`, `segment3d`, `arrow3d`, `polyline3d`, `triangle3d`, `face3d`,
  `surface3d`, `label3d`, `axes3d`), for vectors, surfaces, loss landscapes and rotations without WebGL.
- `projection.ts`: `Camera` and `camera` (orthographic and perspective, orbit angles as in matplotlib's `view_init`),
  `orbit`, `fitCamera`, `projectPoint`, `project` (to a 2-D `Scene` with painter's-algorithm depth sorting and optional
  back-face culling), `layer3d`, `projectLayers` (what renderers call before drawing).
- `structure.ts`: `Structure` (typed nodes with role, label, value badges and ports; typed edges between ports; groups
  such as plates, repeated blocks and clusters; annotations), `structure` builder, `mergeStructures`, `subStructure`,
  `isStructure`.
- `timeline.ts`: `Timeline` (frames keyed by step: scene or structure states with highlights), the data behind every
  animation and export; `timeline`, `sceneTimeline`, `isTimeline`.
- `paint.ts`: `palettes` (the light and dark token colours, tab10 for the categorical tokens), `parsePaint`.
- `raster.ts`: `rasterise`, a small software rasteriser from scenes to RGBA pixels (pixel observations for RL, frames in
  Node): nonzero fills at pixel centres, strokes with dashes, joins and arrow heads, smooth paths, images, 3-D layers,
  supersampling. TODO: text and TeX labels are skipped (they need a bundled bitmap font).

oracle: none (no reference library shares these conventions); checked by unit tests and properties (transforms
compose, projections are affine or perspective-consistent, painter's order, exact pixel coverage)

### 4.3 `tensor/` (tier 1): arrays and differentiable primitives (numpy, torch.Tensor, jax.numpy)

oracle: numpy, jax for derivative rules
- `dtype.ts`: `DType` (float64, float32, int32, bool; complex values are pairs, see `complex.ts`), `Storage<D>`,
  `allocate`, `isFloat`, `resultType` (explicit promotion table), `weakResultType`, `finfo`, `iinfo`.
- `shape.ts`: `Shape`, `sizeOf`, `stridesOf`, `broadcastShapes`, `normaliseAxis`, `normaliseAxes`, `shapesEqual`,
  type-level shape helpers for literal shapes.
- `tensor.ts`: `Tensor<D, S>` (dtype, shape, contiguous row-major storage, backend), `tensor`, `asTensor`
  (`TensorLike` in), `fromStorage`, `isTensor`, `item`, `toArray`, `toFlat`, `clone`.
- `create.ts`: `zeros`, `ones`, `full`, the `*Like` variants, `arange`, `linspace`, `logspace`, `geomspace`, `eye`,
  `diag`, `tri`, `tril`, `triu`, `meshgrid`, `indices`.
- `primitive.ts`: `Primitive` (name, kernel, shape and dtype rules, `jvp`, `vjp`, optional `batch` rule; no separate
  `transpose` rules: every differentiable primitive states its `vjp` directly, and `autodiff.customJvp` transposes a
  forward rule by reverse mode over it), `BatchDim`, `BatchResult`, `definePrimitive` (the extension point), `bind`
  (dispatch: concrete inputs run the kernel; traced inputs go to the highest-level trace among them), `bind1`.
- `batching.ts`: helpers for batch rules: `moveBatchAxis`, `batchedAxis`, `batchAtFront`, `alignBatched`,
  `batchElementwise` (broadcasting elementwise primitives), `batchLeading` (primitives that already map over leading
  axes, such as factorisations).
- `traced.ts`: `Value` (`Tensor` or `Traced`), the `Traced` interface (abstract shape and dtype, trace, level),
  `isTraced`, `lift` (a value into a trace, as a constant when it belongs to a lower one), `concreteOrThrow` (clear
  errors where a concrete value is required), `primalOf`.
- `backend.ts`: the `Backend` interface (the kernel vocabulary: elementwise, reduce, scan, matmul, transpose, slice,
  concat, pad, take and take-along-axis gathers with their scatter-add adjoints, where, cast, sort),
  `BackendCapabilities` (dtypes, synchronous, device, element limit), `supportsDType`. Convolutions are composed in
  `numerics/convolution` (im2col and matmul), so a backend needs no conv kernel; WASM and WebGPU backends implement the
  interface later.
- `cpu.ts` and `_cpu/elementwise.ts`, `_cpu/reduce.ts`, `_cpu/structure.ts`, `_cpu/sort.ts`: the reference CPU backend.
- `ops/arithmetic.ts`: `add`, `sub`, `mul`, `div`, `floorDiv`, `mod`, `pow`, `neg`, `abs`, `sign`, `maximum`,
  `minimum`, `clip`, `reciprocal`.
- `ops/math.ts`: exponentials, logarithms, roots, trigonometric and hyperbolic functions and inverses, `sigmoid`,
  `softplus`, `relu`, `step`, `floor`, `ceil`, `round` (special functions such as `erf` and `lgamma` are primitives defined in
  `numerics/special`).
- `ops/compare.ts`: comparisons, `isnan`, `isfinite`, logical operations, `where`, `any`, `all`, `allclose`.
- `ops/reduce.ts`: `sum`, `prod`, `mean`, `variance` (two-pass), `std`, `max`, `min`, `argmax`, `argmin`, `cumsum`,
  `cumprod`, `logsumexp` (max-shifted), `softmax`, `logSoftmax`, `norm`.
- `ops/shape.ts`: `reshape`, `flatten`, `transpose`, `swapaxes`, `moveaxis`, `expandDims`, `squeeze`, `broadcastTo`,
  `sumToShape`, `concat`, `stack`, `split`, `chunk`, `flip`, `roll`, `tile`, `repeat`, `pad`, `slidingWindowView`
  (numpy's strided windows along an axis, step and dilation; a view-like gather, differentiable).
- `ops/indexing.ts`: `slice`, `take`, `takeAlongAxis` (torch `gather`), `gather` (coordinate tuples, TensorFlow
  `gather_nd`), `scatter` (functional `x.at[...]` with set, add, mul, max, min), `scatterAdd`, `maskedSelect`,
  `nonzero`, `sort`, `argsort`, `topk`, `searchsorted`, `unique`, `bincount`, `oneHot`, `segmentSum`.
- `ops/linear.ts`: `matmul`, `dot`, `outer`, `inner`, `tensordot`, `einsum`, `kron`, `trace`, `diagonal`.
- `complex.ts`: `ComplexValue` (a pair of real values, so every real kernel and derivative rule applies),
  `isComplex`, `complex`, `real`, `imag`, `conj`, `angle`, `absComplex`, `polar`, `cAdd`, `cSub`, `cMul`, `cDiv`,
  `cScale`, `cExp`, `cis`.
- `pytree.ts`: `Tree`, `MapLeaves`, `treeLeaves`, `treeMap`, `treePaths`, `treeFlatten`, `treeUnflatten`, `treeZeros`,
  `treeCount`, `ravel` (returns the flat vector and its `unravel`).
- `format.ts`: `formatTensor` (numpy-style display), `tensorToJson`, `tensorFromJson`.

### 4.4 `random/` (tier 2): keys and samplers (jax.random, numpy.random)

oracle: Random123 known-answer vectors; scipy.stats.qmc for the sequences (unscrambled Sobol' and Halton exactly; mean
discrepancy and fill counts for the randomised ones); scipy.stats quantiles for distributional checks of every sampler
- `philox.ts`: `philox4x32`.
- `key.ts`: `Key`, `key(seed)`, `split`, `foldIn`, `keyFromPath(seed, ...names)` (named derivation), `randomBits`,
  `isKey`.
- `continuous.ts`: `uniform`, `normal`, `truncatedNormal` (Robert's samplers, any interval), `exponential`, `laplace`,
  `logistic`, `gumbel`, `cauchy`, `gamma`, `gammaVariates` (one independent draw per shape, for batched
  distributions), `beta`, `chisquare`, `studentT`, `dirichlet` (beta and Dirichlet in log space for small shapes),
  `multivariateNormal`, `vonMises`.
- `discrete.ts`: `randint`, `bernoulli`, `categorical` (probabilities or logits), `multinomial` (conditional
  binomials), `poisson` (Knuth, PTRS), `binomial` (waiting times, BTPE), `geometric`, `AliasTable`, `aliasTable`,
  `aliasSample`.
- `permutation.ts`: `permutation`, `shuffle`, `choice`, `bootstrapIndices`, `subsample` (Floyd's algorithm, sorted),
  `randomSplit`, `reservoir` (Algorithms R and L over any iterable), `keys`.
- `sequences.ts`: `sobol` (Joe-Kuo new-joe-kuo-6.21201 direction numbers, Gray-code order, `skip` as scipy's
  `fast_forward`, equal to `scipy.stats.qmc.Sobol(d, scramble=False)`), `scrambledSobol` (keyed linear matrix
  scramble plus digital shift, scipy's LMS+shift), `SobolOptions`, `halton` (skips the origin by default; `skip: 0`
  equals scipy's unscrambled Halton), `latinHypercube` (jittered or centred), `poissonDisk` (blue noise by Bridson's
  algorithm in d dimensions), `PoissonDiskOptions`. TODO: Sobol' beyond 1111 dimensions (scipy has 21201; the full
  table is about 300 KB).
- `generator.ts`: `Generator` (numpy-style wrapper that splits on each call), `defaultRng`.
- Private: `_unit.ts` (uniform doubles and streams), `_variates.ts` (scalar truncated normal, gamma, Poisson and
  binomial kernels), `_sobol.ts` (direction numbers, scramble, Gray-code generation) and `sobol-directions.data.ts`
  (the packed Joe-Kuo table for 1111 dimensions, vendored by `oracle/datasets/sobol_directions.py`).

### 4.5 `symbolic/` (tier 2): a small computer algebra system (SymPy)

For the mathematical foundations: exact expressions that can be simplified, differentiated, expanded in series,
integrated where a rule applies, typeset, and compiled into differentiable numeric functions. Every rewrite returns its
steps, so a derivation can be shown step by step.

Expressions are immutable trees in SymPy's canonical form, built by constructors that evaluate as SymPy does on
construction (flattened sums and products, combined coefficients, like terms collected, equal bases merged, exact
rational arithmetic on bigints, perfect powers pulled out of radicals, special values of functions), so structural
equality is a key comparison. Printing follows SymPy's term and factor order, so `toText` equals SymPy's `str()` and
`toLatex` its `latex()` for the same expression. Each multi-step rewrite has a `...Steps` form returning a `Derivation`
(result and `Step`s: rule, before, after).

oracle: sympy (canonical forms, printed text and LaTeX, values at sample points), numpy and scipy (roots, Pade,
lambdify values)
- `expr.ts`: `Expr` (rationals, floats, symbols with assumptions, constants pi, E, I, oo, zoo, nan, sums, products,
  powers, function applications, piecewise, relations, connectives, big-O, unevaluated derivatives and integrals),
  `ExprLike`, `Step`, `Derivation`, `isExpr`, `asExpr`, `sym`, `symbols`, `num`, `rational`, operator helpers (`add`,
  `sub`, `mul`, `div`, `pow`, `neg`, `sqrt`), the constants, `free`, `dependsOn`, `substitute`, structural `equals`.
- `functions.ts`: function constructors with automatic evaluation: `fn` (any name; unknown names are undefined
  functions), `exp`, `log` (with optional base), `root`, `abs`, `sign`, `floor`, `ceiling`, `factorial`, `gamma`,
  `erf`, `re`, `im`.
- `trigonometric.ts`: `sin`, `cos`, `tan`, `cot`, `sec`, `csc` (exact at rational multiples of pi with denominators 1, 2,
  3, 4, 6, 12, reduced by symmetry and shifts of pi/2), their inverses, `atan2`, and the hyperbolic functions and
  inverses.
- `conditions.ts`: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `and`, `or`, `not`, `piecewise` (evaluated to true or false for
  numbers; negated relations flip).
- `parse.ts`: `parse` (SymPy's text syntax with `^`, implicit multiplication and the printed forms `Piecewise`, `Eq`,
  `O`, `Integral`, `Derivative`), `parseLatex` for a useful subset (`\frac`, `\sqrt[n]`, scripts, `\left( \right)`,
  absolute values, Greek letters, named and `\operatorname` functions with powers, floor and ceiling, relations,
  implicit products), `ParseOptions`.
- `simplify.ts`: canonical ordering (`compareExprs`, SymPy's sort key), `countOps`, `expand` (multinomial, power_exp,
  log rules under assumptions), `collect`, `factor` (over the rationals, any number of variables: Berlekamp-Zassenhaus
  with Hensel lifting, Kronecker substitution for several variables), `together`, `cancel`, `apart` (partial fractions
  by undetermined coefficients, symbols as parameters), `trigSimplify` (Fu-style search over Pythagorean, double-angle,
  product-to-sum and tan rewrites, guarded numerically), `simplify` (best of the rewrites by operation count), with
  `expandSteps`, `trigSimplifySteps`, `simplifySteps`, and `rewriteSteps` (any of the rewrites by name, `Rewrite`).
- `polynomials.ts`: `Poly` (univariate and multivariate over the rationals, generators chosen automatically), `poly`,
  `polyToExpr`, `degree`, `coefficients`, `leadingCoefficient`, `polyDiv`, `polyGcd`, `polyLcm`, `exactRoots` (factor
  first, then radicals up to degree four and binomials of any degree; three real roots of a cubic in trigonometric
  form), `Root`, `sturm`, `countRealRoots`, `realRootIntervals`, `resultant`, `discriminant`.
- `differentiate.ts`: `diff` (orders and mixed partials; derivatives of every known function, undefined functions,
  piecewise, Leibniz's rule for integrals), `diffSteps`, `gradient`, `jacobian`, `hessian`, `implicitDiff`,
  `implicitDiffSteps`, `DiffSpec`.
- `series.ts`: `series` (Taylor, Laurent at poles, at points and at infinity, log terms), `seriesSteps`, `taylor`,
  `seriesCoefficients`, `residue`, `pade`, `bigO`, `removeO`, `Direction`. TODO: Puiseux series (fractional powers at
  branch points) and expansions at essential singularities.
- `limits.ts`: `limit` (direct substitution, series leading terms, continuity of composed functions, l'Hopital's rule
  for 0/0, oo/oo and 0*oo, exp(lim g log f) for powers, with a numeric cross-check), one-sided and two-sided limits,
  `limitSteps`, `LimitDirection`.
- `integrate.ts`: `integrate` (linearity; table with linear arguments, Gaussians via erf, quadratic radicals; rational
  functions by partial fractions with the reduction formula for repeated quadratics; powers and products of sines and
  cosines; reduction formulas for tan, cot, sec, csc; tabular integration by parts for polynomials times exp, sin,
  cos, sinh, cosh; derivative-divides substitution; integration by parts; every candidate verified by differentiation;
  the search is bounded in attempts, depth and expression size, so any integrand returns within milliseconds;
  returns an unevaluated `Integral` otherwise, term by term), `integrateSteps`,
  `definiteIntegral` (limits of the antiderivative, checked against adaptive Gauss-Kronrod with a numeric fallback;
  poles of rational integrands in the interval are an error), `definiteIntegralSteps`, `DefiniteOptions`. TODO: the
  Risch algorithm; Rothstein-Trager for rational functions with irreducible denominators of degree above two;
  antiderivatives in special functions other than erf (Ei, li, Si, Ci, erfi, Fresnel S and C, polylogarithms), so
  for example exp(x/2)*log(x^2 + 1), sin(x)/x and exp(x^2) stay unevaluated.
- `solve.ts`: `solve` (polynomials of any degree whose factors over the rationals have degree at most four, rational
  equations, products, polynomials in exp(x) or another single function of x, invertible a*h(u) + b = 0 with SymPy's
  principal solutions), `solveSteps`, `solveLinearSystem` (Gauss-Jordan over expressions, underdetermined systems in
  terms of the free unknowns).
- `matrices.ts`: `Matrix`, `matrix`, `identityMatrix`, `transpose`, `matMul`, `det` and `charpoly` (Berkowitz,
  division-free), `inverse` (adjugate), `nullspace`, `eigenvals`, `eigenvects`, `Eigenvector`.
- `latex.ts`: `toLatex` (expressions and matrices), `toText`, `stepsToLatex` (derivations as `aligned` blocks).
- `compile.ts`: `compile(expr, symbols, options)` to a function over tensors and traced values built from tensor
  primitives (so a symbolic expression can be plotted, differentiated by `grad`, vectorised and used as an objective;
  erf, gamma and undefined functions take implementations through `options.functions`), `evaluate` (real),
  `evaluateComplex`, `CompileOptions`, `TensorFunctions`, `NumericFunctions`.
- Private: `_rational.ts` (bigint rationals, integer roots and factorisation), `_node.ts` (raw nodes and keys),
  `_canon.ts` (the canonicalising constructors), `_assume.ts` (three-valued assumption queries), `_functions.ts`
  (automatic evaluation of known functions), `_order.ts` (SymPy's sort keys and printing order), `_numeric.ts` (real
  and complex evaluation, adaptive Gauss-Kronrod: numerics is a higher tier), `_measure.ts` (operation counts and
  numeric equivalence), `_expand.ts`, `_rational-functions.ts` (together, cancel, apart), `_trigsimp.ts`, `_poly.ts`
  (sparse polynomials, dense univariate helpers, generators), `_factor.ts`, `_series.ts` (truncated Laurent series
  engine), `_integrals.ts` (tables, rational and trigonometric rules), `_latex-parse.ts`.

### 4.6 `autodiff/` (tier 3): differentiation transforms (jax, torch.func)

oracle: jax and torch for derivative values; finite differences in property tests
- `reverse.ts`: the reverse-mode trace (one node per traced value, recorded by the values themselves), `backward` (the
  reverse sweep), higher-order reverse mode by nesting.
- `forward.ts`: the forward-mode trace (dual numbers).
- `taylor.ts`: Taylor-mode differentiation (jets, a truncated power series per value): the jet trace, `jet` (the
  `jax.experimental.jet` convention: derivative coefficients in and out), `taylorCoefficients` (normalised coefficients
  of any order along a direction, in one pass), `taylorPolynomial` for numeric functions; used by Taylor-series figures
  and Taylor ODE methods. Core primitives have recurrences in `_jet-rules.ts` (products, quotients, powers, roots,
  exponentials, logarithms, trigonometric and hyperbolic functions and their inverses, matrix products, `where`;
  linear and structural primitives map each coefficient through their jvp rule); any other primitive propagates the
  series through its own jvp rule (y' = jvp(x, x'), one pass of order K - 1), so the cost stays polynomial in K.
- `transforms.ts`: `vjp`, `jvp`, `linearize`, `grad`, `valueAndGrad` (over pytrees, leaves labelled by path),
  `gradAndAux`, `jacfwd`, `jacrev`, `jacobian`, `hessian`, `hvp` (forward over reverse), `vhp` (reverse over reverse).
- `custom.ts`: `customVjp`, `customJvp` (reverse mode transposes the forward rule by reverse mode over it, so no
  transpose rules are needed), `stopGradient`, `checkpoint` (rematerialisation: the backward pass recomputes the
  function; policies `nothing`, `matmuls`, `everything` or a predicate on primitives choose which intermediate
  results are saved and replayed instead), `straightThrough` (identity-gradient estimator).
- `vmap.ts`: the batching trace and `vmap` (in-axes per argument, out-axes per output; composes with every other
  transform in any order). Primitives with a `batch` rule run once on the whole batch; others loop over the examples
  (slice, apply, stack); a function that needs concrete values of a batched argument (data-dependent control flow)
  is mapped by a loop over the examples.
- `graph.ts`: `computationGraph` (a function traced at an input to a `Structure` with ops, shapes, values and, for a
  scalar output, the gradient with respect to every node), `graphStats` (counts, depth, operations by primitive).
- `check.ts`: `numericalGradient`, `gradCheck`, `dotProductTest` (the adjoint test `<u, J v> = <J^T u, v>`),
  `checkPrimitive` (the conformance harness every primitive must pass: finite-difference vjp and jvp, the dot-product
  test, vmap against a loop with the batch first and last, inputs not mutated).

### 4.7 `trace/` (tier 3): recording what algorithms do

- `events.ts`: event types `run`, `metric`, `tensor` (role: param, grad, state, activation, output, with full values,
  a summary, or a summary plus an indexed subsample), `message`, `episode`, `frame`, `checkpoint`, `log`, `end`; `Run`;
  zod schemas.
- `schema.ts`: `TraceSchema` (declared channels with descriptions, units and directions; tensor roles and shapes with
  named sizes; message payload fields), `channels` builder, `schemaToJson`, `validateAgainstSchema`.
- `policy.ts`: `TracePolicy` (channel selection, frequency: every n, log-spaced, on change with a tolerance, first and
  last; detail: summary or full; gradients on or off through the `grad` role; event and byte budgets; tensor
  subsampling; checkpoint interval; the `inner` policy of child runs), `policies` presets (`minimal`, `standard`,
  `debug`, `teaching`), `tracePolicy`, `shouldRecord` (the single recording rule, built on `channelSelected`,
  `onSchedule`, `selected` and `changed`), `childPolicy` (propagation to inner algorithms).
- `summaries.ts`: tensor summaries (norm, mean, standard deviation, extremes, histogram, sparsity) used at summary
  detail (`summarise`), and `subsample`.
- `tracer.ts`: `Tracer`, `nullTracer`, `recorder`, `tee`, `filtered`, `startRun`, `RunHandle` (applies the policy).
- `algorithm.ts`: `Algorithm<S>` (`init`, `step`, `done`, `observe`, `schema`, `serialise`), `StepContext`,
  `runAlgorithm`, `iterate` (a generator for live views), `runFor` (budgeted slices that continue exactly; the budget
  is a step count or milliseconds of an injected clock, since published code reads no clocks).
- `nesting.ts`: child runs (`subRun`, keyed from the outer step, with the policy `childPolicy` derives); run trees are
  read back with `runTree` in `query.ts`.
- `taps.ts`: `tapContext` (`taps.tap(name, value)` inside model code, `TapContext`), `tapsFor(run, step)` (records
  activations when the policy asks).
- `instrument.ts`: `instrument(fn, options)` (trace calls of any function: numeric inputs and outputs as metrics or
  tensor summaries, with a generated schema; timing excluded unless a clock is injected), `traceCall` (step records
  for direct methods).
- `checkpoint.ts`: `checkpointEvery` (a policy storing checkpoints every n steps), `resume`, `seek`.
- `replay.ts`: `exportRun`, `parseRun`, `replay`, `mergeRuns`, `isRun`.
- `query.ts`: `metricSeries`, `tensorSeries`, `messages`, `episodes`, `frames`, `runTree`, `summary`, `diffRuns`.

oracle: numpy for `summaries.ts` (moments, norms, `np.histogram`); the rest by unit and property tests

### 4.8 `space/` (tier 3): typed spaces (gymnasium.spaces, Optuna distributions, ConfigSpace)

- `types.ts`: `Space` union (`real` with bounds, log scale and step; `int`; `choice`; `bool`; `box`; `discrete`;
  `multiDiscrete`; `multiBinary`; `tuple`; `dict`; conditional dimensions through `when` on a dict child, which may
  chain), metadata (label, description, unit, and a `default` on every kind), with a builder per kind.
- `operations.ts`: `sample(space, key)`, `contains`, `clip`, `defaults`, `flatDim`, `flatten`, `unflatten`,
  `activeDims` (dotted paths of the active dimensions of a value).
- `encoding.ts`: `encode`/`decode` to the unit cube (every kind, inactive keys at 0.5), `encodedDim`, `grid`
  enumeration (tuples, nested and conditional dicts), `neighbours` (one-move neighbourhoods for local search over
  configurations, repairing conditional dimensions), `distance`.
- `_active.ts` (internal): resolution of conditional dict keys shared by the operations.

oracle: none; checked by properties (samples, neighbours and decoded values stay in the space, encodings round-trip)

### 4.9 `context/` (tier 4): reproducibility and defaults

- `context.ts`: `Context` (seed, key derivation, tracer, trace policy, backend, default dtype, tolerances),
  `createContext`, `ctx.key(...names)`, `derive(ctx, name)` (child contexts with a namespaced key path).
- `default.ts`: `defaultContext`, `setDefaultContext` (the one documented piece of module state; read only by
  convenience layers).
- `resolve.ts`: `resolveKey(options, name)`, `resolveTracer(options)`, `resolvePolicy(options)`: the uniform rules every
  high-level function uses.
### 4.10 `numerics/` (tier 5): numerical methods (scipy, numpy.linalg)

oracle: numpy, scipy

Every routine accepts traced values. The derivative rule is stated per folder: "composed" means it follows from the
primitives; "custom" means a dedicated rule.

`special/` (scipy.special; each function is a primitive with an analytic derivative)
- `erf.ts`: `erf`, `erfc`, `erfcx`, `erfinv`, `erfcinv`, `dawsn`.
- `gamma.ts`: `gamma`, `lgamma`, `digamma`, `trigamma`, `polygamma`, `beta`, `lbeta`, `factorial`, `lfactorial`,
  `binom`, `lbinom`, `rgamma`.
- `incomplete.ts`: `gammainc`, `gammaincc`, their inverses, `betainc`, `betaincinv`, log versions.
- `normal.ts`: `ndtr`, `logNdtr`, `ndtri`, `normalPdf`, `normalLogPdf`, `owensT`, `logIntervalProbability`.
- `logistic.ts`: `expit`, `logit`, `softplus`, `logSigmoid`, `log1mexp`, `log1pexp`, `logaddexp`, `xlogy`, `xlog1py`,
  `entr`, `relEntr`.
- `bessel.ts`: `i0`, `i1`, `i0e`, `i1e`, `iv`, `kv`, `jv`, `yv`, `logIv`.
- `elliptic.ts`: complete and incomplete elliptic integrals (`ellipk`, `ellipe`, `ellipkinc`), Jacobi elliptic functions
  (the exact pendulum period and motion).
- `zeta.ts`: `zeta`, `hurwitzZeta`, `lambertW`, `expn`, `expi`, `sici`, `fresnel`.
- `cdf.ts`: CDFs and inverses for test statistics (`stdtr`, `stdtrit`, `chdtr`, `chdtrc`, `fdtr`, `kolmogorov`,
  `smirnov`). TODO: the studentised range distribution (deferred: a nested-quadrature CDF too slow to test).

`linalg/` (numpy.linalg, scipy.linalg; custom rules for every factorisation and solve)
- `blas.ts`: level-1 and level-2 kernels on number arrays (`ddot`, `dnrm2`, `daxpy`, `dscal`, `dgemv`), shared by every
  algorithm that iterates on primal values (Krylov, Lanczos, regularising iterations, root finders).
- `basics.ts`: `norm`, `cond`, `matrixRank`, `matrixPower`, `det`, `slogdet`, `isSymmetric` (`trace` is the tensor op),
  `isPositiveDefinite`.
- `triangular.ts`: `solveTriangular`, `invTriangular`.
- `cholesky.ts`: `cholesky` (optional jitter ladder that reports the jitter used), `choFactor`, `choSolve`,
  `choleskyLogdet`, rank-one `choleskyUpdate` and `choleskyDowndate`.
- `lu.ts`: `lu`, `luFactor`, `luSolve`, `solve`, `inv`.
- `qr.ts`: `qr` (Householder), `lstsq`, `householderReflector`, `givensRotation`.
- `orthogonalisation.ts`: `gramSchmidt` (classical and modified, with projection step records, reorthogonalisation), `orth`.
- `eigh.ts`: `eigh` (tridiagonal reduction and implicit QL by default, or cyclic Jacobi), `eigvalsh`,
  `jacobiEigenSteps` (sweeps and rotations for figures).
- `svd.ts`: `svd` (one-sided Jacobi by default, or Golub-Kahan-Reinsch), `svdvals`, `pinv`, `nullSpace`, `polar`, `lowRankApproximation`.
- `eig.ts`: `hessenberg`, `schur`, `eig`, `eigvals` (complex results), `powerIteration`, `inverseIteration`,
  `rayleighQuotientIteration`.
- `functions.ts`: `expm` (Pade, scaling and squaring, Frechet derivative), `logm`, `sqrtm`, `funm` (Schur-Parlett),
  `funmSymmetric`.
- `equations.ts`: `solveSylvester`, `solveContinuousLyapunov`, `solveDiscreteLyapunov`, `solveCare`, `solveDare`,
  `signm` (the matrix sign function, for the Riccati solver and spectral projectors).
- `structured.ts`: `solveTridiagonal`, `solveBanded`, `toeplitz`, `solveToeplitz`, `circulant`, `hankel`, `blockDiag`,
  `kroneckerSolve`.
- `randomized.ts`: `randomizedSvd`, `randomizedRangeFinder`, `randomProjectionMatrix`, `johnsonLindenstraussDim`,
  `hutchinsonTrace`, `stochasticLogdet`.

`sparse/` (scipy.sparse)
- `csr.ts`: `CsrMatrix`, `csr`, `fromCoo`, `toDense`, `spmv`, `spmm`, `transposeCsr`, `diagCsr`, `csrAdd`,
  `csrDiagonal` (differentiable in the values), `LinearOperator` and `asOperator`.
- `iterative.ts`: `cg`, `bicgstab`, `gmres` (restarted), `minres`, each also an `Algorithm` (`cgAlgorithm`, ...) with
  traced residuals; preconditioners (`jacobiPreconditioner`, `incompleteCholesky`); `sparseSolve` (custom rules by
  adjoint solves).
- `eigen.ts`: `lanczos` (full reorthogonalisation, deflation), `arnoldi`, `eigsh` (top k, with locking for repeated
  eigenvalues), `svds`.

`polynomial/` (numpy.polynomial; composed)
- `power.ts` (lowest degree first, as numpy.polynomial): `polyval`, `polyvander`, `polyfit`, `polyder`, `polyint`,
  `polyadd`, `polysub`, `polymul`, `polydiv`, `polytrim`, `polyroots`, `polyFromRoots`, `companion`, `charpoly`
  (numpy.poly of a matrix, with an exact derivative rule).
- `orthogonal.ts`: Legendre, Chebyshev, Hermite (both), Laguerre and Jacobi families by their recurrences:
  `orthoval` (Clenshaw), `orthovander`, `orthofit`, `orthogauss`, `orthoToPower`, `powerToOrtho`.
- `rational.ts`: `residue`, `residuez`, `invres`, partial fractions.

`quadrature/` (scipy.integrate; differentiable in the integrand's parameters)
- `newton-cotes.ts`: `trapezoid`, `simpson`, `cumulativeTrapezoid`, `romberg` (step records).
- `gauss.ts`: `gaussLegendre`, `gaussHermite`, `gaussLaguerre`, `gaussJacobi`, `fixedQuad`, `normalExpectation`.
- `adaptive.ts`: `quad` (Gauss-Kronrod G7-K15, globally adaptive, infinite limits, breakpoints, traced intervals),
  `adaptiveSimpson`, `tanhSinh`.
- `multidim.ts`: `dblquad`, `tplquad`, `nquad`, `productGaussRule`, `sparseGridRule` (Smolyak), `cubature`.
- `monte-carlo.ts`: `monteCarlo`, `quasiMonteCarlo` (shifted Halton replicates) with standard errors,
  `importanceIntegral`.

`roots/` (scipy.optimize root finding; custom rules by the implicit function theorem)
- `scalar.ts`: `bisect`, `brentq`, `ridder`, `newton`, `secant`, `halley`, `toms748`, with evaluation histories;
  `scalarRootOf` (implicit-function-theorem derivatives).
- `systems.ts`: `fsolve` (damped Newton with backtracking), `broyden`, `fixedPoint` (Steffensen and Anderson
  acceleration), `root`, `rootOf` (implicit-function-theorem derivatives in both modes).
- `continuation.ts`: `homotopy`, `arcLengthContinuation`, fold and branch-point detection.

`interpolate/` (scipy.interpolate; differentiable in data values and query points)
- `piecewise.ts`: `ppoly` (`PPoly`: evaluate, derivative, antiderivative, integrate, roots; periodic extrapolation).
- `splines.ts`: `cubicSpline` (natural, clamped, not-a-knot, periodic, explicit), `cubicHermiteSpline`, `bspline`,
  `makeInterpSpline`, `makeLsqSpline`, `notAKnot`.
- `shape-preserving.ts`: `pchip`, `akima` (and makima).
- `smoothing.ts`: `makeSmoothingSpline` (GCV), `univariateSpline`.
- `polynomial.ts`: `barycentric`, `barycentricWeights`, `lagrange`, `chebyshevNodes`, `equispacedNodes`,
  `chebyshevInterpolant`, `lebesgueConstant`, `runge`.
- `grid.ts`: `interp`, `interp1d`, `regularGridInterpolator`, `griddata` (nearest, linear, natural).
- `scattered.ts`: `rbfInterpolator`, `naturalNeighbourWeights` and `naturalNeighbour` (Sibson),
  `inverseDistanceWeighting`.

`differentiate/`
- `finite-differences.ts`: `derivative` (with Richardson extrapolation), `gradient` (numpy), `jacobianFd`, `hessianFd`,
  `finiteDifferenceWeights` (Fornberg), complex-step derivatives, `differenceMatrix`.

`approximation/` (numerical analysis foundations; chebfun ideas)
- `chebyshev.ts`: `chebfun` (adaptive Chebyshev approximation: construct, evaluate, differentiate, `cumsum`, `sum`,
  roots by the colleague matrix with subdivision, extrema), `chebyshev`, `chebyshevCoefficients`, coefficient decay.
- `best.ts`: `remez` (minimax approximation by the Remez exchange algorithm, polynomial and rational types),
  equioscillation, `chebyshevSeries`.
- `pade.ts`: `pade` (approximants from Taylor coefficients), `continuedFraction` (modified Lentz).
- `acceleration.ts`: `richardson`, `aitken`, `shanks`, `wynnEpsilon`, `eulerTransform`.
- `floating-point.ts`: `ulp`, `nextAfter`, `machineEpsilon`, `toFloat32`, `twoSum`, `twoProduct`, `compensatedSum`
  (naive, Kahan, and base's Neumaier and pairwise sums), `propagateError`, `conditionNumber`, `significantDigits`.

`fourier/` (numpy.fft, scipy.fft; linear, so differentiable; complex pairs come from `tensor/complex.ts`)
- `fft.ts`: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `fftn`, `ifftn`, `fft2`, `ifft2` (radix 2 and Bluestein).
- `helpers.ts`: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `nextFastLen`, `dft`, `dftMatrix`.
- `trig.ts`: `dct`, `idct`, `dst`, `idst` (types I to IV), `mdct`, `imdct`.
- `nonuniform.ts`: non-uniform DFTs (types 1 and 2), Gaussian-gridding NUFFTs, the chirp-z transform and `zoomFft`.

`convolution/` (numpy, scipy.signal, scipy.ndimage; differentiable)
- `one-d.ts`: `convolve`, `correlate`, `fftConvolve`, `oaConvolve`, `correlationLags`.
- `n-d.ts`: `convNd` (stride, dilation, groups, padding), `convTranspose`, `convOutputSize`, `im2col`, `col2im`.
- `padding.ts`: `padArray` (numpy.pad modes: constant, edge, reflect, symmetric, wrap, maximum, minimum, mean),
  `ndimage` border modes (`padModeOf`), `extendIndex`.

`geometry/` (scipy.spatial)
- `polygons.ts`: `convexHull` (2-D), `convexHull3` (3-D, incremental), `polygonArea`, `polygonCentroid`,
  `polygonPerimeter` (differentiable in the vertices), `isConvex`, `windingNumber`, `pointInPolygon`,
  `segmentIntersection`, `polygonClip` (Sutherland-Hodgman), `minkowskiSum` (convex).
- `distances.ts`: `cdist`, `pdist`, `squareform` with the metrics euclidean, sqeuclidean, cityblock, chebyshev, minkowski
  (with weights), seuclidean, cosine, correlation, hamming, jaccard, mahalanobis, canberra, braycurtis and haversine
  (continuous ones differentiable).
- `triangulation.ts`: `delaunay` (neighbours, hull, point location), `voronoi`, `barycentricCoordinates`, `alphaShape`,
  predicates (`orient`, `inCircle`, `circumcentre`).
- `transforms.ts`: `Rotation` (`rotationFromQuat`, `rotationFromMatrix`, `rotationFromRotvec`, `rotationFromEuler`
  with all 24 sequences, `asEuler`, `compose`, `inv`, `apply`, `slerp`), `affine` maps, `homography` (normalised DLT),
  `applyHomography`, `fundamentalMatrix` (normalised eight-point), `normalisePoints`, `orthogonalProcrustes`,
  `procrustes`, `kabsch`.
- `contours.ts`: `contourLines` (marching squares as skimage.find_contours, saddle connectivity, orientation,
  coordinates), `contourLevels`, `isolines`, `covarianceEllipse` (differentiable).

`spatial/` (scipy.spatial, sklearn.neighbors, faiss ideas)
- `brute.ts`: `kNearest` (differentiable distances), `radiusNeighbours`, `kNeighboursGraph`, `radiusNeighboursGraph`.
- `trees.ts`: `kdTree`, `ballTree`, `vpTree` (exact k-NN, radius and count queries; any true metric for ball and VP
  trees), with query logs (visited, pruned, distance evaluations) and node descriptions for figures.
- `hashing.ts`: LSH families (`randomHyperplanes`, `pStableHash`), `collisionProbability`, MinHash (`minHash`,
  `estimateJaccard`, `minHashBanding`, `bandingProbability`), `lshIndex` with multi-probe (`probeSequence`).
- `graphs.ts`: `hnswIndex` (insertion, layered search with logs), `nnDescent` and `nnDescentAlgorithm` (traced).
- `quantisation.ts`: `kmeansCodebook` (k-means++ and Lloyd; the vector-quantisation trainer that
  `methods/clustering/kmeans.ts` builds on), `productQuantiser` (encode, decode, asymmetric and symmetric distances),
  `ivfIndex` (IVF and IVF-PQ).

`inverse/` (ill-posed problems)
- `tikhonov.ts`: `spectralData`, `tikhonov` (standard form by the SVD), `tsvd`, `tikhonovGeneral` (general form with a
  regulariser L, as stacked least squares), `secondDifference`, `filterFactors`, `tikhonovPath`.
- `iterative.ts`: `landweberAlgorithm`, `cglsAlgorithm`, `lsqrAlgorithm` (traced residual, solution and error norms,
  showing semi-convergence), `regularise`.
- `parameter-choice.ts`: `lCurve` (Hansen's curvature), `gcv`, `discrepancyPrinciple`, `quasiOptimality`,
  `lambdaGrid`.

### 4.11 `data/` (tier 5): datasets, generators and truth (torch.utils.data, sklearn.datasets)

One schema per task kind. Each schema declares its per-row fields, so one `take` and one `split` work for all of them.
Every schema file also has a validating constructor that fills defaults (names, metadata). `schema/` and `modifiers/`
are inlined into `whitebox-engine/data`. Each modifier also comes as a recipe `Modifier` (`labelNoiseModifier`,
`missingModifier`, and so on).
- `schema/common.ts`: `DatasetMeta` (name, description, task, source, url, licence, citation, sha256, recipe, key),
  `DatasetKind`, `Dataset` (the union of every schema), `RecipeStamp`, `perRowFields`, `rowCount`.
- `schema/tabular.ts`: `RegressionData`, `ClassificationData`, `MultilabelData`, `MultiOutputData`, `TableData` (named
  mixed-type columns: numeric, categorical, boolean, text, with missing values), `TabularData`, zod schemas
  (`TensorSchema`, `DatasetMetaSchema`, `RegressionDataSchema`, `ClassificationDataSchema`, `MultilabelDataSchema`,
  `MultiOutputDataSchema`, `TableColumnSchema`, `TableDataSchema`).
- `schema/ranking.ts`: `InteractionData` (users, items, values, times, sessions, positions, propensities), `QueryData`;
  `interactions`, `queryData`.
- `schema/corpus.ts`: `CorpusData` (documents or token sequences, labels, vocabulary), `SequenceLabellingData`,
  `ParallelData`; `corpus`, `sequenceLabelling`, `parallel`.
- `schema/sequence.ts`: `SequenceData` (input and target token sequences, masks) for sequence tasks; `sequences`.
- `schema/timeseries.ts`: `TimeSeriesData` (times, values `[n, channels]`, sampling rate, period, events, segment
  labels); `timeSeries`.
- `schema/events.ts`: `EventData` (event times or locations with marks, sequences, window and region) for point
  processes; `events`.
- `schema/comparisons.ts`: `ComparisonData` (pairwise or multi-way outcomes as finishing places, with players, ties and
  times) for rating models; `comparisons`, `pairwise`.
- `schema/graph.ts`: `GraphData` (one or many graphs, node and edge features, labels, masks), `GraphRecord`,
  `TripleData`; `graphRecord`, `graphData`, `triples`.
- `schema/image.ts`: `ImageData` (`[n, h, w, c]`, labels, boxes, masks); `images`.
- `schema/trajectories.ts`: `TrajectoryData` (observations, actions, rewards, terminations, truncations, episodes,
  behaviour propensities); `trajectories`.
- `generator.ts`: `DatasetGenerator` (name, kind, knobs as a `Space`, `generate(key, knobs) -> { data, truth }`),
  `Generated`, `generate` (resolves keys from the context, validates knobs, stamps the recipe and key),
  `describeGenerator`, `asGenerated` (a real dataset as `{ data, truth }` for modifiers).
- `truth.ts`: the `Truth` interface: `program` (the generating probabilistic program), `logProb`, `posterior`,
  `regressionFunction`, `quantile`, `noiseScale`, `bayesError`, `clusters`, `changepoints`, `latents`, `effects`
  (`CausalEffects`: ATE, ATT, CATE, ITE, potential outcomes), `structure`, `relevantFeatures`; helpers that derive
  missing pieces: `completeTruth`, `posteriorFromJoint` (Bayes' rule from the joint density), `bayesErrorEstimate`,
  `bayesPredict`, `clusterAgreement`, `excessRisk`. TODO: generators fill exact `logProb` and `posterior` functions
  but not yet `program`, and `completeTruth` cannot derive from a program, until `lib/probprog` exists.
- `rows.ts`: `take`, `trainTestSplit` (optionally stratified), `trainValTestSplit`, `stratifiedSplit`,
  `stratifiedIndices`, `bootstrap` (with out-of-bag rows), `concatRows`, `shuffleRows`, `labelsOf`.
- `tensors.ts`: `toTensors`, `fromTensors`, `xy`, `xyClasses`, `xyMultilabel`, `xyMultiOutput` (build a dataset from
  arrays), `table`, `tableToXy` (one-hot or code encoding), `describe` (pandas-style feature summaries and class
  counts).
- `loader.ts`: `DatasetRef`, `load` (embedded, remote or several remote files; sha256 verification with Web Crypto),
  `AssetCache` (`memoryCache`, `indexedDbCache`), `sha256Hex`.
- `archives.ts`: `gunzip`, `untar`, `unzip`, `parseCsv`, `parseNpy`, for parsing remote assets in the browser.
- `samplers.ts`: `Sampler`, `sequentialSampler`, `randomSampler`, `weightedRandomSampler`, `subsetRandomSampler`,
  `batchSampler`, `balancedWeights` (torch.utils.data samplers, reshuffled per epoch by derived keys).
- `batching.ts`: `batches` (batch size, shuffle by key or sampler, drop last), `epochBatches`, `epochs`, `dataLoader`,
  `collate`, `padSequences`, `sequenceLengths`, `bucketBy`.
- `recipes.ts`: `Recipe` (a generator plus modifiers with their keys, as data), `Modifier`, `builtinModifiers`,
  `makeFromRecipe`, `describeRecipe`, `encodeRecipe`, `parseRecipe`, `RecipeSchema`.
- `modifiers/labels.ts`: `withLabelNoise` (symmetric, pair-flip or a transition matrix; `noiseTransition`),
  `withPrevalence`, `withLabelShift`, `withClassImbalance` (exponential and step profiles); posteriors adjusted.
- `modifiers/features.ts`: `withMissing` (MCAR, MAR, MNAR), `withOutliers`, `withNuisanceFeatures`, `withTransform`
  (keeping the density truth consistent; `featureTransform`), `withCorrelatedFeatures` (Toeplitz and equicorrelated).
- `modifiers/shift.ts`: `withCovariateShift` (with the density ratio), `withConceptDrift` (sudden, gradual,
  recurring), `shiftedSplit`, `semiSupervisedSplit`.

### 4.12 `foundations/` (tier 6): mathematical foundations

The mathematics underneath machine learning, written so each idea can be computed, drawn and stepped through.
Functions that teach a procedure (elimination, Euclid's algorithm, Newton's method) return their steps. Geometry and
calculus on values are built from tensor operations (vectorised, differentiable, checked by `checkPrimitive`);
procedures on small exact data take plain numbers, and exact counts are BigInts. Figures come out as point data,
`Scene`/`Scene3D` nodes or `Pixels`.

oracle: sympy, numpy, scipy (and scipy.spatial.transform, sympy.geometry, sympy.stats, sympy.ntheory)
- `trigonometry.ts`: `degrees`, `radians`, `wrapAngle`, `unitCirclePoint`, `quadrant`, `referenceAngle`; quadrant-aware
  inverses (`directionAngle`, `inverseInQuadrant`, `trigSolutions` over an interval); `trigIdentities` (Pythagorean,
  double, half and triple angle, sum and difference, product to sum) and `checkIdentity` (grid evidence or a
  counterexample); `triangleCase` and `solveTriangle` (SSS, SAS, ASA, AAS and the ambiguous SSA case with 0, 1 or 2
  triangles, vertices for drawing); `lawOfSines`, `lawOfCosines`, `lawOfCosinesAngle`, `heronArea` (Kahan's stable form).
- `vectors.ts`: `dotProduct`, `crossProduct` (3-D, and the 2-D scalar), `tripleProduct`, `vectorTripleProduct`,
  `magnitude`, `unitVector`, `projection`, `scalarProjection`, `rejection`, `angleBetween` (Kahan's atan2 form),
  `orthonormalise` (MGS with reorthogonalisation, dropping dependent vectors), `isLinearlyIndependent`,
  `spanDimension`, `inSpan`, `linearCombination`; lines and planes (`lineThrough`, `planeThrough`,
  `planeFromEquation`, `planeEquation`, `pointLineDistance`, `pointPlaneDistance`, `closestPoints` (skew lines),
  `lineIntersection`, `linePlaneIntersection`, `planeIntersection`).
- `rotations.ts`: `rotation2d`, `rotate2d`, `elementaryRotation`, `isRotationMatrix`; axis-angle (`axisAngleToMatrix`,
  Rodrigues' `rotateAboutAxis`, `matrixToAxisAngle`); Euler angles (`eulerSequences`: all 24 conventions,
  `eulerToMatrix`, `matrixToEuler`, `gimbalLock` with the surviving outer-angle combination); quaternions
  (`quatMultiply`, `quatConjugate`, `quatInverse`, `quatNormalize`, `quatFromAxisAngle`, `quatToAxisAngle`,
  `quatToMatrix`, `matrixToQuat`, `quatRotate`, `quatSlerp`, `rotationDistance`); rotation averaging (`meanRotation`,
  the chordal mean of scipy's `Rotation.mean`, and `karcherMean`, the geodesic mean). Number-level conversions reuse
  `numerics/geometry/transforms.ts`.
- `coordinates.ts`: polar, cylindrical, spherical (physics convention) and geographic coordinates both ways;
  homogeneous coordinates (`toHomogeneous`, `fromHomogeneous`, `homogeneousMatrix`, `applyHomogeneous`); barycentric
  coordinates in any simplex (`barycentric`, `fromBarycentric`); change of basis (`coordinatesInBasis`,
  `changeOfBasis`, `matrixInBasis`, also the home of linear-algebra.ts's change of basis); Jacobians and Jacobian
  determinants of named or user coordinate maps by autodiff.
- `curves.ts`: parametric curves (`curveDerivatives` by nested forward mode, `sampleCurve`, `curveSpeed`, `arcLength`,
  `cumulativeArcLength`, `curvature` (signed in 2-D), `torsion`, `frenetFrame`, `osculatingCircle`, `evolute`);
  implicit curves (`traceImplicitCurve` by marching squares, `implicitCurveGeometry`); conic sections from the general
  equation (`classifyConic`: type including degenerate cases, centre, semi-axes, axis angle, vertices, foci,
  eccentricity, directrices; `conicPoints` for drawing); Bezier curves (`bernstein`, `deCasteljau` with its levels,
  `bezier`, `bezierSplit`, `bezierDerivative`, `bezierElevate`).
- `surfaces.ts`: parametric surfaces (`surfaceDerivatives`, `sphere`, `torus`, `graphSurface`, `surfaceOfRevolution`),
  `surfaceNormal`, `tangentPlane`, `firstFundamentalForm`, `secondFundamentalForm`, `gaussianCurvature`,
  `meanCurvature`, `principalCurvatures` (with directions), `surfaceArea`, `christoffelSymbols`, `geodesic` (RK4 on the
  geodesic equations), `surfaceFaces` (scene faces, optionally coloured by a function of the parameters).
- `calculus.ts`: `derivatives` (Taylor mode), `limit` (one- and two-sided, at infinity; Richardson-extrapolated, with the
  raw sequence), `differenceQuotient`, `secantTable`, `riemannSum` (left, right, midpoint, trapezoid, with the pieces),
  `fundamentalTheorem`, `definiteIntegral`, `taylorApproximation` (coefficients through Taylor-mode autodiff and the
  Lagrange remainder bound), `newtonSteps`, `tangentLine`, `secondDerivativeTest`, `criticalPoints`,
  `meanValuePoints`. TODO: exact Taylor coefficients and remainders through `symbolic` once it lands (built against
  Taylor-mode autodiff for now).
- `multivariable.ts`: `gradientAt`, `directionalDerivative`, `jacobianAt`, `hessianAt` (vectorised over points, forward
  mode), `quadraticApproximation` (linear and quadratic models), `graphTangentPlane`, `classifyCriticalPoint`,
  `lagrangeMultipliers` (Newton on the stationarity system), `divergence`, `curl`, line integrals
  (`scalarLineIntegral`, `vectorLineIntegral`, `planeFlux`), surface integrals (`scalarSurfaceIntegral`,
  `surfaceFlux`), and the theorems checked numerically (`greensTheorem`, `stokesTheorem`, `divergenceTheorem`).
- `series.ts`: `sequenceTerms`, `partialSums` (compensated), `seriesSum` (Wynn's epsilon, or Richardson in 1/N for
  logarithmic convergence); convergence tests with their evidence (`nthTermTest`, `ratioTest`, `rootTest`,
  `integralTest`, `pSeriesTest`, `limitComparisonTest`, `alternatingSeriesTest`), `radiusOfConvergence`; Fourier
  series (`fourierCoefficients`, `fourierPartialSum`, `squareWaveCoefficients`, `gibbsOvershoot`); generating functions
  (`rationalSeries`, `seriesProduct`, `linearRecurrence`, `recurrenceGeneratingFunction`).
- `linear-algebra.ts`: step-by-step procedures: `rowEchelon` and `rref` (every row operation), `solveByElimination`
  (general solution), `luSteps`, `detByCofactors`, `detByElimination`, `fundamentalSubspaces` (rank, nullity, null,
  column, row and left null spaces), `basisFromVectors`, `projectionMatrix`, `leastSquaresProjection` (with the normal
  equations), `gridTransformation`, `linearMapScene` and `linearMapScene3d` (2-D and 3-D maps as grid pictures),
  `characteristicPolynomial` (Faddeev-LeVerrier), `smallEigen` (multiplicities and eigenspaces), `diagonalise`,
  `quadraticForm`, `classifyQuadraticForm`, `principalAxes`, `svdGeometry` (circle, V^T, Sigma, U stages). Change of
  basis lives in `coordinates.ts`.
- `number-theory.ts`: `euclid`, `gcd`, `lcm`, `extendedEuclid` (the Bezout table), `modulo`, `modInverse`, `modPow`
  (square and multiply, BigInt-exact), `sieve` (with smallest factors), `millerRabin` (with its rounds; deterministic
  for safe integers), `isPrime`, `trialDivision`, `pollardRho` (with its walk), `factorise`, `eulerPhi`, `divisors`,
  `chineseRemainder` (non-coprime moduli too), `continuedFractionExpansion`, `convergents`, `fibonacci` and `lucas`
  (fast doubling, BigInt).
- `combinatorics.ts`: `factorialExact`, `binomialExact`, `multinomialExact`, `permutations`, `combinations`,
  `permutationRank`/`permutationUnrank` (Lehmer codes), `combinationRank`/`combinationUnrank`, `derangements`,
  `stirling1` (signed or not), `stirling2`, `bell`, `catalan`, `partitionCount` (pentagonal recurrence),
  `integerPartitions`, `inclusionExclusion` (with Bonferroni partial sums), `pascalTriangle` (optionally modulo m).
- `complex.ts`: on complex pairs (`tensor/complex.ts` supplies the arithmetic): `toPolarForm`, `complexLog`,
  `complexPow`, `complexSqrt`, `complexSin`, `complexCos`, `nthRoots`, `rootsOfUnity`; complex functions on grids
  (`complexGrid`, `domainColouring` as scene `Pixels`); Mobius transformations (`mobius`, `mobiusApply`,
  `mobiusCompose`, `mobiusInverse`, `mobiusFixedPoints`, `classifyMobius`, `mobiusFromPoints`); conformal maps
  (`conformalMaps`, `conformalGrid`, `cauchyRiemann`); contour integrals and residues (`circleContour`,
  `contourIntegral`, `residueAt`, `laurentCoefficient`, `windingNumber`, `argumentPrinciple`).
- `probability.ts`: finite spaces (`probabilitySpace`, `productSpace`, `powerSpace`, `conditionSpace`), events
  (`eventProbability`, `conditionalProbability`), Bayes' rule calculators (`bayesRule` with joint and evidence,
  `diagnosticTest`), independence (`independent`, `mutuallyIndependent`), discrete variables (`distributionOf`,
  `expectation`, `varianceOf`, `covarianceOf`), transformations (`transformPmf`, `transformDensity` by change of
  variables), sums by convolution (`sumPmf`, `sumDensity`, `diceSum`).
- `_extrapolate.ts` (private): Richardson extrapolation of sampled sequences, shared by limits and the series tests.

### 4.13 `graph/` (tier 6): graphs and their algorithms (NetworkX, scipy.sparse.csgraph)

oracle: networkx, scipy.sparse.csgraph, numpy and scipy (heat kernels, Chebyshev filters, spectra), checked as
canonical forms where answers are not unique (sorted sets, partitions up to relabelling, modularity values, path costs,
flow and cut values with feasibility checks)

Nodes are the indices `0 .. n - 1` with labels; every function that takes a node also takes its label. Adjacency is
sorted CSR in both directions, so traversal orders match scipy.sparse.csgraph and NetworkX on graphs built in sorted
order. Iterative methods (traversals, power iterations, multilevel community searches, label propagation,
Girvan-Newman, Weisfeiler-Lehman, subspace iteration) are `Algorithm`s run by `runAlgorithm`. Matrix and spectral
functions take a graph or a dense (possibly traced, possibly batched) matrix and are differentiable and vectorisable
in it (`checkPrimitive`).
- `graph.ts`: `Graph` (nodes with labels and data, weighted edges with data, directedness, attributes, CSR adjacency
  `out`/`in`), `graph`, `digraph` (repeated edges: last, sum, min, max or error), `isGraph`, `nodeIndex`, `neighbours`,
  `successors`, `predecessors`, `degree`, `inDegree`, `outDegree`, `edgeWeight`, `hasEdge`, `edgeList`, `subgraph`,
  `reverse`, `toUndirected`, `toDirected`, `toStructure` (for drawing, with community groups and highlights),
  `GraphRunOptions`.
- `structures.ts`: `BinaryHeap` (a class over `base/heap.ts`), `UnionFind` (rank and path halving), `IndexedQueue`
  (indexed min-priority queue with decrease-key).
- `traversal.ts`: `bfs`/`bfsAlgorithm` (layer per step, several sources, direction, depth limit), `dfs`/`dfsAlgorithm`
  (tree, back, forward and cross edges, discovery and finish times), `topologicalSort` (lexicographic Kahn),
  `isDag`, `findCycle`, `bipartiteColouring`, `isBipartite`, `kHopNeighbourhood`.
- `components.ts`: `connectedComponents` (weak or strong), `isConnected`, `stronglyConnectedComponents` (Tarjan,
  Kosaraju), `isStronglyConnected`, `condensation`, `articulationPoints`, `bridges`, `biconnectedComponents`.
- `paths.ts`: `dijkstra`, `bellmanFord`, `aStar` (with the expanded set), `floydWarshall`, `johnson`, `kShortestPaths`
  (Yen), `shortestPath`, `reconstructPath` (single-source and all-pairs predecessors).
- `trees.ts`: `minimumSpanningTree` (Kruskal, Prim, Boruvka; minimum or maximum; forests), `isTree`, `isForest`,
  `rootTree` (parents, children, depths, distances), `subtreeSizes`, `treeOrder` (pre, post, level), `eulerTour`,
  `lcaIndex` (binary lifting), `lowestCommonAncestor`.
- `flows.ts`: `maxFlow` (Dinic, Edmonds-Karp), `minCut`, `minCostFlow` (negative costs saturated, then successive
  shortest paths with potentials), `maxFlowMinCost`, `bipartiteMatching` (Hopcroft-Karp).
- `centrality.ts`: `degreeCentrality`, `closenessCentrality` (Wasserman-Faust), `betweennessCentrality` (Brandes,
  weighted, endpoints), `edgeBetweennessCentrality`, `shortestPathCounts`, and traced power iterations
  `eigenvectorCentrality`, `katzCentrality`, `pagerank` (personalisation, dangling distribution), `hits`, each with
  its `*Algorithm`.
- `community.ts`: `partition`, `modularity` (resolution, directed), `louvain`/`louvainAlgorithm`, `leiden`/
  `leidenAlgorithm` (modularity or CPM, refinement with randomness theta), `labelPropagation`/
  `labelPropagationAlgorithm` (asynchronous), `girvanNewman`/`girvanNewmanAlgorithm`, `spectralBisection`;
  `_levels.ts` holds the shared level graphs and moves.
- `matrices.ts`: `adjacencyMatrix`, `adjacencyCsr`, `fromAdjacency`, `incidenceMatrix`, `degreeMatrix`, `laplacian`,
  `normalisedLaplacian`, `randomWalkLaplacian`, `laplacianCsr`, `transitionMatrix` (lazy, dangling rules),
  `modularityMatrix`, `normalisedAdjacency` (GCN and mean aggregation), `laplacianPositionalEncoding`,
  `randomWalkPositionalEncoding`.
- `spectral.ts`: graph signal processing: `graphFourierTransform`, `inverseGraphFourierTransform`, `spectralFilter`,
  `heatKernel`, `heatDiffusion`, `chebyshevFilter`, `chebyshevFilterCoefficients`, `graphWavelets` (spectral graph
  wavelets), `propagate` (weighted message passing, SGC and APPNP), `dirichletEnergy`, `fiedlerVector` (dense or
  Lanczos), `spectralEmbedding`, and the traced subspace iteration `spectralEmbeddingAlgorithm`/
  `iterativeSpectralEmbedding`.
- `measures.ts`: `density`, `eccentricity`, `diameter`, `radius`, `center`, `periphery`, `triangles`, `clustering`
  (weighted, directed), `averageClustering`, `transitivity`, `degreeAssortativity`, `numericAssortativity`,
  `attributeAssortativity`, `homophily` (edge, node, class-insensitive), `conductance`, `edgeExpansion`,
  `cheegerConstant` (exhaustive Gray-code search up to 24 nodes), `algebraicConnectivity`, `spectralGap`.
- `generators.ts`: deterministic families (`completeGraph`, `pathGraph`, `cycleGraph`, `starGraph`, `wheelGraph`,
  `gridGraph`, `hypercubeGraph`, `completeBipartiteGraph`, `balancedTree`, `barbellGraph`, `petersenGraph`,
  `karateClubGraph`) and random models from keys (`erdosRenyiGraph`, `gnmRandomGraph`, `wattsStrogatzGraph`,
  `barabasiAlbertGraph`, `stochasticBlockModel`, `randomDag`, `randomGeometricGraph`, `kNearestNeighbourGraph`,
  `randomKnnGraph`).
- `isomorphism.ts`: `weisfeilerLehmanAlgorithm`/`weisfeilerLehman` (traced colour refinement), `wlHash`, `vf2`
  (isomorphism, induced subgraph isomorphism and monomorphism, with node matching), `isIsomorphic`,
  `subgraphIsomorphisms`, `isSubgraphIsomorphic`.
- `colouring.ts`: `greedyColouring` (largest-first, smallest-last, random-sequential, independent-set,
  connected-sequential BFS/DFS, DSATUR), `isProperColouring`, `maximalIndependentSet`, `maximalCliques`
  (Bron-Kerbosch with Tomita pivoting), `maximumClique`, `cliqueNumber`.

The graph machinery in `pgm/graphs.ts` and `pgm/triangulation.ts` (tier 8) predates this domain and works on
string-named adjacency maps; `pgm.connectedComponents`, `topologicalOrder` and `maximalCliques` can later delegate to
`graph.connectedComponents`, `graph.topologicalSort` and `graph.maximalCliques`.

### 4.14 `dists/` (tier 6): distributions (scipy.stats, torch.distributions, NumPyro)

oracle: scipy.stats, torch.distributions

`continuous/`, `discrete/` and `multivariate/` are inlined into the entry point (`import { normal, gamma } from
'whitebox-engine/dists'`); the other subfolders are namespaces (`bijectors`, `families`, `compose`). Sampling is a
primitive: its kernel draws from the concrete parameter values under any trace, and its derivative rule is the
reparameterisation (`dists/_sampling.ts`).

Every distribution is a plain object from a factory with `name`, `params`, `support`, `batchShape`, `eventShape` and
pure methods `logProb`, `prob`, `cdf`, `logCdf`, `sf`, `logSf`, `ppf`, `isf`, `sample(key, shape)`, `rsample`
(reparameterised), `mean`, `variance`, `stddev`, `mode`, `entropy`. Parameters are values (tensors or traced) with batch
shapes, so densities differentiate with respect to parameters and broadcast over batches.

`rsample(key, shape)` draws samples that are differentiable functions of the parameters (pathwise gradients), and
`hasRsample` says whether a distribution supports it. `sample` is always `stopGradient(rsample)` where `rsample` exists.
Reparameterisation comes in three forms:
- Location-scale and transforms: `loc + scale * eps` (normal, laplace, logistic, cauchy, gumbel), or the inverse CDF
  applied to uniform noise (exponential, weibull, pareto, kumaraswamy). Transformed distributions inherit `rsample`
  from their base.
- Implicit reparameterisation (Figurnov et al., 2018) for distributions without a usable inverse CDF: gamma, beta,
  dirichlet, chi2 and inverse gamma, plus studentT through its gamma mixture. The sample comes from the usual sampler;
  its gradient is `-(dF/dθ) / p(x)`, which needs the derivatives of the incomplete gamma and beta functions with
  respect to their shape parameters (`numerics/special`).
- Discrete distributions have no `rsample` and throw with a pointer to the alternatives: the relaxed distributions
  `relaxedBernoulli` and `relaxedOneHotCategorical` (Concrete / Gumbel-softmax, with a temperature), straight-through
  estimators (`straightThrough`), and score-function estimators (`elbo.ts`, `policy-gradient.ts`).
- `distribution.ts`: the `Distribution` interface and `defineDistribution`, `Support` (elementwise, vector and matrix
  kinds) with `inSupport`, parameter validation (`validateParams`), batch-shape helpers (`batchShapeOf`,
  `extendedShape`, `broadcastBatch`, `sumRightmost`). Private `_numeric.ts` holds the numerical fallbacks every family shares
  (quantiles by safeguarded Newton with an implicit-function derivative, modes by search, moments and entropies by
  batched double-exponential quadrature); `_sampling.ts` the sampling primitives and `implicitReparam`.
- `continuous/normal.ts`: `normal`, `halfNormal`, `truncatedNormal`, `logNormal`, `skewNormal`.
- `continuous/heavy-tailed.ts`: `studentT`, `cauchy`, `halfCauchy`, `laplace`, `logistic`, `pareto`, `johnsonSU`,
  `sinhArcsinh`.
- `continuous/gamma.ts`: `gamma`, `inverseGamma`, `exponential`, `chi2`, `weibull`, `erlang`, `inverseGaussian`.
- `continuous/bounded.ts`: `uniform`, `beta`, `kumaraswamy`, `triangular`, `logitNormal`.
- `continuous/extreme.ts`: `gumbel`, `generalisedExtremeValue`, `generalisedPareto`, `frechet`.
- `continuous/sampling-distributions.ts`: `fDistribution`, `noncentralChi2`, `noncentralT` (Poisson-weighted series).
- `continuous/studentised-range.ts`: `studentisedRange` (double integrals by a product double-exponential rule).
- `discrete/binary.ts`: `bernoulli`, `binomial`, `betaBinomial`.
- `discrete/categorical.ts`: `categorical`, `oneHotCategorical`, `discreteUniform`, `orderedLogistic`, `orderedProbit`,
  `orderedCategorical` (any cumulative link).
- `relaxed.ts`: `relaxedBernoulli`, `relaxedOneHotCategorical` (Concrete / Gumbel-softmax) and `straightThrough`.
- `discrete/counts.ts`: `poisson`, `negativeBinomial`, `geometric`, `hypergeometric`, `zeroInflated` (any count
  base), `zeroInflatedPoisson`, `zeroInflatedNegativeBinomial`.
- `multivariate/normal.ts`: `multivariateNormal` (covariance, precision or Cholesky, batched), `condition`,
  `marginal`, `lowRankMultivariateNormal`, `matrixNormal`.
- `multivariate/simplex.ts`: `dirichlet`, `multinomial`, `dirichletMultinomial`, `logisticNormal`.
- `multivariate/matrix.ts`: `wishart`, `inverseWishart` (Bartlett reparameterised draws), `lkjCholesky`, `lkj`,
  `multigammaln`.
- `multivariate/directional.ts`: `vonMises`, `vonMisesFisher`, `powerSpherical`.
- `multivariate/copulas.ts`: `Copula`, `defineCopula`, `gaussianCopula`, `studentTCopula` (densities, reparameterised
  sampling, bivariate CDFs), `copulaDistribution` (joining marginals). TODO: elliptical copula CDFs for d > 2 need a
  multivariate normal or t CDF (Genz quasi-Monte Carlo); deferred as expensive and rarely needed.
- `multivariate/archimedean-copulas.ts`: `claytonCopula`, `gumbelCopula`, `frankCopula` in any dimension (CDFs,
  closed-form or mixed-partial densities, Marshall-Olkin and conditional sampling, Kendall's tau).
- `multivariate/copula-fitting.ts`: `pseudoObservations`, `kendallTauMatrix`, `fitCopula` (inversion of Kendall's
  tau).
- `compose/mixture.ts`: `Mixture`, `mixture`, `mixtureSameFamily`, `responsibilities` (implicitly reparameterised
  draws, numerical quantiles, mode and entropy for univariate mixtures).
- `compose/structure.ts`: `independent`, `truncated` (continuous and discrete bases), `censored`, `transformed`,
  `empirical`, `deterministic`.
- `compose/conditional.ts`: `ConditionalDistribution` (inputs to distribution), `conditional`, `linearGaussian`
  (marginal, Kalman-form posterior and joint), `glmConditional` (family plus link), `distributionalConditional`
  (GAMLSS), and heads that turn network outputs into distributions (`normalFromOutput`, `bernoulliFromOutput`,
  `categoricalFromOutput`, `poissonFromOutput`, `negativeBinomialFromOutput`, `betaFromOutput`,
  `gammaFromOutput`, `mixtureDensityFromOutput`, `multivariateNormalFromOutput`).
- `bijectors/bijector.ts`: `Bijector`, `defineBijector`, `chain`, `invert` (ranks, supports, monotonicity and shape
  maps tracked through compositions).
- `bijectors/elementwise.ts`: `identity`, `exp`, `softplus`, `sigmoid`, `tanh`, `affine`, `power`, `normalCdf`.
- `bijectors/structured.ts`: `ordered`, `stickBreaking`, `correlationCholesky`, `lowerCholesky`,
  `choleskyOuterProduct`, `affineCoupling`, `permute`, `autoregressive` (MAF-style), `madeMasks`.
- `bijectors/unconstrain.ts`: `bijectorForSupport`, `transformLogDensity`, `unconstrained`.
- `exponential-family.ts`: `exponentialFamily` (25 families): natural parameters, sufficient statistics, log
  partition, base measure, mean parameters and Fisher information (by differentiating the log partition), and
  `fromNatural`.
- `conjugacy.ts`: the conjugate table `CONJUGATES` (used by `probprog` to detect conjugacy), `conjugatePrior`,
  `conjugatePosterior` (17 likelihood-parameter pairs, including the multivariate normal mean, covariance and
  precision).
- `divergences.ts`: `klDivergence` (closed forms for 26 family pairs, independent sums, shared-bijector invariance,
  the exponential-family identity, then a Monte Carlo fallback), `klMonteCarlo`, `bhattacharyyaCoefficient`,
  `bhattacharyyaDistance`, `hellinger`, `wasserstein2Gaussian` (normal and multivariate normal).
- `families/links.ts`: `Link`, GLM link functions with inverses and first and second derivatives (`identity`, `log`,
  `logit`, `probit`, `cauchit`, `cloglog`, `loglog`, `inverse`, `inverseSquared`, `sqrt`, `power`,
  `negativeBinomialLink`, `linkByName`), numerically guarded as statsmodels guards them.
- `families/dispersion.ts`: `DispersionFamily`, exponential dispersion families for GLMs (`gaussian`, `binomial`,
  `poisson`, `gamma`, `inverseGaussian`, `negativeBinomial`, `tweedie`): variance and its derivative, unit deviance,
  deviance residuals, log-likelihood with scale and weights, starting means, IRLS weights and the response
  distribution; `tweedieDistribution` (compound Poisson-gamma, Dunn and Smyth's series).
- `families/distributional.ts`: `DistributionalFamily`, `defineDistributionalFamily` and GAMLSS-style families with
  per-parameter links (`gamlssNormal`, `gamlssStudentT`, `gamlssSkewNormal`, `gamlssGamma`, `gamlssLogNormal`,
  `gamlssWeibull`, `gamlssBeta`, `gamlssPoisson`, `gamlssNegativeBinomial`, `gamlssZeroInflatedPoisson`);
  cumulative-link ordinal families (`ordinalFamily` with logit, probit, cloglog, loglog and cauchit links).

### 4.15 `information/` (tier 6): information theory and coding

oracle: scipy.stats.entropy, scipy.special (rel_entr, kl_div), sklearn mutual information, pyitlib (plug-in
conditional and co-information), npeet for k-NN estimators, zlib and gzip (byte-exact DEFLATE), bitarray (Huffman),
reedsolo (Reed-Solomon), scikit-commpy (convolutional encoding), antropy (Lempel-Ziv, approximate and sample entropy)
- `discrete.ts`: entropy (any base, scipy.stats.entropy semantics), binary entropy, joint and conditional entropy,
  cross-entropy, KL (`relativeEntropy`; `dists.klDivergence` is the one for distribution objects), Jensen-Shannon
  (weighted divergence and distance), total variation, mutual information from tables, conditional and interaction
  (co-)information, PMI, NPMI, information gain and gain ratio; all differentiable.
- `estimators.ts`: estimators from samples: plug-in with Miller-Madow correction (entropy and mutual information),
  kernel (Gaussian KDE) estimators, transfer entropy (plug-in and KSG), variational bounds from critic scores (InfoNCE,
  NWJ, MINE).
- `neighbour-estimators.ts`: k-nearest-neighbour estimators on `numerics/spatial` KD trees: Kozachenko-Leonenko
  entropy, KSG mutual information (algorithms 1 and 2) and conditional mutual information, k-NN KL divergence, Ross's
  mixed continuous-discrete estimator.
- `divergences.ts`: the f-divergence family (named generators or any convex f), Renyi divergence and entropy, alpha
  and beta divergences, Bregman divergences (named or any convex function, gradients by autodiff), on discrete
  distributions and on grids (trapezoidal rule); all differentiable.
- `source-coding.ts`: Kraft-McMillan (Kraft sums, prefix-freeness, Sardinas-Patterson unique decodability), Huffman
  (traced tree construction, D-ary), canonical codes, Shannon-Fano (Fano's splits and Shannon's cumulative code),
  expected length, prefix encoding and decoding.
- `stream-coding.ts`: arithmetic coding (32-bit integer, pending bits) and carry-less range coding, with static and
  adaptive frequency models.
- `integer-coding.ts`: Elias gamma, delta and omega, Golomb-Rice (and the optimal parameter), run-length.
- `dictionary-coding.ts`: LZ77, LZ78, LZW, BWT with move-to-front.
- `deflate.ts`: DEFLATE (a port of zlib's compressor, byte-identical to zlib at levels 0 to 9 and every strategy, and
  an inflater with puff's checks), the zlib and gzip formats (compress and decompress), CRC-32 and Adler-32.
- `channels.ts`: channel mutual information, capacity by Blahut-Arimoto (traced, with its bounds and an active-set Newton
  refinement for optima on the boundary of the simplex), binary symmetric, erasure and Z channels, Gaussian channels
  (Shannon-Hartley) and water-filling.
- `rate-distortion.ts`: rate-distortion by Blahut's algorithm (traced) and R(D) curves, reverse water-filling, binary
  and Gaussian R(D).
- `error-correction.ts`: binary linear block codes (generator and parity-check matrices, syndromes, coset-leader
  decoding, minimum distance), parity, repetition and Hamming codes (extended too), bounds (Hamming, Singleton,
  Gilbert-Varshamov, Plotkin).
- `reed-solomon.ts`: GF(2^m) arithmetic, Reed-Solomon encoding and decoding of errors and erasures (Berlekamp-Massey,
  Chien, Forney), matching reedsolo.
- `convolutional.ts`: feed-forward convolutional codes and Viterbi decoding (traced, hard and soft decisions).
- `ldpc.ts`: Gallager's regular LDPC codes, encoding, channel LLRs, belief-propagation decoding (sum-product and
  min-sum, traced).
- `complexity.ts`: normalised compression distance (any compressor) and distance matrices, Lempel-Ziv complexity,
  block entropies and entropy rate (plug-in and of a Markov chain), approximate and sample entropy.

### 4.16 `systems/` (tier 6): linear time-invariant systems (scipy.signal LTI, python-control)

oracle: scipy.signal, python-control

Systems carry a sample time (0 continuous, positive discrete, as python-control); polynomials are highest degree
first, as scipy.signal. State-space and transfer-function values are differentiable in their matrices and
coefficients, and so are the conversions between them, the responses and the discretisations.
- `representations.ts`: transfer functions, zero-pole-gain, state space and second-order sections (`tf`, `zpk`, `ss`,
  `sos`); continuous and discrete; conversions between them (`toTf`, `toZpk`, `toSs`, `toSos`), `dimensions`,
  `isDiscrete`; minimal realisation (`minreal`: staircase reductions, pole-zero cancellation).
- `conversions.ts`: scipy's pairwise conversions: `normalize`, `abcdNormalize`, `tf2ss` (controller canonical form),
  `ss2tf` (differentiable through `charpoly`), `tf2zpk`, `zpk2tf`, `zpk2ss`, `ss2zpk`.
- `sections.ts`: second-order sections: `zpk2sos` (nearest, keep-odd and minimal pairing; analog), `sos2zpk`,
  `sos2tf`, `tf2sos`.
- `interconnect.ts`: `series`, `parallel`, `feedback`, `append`, `connect` (python-control semantics; static gains
  accepted; transfer functions combine as polynomials, multivariable systems in state space).
- `responses.ts`: impulse, step, initial-condition and arbitrary-input responses (`lsim` and the traced
  `lsimAlgorithm`, zero- and first-order hold, continuous and discrete), `defaultResponseTimes`.
- `frequency.ts`: frequency response (`freqresp`, `evalfr`, `dcgain`, `findfreqs`), Bode, Nyquist (with the
  encirclement count) and Nichols data.
- `filters.ts`: frequency responses of filters given by coefficients (`freqs` analog, `freqz` digital, FFT for FIR).
- `stability.ts`: poles and zeros (invariant zeros of square systems), `isStable`, Routh-Hurwitz (`routh`) and Jury
  (`jury`) tests, root locus (`rootLocus`: continuous branches, asymptotes, break points).
- `margins.ts`: gain, phase, stability and delay margins (`stabilityMargins`, `allMargins`, `margin`,
  `phaseCrossoverFrequencies`; python-control's polynomial method, continuous and discrete).
- `discretise.ts`: `cont2discrete` (zero-order and first-order hold, Tustin with prewarping, matched z-transform, Euler
  and backward difference, the generalised bilinear transform, impulse invariance), `c2dMatrices`.
- `analysis.ts`: controllability and observability (`ctrb`, `obsv`, `isControllable`, `isObservable`), Gramians
  (`gram`), Kalman decomposition (`kalmanDecomposition`).
- `reduction.ts`: Hankel singular values (`hsvd`), balanced realisation (`balancedRealization`) and truncation
  (`balred`: truncate or match dc, unstable parts kept).
- `norms.ts`: H2 and H-infinity norms (`h2Norm`, `hinfNorm` by Bruinsma-Steinbuch, `systemNorm`).
- `transforms.ts`: difference and differential equations to transfer functions and back (`differenceToTf`,
  `tfToDifference`, `differentialToTf`, `tfToDifferential`, with LaTeX), partial-fraction inversion
  (`inverseLaplace`, `inverseZ`).
- `pairs.ts`: z-transform and Laplace transform pairs as data with evaluators and rational forms (`laplacePairs`,
  `zTransformPairs`, `pairToTf`). TODO: derive them with `symbolic` once that domain exists.
- TODO: invariant zeros of non-square multivariable systems (they need the staircase reduction of the system pencil).
- Pole placement and LQR live in `dynamics/control` (section 4.26).

### 4.17 `stats/` (tier 7): statistics (scipy.stats, statsmodels, pingouin)

oracle: scipy.stats, statsmodels, pingouin

Every test in `tests/` returns a `TestResult` with the statistic, p-value, effect size and interval, the alternative,
the method name and the assumptions it checked.
- `descriptive.ts`: `mean`, `variance` (two-pass, `ddof`), `std`, `skewness`, `kurtosis`, `moment`, weighted versions,
  `describe`, `zscore`, `sem`, `mode`, `geometricMean`, `harmonicMean`.
- `running.ts`: Welford moments (`push`, `merge`), P-squared quantiles, exponentially weighted moments.
- `quantiles.ts`: `quantile` (the nine Hyndman-Fan methods), `median`, `iqr`, `percentileOfScore`, `ecdf`,
  `medianAbsoluteDeviation`.
- `histograms.ts`: `histogram`, `histogramBinEdges` (auto, Freedman-Diaconis, Scott, Sturges, Doane, Knuth),
  `histogram2d`, `binnedStatistic`.
- `correlation.ts`: `pearson`, `spearman`, `kendallTau`, `cov`, `corrcoef`, `partialCorrelation`, `pointBiserial`,
  `distanceCorrelation`, each with intervals.
- `covariance.ts`: `empiricalCovariance`, `ledoitWolf`, `oas`, `shrunkCovariance`.
- `density.ts`: `gaussianKde` (univariate and multivariate), bandwidth rules and cross-validation, boundary correction.
- `ranks.ts`: `rankdata`, tie corrections.
- `transforms.ts`: `boxcox`, `yeojohnson` (maximum-likelihood lambda), inverses, `standardise`.
- `tests/result.ts`: `TestResult` helpers, alternatives, `formatTest`, assumption checks shared by tests.
- `tests/means.ts`: one-sample, independent (pooled and Welch) and paired t tests, z tests, one-way and Welch ANOVA,
  Tukey HSD, repeated-measures ANOVA.
- `tests/contingency.ts`: chi-square and G tests, Fisher's exact test, McNemar, Cochran's Q, odds and risk ratios.
- `tests/nonparametric.ts`: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman, Nemenyi, sign test,
  Brunner-Munzel.
- `tests/distribution.ts`: one-sample Kolmogorov-Smirnov, Anderson-Darling, Shapiro-Wilk, Jarque-Bera, chi-square
  goodness of fit, Lilliefors.
- `tests/two-sample.ts`: two-sample Kolmogorov-Smirnov, Cramer-von Mises, k-sample Anderson-Darling, energy distance,
  Wasserstein permutation tests, Epps-Singleton, classifier two-sample tests (any classifier passed in).
- `tests/kernel.ts`: kernel two-sample (MMD with permutation and asymptotic nulls) and independence (HSIC) tests; they
  take any kernel and default to a Gaussian kernel with the median heuristic.
- `tests/independence.ts`: distance covariance test, Hoeffding's D, mutual-information permutation test, conditional
  independence tests (partial correlation, G-test, kernel CI) used by causal discovery.
- `tests/multivariate.ts`: Hotelling's T-squared, MANOVA, Box's M, Mardia's multivariate normality.
- `tests/proportions.ts`: binomial test, proportion z tests, proportion intervals (Wald, Wilson, Agresti-Coull,
  Clopper-Pearson, Jeffreys), difference of proportions.
- `tests/timeseries.ts`: Ljung-Box, Durbin-Watson, augmented Dickey-Fuller, KPSS, runs test.
- `tests/equivalence.ts`: two one-sided tests (TOST) for means and proportions.
- `tests/bayesian.ts`: Bayes factors for t tests (JZS), binomial and correlation tests, Bayesian A/B comparison of
  proportions, region of practical equivalence.
- `tests/sequential.ts`: SPRT, group-sequential alpha spending, always-valid p-values and confidence sequences.
- `tests/multiple.ts`: Bonferroni, Holm, Hochberg, Benjamini-Hochberg, Benjamini-Yekutieli, combining p-values.
- `effect-size.ts`: Cohen's d, Hedges' g, Glass's delta, Cohen's h, Cramer's V, eta and omega squared, Cliff's delta,
  with intervals.
- `power.ts`: power and sample size for t, proportion, ANOVA and chi-square tests, simulation-based power.
- `resampling.ts`: `bootstrap` (percentile, basic, BCa, studentised), `permutationTest`, `jackknife`, Monte Carlo
  tests.
- `extremes.ts`: generalised Pareto and GEV fitting, peaks over threshold, return levels, mean-excess plots.
- `robust.ts`: trimmed and winsorised statistics, Huber and Tukey M-estimators, `ransac` (traced, generic).
- `smoothing/bases.ts`: B-splines (`bs`), natural cubic splines (`ns`), P-splines (`ps`), cubic regression splines
  (`cr`), cyclic splines (`cc`), thin-plate regression splines (`tp`, `ts`), Duchon splines, Gaussian-process smooths,
  random-effect and factor-smooth bases, tensor products (`te`, `ti`), with knot placement rules.
- `smoothing/penalties.ts`: difference and derivative penalties, null spaces, shrinkage, shape-constraint
  reparameterisations (monotone, convex), sum-to-zero constraints.
- `smoothing/smoothers.ts`: `loess`, `lowess`, kernel smoothers (Nadaraya-Watson, local linear), running statistics,
  smoothing-spline smoothers.
- `fit.ts`: closed-form and Newton maximum-likelihood fits for the distribution families, with standard errors from the
  observed information.

### 4.18 `optim/` (tier 7): optimisation (optax, torch.optim, scipy.optimize)

oracle: torch.optim, optax, scipy.optimize, cvxpy for small LPs and QPs, pot for transport

Objectives are plain functions of values; gradients come from `autodiff`. Every iterative method is a traced algorithm
that records the objective, gradient norm, step size and (by policy) the iterates.
- `transform.ts`: `GradientTransformation`, `applyUpdates`, `chain`, `scaleByLearningRate`, `clipByGlobalNorm`,
  `clipByValue`, `addDecayedWeights`, `masked`, `multiTransform` (different optimisers per parameter group),
  `globalNorm`, `identity`, `setToZero`, `scale`, `scaleByTrustRatio`, and the scalings `traceMomentum` (dampening),
  `scaleByAdam`, `scaleByRms` (centred), `scaleByAdagrad`.
- `first-order.ts`: `sgd` (momentum, dampening, Nesterov), `adagrad` (lr decay), `rmsprop` (centred), `adadelta`,
  `rprop`, `asgd` (averaged iterate), `signSgd` (signSGD and signum), `ema`, `scaleBySign`, `scaleByAdadelta`; torch.optim
  and optax trajectories.
- `sparse-updates.ts`: optimisers for sparse gradients (embeddings and very wide linear models): `sparseAdam` and
  `lazyAdagrad` (moments updated only on the active set, element-wise or row-wise; `activeMask`), `sgdCumulativeL1`
  (Tsuruoka et al.) and `truncatedGradient` with `truncate` (Langford et al.), `ftrlProximal` and `ftrlSupport`
  (McMahan et al.) for per-coordinate L1 and L2; torch.optim on COO gradients.
- `adaptive.ts`: `adam`, `adamw`, `adamax`, `nadam`, `radam` (both with decoupled decay), `amsgrad`, `lion`, `lamb`,
  `adafactor` (factored, torch.optim.Adafactor), `sophiaLite` with `scaleBySophia` and `hutchinsonDiagonal`, and the
  scalings `scaleByAdamax`, `scaleByNadam`, `scaleByRadam`, `scaleByLion`.
- `preconditioned.ts`: natural gradient (`naturalGradient` fed Fishers by `observe`, `empiricalFisher`,
  `naturalGradientDirection`), K-FAC for dense layers (`kfac`, `kfacFactors`, `kfacPrecondition` with factored
  Tikhonov damping and joint bias blocks), `shampoo` (tensors of any order), gradient preconditioning helpers
  (`matrixInversePthRoot`, `preconditionBy`, `diagonalPrecondition`).
- `variance-reduced.ts`: `sag`, `saga`, `svrg` (proximal) over a `FiniteSum`, each a traced algorithm (one epoch per
  step) with a convenience function; stochastic average gradient for finite sums (used by linear models).
- `schedules.ts`: constant (`constantSchedule`, `constantFactor`), step (`stepDecay`, `multiStepDecay`), exponential,
  inverse-time, cosine (`cosineDecay`, `cosineWarmRestarts`, `sgdr`), warmup (`warmupCosineDecay`,
  `warmupExponentialDecay`, `warmupConstant`, `linearSchedule`), one-cycle, piecewise (`piecewiseConstant`,
  `piecewiseInterpolate`, `joinSchedules`), polynomial, cyclical, `reduceOnPlateau`; torch.optim.lr_scheduler and optax.
- `line-search.ts`: the line-search contract (`LineFunction`, `LineStart`, `LineSearchResult` with every trial point
  recorded, `LineSearch`), `backtracking` (Armijo, scipy's quadratic-cubic interpolation, optionally safeguarded, or
  halving; value-only evaluations), `strongWolfe` (Nocedal-Wright bracketing and zoom by cubic interpolation, scipy's
  line_search), `withFallback`, `lineFunction` and the vector front `lineSearch`; the one-dimensional searches every
  second-order method uses.
- `line-search-interval.ts`: `moreThuente` (MINPACK-2 dcsrch, scipy's scalar_search_wolfe1), `hagerZhang` (CG_DESCENT:
  approximate Wolfe conditions, secant^2 with bisection safeguards), `wolfe12` (scipy's More-Thuente-then-zoom search
  with an extra acceptance condition).
- `minimize/objective.ts`: `flatObjective` (an objective of any parameter tree as a function of one flat vector, with
  gradients, Hessian-vector products and Hessians by autodiff), `flatResiduals` (residuals and their Jacobian),
  `MinimizeResult` (scipy's OptimizeResult), `MinimizeMethod`, `VectorMethodOptions`, `Bounds`, `boxOf`, `clipToBox`,
  `infNorm`.
- `minimize/minimizer.ts`: `minimizer` (a transformation as an algorithm), `minimize(f, x0, { method })` (methods are
  function values, not strings: any vector method below with its own options, or a gradient transformation; BFGS by
  default).
- `minimize/gradient.ts`: `gradientDescent` (line-search, fixed or Barzilai-Borwein steps), `conjugateGradient` (a port
  of scipy's CG: Polak-Ribiere+ with the descent condition; Fletcher-Reeves, Hestenes-Stiefel, Dai-Yuan and Hager-Zhang
  betas, restarts), `coordinateDescent` (cyclic, keyed random or greedy order; exact, Newton or gradient coordinate
  updates).
- `minimize/quasi-newton.ts`: `bfgs` (a port of scipy's BFGS: inverse-Hessian updates, optional Powell damping, the
  final inverse Hessian; `bfgsInverseUpdate`), `lbfgs` (the two-loop recursion `twoLoopRecursion`, memory m, the
  More-Thuente search), each traced (objective, gradient norm, step, line-search trials, curvature pairs).
- `minimize/lbfgsb.ts`: `lbfgsb` (Byrd-Lu-Nocedal-Zhu: the generalised Cauchy point on the compact limited-memory model,
  subspace minimisation with the Morales-Nocedal projection, the bounded More-Thuente search, restarts on failure;
  scipy's stopping tests), `projectedGradientNorm`.
- `minimize/owlqn.ts`: `owlqn` (Andrew and Gao 2007: the pseudo-gradient of f + C |x|_1 with per-coordinate weights,
  orthant-wise projection of the search direction and of the backtracking line search, exact zeros; traced nonzeros),
  `pseudoGradient`.
- `minimize/sr1.ts`: `sr1Hessian` and `bfgsHessian` (quasi-Newton curvature models for trust regions), `sr1` (SR1 with
  a trust region, Nocedal-Wright 6.2: updated after every step, Steihaug or exact subproblems).
- `minimize/newton.ts`: `newton` (modified Cholesky of Nocedal-Wright 3.3, `modifiedCholesky`, or eigenvalue flooring,
  with safeguarded backtracking), `newtonCg` (a port of scipy's Newton-CG on Hessian-vector products or the dense
  Hessian).
- `minimize/trust-region.ts`: the trust-region driver (`trustRegionAlgorithm`, `trustRegionOn`) over pluggable
  curvature models (`HessianModel`, `exactHessian` dense or by products), `trustRegion`, `trustDogleg`, `trustNcg`,
  `trustExact` (scipy's dogleg, trust-ncg and trust-exact). TODO: trust-krylov (GLTR, Lanczos on the subproblem).
- `minimize/trust-subproblems.ts`: `doglegStep`, `steihaugStep` (Steihaug-Toint CG), `exactStep` (More-Sorensen with
  the hard case), `quadraticModelValue`.
- `minimize/least-squares.ts`: `gaussNewton` (damped, minimum-norm steps, robust losses), `leastSquares` (scipy's
  least_squares over method function values, default `trf`), `curveFit` (sigma vectors or covariances, absolute sigma,
  bounds, the SVD covariance), the result shape `LeastSquaresResult` (cost, residuals, Jacobian, optimality, active
  mask).
- `minimize/levenberg-marquardt.ts`: `levenbergMarquardt` (MINPACK's lmder and lmpar: scaled trust region on |D p|,
  the Levenberg-Marquardt parameter by safeguarded Newton, ftol, xtol and gtol tests).
- `minimize/least-squares-trf.ts`: `trf` (scipy's trust region reflective method: Coleman-Li scaling, reflected and
  truncated steps, robust losses).
- `minimize/least-squares-bounded.ts`: `dogbox` (scipy's rectangular trust-region dogleg with active variables
  frozen, robust losses).
- `minimize/least-squares-linear.ts`: `nnls` (Lawson-Hanson), `lsqLinear` (bounded-variable least squares, BVLS), each a
  traced active-set algorithm.
- `minimize/derivative-free.ts`: `nelderMead` (scipy's moves exactly: adaptive coefficients, xatol and fatol, initial
  simplex, evaluation budget, clipping into bounds), the `Bounds` type.
- `minimize/powell.ts`: `powell` (conjugate directions with scipy's bracket, Brent and bounded line minimisations,
  initial direction set, bounds).
- `minimize/pattern-search.ts`: `hookeJeeves` (exploratory and pattern moves), `compassSearch` (a generalised pattern
  search: coordinate or minimal positive spanning set, mesh expansion and contraction, opportunistic or complete polls,
  bounds).
- `minimize/cobyla.ts`: `cobyla` (Powell's linear-approximation trust-region method for inequality constraints c(x) >= 0
  and bounds: simplex of interpolation points, least-violation then objective trust-region subproblem, merit penalty
  updates, geometry steps, rhobeg and rhoend), `InequalityConstraint`.
- `minimize/global.ts`: `simulatedAnnealing` (exponential, logarithmic, fast or custom cooling, Corana step
  adaptation, bounds), `basinHopping` (scipy's: random displacement, Metropolis acceptance, adaptive step size,
  niterSuccess, any `LocalMethod` as a child run).
- `minimize/dual-annealing.ts`: `dualAnnealing` (scipy's generalised simulated annealing: Tsallis visiting
  distribution, generalised acceptance, reannealing, local searches).
- `minimize/differential-evolution.ts`: `differentialEvolution` (scipy's: twelve strategies, dithering, binomial and
  exponential crossover, immediate or deferred updating, Latin hypercube, Halton or given initial populations,
  constraints by feasibility rules, polishing).
- `minimize/cmaes.ts`: `cmaes` ((mu/mu_w, lambda)-CMA-ES with cumulative step-size adaptation, rank-one and rank-mu
  updates, bounds by repair, tolFun and tolX).
- `minimize/particle-swarm.ts`: `particleSwarm` (inertia or constriction, global or ring topology, velocity clamping,
  absorbing walls, stagnation stop).
- `minimize/constrained.ts`: `Constraint` (scipy's eq and ineq form) and `constraintSet` (stacked values and autodiff
  Jacobians), `projectedGradient` (spectral projected gradient SPG2 with a nonmonotone line search, Armijo along the
  projection arc, or fixed steps; over bounds or any projection).
- `minimize/penalty.ts`: `penaltyMethod` (quadratic penalty), `barrierMethod` (logarithmic barrier with the duality-gap
  bound), `augmentedLagrangian` (Powell-Hestenes-Rockafellar with shifted inequality penalties and multipliers), each
  an outer traced algorithm whose inner solves (L-BFGS-B or Newton by default, any `InnerMethod`) are child runs.
- `minimize/slsqp.ts`: `slsqp` (sequential quadratic programming after Kraft: QP subproblems by the active-set QP
  solver with elastic relaxation of inconsistent linearisations, damped BFGS on the Lagrangian, an L1 merit line
  search with Powell's penalties, the multipliers).
- `minimize/convex.ts`: `subgradientMethod` (projected, step rules constant, constant length, square summable,
  diminishing, Polyak, or custom; best-point tracking), `mirrorDescent` (`entropicMirror` for exponentiated gradient on
  the simplex, `euclideanMirror`, any `Mirror`; last, best or averaged output), `dualAscent` (and the method of
  multipliers; numeric or closed-form inner minimisation), `dualDecomposition` (separable blocks).
- `minimize/frank-wolfe.ts`: `frankWolfe` (conditional gradient: standard 2 / (k + 2), adaptive short step, away
  steps and pairwise steps with active sets, the Frank-Wolfe gap) and the oracles `simplexLmo`, `l1BallLmo`, `boxLmo`,
  `l2BallLmo`, `nuclearBallLmo`.
- `scalar.ts`: `minimizeScalar` over method function values: `brent`, `golden`, `fminbound` (scipy's ports, iteration
  and evaluation counts included; each also a traced algorithm with the interval of uncertainty), `bracket`.
- `proximal.ts`: proximal operators on values, differentiable (`proxL1`, `proxL2`, `proxSquaredL2`, `proxElasticNet`,
  `proxGroupLasso`, `proxNuclear`, `proxHuber`, `proxLinf`, `proxConjugate` by the Moreau decomposition) and the
  projections every optim file uses (`projectBox`, `projectNonnegative`, `projectSimplex`, `projectL1Ball`,
  `projectL2Ball`, `projectAffine`, `projectHalfspace`, `projectNuclearBall`, `projectPsd`).
- `total-variation.ts`: `proxTv1d` (Condat's direct algorithm, differentiable in the signal and the weight),
  `proxTv2d` and `tv2dAlgorithm` (Beck-Teboulle fast gradient projection, isotropic or anisotropic, box constraints).
- `splitting.ts`: `proximalGradientAlgorithm`, `ista` and `fista` (backtracking, adaptive restart, gradient mapping),
  `chambollePock` (primal-dual hybrid gradient with residuals, operator norm by power iteration).
- `admm.ts`: `admm` (scaled form, A x + B z = c, over-relaxation, residual balancing, Boyd's stopping rule),
  `admmProx` (Douglas-Rachford splitting from two proximal operators), `admmLasso` (cached Cholesky),
  `consensusAdmm` (global consensus with a shared regulariser).
- `smo.ts`: sequential minimal optimisation (`smo`, `smoAlgorithm`) for quadratic programs with box constraints and
  one linear equality (the SVM dual family, libsvm's solver): working-set selection (maximal violating pair, second
  order WSS3), shrinking with gradient reconstruction, libsvm's rho, traced working pairs and active sets; `svcDual`
  (C-SVC with per-sample C) and `svrDual` (epsilon-SVR over 2n variables).
- `sparse.ts`: sparse approximation, each a traced algorithm: `matchingPursuit`, `orthogonalMatchingPursuit`
  (sklearn's Cholesky OMP, atoms or squared-residual tolerance), `cosamp`, `iterativeHardThresholding` (normalised
  IHT or a fixed step), `basisPursuit` (ADMM with a cached projection), `basisPursuitDenoising` and
  `lassoCoordinateDescent` (sklearn's coordinate descent with the duality-gap test, elastic net).
- `dictionary.ts`: `sparseEncode` (sklearn's sparse_encode: OMP, coordinate-descent lasso, thresholding; TODO: LARS
  and lasso-LARS coding), dictionary learning (`ksvd`, `mod`, `onlineDictionaryLearning` of Mairal et al. with
  minibatches from keys), recovery guarantees (`mutualCoherence`, `babelFunction`, `coherenceSparsityBound`,
  `ripEstimate` by sampled supports).
- `linear-programming.ts`: `linprog` in scipy's form (the method a function value: `linprogSimplex`,
  `linprogInteriorPoint`), statuses (optimal, infeasible, unbounded), marginals (duals), slacks and residuals,
  sensitivity ranges for costs and right-hand sides, the simplex path for figures.
- `lp-standard-form.ts`: `standardForm` (shifted, reflected and split variables, bound rows, slacks, non-negative
  right-hand sides) and `fromStandard`.
- `lp-simplex.ts`: `simplexAlgorithm`, the two-phase tableau simplex (Bland's or Dantzig's rule, redundant rows
  dropped after phase 1), traced pivot by pivot.
- `lp-interior-point.ts`: `lpInteriorPointAlgorithm`, Mehrotra's predictor-corrector on the homogeneous self-dual
  embedding (certifies infeasible and unbounded programs), with dependent equality rows removed first.
- `quadratic.ts`: `quadprog` (inequalities, equalities and bounds; the method a function value: `qpActiveSet`,
  `qpInteriorPoint`) with multipliers split by constraint type, `qpArrays` and `QpArrays` (the reusable number-array
  form), `kktResiduals`, box QP by Bertsekas' projected Newton (`boxQp`, `boxQpAlgorithm`).
- `qp-active-set.ts`: `qpActiveSetAlgorithm`, the primal active-set method (Nocedal and Wright 16.3) from a phase-1
  linear program, traced working-set changes.
- `qp-interior-point.ts`: `qpInteriorPointAlgorithm`, primal-dual Mehrotra predictor-corrector on the reduced KKT
  system.
- `conic.ts`: small conic programs (linear inequalities, second-order cones, linear matrix inequalities, equalities) by
  the barrier interior-point method (`conicProgram`, `conicBarrierAlgorithm`: phase 1, centring by Newton with
  equality constraints, duals from the central path), and standard-form semidefinite programs (`sdp`, with the dual y
  and slack Z); for teaching and small models.
- `integer.ts`: `milp` and `milpAlgorithm` (branch and bound on simplex relaxations as child runs, best-first or
  depth-first, incumbent, bound and gap, Gomory mixed-integer cuts at the root, the search tree traced), `knapsack`
  (0-1, by dynamic programming, or by `knapsackAlgorithm`: traced branch and bound with the fractional bound),
  `setCover` (greedy or exact).
- `assignment.ts`: `linearSumAssignment` (rectangular, maximise) by `hungarianAlgorithm` (shortest augmenting paths
  with potentials), `auction` and `auctionAlgorithm` (Bertsekas, epsilon scaling), `stableMatching` and
  `galeShapleyAlgorithm` (incomplete lists, unequal sides), `blockingPairs`.
- `online.ts`: online gradient descent, FTRL, hedge, exponentiated gradient, online Newton step (`OnlineLearner`s),
  regret records (`onlineAlgorithm`, `runOnline`, `linearLosses`).
- `transport.ts`: `emd` (exact transport by the transportation simplex with dual potentials, traced pivots),
  `sinkhorn` (log domain or scaling, traced marginal error), `entropicTransportLoss` (differentiable, envelope
  derivatives: the plan and the potentials), `sinkhornUnbalanced` (KL-relaxed marginals, product or uniform reference).
- `wasserstein.ts`: `wasserstein1d` (exact quantile merge, any p, weights, differentiable in positions),
  `slicedWasserstein` (explicit or keyed random directions), barycentres (`sinkhornBarycenter` by iterative Bregman
  projections in the log domain, `freeSupportBarycenter` by exact transport).
- `gromov.ts`: `gromovWasserstein` (conditional gradient with exact line search, square or KL loss, asymmetric
  structures) and `entropicGromovWasserstein` (projected gradient with Sinkhorn steps).
- `multiobjective.ts`: `dominates`, `paretoFront`, `nonDominatedSort` (Deb's fast sort), `crowdingDistance`,
  `hypervolume` (exact: a sweep in 2-D, WFG above), `hypervolumeContributions`, scalarisations (`weightedSum`,
  `chebyshev`, `augmentedChebyshev`, `epsilonConstraint`), `VectorObjective`.
- `nsga2.ts`: `nsga2` (binary tournaments by rank and crowding, elitist survival, constrained domination, hypervolume
  traced against a reference point), the operators `sbxCrossover` and `polynomialMutation` (bounded, as Deb's code).
- `implicit.ts`: differentiation through solutions by the implicit function theorem, with any solver: `implicitRoot`
  (F(x, ...thetas) = 0, reverse and forward rules, dense solves that differentiate again or matrix-free GMRES/CG),
  `implicitFixedPoint` (equilibrium layers, default fixed-point solver), `implicitArgmin` (optimisation layers through
  the stationarity condition, default inner `minimize`).

### 4.19 `kernels/` (tier 7): kernels and feature maps (sklearn kernels, GPyTorch)

oracle: scikit-learn, gpytorch
- `kernel.ts`: `Kernel` (`k(x, y)`, `diag`, hyperparameters as a differentiable pytree), `gram`, `crossGram`, sum,
  product and scale composition, ARD, log-parameterisation.
- `stationary.ts`: RBF, Matern (1/2, 3/2, 5/2 and general), rational quadratic, exponential, periodic, locally
  periodic, spectral mixture, cosine.
- `dot-product.ts`: linear, polynomial, sigmoid, arc-cosine, neural tangent kernel of an MLP.
- `structured.ts`: white noise, constant, string kernels, set kernels, graph kernels (diffusion, regularised Laplacian,
  WL subtree), chi-square, histogram intersection.
- `expansion.ts`: random Fourier features, orthogonal random features, Nystroem, additive chi-square sampler,
  polynomial count sketch, tensor sketch, approximation error helpers.
- `matrices.ts`: centring, kernel alignment, HSIC and MMD statistics, effective dimension, kernel PCA projections.

### 4.20 `signal/` (tier 7): signal and image processing (scipy.signal, scipy.ndimage, pywt, librosa)

oracle: scipy.signal, scipy.ndimage, pywt, librosa, scikit-image, stumpy

The files follow the encyclopedia's signal-processing chapters. LTI system representations are in `systems`; sparse
representations are in `optim/sparse.ts`.
- `signal.ts`: `Signal` (samples, sampling rate, start, units, channels), `Spectrum`, `TimeFrequency`, `sampleTimes`,
  `magnitudeDb`, `unwrap`.
- `sampling.ts`: sampling and reconstruction (sinc interpolation), aliasing, quantisation (uniform, mu-law, dither),
  zero-order hold, signal-to-quantisation-noise.
- `fourier.ts`: DTFT evaluation, zero padding and frequency resolution, spectral leakage, Parseval checks, the analytic
  signal and Hilbert transform.
- `windows.ts`: `getWindow` (boxcar, Hann, Hamming, Blackman, Blackman-Harris, Nuttall, flat top, Bartlett, Kaiser,
  Gaussian, Tukey, cosine, DPSS), window metrics (main-lobe width, sidelobe level, ENBW).
- `filters/fir.ts`: `firwin`, `firwin2`, `remez` (Parks-McClellan), `kaiserord`, `minimumPhase`.
- `filters/iir.ts`: Butterworth, Chebyshev I and II, elliptic, Bessel, notch, peak, comb filters; `ba`, `zpk` and `sos`
  outputs.
- `filters/apply.ts`: `lfilter`, `sosfilt`, `filtfilt`, `sosfiltfilt`, initial conditions, `medfilt`, `savgolFilter`,
  moving averages.
- `filters/response.ts`: frequency response, group delay, phase delay, filter specification checks.
- `multirate.ts`: decimation and interpolation, `resamplePoly`, `upfirdn`, polyphase decomposition, noble identities.
- `filter-banks.ts`: DFT filter banks, quadrature-mirror and perfect-reconstruction filter banks, wavelet packets.
- `spectral.ts`: periodogram, Welch, Bartlett, Blackman-Tukey, cross-spectral density, coherence, multitaper,
  Lomb-Scargle (with false-alarm probabilities), spectral confidence intervals.
- `parametric.ts`: Yule-Walker, Burg, AR spectra, Levinson-Durbin, MUSIC, ESPRIT, Prony.
- `time-frequency.ts`: STFT and inverse, spectrogram, constant-Q transform, Wigner-Ville and pseudo variants,
  reassignment, synchrosqueezing, instantaneous frequency, uncertainty-bound helpers.
- `decompositions.ts`: EMD, EEMD, CEEMDAN, VMD, singular spectrum analysis.
- `wavelets.ts`: continuous wavelet transform, DWT and inverse, multilevel decomposition, stationary wavelet transform,
  families (Haar, Daubechies, Symlets, Coiflets, biorthogonal), threshold denoising.
- `estimation.ts`: Wiener filtering (causal and non-causal), linear prediction and LPC, the matched filter.
- `detection.ts`: energy and matched-filter detectors, CFAR, detector ROC curves.
- `adaptive.ts`: LMS, NLMS, RLS, adaptive noise cancellation, traced tap weights.
- `cepstrum.ts`: real and complex cepstra, liftering, cepstral pitch.
- `peaks.ts`: `findPeaks` (height, prominence, distance, width), prominences, widths, zero crossings.
- `similarity.ts`: dynamic time warping (with path and lower bounds), PAA, SAX, cross-correlation lag.
- `matrix-profile/sliding.ts`: the shared machinery of matrix profiles (as STUMPY): rolling means and standard
  deviations by Welford updates with periodic exact recomputation (and STUMPY's precision checks for near-constant
  windows), sliding dot products QT by FFT (`fftConvolve` and overlap-add for long series) and by the O(1) diagonal
  update, z-normalised and non-normalised distances from QT, exclusion zones, NaN and constant-subsequence handling.
- `matrix-profile/mass.ts`: MASS distance profiles (v2 FFT, v3 chunked by overlap-add), the non-normalised variant,
  `distanceProfile`.
- `matrix-profile/algorithms.ts`: `stomp` (row order), `stamp` (anytime, random order, traced convergence), `scrimp`
  (PreSCRIMP then SCRIMP++ diagonals), diagonal-order STOMP for AB-joins, left and right profiles and indices,
  `stumpi` (incremental and streaming updates, egress), `aamp` (non-normalised), top-k profiles; each an `Algorithm`
  with traced progress.
- `matrix-profile/multidimensional.ts`: mSTAMP (k-dimensional profiles by sorted per-dimension distances), MDL-based
  dimension selection, constrained (included and excluded) dimensions.
- `matrix-profile/discovery.ts`: motifs and discords (top-k with exclusion zones, `motifs`, `match`), time-series chains
  (anchored and all-chain sets, ATSC and ALLC), FLUSS and FLOSS semantic segmentation (corrected arc curve), snippets
  (MPdist-based), consensus motifs (Ostinato), `mpdist`, pan matrix profiles over many window lengths (SKIMP order).
- `audio/scales.ts`: mel, Bark and ERB scales.
- `audio/filterbanks.ts`: mel and gammatone filter banks.
- `audio/features.ts`: mel spectrograms, MFCCs, spectral centroid, bandwidth, roll-off, flux and flatness, chroma,
  zero-crossing rate, RMS energy.
- `audio/pitch.ts`: YIN and probabilistic YIN, autocorrelation and cepstral pitch, harmonic product spectrum.
- `audio/rhythm.ts`: onset strength and detection, tempo estimation, beat tracking.
- `audio/enhancement.ts`: spectral subtraction, Wiener denoising, voice activity detection.
- `audio/separation.ts`: ideal binary and ratio masks, mask application, separation scores (NMF and ICA separation are
  in `methods/audio`).
- `audio/coding.ts`: psychoacoustic masking thresholds, MDCT transform coding, perceptual quantisation.
- `audio/augmentation.ts`: SpecAugment, time stretching and pitch shifting (phase vocoder), mixing and noise.
- `image/filters.ts`: Gaussian, uniform, median and bilateral filters, Sobel, Scharr, Prewitt, Laplacian of Gaussian,
  Gabor kernels and Gabor filter banks.
- `image/features.ts`: Canny, Harris and Shi-Tomasi corners, peak finding, LoG and DoG blobs, Hough lines and circles,
  HOG, local binary patterns, Gabor features, structure tensors.
- `image/morphology.ts`: erosion, dilation, opening, closing, gradients, top-hats, connected-component labelling,
  distance transforms, skeletonisation.
- `image/geometry.ts`: resizing, rotation, affine and perspective warps, pyramids, integral images.
- `image/segmentation.ts`: Otsu thresholds, watershed, SLIC superpixels, Felzenszwalb, graph cuts.
- `image/motion.ts`: Lucas-Kanade and Horn-Schunck optical flow, block matching, template matching.

### 4.21 `text/` (tier 7): text processing (HF tokenizers, nltk, sklearn.feature_extraction.text)

oracle: HF tokenizers, tiktoken, nltk, scikit-learn, rapidfuzz, datasketch
- `normalise.ts`: Unicode normalisation, case folding, accent stripping, whitespace collapsing, `AlignedText` (offsets
  back to the original).
- `tokenise.ts`: regex, word (sklearn pattern, Treebank), tweet, character and grapheme tokenisers, sentence splitting,
  detokenisation, pre-tokeniser patterns (GPT-2, cl100k, BERT).
- `stem.ts`: Porter (with a rule trace) and Snowball English stemmers, rule-based lemmatisation, stop-word lists.
- `vocabulary.ts`: `Vocabulary`, building (minimum count, maximum size, specials, ordering), encoding and decoding.
- `subword/bpe.ts`: BPE training (character and byte level, traced merges), encoding with dropout, the GPT-2 byte
  alphabet.
- `subword/wordpiece.ts`: WordPiece training and greedy longest-match encoding.
- `subword/unigram.ts`: Unigram language-model training (EM with pruning), Viterbi encoding, n-best and sampling.
- `tokenizer.ts`: an HF-tokenizers-style pipeline (normaliser, pre-tokeniser, model, post-processor, decoder), batch
  encoding with padding and truncation, loading HF tokenizer files.
- `shingles.ts`: character k-shingles, word n-gram shingles, w-shingling, skip-grams, the hashed character n-grams of
  fastText (bucketed with FNV hashing), weighted shingles; one extractor shared by matching, fingerprinting and fastText.
- `features.ts`: count, TF-IDF and hashing vectorisers, word and character n-grams, BM25 and BM25+ weighting.
- `templates.ts`: feature-template languages for sequence labellers. CRF++ templates (Kudo 2005): `U` (unigram, conjoined
  with the label) and `B` (bigram, conjoined with the label pair) lines, `%x[row,col]` macros over a token-by-attribute
  table, boundary padding (`_B-1`, `_B+1`), comments, and syntax errors with line and column. CRFsuite-style attribute
  templates: windows of `(attribute, offset)` tuples joined into attribute strings (`w[-1]|w[0]`). Expansion over a
  sequence, a `FeatureIndex` (string to id with counts and a minimum frequency, as CRF++'s `-f`), feature firing for
  inspection.
- `representations.ts`: co-occurrence counts, PMI and PPMI, LSA, SVD word vectors, random indexing, analogies, nearest
  words.
- `similarity.ts`: Levenshtein, Damerau-Levenshtein, Jaro-Winkler, longest common subsequence, Needleman-Wunsch,
  Smith-Waterman, Jaccard and containment over shingles, Soundex and Metaphone, token-set and token-sort ratios.
- `approximate-search.ts`: approximate string search: q-gram indexes with count filtering, BK-trees, tries with fuzzy
  lookup, Sellers' approximate substring matching, top-k fuzzy matches with traced candidate pruning.
- `fingerprinting.ts`: Rabin-Karp rolling hashes, winnowing (MOSS), SimHash, MinHash signatures for text, near-duplicate
  detection.
- `parsing.ts`: context-free grammars, CYK and Earley parsing, probabilistic CFGs (Viterbi parse, inside-outside),
  dependency parsing (Eisner, arc-standard transitions).
- `hyphenation.ts`: Liang pattern hyphenation with matching traces.
- `statistics.ts`: Zipf and Heaps fits, type-token ratios, tokeniser fertility, compression ratio, topic coherence
  (NPMI, UMass).

### 4.22 `search/` (tier 7): search, constraint satisfaction and planning (AIMA)

oracle: aima-python results and hand-checked fixtures; python-constraint; pyperplan

One `SearchProblem` contract (initial state, actions, result, goal test, step cost, heuristic) and traced search
algorithms that record the frontier and explored set, so every search can be animated.
- `problem.ts`: `SearchProblem`, `SearchNode`, path reconstruction, search statistics (expanded, generated, effective
  branching factor).
- `uninformed.ts`: breadth-first, depth-first, depth-limited, iterative deepening, uniform cost, bidirectional search.
- `informed.ts`: greedy best-first, A*, weighted A*, IDA*, RBFS, beam search; heuristic admissibility and consistency
  checks.
- `local.ts`: hill climbing (steepest, first choice, random restarts), simulated annealing, tabu search, local beam
  search, late acceptance.
- `evolutionary.ts`: genetic algorithms (selection, crossover and mutation for bit strings, permutations and real
  vectors), evolution strategies, genetic programming over expression trees, MAP-Elites, novelty search.
- `swarm.ts`: ant colony optimisation, discrete particle swarm.
- `adversarial.ts`: minimax, alpha-beta with move ordering, negamax, expectimax, Monte Carlo tree search (UCT and PUCT
  with priors).
- `csp.ts`: constraint satisfaction problems: backtracking with MRV, degree and LCV heuristics, forward checking, AC-3,
  maintaining arc consistency, min-conflicts.
- `planning.ts`: STRIPS and a PDDL subset, forward and regression planning with h-add, h-max and FF heuristics,
  GraphPlan, partial-order planning, simple HTN decomposition.
- `refinement.ts`: refinement-operator search over pattern and rule languages (beam, best-first, exhaustive with
  optimistic-estimate pruning), top-k result sets with redundancy filtering; used by rule learning, subgroup discovery,
  exceptional model mining and ILP.

### 4.23 `stochastic/` (tier 7): stochastic processes and stochastic calculus

oracle: closed-form moments and transition densities (scipy.stats); sdeint with shared Brownian increments; scipy.linalg
for chains; scikit-learn for GP regression; nolds for Hurst estimators; astropy for Ripley's K; DPPy for DPPs. Simulated
paths are checked statistically (moments, convergence orders), and pathwise derivatives with `checkPrimitive` (implicit
reparameterisations, which finite differences cannot see, by the unbiasedness of the mean gradient).
- `paths.ts`: the `Paths` shape every simulator returns (`[paths, times, ...state]`), time grids, the traced Markov
  path simulation (`pathAlgorithm`, `simulatePaths`), persistent histories for long event simulations, and increment
  utilities (coarsening for convergence studies).
- `brownian.ts`: Brownian motion (multi-dimensional, with drift or a covariance), Brownian increments, Brownian bridges
  (exact transitions, moments, Levy midpoint refinement), quadratic variation and covariation, the reflection principle
  (running maxima, reflected paths, the law of the maximum), first-passage times (simulated, CDF, density, law).
- `calculus.ts`: Ito and Stratonovich integrals by Riemann sums, Ito's lemma checked numerically, the Ito isometry,
  Girsanov reweighting, Feynman-Kac by Monte Carlo (with the Black-Scholes price as its closed-form check).
- `sde.ts`: SDE solvers (Euler-Maruyama, Milstein, Platen's stochastic Runge-Kutta, stochastic Heun for Stratonovich;
  diagonal, scalar and general noise), Ito-Stratonovich conversion, strong and weak convergence studies, reverse-time
  SDEs and probability-flow drifts given a score. `sdeSolver` fits `nn.blocks.SdeSolver`.
- `diffusions.ts`: named processes with exact transitions: Ornstein-Uhlenbeck, geometric Brownian motion, CIR, Vasicek
  (with Vasicek and CIR bond prices), Bessel, overdamped and underdamped Langevin dynamics, jump diffusions (generic, and
  Merton's with its exact simulation, log-return density and option price).
- `markov-chains.ts`: discrete-time Markov chains: stationary distributions, classification, absorption, hitting
  times, mixing times, spectral gaps, reversibility, simulation.
- `ctmc.ts`: continuous-time Markov chains (split from `markov-chains.ts` for size): transition matrices exp(tQ)
  (the Kolmogorov equations), stationary laws, embedded jump chains, uniformisation, hitting times, Gillespie
  simulation and exact grid sampling.
- `random-walks.ts`: simple and biased walks (with their n-step law), lattice walks, walks on graphs, self-avoiding
  walks (Rosenbluth growth, exact counts), Levy flights, gambler's ruin (closed forms and simulation).
- `branching.ts`: Galton-Watson processes, generating functions, extinction probabilities, birth-death processes
  (generators, product-form stationary laws, linear birth-death and Yule laws, Gillespie simulation).
- `queues.ts`: M/M/1, M/M/c (Erlang B and C) and M/G/1 queues (formulas and G/G/c simulation), Little's law.
- `point-processes.ts`: homogeneous and inhomogeneous Poisson processes (thinning), renewal processes and renewal
  functions, Hawkes (Ogata thinning, maximum likelihood by EM and Newton), Cox (including log-Gaussian Cox) and marked
  processes.
- `spatial-processes.ts` (split from `point-processes.ts` for size): spatial Poisson and cluster processes (Thomas,
  Matern), Ripley's K and L with the usual edge corrections.
- `dpp.ts`: determinantal point processes: L-ensembles, exact and k-DPP sampling, greedy MAP, marginals, conditioning.
- `random-measures.ts`: Dirichlet processes (stick-breaking, Chinese restaurant, Polya urn), Pitman-Yor, Indian buffet,
  beta and gamma processes (as random measures and as a subordinator in time).
- `gaussian-processes.ts`: Gaussian processes as random functions: sampling paths from covariance functions (the
  process covariances of Brownian motion, bridges and Ornstein-Uhlenbeck here; stationary kernels in `kernels`),
  conditioning, Karhunen-Loeve expansions.
- `fractional.ts`: fractional Brownian motion (Davies-Harte), Hurst exponent estimation (R/S, DFA, aggregated
  variance), long-range dependence.
- `levy.ts`: compound Poisson, alpha-stable (Chambers-Mallows-Stuck sampling, densities and CDFs by Fourier inversion)
  and variance-gamma processes.

### 4.24 `sampling/` (tier 7): Monte Carlo (NumPyro and BlackJAX kernels, ArviZ diagnostics)

oracle: statistical agreement with analytic targets; arviz for diagnostics
- `target.ts`: `LogDensity` (log density, gradient by autodiff, dimension, constraints), from distributions, unconstrained
  targets.
- `metropolis.ts`: random-walk, independence and general Metropolis-Hastings, adaptive Metropolis.
- `hamiltonian.ts`: leapfrog, HMC, NUTS (multinomial, generalised U-turn, divergences), step-size adaptation (dual
  averaging), mass-matrix adaptation (diagonal and dense, windowed), Riemannian HMC for small problems.
- `langevin.ts`: MALA, ULA, SGLD, SVGD.
- `gibbs.ts`: blocked Gibbs, slice sampling, elliptical slice sampling.
- `importance.ts`: importance sampling, self-normalised estimates, Pareto-smoothed weights, weight diagnostics.
- `smc.ts`: SMC samplers with adaptive tempering, resampling schemes, annealed importance sampling.
- `rejection.ts`: rejection sampling, adaptive rejection sampling, ratio of uniforms.
- `exact.ts`: inverse-transform sampling, Box-Muller and Marsaglia polar methods, alias sampling (step records for
  teaching).
- `chains.ts`: running several chains with independent keys, `Chains` (draws by chain), thinning, warm-up handling.
- `diagnostics.ts`: R-hat (rank-normalised split), bulk and tail ESS, MCSE, autocorrelation, summaries, divergence and
  tree-depth reports.
### 4.25 `logic/` (tier 8): logic programming, satisfiability and ILP

oracle: recorded SWI-Prolog outputs, pysat, Aleph and Popper results on small tasks
- `terms.ts`: atoms, variables, compounds, lists, numbers, groundness, term variables, equality, printing.
- `parse.ts`: Prolog syntax for clauses and queries (operators, lists, comments).
- `unify.ts`: unification (iterative, optional occurs check), substitutions, composition, renaming apart,
  Martelli-Montanari step traces.
- `resolution.ts`: SLD resolution with backtracking and depth limits (traced), built-ins, negation as failure, a small
  standard library, SLD trees as structures.
- `datalog.ts`: naive and semi-naive bottom-up evaluation, least Herbrand models, stratified negation, derivation trees.
- `propositional.ts`: formulas, evaluation, truth tables, models, entailment, CNF (Tseitin) and DNF, simplification.
- `sat.ts`: DPLL, CDCL (clause learning, watched literals, traced decisions and conflicts), WalkSAT, DIMACS.
- `first-order.ts`: clausal form (Skolemisation), given-clause resolution proving, subsumption.
- `answer-sets.ts`: stable models of small normal programs (guess and check), for comparison with Datalog semantics.
- `ilp/generalisation.ts`: theta-subsumption, least general generalisation and relative LGG, clause reduction.
- `ilp/version-spaces.ts`: Find-S, candidate elimination, boundaries and voting.
- `ilp/foil.ts`: FOIL (gain, refinement, coverage), traced.
- `ilp/progol.ts`: mode declarations, bottom clauses, inverse entailment.
- `ilp/meta-interpretive.ts`: Metagol-style learning with metarules; learning from failures (Popper-style).

### 4.26 `dynamics/` (tier 8): differential equations, nonlinear dynamics, control and filtering (scipy.integrate, diffrax, python-control, filterpy)

oracle: scipy.integrate, diffrax and torchdiffeq (including adjoint gradients), python-control, filterpy, statsmodels

Every solver works on traced values, so solutions are differentiable: by differentiating through the solver steps, or
by the adjoint method in `ode/adjoint.ts`. Neural ODEs and PINNs in `methods/scientific/` are built on this folder.
- `ode/problem.ts`: `OdeProblem` (vector field, initial state, time span, parameters, mass matrix), solution type with
  dense output, event specification.
- `ode/explicit.ts`: fixed-step explicit Runge-Kutta methods by order: Euler (1), Heun and midpoint and Ralston (2),
  Kutta's third order (3), classical RK4 and the 3/8 rule (4), any Butcher tableau.
- `ode/adaptive.ts`: embedded pairs: Heun-Euler (2/1), Bogacki-Shampine (3/2), Fehlberg (4/5), Cash-Karp,
  Dormand-Prince (5/4), Tsitouras (5/4), DOP853 (8); PI step-size control, rejected steps recorded as events.
- `ode/multistep.ts`: Adams-Bashforth (orders 1 to 5), Adams-Moulton, predictor-corrector (PECE), Nordsieck form.
- `ode/implicit.ts`: backward Euler, trapezoid (Crank-Nicolson), BDF orders 1 to 5 with variable order, Radau IIA (5),
  SDIRK, Rosenbrock methods, stiffness detection.
- `ode/symplectic.ts`: symplectic Euler, Stormer-Verlet (leapfrog), velocity Verlet, Forest-Ruth and Yoshida fourth
  order; energy-drift diagnostics.
- `ode/exponential.ts`: exponential Euler and ETDRK4 for stiff semilinear problems.
- `ode/taylor.ts`: Taylor-series integrators using Taylor-mode differentiation.
- `ode/solve-ivp.ts`: `solveIvp` (scipy semantics: method, evaluation times, dense output, events with direction and
  termination), `odeint`, `diffeqsolve` (diffrax style: solver, step-size controller, saving policy).
- `ode/adjoint.ts`: continuous adjoint sensitivities (backsolve, interpolated and checkpointed adjoints), forward
  sensitivities, gradient-method comparison helpers (memory and accuracy).
- `ode/boundary.ts`: shooting and multiple shooting, collocation (`solveBvp`).
- `ode/dae.ts`: index-1 differential-algebraic equations by BDF and Radau.
- `ode/convergence.ts`: order verification (error against step size, observed order), stability regions and
  A-stability checks, work-precision diagrams.
- `pde/grids.ts`: 1-D and 2-D grids, boundary-condition types (Dirichlet, Neumann, Robin, periodic).
- `pde/finite-difference.ts`: heat (explicit, implicit, Crank-Nicolson), wave (leapfrog), advection (upwind,
  Lax-Friedrichs, Lax-Wendroff), Poisson (Jacobi, Gauss-Seidel, SOR, conjugate gradients, multigrid).
- `pde/finite-volume.ts`: conservation laws, Godunov and MUSCL schemes, flux limiters, Burgers shocks and rarefactions.
- `pde/finite-element.ts`: 1-D linear and quadratic elements, 2-D linear triangles, assembly, Poisson and heat problems.
- `pde/spectral.ts`: Fourier pseudo-spectral methods (dealiasing), Chebyshev collocation, ETDRK4 for KdV,
  Kuramoto-Sivashinsky and Allen-Cahn.
- `pde/method-of-lines.ts`: semi-discretisation of PDEs into ODE systems for `solveIvp`.
- `pde/operators.ts`: differential operators applied to functions by autodiff (gradient, divergence, Laplacian,
  time derivatives of networks): the residual machinery used by physics-informed networks.
- `pde/stability.ts`: von Neumann analysis, CFL conditions, numerical dispersion and dissipation.
- `pde/fokker-planck.ts`: density evolution for SDEs (Scharfetter-Gummel), stationary densities.
- `fields.ts`: Jacobians, divergence, curl, fixed points (seeded Newton), classification (node, spiral, saddle,
  centre), nullclines, streamlines, flow maps, invariant manifolds.
- `maps.ts`: iterated maps, orbits, cobwebs, bifurcation diagrams, Lyapunov exponents.
- `nonlinear.ts`: Poincare sections, limit cycles by shooting, Floquet multipliers, Lyapunov spectra of flows,
  correlation dimension, recurrence quantification, bifurcation continuation (fold and Hopf detection).
- `control/optimal.ts`: LQR, discrete LQR, LQE, LQG, finite-horizon Riccati recursions.
- `control/placement.ts`: pole placement (Kautsky-Nichols-Van Dooren, Ackermann).
- `control/predictive.ts`: linear and nonlinear MPC (condensed QP, constraints, receding horizon as a traced algorithm).
- `control/pid.ts`: PID control with anti-windup and derivative filtering, Ziegler-Nichols and Cohen-Coon tuning,
  relay auto-tuning.
- `control/trajectory.ts`: trajectory optimisation: single and multiple shooting, direct collocation, iLQR and DDP.
- `filtering/kalman.ts`: Kalman filter (Joseph form, missing data), RTS smoother, steady-state filter, log-likelihood,
  consistency checks.
- `filtering/nonlinear.ts`: extended, unscented and ensemble Kalman filters.
- `filtering/particle.ts`: bootstrap and auxiliary particle filters, particle smoothing.
- `filtering/changepoint.ts`: Bayesian online changepoint detection (conjugate predictives, run-length posteriors,
  hazard functions), PELT, binary segmentation.
- `filtering/em.ts`: EM for linear-Gaussian state-space parameters (traced).
- `identification.ts`: ARX, ARMAX, prediction-error methods, N4SID, dynamic mode decomposition, SINDy, extended DMD.

### 4.27 `pgm/` (tier 8): probabilistic graphical models (pgmpy, libDAI)

oracle: pgmpy, networkx, pyAgrum (loopy BP, influence diagrams), hmmlearn, sklearn-crfsuite and pytorch-crf (CRFs),
causal-learn (GES, FCI, DirectLiNGAM), numpy and scipy (Gaussian models), brute-force enumeration

A model is plain data. Every model type describes itself as a `Structure` (roles: latent, observed, factor,
deterministic, decision, utility; plates and slices), and inference algorithms emit `message` events, so the drawing of
a model can animate its inference. The `probprog` engines reuse these algorithms. Every discrete model (Bayes network,
Markov network, factor graph) is accepted by every discrete inference routine. `models` and `inference` are inlined
into `whitebox-engine/pgm`; `learning` and `causal` are namespaces.
- `factor.ts`: discrete factors in log space (`factor`, `logFactor`, `unitFactor`, `DiscreteVariable`): product
  (`factorProduct`, `factorProductAll`), sum and max marginalisation (`marginalise`, `maxMarginalise`,
  `factorMarginal`), reduction by evidence (`reduceFactor`), normalisation, division (`factorDivide`, 0/0 = 0), powers,
  permutation, lookups, argmax, entropy, `allAssignments`.
- `gaussian-factor.ts`: canonical-form Gaussian factors: product, division, marginalisation (Schur complement),
  conditioning (`gaussianReduce`), conversion to and from moment form, log mass and log value, permutation, and the
  factor of a linear-Gaussian conditional.
- `graphs.ts`: DAGs (`dag`, `topologicalOrder`, `dagEdges`, `ancestors`, `descendants`), undirected graphs, moral
  graphs (`moralGraph`), interaction graphs of factors, connected components, graph separation.
- `triangulation.ts`: elimination costs and greedy orderings (min-degree, min-fill, min-weight, weighted min-fill),
  `triangulate` (fill-ins and cliques), maximum cardinality search, `isChordal`, maximal cliques (Bron-Kerbosch),
  induced width, `treewidth` (exact by subset DP up to about 20 vertices, or heuristic), `junctionTree` (maximum
  spanning tree on separators), `hasRunningIntersection`.
- `structure.ts`: `toStructure` for every model type (Bayes networks, DAGs, Markov networks, factor graphs,
  linear-Gaussian networks, GMRFs, DBNs, HMMs, LDSs, linear-chain and general CRFs, plate models, influence diagrams,
  SCMs, PDAGs and PAGs).
- `models/bayes-net.ts`: Bayes networks (nodes, states, parents, CPTs, validation), d-separation, Bayes-ball active
  trails, minimal d-separators, local and pairwise independencies, Markov blankets, children, CPT rows, ancestral
  (forward) sampling, joint probabilities, factors.
- `models/markov-network.ts`: Markov networks (cliques and potentials), moralisation (`moralise`), neighbours and
  separation, maximal cliques, Ising and Potts models on grids or graphs, Gibbs distributions and energies.
- `models/factor-graph.ts`: factor graphs, `DiscreteModel`, conversions from the other model types (`toFactorGraph`,
  `toMarkovNetwork` in markov-network.ts), evidence coding, state names, variable-factor adjacency, `isFactorTree`.
- `models/linear-gaussian.ts`: linear-Gaussian Bayes networks (joint Gaussian, factors, posteriors, sampling,
  log-likelihood) and Gaussian Markov random fields in moment and information form (from a linear-Gaussian network,
  conditionals, covariance, log density, Cholesky sampling, graph, factor).
- `models/dynamic.ts`: dynamic Bayes networks (two-slice templates), unrolling, interface variables, exact filtering by
  the interface algorithm (`dbnFilter`).
- `models/state-space.ts`: HMMs (categorical and full-covariance Gaussian emissions: chain potentials, sampling, the DBN
  form) and linear dynamical systems (sampling, the unrolled linear-Gaussian network with correlated noise as
  within-slice arcs, smoothing with the log-likelihood) as DBNs.
- `models/crf.ts`: conditional random fields: CRFsuite-style attribute templates over token columns (offset windows,
  skipped or CRF++-padded boundaries, bias) and attribute vocabularies with frequency cut-offs; linear-chain CRFs
  (potentials as chains, scores, conditional log-likelihoods, marginals, Viterbi decoding, the penalised objective with
  its gradient, first-order training as a traced run); general CRFs over arbitrary cliques with per-assignment feature
  tables (conditional factor graphs, objective with exact or loopy-BP marginals, training). The CRF++ template parser
  and the estimators' training algorithms stay in `text/templates.ts` and `methods/models/crf.ts`.
- `models/plates.ts`: plate notation and template models, grounding against plate elements and relations, grounded
  evidence, pooled (parameter-sharing) maximum likelihood.
- `models/influence.ts`: influence diagrams (decision and utility nodes, no-forgetting information sets), policy
  networks, expected utility, optimal policies by backward induction, value of information.
- `inference/enumerate.ts`: brute-force enumeration.
- `inference/elimination.ts`: variable elimination (sum-product and max-product with traceback) with intermediate factor
  events, `logPartition`, `mapQuery`.
- `inference/junction-tree.ts`: Shafer-Shenoy and HUGIN propagation (sum and max), cross-clique queries, marginals, MAP.
- `inference/belief-propagation.ts`: sum- and max-product belief propagation (tree, flooding, sequential and residual
  schedules, damping, loopy convergence), Bethe free energy.
- `inference/gaussian-bp.ts`: Gaussian belief propagation in information form, walk-summability.
- `inference/chain.ts`: forward-backward (scaled and log space, pairwise posteriors), Viterbi, posterior decoding,
  forward filtering and backward sampling.
- `inference/sampling.ts`: forward (`ancestralSample` in models/bayes-net.ts), rejection and likelihood-weighted
  sampling with effective sample sizes, Gibbs sampling on factor graphs, weighted marginals, approximate queries.
- `inference/variational.ts`: naive mean field, variational message passing (on discrete factor graphs, where it has
  the fixed points of mean field), tree-reweighted belief propagation and its upper bound with spanning-tree edge
  probabilities. TODO: VMP on conjugate-exponential (Dirichlet, Gaussian) nodes.
- `inference/ep.ts`: expectation propagation on factor graphs (discrete fully factorised sites; Gaussian sites of a
  latent-Gaussian model: probit, Gaussian and quadrature sites; damping, power EP).
- `inference/map.ts`: MAP and MPE by max-product (elimination and junction trees elsewhere; loopy BP decoding here), by
  LP relaxation (MPLP on its dual, with the integrality gap), and by local search (ICM with restarts).
- `learning/parameters.ts`: state counts, maximum-likelihood and Bayesian (BDeu, K2, Dirichlet) parameter estimation,
  data log-likelihood, EM with missing data and latent variables (traced), linear-Gaussian least squares.
- `learning/scores.ts`: log-likelihood, BIC, AIC, K2, BDeu and BDs local and structure scores, Gaussian log-likelihood,
  BIC and AIC.
- `learning/independence.ts`: chi-square and G tests (stratified), Fisher z and partial-correlation tests, oracles.
- `learning/equivalence.ts`: PDAGs, skeletons, immoralities, Markov equivalence, Meek's rules, CPDAGs, consistent
  extensions, structural Hamming distance.
- `learning/structure.ts`: Chow-Liu trees (and TAN), the PC algorithm (stable; first or union separating sets), hill
  climbing with any score (BIC, BDeu, ...), tabu list and parent limit, mutual information.
- `causal/scm.ts`: structural causal models (discrete and linear assignments plus noise), sampling, induced Bayes and
  linear-Gaussian networks, `scmStructure` (also through `toStructure`).
- `causal/interventions.ts`: the do-operator (graph surgery), mutilated models, interventional distributions and
  moments, average causal effects.
- `causal/identification.ts`: back-door and front-door criteria, adjustment sets (all minimal, minimal with latents),
  the adjustment formulas, simple ID-algorithm cases (`identifyEffect`), instrumental-variable checks.
- `causal/counterfactuals.ts`: abduction, action and prediction; twin networks; probabilities of necessity and
  sufficiency; linear-Gaussian counterfactuals.
- `causal/discovery.ts`: PC (re-exported from `learning`), GES, DirectLiNGAM (least-squares effects; TODO:
  causal-learn's adaptive-lasso pruning, which needs LARS) and NOTEARS (projected-gradient inner solves; switch to
  `optim`'s L-BFGS-B once it exists).
- `causal/fci.ts`: FCI with Possible-D-Sep pruning and Zhang's rules R0-R10 (R4 from the separating sets), PAGs.

### 4.28 `metrics/` (tier 8): evaluation metrics (scikit-learn, torchmetrics)

oracle: scikit-learn, torchmetrics, scipy, statsmodels, fairlearn, pingouin, krippendorff, properscoring, sacrebleu,
rouge-score, jiwer, pycocotools, torchvision, MONAI, scikit-image, sewar, prdc, pystoi, dtw-python

The files follow the encyclopedia's evaluation chapters, in its order. Every metric has a `MetricSpec` and returns a
`MetricResult` (a few return richer records: curves, reports, PRDC, permutations); per-sample metrics accept sample
weights where the reference does, and metrics over rows of samples (targets, predictions) can attach a confidence
interval (`withInterval`). Metrics that need a pretrained network (LPIPS, DISTS, FID, KID, CLIPScore, BERTScore,
Inception score) take its features, embeddings or outputs, and DNSMOS takes the networks as functions; whitebox ships
none of those weights.
- `foundations.ts`: choosing a metric for a decision (`expectedCost`, `expectedUtility`, `bayesDecision`,
  `costThreshold`), `withInterval` (bootstrap: percentile, basic and BCa; analytic (Wald); Wilson; DeLong, with
  `delongVariance`), paired comparisons of two models on one test set (`pairedBootstrap`, `mcnemarTest`,
  `delongTest`), `evaluate` (many metrics in one pass, with shared bootstrap intervals).
- `metric.ts`: `Metric`, `defineMetric`, `metricResult`, sample weights, `weightedPercentile`, row selection and
  `streaming` (update / compute over batches).
- `classification.ts`: confusion matrices (`confusionMatrix`, `multilabelConfusionMatrix`) and the rates computed from
  them: `accuracyScore`, `subsetAccuracy`, `balancedAccuracyScore`, `precisionScore`, `recallScore`, `fbetaScore`,
  `f1Score`, `precisionRecallFscoreSupport`, `sensitivityScore`, `specificityScore`, `negativePredictiveValue`,
  `informedness`, `markedness`, `jaccardScore`; binary, micro, macro, weighted, samples and per-class averaging, label
  subsets, string labels and zero-division control.
- `classification-summary.ts`: `matthewsCorrcoef`, `cohenKappaScore` (unweighted, linear, quadratic),
  `quadraticWeightedKappa` (ordinal), likelihood ratios (`positiveLikelihoodRatio`, `negativeLikelihoodRatio`),
  `diagnosticOddsRatio`, `hammingLoss`, `zeroOneLoss`, `topKAccuracyScore`, multi-label ranking
  (`labelRankingAveragePrecisionScore`, `coverageError`, `labelRankingLoss`) and `classificationReport` (data and
  sklearn's text).
- `curves.ts`: `confusionMatrixAtThresholds`, ROC curves (`rocCurve`) and AUROC (`rocAucScore`: binary with the
  McClish partial AUC, one-vs-rest, Hand and Till one-vs-one, multilabel; macro, weighted and micro), `auc`, DET
  curves (`detCurve`).
- `precision-recall.ts`: precision-recall curves (`precisionRecallCurve`) and average precision
  (`averagePrecisionScore`), PR-gain (`prGainCurve`, `prGainAuc`, ported from Flach and Kull's reference code).
- `operating-points.ts`: operating points at a fixed error rate (`recallAtFixedPrecision`, `precisionAtFixedRecall`,
  `sensitivityAtSpecificity`, `specificityAtSensitivity`, `fprAtTpr`), the equal error rate (`equalErrorRate`:
  interpolated or nearest), Youden's index (`youdenIndex`) and minimum expected cost (`minimumExpectedCost`, minDCF);
  cost curves (`costCurve`).
- `probabilistic.ts`: `logLoss`, `d2LogLossScore`, `brierScoreLoss` and its decomposition (`brierDecomposition`),
  `calibrationCurve`, expected, maximum and classwise calibration errors (`expectedCalibrationError` with l1, l2 and
  max norms, `maximumCalibrationError`, `classwiseCalibrationError`; uniform and quantile binning).
- `scoring-rules.ts`: proper scoring rules for any `Distribution`: logarithmic (`logScore`), spherical
  (`sphericalScore`), CRPS (`crps`: closed forms for the normal, Laplace, logistic, exponential and uniform, quadrature
  otherwise) and the energy score (`energyScore`) from samples.
- `fairness.ts`: group rates (`groupRates`); demographic parity, equalised odds, equal opportunity (differences and
  ratios, between groups or to overall), predictive parity and calibration by group (`calibrationByGroup`); disparate
  impact.
- `regression.ts`: MSE, RMSE, MAE, median absolute error, maximum error, R-squared (`r2Score`, `adjustedR2Score`),
  explained variance, MSLE, MAPE, log-cosh (`meanLogCoshError`), Huber (`meanHuberLoss`).
- `deviance.ts`: Poisson, gamma and Tweedie deviances, pinball loss, D-squared scores (`d2AbsoluteErrorScore`,
  `d2PinballScore`, `d2TweedieScore`).
- `correlation-agreement.ts`: Pearson (weighted), Spearman and Kendall (tau-b and tau-c) as metrics (TODO: make
  them aliases of `stats/correlation.ts` once the stats domain exists; for now they are implemented here), Cramer's
  V, Theil's U, Lin's concordance correlation, Bland-Altman limits, intraclass correlation (all six Shrout-Fleiss
  forms), weighted Cohen's kappa, Fleiss' kappa (and Randolph's), Krippendorff's alpha (nominal, ordinal, interval,
  ratio, missing data).
- `distances.ts`: Minkowski distances, cosine similarity, Mahalanobis distance, Procrustes disparity (metrics over
  the canonical `numerics/geometry` functions), Hausdorff and Chamfer distances between point sets.
- `forecasting.ts`: MAPE (`mape`, an alias of the regression metric), sMAPE, WMAPE, MASE, RMSSE, MSIS, CRPS for
  samples (`crpsEnsemble`, fair or not) and closed forms (`crpsNormal`), interval score, coverage
  (`intervalCoverage`), quantile loss (mean and weighted), Theil's U2.
- `ranking.ts`: `precisionAtK`, `recallAtK`, `hitRate`, `rPrecision`, `meanAveragePrecision` (trec_eval, min and
  retrieved denominators), `meanReciprocalRank`, `dcgScore` and `ndcgScore` (linear and exponential gains,
  tie-averaged), `expectedReciprocalRank`, `rankBiasedPrecision`; beyond-accuracy measures (`catalogCoverage`,
  `intraListDiversity`, `novelty`, `serendipity`, `exposureGini`).
- `clustering.ts`: Rand, adjusted Rand and Fowlkes-Mallows (`randScore`, `adjustedRandScore`,
  `fowlkesMallowsScore`, `pairConfusionMatrix`); `mutualInfoScore`, NMI variants (`normalizedMutualInfoScore`:
  arithmetic, geometric, min, max), AMI (`adjustedMutualInfoScore`), V-measure, homogeneity and completeness;
  `purityScore`; B-cubed (`bCubed`, `bCubedF1`); contingency tables (`contingencyMatrix`).
- `cluster-validity.ts`: silhouette (`silhouetteScore`, `silhouetteSamples`, any distance or precomputed),
  Calinski-Harabasz, Davies-Bouldin and Dunn indices.
- `detection.ts`: IoU, GIoU, DIoU and CIoU of boxes (`boxIou`, `generalizedBoxIou`, `distanceBoxIou`,
  `completeBoxIou`); mean average precision (`detectionMeanAveragePrecision`: the COCO protocol with all twelve
  COCOeval statistics, crowds and size ranges; VOC all-point and 2007 11-point, with difficult objects).
- `segmentation.ts`: Dice (`diceCoefficient`), `maskIou`; Hausdorff distance (and HD95) for mask boundaries
  (`maskHausdorffDistance`, MONAI's definition with spacing); boundary F1; panoptic quality (with void and modified
  stuff classes); per-class mean IoU (`meanIou`).
- `image-quality.ts`: MSE (`imageMeanSquaredError`), PSNR, SSIM (skimage's uniform and Gaussian windows), MS-SSIM,
  VIF (pixel domain), LPIPS and DISTS from supplied features (`lpipsFromFeatures`, `distsFromFeatures`), SAM, ERGAS,
  RASE, UIQI; images `[H, W]` or `[H, W, C]`, channels last. TODO: move the private filters of `_images.ts` to
  `signal/` once that domain exists.
- `generative.ts`: Inception score (from class probabilities), FID (`frechetDistance`) and KID
  (`kernelInceptionDistance`) from features, precision, recall, density and coverage
  (`precisionRecallDensityCoverage`), CLIPScore from embeddings (`clipScore`), Vendi score (any order).
- `text.ts`: WER, CER, MER, WIL and WIP (jiwer's alignment); BLEU (corpus and sentence, smoothing, the 13a tokeniser
  `tokenize13a`); chrF and chrF++ (`chrfScore`, `wordOrder: 2`); TER (Tercom shifts and normalisation flags); ROUGE-N,
  ROUGE-L and ROUGE-Lsum (rouge-score; a stemmer can be supplied); SQuAD exact match and token F1; BERTScore from
  embeddings (idf weights, baseline rescaling); perplexity; distinct-n. Private `_sequences.ts` (edit alignment, LCS,
  n-gram counts) and `_ter.ts`. TODO: move the alignment and LCS to `text/similarity.ts` and default ROUGE's stemmer to
  `text/stem.ts` once the text domain exists.
- `audio.ts`: SNR, segmental SNR, SDR (BSS Eval with a distortion filter), SI-SDR and SI-SNR, permutation-invariant
  matching (`permutationInvariant`, Hungarian algorithm via `optim` `linearSumAssignment`), STOI and ESTOI (`shortTimeObjectiveIntelligibility`, pystoi with its
  resampling), log-spectral distance, mel-cepstral distortion from cepstra (optionally DTW-aligned), DNSMOS from a
  supplied model (torchmetrics' windows and calibration, 16 kHz). Private `_audio.ts` (windows, `upfirdn`,
  `resamplePoly`, framed power spectra). TODO: PESQ (ITU-T P.862), whose reference implementation is ITU-copyrighted
  C that cannot be ported under the licence; move `_audio.ts` to `signal/` once it exists.

### 4.29 `losses/` (tier 8): differentiable objectives (torch.nn losses, optax losses)

oracle: torch.nn.functional, optax, MONAI, POT, torchvision, scikit-learn, scipy

Training objectives with a uniform `reduction` option. Evaluation metrics live in `metrics`; where a loss and a metric
share a concept they share a definition but not a name.
- `regression.ts`: `mse`, `mae`, `huber`, `smoothL1`, `logCosh`, `pinball`, `expectile`, `poissonNll`, `gaussianNll`,
  `tweedieDeviance`.
- `classification.ts`: binary and categorical cross-entropy (`binaryCrossEntropy`, `binaryCrossEntropyWithLogits`,
  `crossEntropy`: with logits, label smoothing, class weights and ignore index), `focal` (sigmoid) and
  `categoricalFocal`, `nllLoss`, `multiLabelSoftMargin`.
- `large-output.ts`: losses for very many classes: hierarchical softmax (`huffmanCoding`, `hierarchicalSoftmax`,
  `hierarchicalSoftmaxLogProbs`), sampled softmax (`sampledSoftmax`, `sampleClasses` with uniform and log-uniform
  proposals). TODO: take the Huffman tree from `information/source-coding.ts` once that domain exists.
- `margin.ts`: `hinge`, `squaredHinge`, `logistic`, `exponential`, `modifiedHuber`, `perceptron`, `crammerSinger`,
  `westonWatkins`.
- `density.ts`: negative log-likelihood of any `Distribution` (`negativeLogLikelihood`) or conditional distribution
  (`conditionalNll`), mixture-density NLL (`mixtureDensityNll`), heteroscedastic Gaussian NLL with beta-NLL weighting
  (`heteroscedasticGaussianNll`), censored likelihoods (`censoredNll`: exact, left, right and interval).
- `ranking.ts`: `marginRanking`, `rankNet`, `pairwiseHinge`, `bpr`, `lambdaRank` (NDCG@k deltas), `listNet`,
  `softmaxRanking`, `listMle`, `approxNdcg`; padded lists through a mask.
- `contrastive.ts`: `infoNce` (in-batch, shared or per-query negatives), `ntXent`, `tripletMargin`, `contrastive`,
  `cosineEmbedding`, `hingeEmbedding`, `supervisedContrastive`, `clipLoss`, ArcFace and CosFace margins
  (`arcFaceLogits`, `cosFaceLogits`, `arcFace`, `cosFace`).
- `segmentation.ts`: `dice` (soft and squared), `tversky`, `focalTversky`, `lovaszHinge`, `lovaszSoftmax`; classes last,
  index or one-hot targets, activations, background, batch pooling, ignore index and class weights as MONAI.
- `generative.ts`: GAN losses (`discriminatorLoss`, `generatorLoss`: minimax, non-saturating, Wasserstein, hinge, least
  squares; `gradientPenalty` (WGAN-GP) and `r1Penalty`), ELBO terms (`gaussianKl`, `sampledKl`, `negativeElbo`,
  `bernoulliLogLikelihood`, `gaussianLogLikelihood`, `importanceWeightedBound`), `denoisingScoreMatching`, diffusion
  objectives (`diffusionTarget`, `diffusionLoss`: epsilon, x0 and v), flow matching (`flowMatchingPath`,
  `flowMatching`).
- `divergence.ts`: `klDiv` (torch, with 'batchmean'), `klDivergence`, `jensenShannon`, `distillation`, `mmd` (Gaussian,
  Laplacian and energy kernels, multi-scale, median heuristic, biased and unbiased) and `sinkhorn` (log domain, weighted,
  debiased) losses.
- `physics.ts`: `fieldDerivatives` (gradient, second derivatives and Laplacian by autodiff), `pdeResidualLoss`,
  `boundaryLoss` (Dirichlet and Neumann) and `initialConditionLoss`, adaptive loss balancing (`gradientNormWeights`,
  `ntkWeights`) for physics-informed training.
- `regularisers.ts`: `l1Penalty`, `l2Penalty`, `elasticNet`, `groupLasso` (and sparse group lasso), `nuclearNorm`,
  `orthogonality`, `entropyPenalty`; penalties take any pytree of parameters.

### 4.30 `rl/` (tier 8): reinforcement learning infrastructure (gymnasium, PettingZoo, stable-baselines3 building blocks)

oracle: gymnasium step-for-step dynamics, stable-baselines3 for returns and advantages, rliable for aggregate metrics
- `env.ts`: `Env<S, O, A>` (spec, observation and action spaces, `reset`, `step`, `render` required, optional `model`
  (the tabular `P` table: states, actions, initial distribution, outcomes) for planning, `oracle` for regret,
  `resetFrom` for state that persists across episodes, `observe`), `EnvSpec`, `EnvDefinition` (factory with an
  options `Space`), `resetAfter`, `unwrapped`, `isEnvDefinition`.
- `gym.ts`: `GymEnv` (gymnasium-style wrapper), `make` (by definition or by id in a registry, time-limit override,
  wrappers), `listEnvironments` (finds definitions in an imported module).
- `multi-agent.ts`: multi-agent environments in turn-based (`AecEnv`) and simultaneous (`ParallelEnv`) forms, per-agent
  spaces, `aecToParallel`, `parallelToAec`, `parallelTimeLimit`, `singleAgentView` (one agent against fixed policies),
  `playAec`, `playParallel`.
- `wrappers.ts`: time limits, episode statistics, reward normalisation (`RunningMeanStd`), reward, action and
  observation transforms, reward clipping, action clipping and rescaling, action repeat, sticky actions.
- `observation-wrappers.ts`: observation normalisation, frame stacking, flattening, one-hot observations, time-aware
  observations, pixel observations rendered from scenes (`scene/raster.ts`).
- `vector.ts`: vectorised environments with independent keys and auto-reset (same-step and next-step), `GymVectorEnv`.
- `agent.ts`: `Agent<G, O, A>` (`init`, `act`, `learn`, `greedy`, `stats`), `Policy`, `Transition`,
  `behaviourPolicy`, `greedyPolicyOf`, `policyAgent`.
- `rollout.ts`: rollouts as traced algorithms (episode events, optional frames), `runEpisode`, `evaluatePolicy`,
  `compareAgents` (common random numbers).
- `buffers.ts`: replay buffers (persistent, chunked), prioritised replay (sum trees), n-step buffers, on-policy rollout
  buffers, episode storage for sequence models.
- `returns.ts`: discounted returns, GAE, TD targets, lambda returns, n-step returns, V-trace, Retrace.
- `policies.ts`: epsilon-greedy, Boltzmann, Gaussian (with tanh squashing), categorical and deterministic policies,
  Gaussian and Ornstein-Uhlenbeck exploration noise.
- `mdp.ts`: tabular MDPs (from arrays or environment models), value iteration, policy iteration, policy evaluation (exact
  and iterative), Q from V, greedy policies, occupancy measures, `tabularEnv` (environments from `P` tables). TODO:
  linear-programming solutions, once `optim/linear-programming.ts` provides `linprog`.
- `planning.ts`: planning with models (MCTS over environment models, UCT and PUCT with priors and values, and two-player
  game search with negamax backups, written here while `search/adversarial` is not built; rollout planners, CEM over
  action sequences, MPPI).
- `models.ts`: learned-dynamics interfaces (deterministic and probabilistic ensembles) for model-based agents.
- `exploration.ts`: count-based bonuses, pseudo-counts, random network distillation and curiosity interfaces.
- `off-policy.ts`: IPS, SNIPS, clipped IPS, direct method, doubly robust, fitted Q evaluation.
- `bandits.ts`: bandit problem interface, regret, Lai-Robbins bounds, KL helpers.
- `evaluation.ts`: aggregate metrics across runs (interquartile mean, optimality gap, probability of improvement) with
  bootstrap intervals, learning-curve aggregation.
- `training.ts`: `trainAgent` (episodes or steps as a traced algorithm, checkpoints with doubling spacing, greedy
  evaluation), `replayEpisode` (rebuild any episode, including its rendered frames, from the nearest checkpoint).

### 4.31 `games/` (tier 8): game theory (nashpy, OpenSpiel)

oracle: nashpy, OpenSpiel
- `normal-form.ts`: normal-form games, dominance, best responses, pure and mixed Nash equilibria (support enumeration,
  Lemke-Howson), zero-sum solutions by LP, correlated equilibria.
- `learning.ts`: fictitious play, replicator dynamics, regret matching, multiplicative weights in games, convergence
  traces.
- `extensive-form.ts`: game trees with information sets, backward induction, sequence form, CFR, CFR+, Monte Carlo CFR;
  Kuhn and Leduc poker.
- `cooperative.ts`: coalitional games, Shapley values (exact and sampled), the core, Banzhaf indices; used by
  `learn/explain/shapley.ts`.
- `mechanisms.ts`: auctions (first price, second price, VCG), voting rules (plurality, Borda, Condorcet, instant
  runoff), social choice paradoxes.
### 4.32 `probprog/` (tier 9): probabilistic programming and automatic inference (NumPyro, Pyro)

oracle: numpyro and pyro on small models (posterior moments within Monte Carlo error), pgmpy for discrete programs

A model is a function that calls the primitives below inside an effect-handler context. `analyse` traces it once and
describes it; every engine says whether it applies and at what cost; `infer` picks one, records the reason, runs it as
a traced algorithm and returns a `Posterior`. Adding an engine is adding a file to `engines/`.
- `primitives.ts`: `sample`, `observe`, `deterministic`, `param`, `plate`, `factor`, `markov` (sequential dependence),
  the `Model` type.
- `handlers.ts`: `trace`, `seed`, `condition`, `substitute`, `replay`, `block`, `scale`, `mask`, `doIntervention`,
  `enumerate` (mark discrete sites for exact summation).
- `density.ts`: `logJoint`, `potentialEnergy` (to an unconstrained `LogDensity` with Jacobian corrections),
  initialisation strategies, constrain and unconstrain maps.
- `analysis.ts`: `analyse(model, data)`: sites with supports, shapes and observed status; plates; the dependency graph;
  discrete cardinalities and the size of their joint space; conjugate prior-likelihood pairs (from
  `dists/exponential-family.ts`); linear-Gaussian structure; chain, tree or loopy structure; whether a likelihood
  exists (or only a simulator); differentiability.
- `graph.ts`: `toStructure` for programs (plates, roles, dependencies), so any program draws like a graphical model.
- `engines/engine.ts`: the `InferenceEngine` contract (`name`, `applies(analysis)` returning feasibility, estimated cost
  and a reason; `run`; the posterior form it produces).
- `engines/enumeration.ts`: exact inference for discrete programs through `pgm` variable elimination and junction trees.
- `engines/conjugate.ts`: closed-form posteriors and predictives for conjugate programs.
- `engines/gaussian.ts`: linear-Gaussian programs (exact Gaussian conditioning, Kalman filtering and smoothing for
  chains, Gaussian belief propagation).
- `engines/chain.ts`: discrete-state chains (HMM-like programs) by forward-backward and FFBS.
- `engines/em.ts`: EM and MAP-EM for programs whose latents are discrete or conjugate (mixtures, HMMs, factor analysis,
  mixtures of experts); closed-form M-steps where they exist, gradient M-steps otherwise.
- `engines/gibbs.ts`: Gibbs sampling with conjugate and discrete conditionals, collapsed where possible (topic models).
- `engines/nuts.ts`: NUTS and HMC on the unconstrained density, with enumerated discrete sites summed out.
- `engines/svi.ts`: stochastic variational inference (automatic guides, ELBO estimators, subsampling over plates).
- `engines/laplace.ts`: Laplace approximation at the MAP estimate.
- `engines/smc.ts`: sequential Monte Carlo for programs with `markov` structure.
- `engines/importance.ts`: likelihood weighting and importance sampling, marginal-likelihood estimates.
- `engines/abc.ts`: approximate Bayesian computation for simulator programs without likelihoods (rejection ABC, ABC-SMC,
  summary statistics).
- `infer.ts`: `plan(model, data)` (ranked engines with reasons), `infer(model, data, { engine: 'auto' | engine,
  budget })`, `explainPlan`.
- `guides.ts`: automatic guides (`autoDelta`, `autoNormal`, `autoMultivariateNormal`, `autoLowRankMultivariateNormal`,
  `autoLaplace`, normalising-flow guides).
- `elbo.ts`: ELBO estimators (reparameterised, score function with baselines, mean-field analytic, Renyi, importance
  weighted).
- `posterior.ts`: `Posterior` (exact distributions, variational approximations or chains, behind one interface:
  `marginal(site)`, `sample`, `mean`, `quantiles`, `logProb` where available), summaries.
- `predictive.ts`: prior and posterior predictive distributions, per-observation log-likelihoods.
- `criticism.ts`: WAIC, PSIS-LOO (with Pareto k diagnostics), posterior predictive checks, model comparison and
  stacking weights.

### 4.33 `learn/` (tier 9): the toolkit around models (scikit-learn's non-model modules)

oracle: scikit-learn, mapie for conformal, captum and shap for explanations
- `estimator.ts`: the estimator contract (`name`, `hyper`, `hyperSpace`, `fit`), `Fitted` (parameters as data, verbs as
  methods, `revive` after deserialisation), capability guards, `cloneEstimator`, `EstimatorClass` (the base of
  sklearn-style class wrappers).
- `validation.ts`: input checks with clear shape errors, sample weights, `asFloatMatrix`.
- `preprocessing/scalers.ts`: standard, min-max, max-abs and robust scalers, normaliser, quantile and power
  transformers.
- `preprocessing/encoders.ts`: one-hot, ordinal, label, target (cross-fitted), multi-label binarisers.
- `preprocessing/features.ts`: polynomial features, spline features, binning, function transformers, interaction
  features.
- `preprocessing/imputation.ts`: simple, k-NN and iterative imputers, missing indicators.
- `preprocessing/resampling.ts`: random over- and under-sampling, SMOTE, borderline SMOTE, ADASYN, Tomek links,
  NearMiss.
- `feature-selection.ts`: variance threshold, univariate selection (F test, chi-square, mutual information), RFE and
  RFECV, sequential selection, selection from model importances, stability selection.
- `compose.ts`: `pipeline`, `featureUnion`, `columnTransformer`, `transformedTargetRegressor`, step inspection
  (intermediate outputs recorded by policy), caching of fitted steps.
- `model-selection/splitters.ts`: holdout and train/validation/test splits; k-fold, stratified, group, stratified
  group, shuffle, stratified shuffle, group shuffle, leave-one-out, leave-p-out, leave-one-group-out, repeated variants;
  time-series splits (expanding window, sliding window, blocked, purged and embargoed); spatial and cluster blocks;
  predefined splits.
- `model-selection/cross-validation.ts`: `crossValidate` (traced, one child run per fold), `crossValPredict`, learning
  and validation curves, permutation-test scores, nested cross-validation.
- `model-selection/criteria.ts`: AIC, AICc, BIC, Mallows' Cp, GCV, leave-one-out shortcuts for linear smoothers, the
  one-standard-error rule, the .632 bootstrap.
- `calibration.ts`: Platt scaling, isotonic calibration, temperature and vector scaling, beta calibration, histogram
  binning, cross-fitted calibrated classifiers.
- `conformal.ts`: split conformal regression, conformalised quantile regression, conformal classification (LAC, APS,
  RAPS), Mondrian conformal, jackknife+ and CV+, conformal for time series (adaptive conformal inference).
- `inspection.ts`: permutation importance, partial dependence, ICE, accumulated local effects, H-statistics,
  functional ANOVA.
- `explain/gradients.ts`: saliency, input times gradient, integrated gradients, SmoothGrad, DeepLIFT, expected
  gradients.
- `explain/shapley.ts`: exact, kernel, sampling and tree SHAP, interaction values (games from `games/cooperative.ts`).
- `explain/surrogates.ts`: LIME, anchors, global surrogate trees, fidelity.
- `explain/counterfactuals.ts`: Wachter counterfactuals, diverse counterfactuals, FACE, actionable recourse.
- `valuation.ts`: influence functions, TracIn, data Shapley (truncated Monte Carlo), KNN-Shapley.
- `active-learning.ts`: the pool-based and stream-based active-learning loop as a traced algorithm (query, label,
  refit), query-strategy contract, budget handling.
- `privacy.ts`: Laplace, Gaussian (classic and analytic) and exponential mechanisms, randomised response, composition,
  RDP accounting for the subsampled Gaussian.

### 4.34 `nn/` (tier 9): neural networks (torch.nn, torch.nn.functional, Flax, Haiku)

oracle: PyTorch in float64 with exported weights (forward values and gradients)

Two equal forms. Functional: `init(key, ...) -> params` and `apply(params, x, ctx) -> y` over pytrees. Module:
PyTorch-shaped classes (`nn.Linear`, `nn.Sequential`, `forward`, `parameters`, `stateDict`) that hold params and call
the functional form. `ctx` carries the train flag, the dropout key, taps and buffer updates, all returned as data.
- `functional/activations.ts`: ReLU, ReLU6, leaky ReLU, ELU, CELU, SELU, GELU, SiLU, Mish, softplus, softsign,
  log-sigmoid, hard tanh, sigmoid and swish, PReLU, shrinkage functions, threshold, GLU, SwiGLU, GeGLU, ReGLU,
  `activation(name)`.
- `functional/linear.ts`: linear, bilinear, embedding (padding index, max norm), embedding bag (sum, mean, max;
  offsets, per-sample weights).
- `functional/conv.ts`: `conv` and `conv1d/2d/3d` (stride, padding or 'same', padding modes, dilation, groups),
  `convTransposed` and `convTranspose1d/2d/3d` (output padding or size), `depthwiseConv`, `separableConv`, `unfold`,
  `fold`; kernels from `numerics/convolution`.
- `functional/pooling.ts`: max (padding, dilation, ceil mode, indices), `maxUnpool`, average (padding counted or not,
  divisor override), adaptive average and max, global and Lp pooling, over any number of spatial axes.
- `functional/normalisation.ts`: batch (returning updated statistics), layer, RMS, group, instance, local response,
  `normalize`, weight and spectral normalisation.
- `functional/regularisation.ts`: dropout, channel dropout (1-D, 2-D, 3-D), alpha and feature alpha dropout, drop
  path, Gaussian noise (additive or multiplicative), cutout.
- `functional/attention.ts`: `attention` and `scaledDotProductAttention` (boolean or additive masks, causal, sliding
  window, bias, soft-capping, dropout, grouped-query heads), `maskedSoftmax`, `causalMask`, `localMask`,
  `splitHeads`/`mergeHeads`, `multiHeadAttention` (PyTorch's packed projections, key padding, `add_bias_kv`,
  `add_zero_attn`, per-head or averaged weights), linear attention (causal by prefix sums); weights are tapped by the
  modules.
- `functional/positional.ts`: sinusoidal and learned positions, rotary angles and `applyRotary` (half or interleaved
  pairs, partial rotary, position interpolation), ALiBi slopes and bias.
- `functional/recurrent.ts`: RNN, GRU and LSTM cells (LSTM projection), `scan` over time, `recurrent` (stacked,
  bidirectional runs with per-sequence lengths and dropout between layers), packed sequences (`packPaddedSequence`,
  `padPackedSequence`, `packedLengths`, `lengthsFromMask`).
- `functional/state-space.ts`: SSM discretisation (`discretise` by zero-order hold or bilinear, `discretiseDiagonal`),
  S4 convolution kernels (`ssmKernel`, Vandermonde for diagonal models), `causalConvolution`, `ssmRecurrence`,
  `linearRecurrence`, selective scan (Mamba, with gating), delta rule (DeltaNet), LRU (`lruInit`, `lruEigenvalues`,
  `lru`). TODO: complex-valued diagonal S4D (the LRU covers complex recurrences).
- `functional/graph.ts`: `edgeList`, `aggregate` and `messagePassing` (sum, mean, max), edge softmax, `addSelfLoops`,
  `gcnNorm`, GCN, GAT, GraphSAGE and GIN convolutions, graph pooling (`globalPool` by batch, `topKPool`).
- `functional/vision.ts`: `interpolate` (nearest, nearest-exact, linear, bilinear, bicubic, trilinear, area; size or
  scale factor, aligned corners), pixel shuffle and unshuffle, `pad` (PyTorch's F.pad modes), patch embedding.
- `functional/sampling.ts`: spatial transformers (`affineGrid` 2-D and 3-D, `gridSample` bilinear, nearest and
  bicubic with zeros, border or reflection padding, `identityTheta`), ROI Align and ROI pooling. TODO: volumetric
  (5-D) `gridSample`.
- `functional/sets.ts`: masked set pooling (`setPool`, `maskSet`), set attention (`mab`, `sab`, `isab`, `pma` of the
  Set Transformer).
- `init.ts`: Xavier, Kaiming, LeCun, orthogonal, truncated normal, uniform, normal, constant, zeros and ones, eye,
  Dirac, sparse, PyTorch's default uniform; gains and fans.
- `module/module.ts`: `Module` (forward, parameters, named parameters, buffers, state dicts, train and eval modes,
  `initParams`/`resetParameters`, freezing with `requiresGrad` and `trainableMask`), `ApplyContext` (train flag, key,
  buffers in and updated buffers out, taps, path), `childScope` and `assemble` for composites, `tap`, `functional`
  (conversion to the functional form: `init`, `apply`, `applyWithState`), `Parameter`.
- `module/containers.ts`: `Sequential`, `ModuleList`, `ModuleDict`, `Residual`, `Parallel`, `Lambda`, `Composite` (named
  parts with routed contexts, the base of the blocks), `mlp`.
- `module/layers.ts`: `Linear`, `Bilinear`, `Embedding` (with `fromPretrained`), `EmbeddingBag`, `Flatten`,
  `Unflatten`, `Identity`.
- `module/activations.ts`: a module per activation (`ReLU` to `Threshold`, `Softmax`, `LogSoftmax`, `Softmin`, `GLU`,
  `PReLU` with learnt slopes), `activationModule`.
- `module/conv.ts`: `Conv1d`, `Conv2d`, `Conv3d` (padding modes, groups, 'same'), `ConvTranspose1d/2d/3d`, `Unfold`,
  `Fold`.
- `module/pooling.ts`: `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `LPPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`,
  `AdaptiveMaxPool1d/2d/3d`, `GlobalPool`, `MaxUnpool`.
- `module/regularisation.ts`: `Dropout`, `Dropout1d/2d/3d`, `AlphaDropout`, `FeatureAlphaDropout`, `DropPath`,
  `GaussianNoise`, `Cutout`.
- `module/normalisation.ts`: `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d` (running statistics, momentum or cumulative
  average, train and eval), `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm1d/2d/3d` (optional affine and running
  statistics), `LocalResponseNorm`.
- `module/parametrisation.ts`: `WeightNorm` and `SpectralNorm` wrappers (PyTorch's `weight_g`/`weight_v` and
  `weight_orig`/`weight_u`/`weight_v` layouts; power-iteration vectors as buffers), `spectralNormEstimate`.
- `module/recurrent.ts`: `RNN`, `GRU`, `LSTM` (stacked, bidirectional, batch-first, initial states, packed or padded
  inputs with lengths, LSTM `projSize`; PyTorch's weight names) and `RNNCell`, `LSTMCell`, `GRUCell`.
- `module/attention.ts`: `MultiheadAttention` (PyTorch's layout, sequence- or batch-first), `GroupedQueryAttention`
  (grouped- and multi-query heads, rotary positions, sliding windows, soft-capping, cross-attention, `KVCache`),
  `emptyKVCache`.
- `module/transformer.ts`: `TransformerEncoderLayer`, `TransformerDecoderLayer` (post- or pre-norm), the stacks
  `TransformerEncoder` and `TransformerDecoder`, `Transformer`, `squareSubsequentMask`.
- `module/positional.ts`: `SinusoidalPositionalEncoding`, `LearnedPositionalEmbedding`, `RotaryEmbedding`.
- `module/vision.ts`: `Upsample`, `UpsamplingNearest2d`, `UpsamplingBilinear2d`, `PixelShuffle`, `PixelUnshuffle`,
  the padding modules (`ZeroPad`, `ConstantPad`, `ReflectionPad`, `ReplicationPad`, `CircularPad`, 1-D to 3-D),
  `PatchEmbedding`, `SpatialTransformer`.
- `module/graph.ts`: `GCNConv`, `GATConv`, `SAGEConv`, `GINConv` (PyTorch Geometric's parameter names).
- `module/sets.ts`: `SetAttentionBlock`, `InducedSetAttentionBlock`, `PoolingByMultiheadAttention`.
- `blocks/` (namespace `blocks`, on `Composite` from `module/containers.ts`):
  - `transformer.ts`: `FeedForward`, `GatedFeedForward` (SwiGLU, GeGLU), `TransformerBlock` (pre- or post-norm, RMS
    or layer norm, grouped-query self-attention with rotary positions, sliding windows, cross-attention, KV caches),
    `TransformerStack` (encoder, decoder or decoder-only, one cache per block), `EncoderDecoder` and
    `encoderDecoder`, `TransformerLM` (decoder-only language model, tied embeddings).
  - `vision.ts`: `BasicBlock` and `Bottleneck` (torchvision's names), `SqueezeExcitation`, `ConvNeXtBlock`, `MBConv`,
    `DoubleConv`, U-Net stages `UNetDown` and `UNetUp`, `FeaturePyramid`, `toTokens`.
  - `sequence.ts`: `causalPad`, `TemporalBlock` and `TCN` (dilated causal, weight-normalised), `WaveNetBlock` and
    `WaveNet` (gated, residual and skip, conditioning), `StackedRecurrent` (residual LSTM or GRU layers).
  - `experts.ts`: routing (`softmaxRouting`, `topKRouting`, `noisyTopKRouting`, `switchRouting`,
    `expertChoiceRouting`), `MixtureOfExperts` (sparse dispatch), `loadBalancingLoss`, `routerZLoss`,
    `importanceLoss`.
  - `continuous.ts`: `OdeSolver` and `SdeSolver` contracts (filled by `dynamics/ode` and `stochastic` solvers),
    `ODEBlock` (gradients by backpropagation through the solver or by the adjoint), `AugmentedODEBlock`,
    `SecondOrderODEBlock`, `CDEBlock`, `SDEBlock`.
  - `generative.ts`: `timestepEmbedding`, `noiseLevelEmbedding` and `fourierFrequencies`, `TimeEmbedding`,
    `DiffusionResBlock`, `SpatialSelfAttention`, `DiffusionDownStage` and `DiffusionUpStage`, `AffineCoupling`
    (RealNVP) and `AdditiveCoupling` (NICE).
- `describe.ts`: `describe` (a module to a `Structure`: blocks, tensor shapes, repeats, skip connections into `add`
  nodes, parallel branches, parameter counts, FLOPs, optionally composite blocks' parts), `describeFunction` (any
  functional model, by its computation graph), `layerTable`, `summary` (torchinfo-style, with trainable and frozen
  counts and sizes), FLOP estimates (`flops`, `functionFlops`, counted from traced primitives) and `memoryEstimate`.
- `training/state.ts`: `TrainState` (params, optimiser state, buffers, step and update counts, key, update scale,
  accumulation window), `trainState`, `lossAndGrads`, `applyGradients` (masking frozen parameters), `trainStep` (the
  functional step), `accumulate`, `hasFrozen`.
- `training/fit.ts`: `fitAlgorithm` and `fit` (minibatch epochs, buffers carried, gradient clipping and accumulation,
  frozen parameters, early stopping, validation metrics, divergence detection); records loss, metrics, learning rate
  and, by policy roles, per-layer gradient and parameter norms, gradients and tapped activations.
- `training/callbacks.ts`: Lightning-style hooks without Lightning's machinery: a `Callback` implements any of
  `onTrainStart`, `onEpochStart`, `onBatchStart`, `onAfterBackward` (with gradients), `onBeforeOptimizerStep` (with
  gradients and updates), `onBatchEnd`, `onEpochEnd`, `onValidationEnd` (with validation metrics) and `onTrainEnd`; each
  hook gets `log(name, value)` and `logTensor` (into the run's trace, like `self.log`), `stop(reason)` and
  `setLearningRateScale`; `onBatchEnd` gets the tapped activations. Built-ins: `gradNormLogger` (global and per
  parameter), `paramNormLogger`, `valueLogger` (gradient and parameter tensors), `learningRateMonitor`,
  `earlyStopping`, `bestCheckpoint`, `reduceLROnPlateau` (PyTorch's rule), `activationLogger` (activation taps).
- `training/serialise.ts`: `saveState`/`loadState` and `stateToText`/`stateFromText` (parameters and buffers as JSON
  with dtype and shape), `loadTorchStateDict` (weights exported by the oracle, with renaming).
- `language-model.ts`: the `LanguageModel` contract (a structural `Tokenizer`, `logits(tokens)`, optional embeddings
  and a cached `step`), implemented by local models and by adapters in `methods/llm`; `languageModelFromModule`,
  `bigramLanguageModel` and `characterTokenizer` (reference models), `nextTokenLogits`, `sequenceLogProb`,
  `perplexity`.
- `decoding/` (inline): `processors.ts` (temperature, top-k, top-p, min-p, repetition, frequency and presence
  penalties, composition), `constraints.ts` (allowed-token masks from automata and tries, so regular grammars; a
  pushdown automaton fits the same interface. TODO: compiling context-free grammars and JSON schemas to automata),
  `generate.ts`
  (`decodingAlgorithm` and `generate`: greedy or sampled, traced one token per step), `beam.ts` (beam search with
  GNMT's length penalty), `speculative.ts` (speculative decoding with exact acceptance and residual resampling).
- `quantise.ts`: quantisation (symmetric, affine, per-tensor and per-channel), dequantisation, fake quantisation with a
  straight-through gradient, GPTQ, AWQ, error reports (MSE, max error, SQNR, output error), `quantiseModule`.
- `pruning.ts`: magnitude (per tensor and global), structured (row and channel norms) and N:M pruning, masks,
  lottery-ticket rewinding and iterative magnitude pruning, sparsity schedules (constant, linear, polynomial),
  sparsity reports.
- `equivariant.ts`: p4 lifting and group convolutions (functions and modules), orientation pooling, `rotate90`,
  equivariance and invariance errors.

### 4.35 `tune/` (tier 10): parameter selection for anything (Optuna, Ray Tune ideas)

oracle: statistical agreement with optuna and scikit-optimize on fixed seeds; exact checks for grid and halving
schedules

A study runs trials over a `Space` against an `Objective`. Objectives can be cross-validated estimators, training runs
with epochs as fidelity, agents in environments, probabilistic programs (evidence, ELBO, LOO) or any function. The same
samplers, pruners and schedules work for all of them. A study is a traced algorithm whose steps are trials, each a
child run.
- `study.ts`: `Study`, `Trial` (params, intermediate values, status), `tune(objective, space, options)`, best and Pareto
  trials, resuming a study, exporting trials as data.
- `objectives.ts`: `estimatorObjective` (splitter plus metric), `holdoutObjective`, `trainingObjective` (reports per
  epoch for pruning and fidelity), `agentObjective`, `programObjective`, `functionObjective`, multi-objective objectives.
- `samplers.ts`: the `Sampler` contract; grid, random, Sobol and Latin hypercube samplers; CMA-ES and Nelder-Mead
  samplers (from `optim`); evolutionary samplers (from `search`).
- `pruners.ts`: median, percentile, threshold, patience, successive-halving and Hyperband pruners.
- `schedules.ts`: multi-fidelity schedules (successive halving, Hyperband brackets, ASHA) as data.
- `wrappers.ts`: familiar entry points built on studies: `gridSearchCv`, `randomizedSearchCv`, `halvingGridSearchCv`,
  `halvingRandomSearchCv`.
- `analysis.ts`: trial tables, optimisation history, parameter importance (permutation and correlation based),
  parallel-coordinate, slice and contour data.
## 5. whitebox-engine methods, `packages/engine/src/methods/`

Methods are the named models, algorithms, environments, datasets and studies, each composed from the core. Each area
lists what it is built from, which is also its import list. Areas are grouped below as a reader looks for them; on disk
they are flat folders. Every estimator comes as a factory (`pca({ nComponents: 2 })`) and as a class
(`new PCA({ nComponents: 2 })`), declares its `hyperSpace`, and offers `predictDistribution` where it predicts.

### 5.1 Supervised learning

oracle: scikit-learn, statsmodels, mgcv and pygam (recorded once), lifelines, xgboost for boosting agreement

`linear/` (linear and generalised linear models on one core; sklearn.linear_model, statsmodels)
built from: `optim` (solvers), `losses`, `dists/families`, `numerics/linalg`, `stats`, `learn`
- `model.ts`: `linearModel({ family, link, penalty, solver })`, the one penalised GLM estimator every named model below
  configures; `predictDistribution` from the family.
- `solvers.ts`: normal equations (Cholesky), QR, SVD, IRLS (traced), coordinate descent with duality gaps (traced),
  proximal gradient, L-BFGS, SAG and SAGA; `chooseSolver` (and the reason it chose).
- `regression.ts`: `linearRegression`, `ridge`, `lasso`, `elasticNet`, their cross-validated forms, regularisation
  paths, LARS and lasso-LARS, orthogonal matching pursuit regression, group lasso, best subset, stepwise and stagewise
  selection.
- `classification.ts`: `logisticRegression` (binary and multinomial, L1, L2 and elastic net), `probitRegression`,
  `ridgeClassifier`, `perceptron`, `sgdClassifier`, `passiveAggressive`.
- `glm.ts`: Poisson, gamma, inverse Gaussian, Tweedie and negative binomial regression, zero-inflated and hurdle models,
  quasi-likelihood.
- `ordinal.ts`: cumulative-link (logit, probit), continuation-ratio and adjacent-category ordinal regression.
- `robust.ts`: Huber regression, RANSAC regression, Theil-Sen, quantile regression.
- `bayesian.ts`: Bayesian ridge, ARD, conjugate Bayesian linear regression (a program solved by the conjugate engine).
- `mixed.ts`: linear mixed models (random intercepts and slopes by REML), GLMMs by the Laplace approximation.
- `inference.ts`: statsmodels-style summaries: standard errors (classical, HC0 to HC3, cluster-robust), t, F, Wald and
  likelihood-ratio tests, ANOVA tables, confidence and prediction intervals, diagnostics (leverage, Cook's distance,
  VIF, residual analysis, Breusch-Pagan).
- `isotonic.ts`: isotonic regression (pool adjacent violators, traced).

`gam/` (generalised additive models; mgcv, pygam, gratia)
built from: `linear` (the GLM core), `stats/smoothing`, `optim`, `numerics/linalg`, `probprog` (fully Bayesian GAMs)
- `terms.ts`: `s` with every basis of `stats/smoothing/bases.ts` (`tp`, `ts`, `cr`, `cs`, `cc`, `ps`, `bs`, `ns`, `re`,
  `fs`, `gp`, `ad`), `by` variables, `te` and `ti` tensor products, linear, factor and random-effect terms,
  shape-constrained smooths (monotone, convex).
- `design.ts`: model matrices and penalties, identifiability constraints, null spaces.
- `fit.ts`: `gam` with interchangeable traced fitters from one objective (penalised IRLS, backfitting, gradient descent,
  Adam, L-BFGS, SGD), so their paths can be compared.
- `smoothness.ts`: smoothing-parameter selection by REML, ML, GCV and UBRE.
- `inference.ts`: effective degrees of freedom per term, Bayesian credible bands, simultaneous intervals, approximate
  p-values, summaries.
- `derivatives.ts`: derivatives and curvature of fitted smooths with simultaneous intervals (gratia style), periods of
  significant change.
- `checks.ts`: basis-dimension checks (k-index), concurvity, residual diagnostics, posterior simulation of smooths.
- `gamlss.ts`: distributional regression (location, scale and shape each with an additive predictor).
- `bayesian.ts`: GAMs as probabilistic programs (smoothness priors from penalties), solved by NUTS or Laplace.

`svm/` (sklearn.svm, LIBSVM, LIBLINEAR)
built from: `optim/smo.ts`, `optim/quadratic.ts`, `kernels`, `learn/calibration.ts`
- `classifier.ts`: SVC and nu-SVC (one-versus-one multiclass, Platt probabilities), decision functions, support
  vectors, margins.
- `regressor.ts`: SVR and nu-SVR.
- `linear.ts`: linear SVM by dual coordinate descent, Pegasos, Crammer-Singer multiclass.
- `one-class.ts`: one-class SVM, support vector data description.

`kernel-methods/`
built from: `kernels`, `numerics/linalg`, `optim`, `learn`
- `kernel-ridge.ts`: kernel ridge regression, its equivalence to the GP posterior mean.
- `kernel-classifiers.ts`: kernel logistic regression, kernel Fisher discriminant.
- `random-features.ts`: random-feature models (and their link to double descent).
- `embeddings.ts`: kernel mean embeddings, conditional mean embeddings, the representer theorem in figures.

`trees/` (sklearn.tree)
built from: `learn`, `metrics`, `scene` (tree structures)
- `tree.ts`: the `Tree` data structure, prediction, `apply`, decision paths, node regions, `toStructure`, text export.
- `criteria.ts`: Gini, entropy, log loss, squared error (two-pass), absolute error and Poisson criteria; exact and
  histogram split search; surrogate splits.
- `grow.ts`: depth-first, breadth-first and best-first growth (one node per step, traced), classifiers and regressors.
- `classic.ts`: ID3, C4.5 (gain ratio, multiway splits, pruning) and CHAID, for comparison with CART.
- `prune.ts`: cost-complexity paths and pruning, reduced-error pruning, impurity importances.

`ensemble/` (sklearn.ensemble, XGBoost and LightGBM ideas)
built from: `trees`, `learn`, `losses`
- `bagging.ts`: bagging classifiers and regressors, out-of-bag estimates.
- `forest.ts`: random forests, extra trees, random trees embedding, importances.
- `adaboost.ts`: AdaBoost (SAMME, R2), staged predictions.
- `gradient-boosting.ts`: gradient boosting (squared, absolute, Huber, quantile and log losses; Newton leaves;
  subsampling), staged predictions.
- `histogram-boosting.ts`: histogram gradient boosting (binning, leaf-wise growth, missing values, early stopping).
- `stacking.ts`: voting and stacking ensembles.

`neighbours/` (sklearn.neighbors)
built from: `numerics/spatial`, `learn`
- `knn.ts`: k-nearest-neighbour and radius classifiers and regressors.
- `centroid.ts`: nearest centroids, kernel-density classifiers.
- `metric-learning.ts`: neighbourhood components analysis, LMNN.

`naive-bayes/` (sklearn.naive_bayes, pgmpy classifiers)
built from: `pgm` (models and parameter learning), `dists`, `learn`
- `naive-bayes.ts`: Gaussian, multinomial, Bernoulli, complement and categorical naive Bayes (as Bayes networks, so
  they draw).
- `bayes-net-classifiers.ts`: tree-augmented naive Bayes (Chow-Liu), AODE, k-dependence Bayes classifiers.

`discriminant/` (sklearn.discriminant_analysis)
built from: `numerics/linalg`, `stats/covariance.ts`, `learn`
- `discriminant.ts`: linear, quadratic and regularised discriminant analysis, Fisher's discriminant, shrinkage, LDA as
  a transformer.

`gp/` (sklearn.gaussian_process, GPyTorch)
built from: `kernels`, `numerics/linalg` (differentiable Cholesky), `optim`, `dists`, `probprog`, `stochastic`
- `regression.ts`: priors and posteriors, log marginal likelihood and its gradient, hyperparameter fitting with
  restarts, posterior samples.
- `classification.ts`: Laplace and EP approximations, predictive probabilities.
- `sparse.ts`: inducing-point methods (SoR, DTC, FITC, VFE) and stochastic variational GPs.
- `multi-output.ts`: intrinsic and linear coregionalisation models.
- `latent.ts`: GPLVM and Bayesian GPLVM.
- `relevance-vector.ts`: relevance vector machines.

`multiclass/` (sklearn.multiclass, sklearn.multioutput)
built from: `learn`
- `reductions.ts`: one-versus-rest, one-versus-one, output codes, nested dichotomies.
- `multilabel.ts`: binary relevance, classifier chains, label powerset.
- `multioutput.ts`: multi-output regressors, regressor chains.

`survival/` (lifelines, scikit-survival)
built from: `optim`, `stats`, `dists`, `metrics`
- `nonparametric.ts`: Kaplan-Meier, Nelson-Aalen, log-rank tests, median survival.
- `cox.ts`: Cox proportional hazards (Breslow and Efron ties), baseline hazards, survival curves, concordance.
- `parametric.ts`: accelerated failure time models (Weibull, log-normal, log-logistic).
- `metrics.ts`: time-dependent Brier score, integrated Brier score, time-dependent AUC.

`rules/` (rule learners; wittgenstein, imodels)
built from: `search/refinement.ts`, `learn`, `metrics`
- `covering.ts`: CN2, RIPPER (traced), sequential covering, rule-quality measures.
- `tree-rules.ts`: rules from trees (C4.5 rules), RuleFit.
- `rule-lists.ts`: Bayesian rule lists, certifiably optimal rule lists (small), falling rule lists.
- `associative.ts`: classification by association rules (CBA, CMAR), built on `patterns/association-rules.ts`.

### 5.2 Unsupervised learning

oracle: scikit-learn, umap-learn, hdbscan, pacmap, gensim, pykeen

`cluster/` (sklearn.cluster)
built from: `numerics/spatial`, `numerics/linalg`, `graph`, `learn`, `metrics/clustering.ts`
- `kmeans.ts`: k-means (k-means++, Lloyd and Elkan, traced; seeding and Lloyd steps from
  `lib/numerics/spatial/quantisation.ts`), mini-batch, bisecting, kernel k-means, elbow data.
- `medoids.ts`: PAM (build and swap, traced), CLARA.
- `fuzzy.ts`: fuzzy c-means.
- `hierarchical.ts`: agglomerative clustering (single, complete, average, Ward; merge events), linkage matrices,
  dendrograms, cophenetic correlation.
- `density.ts`: DBSCAN, HDBSCAN, OPTICS, mean shift (traced).
- `spectral.ts`: spectral clustering, spectral co-clustering.
- `exemplar.ts`: affinity propagation, BIRCH.
- `constrained.ts`: COP-k-means.

`mixture/` (sklearn.mixture, as programs)
built from: `probprog` (programs and engines), `dists`, `learn`
- `programs.ts`: mixture models as probabilistic programs (Gaussian with full, tied, diagonal or spherical covariance;
  Bernoulli; multinomial; Student-t; mixtures of regressions).
- `gaussian.ts`: `gaussianMixture` (the familiar estimator; EM by default through `probprog/engines/em.ts`, with VI,
  Gibbs or NUTS by option; traced responsibilities).
- `bayesian.ts`: Bayesian and Dirichlet-process mixtures (variational and collapsed Gibbs).
- `selection.ts`: choosing the number of components (BIC, ICL, held-out likelihood, stability).

`decomposition/` (sklearn.decomposition, cross_decomposition, random_projection)
built from: `numerics/linalg`, `optim`, `probprog` (latent-variable programs), `learn`
- `pca.ts`: PCA (SVD, randomised, whitening, inverse transform), incremental PCA, sparse PCA.
- `probabilistic.ts`: probabilistic PCA and factor analysis as programs (EM through the EM engine), rotations
  (varimax, quartimax).
- `kernel-pca.ts`: kernel PCA with pre-images.
- `ica.ts`: FastICA (parallel and deflation, traced), Infomax.
- `nmf.ts`: NMF (multiplicative updates, HALS, coordinate descent; Frobenius and KL), online NMF.
- `svd.ts`: truncated SVD (LSA), dictionary learning and sparse coding (from `optim/sparse.ts`).
- `cross-decomposition.ts`: CCA, PLS regression, PLS canonical, PLS-SVD.
- `random-projection.ts`: Gaussian and sparse random projections with distortion reports.

`manifold/` (sklearn.manifold, umap-learn)
built from: `numerics/spatial`, `graph`, `numerics/linalg`, `optim` (embedding updates as gradient transformations)
- `mds.ts`: classical, metric (SMACOF, traced stress) and non-metric MDS.
- `isomap.ts`: Isomap.
- `lle.ts`: locally linear embedding (standard, modified, Hessian, LTSA).
- `spectral.ts`: Laplacian eigenmaps, diffusion maps.
- `tsne.ts`: t-SNE (perplexity calibration, exact and Barnes-Hut gradients, traced embeddings).
- `umap.ts`: UMAP (fuzzy simplicial sets, spectral initialisation, negative sampling, traced).
- `pacmap.ts`: PaCMAP and TriMap.
- `som.ts`: self-organising maps (traced), U-matrices.
- `quality.ts`: trustworthiness, continuity, neighbourhood preservation, co-ranking matrices.

`embedding/` (gensim, fastText, node2vec, PyKEEN)
built from: `text/shingles.ts`, `text/vocabulary.ts`, `graph`, `optim`, `losses`, `nn`
- `word2vec.ts`: skip-gram and CBOW with negative sampling and hierarchical softmax (traced), subsampling of frequent
  words.
- `fasttext.ts`: fastText: subword skip-gram and CBOW over hashed character n-grams (the same extractor as shingle
  matching), vectors for unseen words, the supervised fastText classifier (bags of word and character n-grams,
  hierarchical softmax), product-quantised models.
- `glove.ts`: GloVe (weighted least squares on co-occurrences).
- `graph.ts`: DeepWalk, node2vec, NetMF, spectral node embeddings.
- `knowledge-graph.ts`: TransE, DistMult, ComplEx, RotatE, link-prediction ranking.
- `evaluation.ts`: word similarity and analogy benchmarks, nearest neighbours, embedding projections for figures.

`anomaly/` (sklearn outlier detection, PyOD)
built from: `trees`, `numerics/spatial`, `stats`, `svm/one-class.ts`
- `isolation.ts`: isolation forests and extended isolation forests.
- `neighbours.ts`: local outlier factor, k-NN distances, connectivity-based outlier factor.
- `statistical.ts`: elliptic envelope (minimum covariance determinant), Mahalanobis and PCA reconstruction scores,
  HBOS.
- `sequences.ts`: discords from matrix profiles, forecasting-residual detectors.
- `combine.ts`: score normalisation and combination, thresholds (contamination, extreme-value, quantile).

`covariance/` (sklearn.covariance)
built from: `stats/covariance.ts`, `optim`
- `robust.ts`: minimum covariance determinant (fast MCD, traced).
- `graphical-lasso.ts`: graphical lasso (traced), cross-validated selection, precision-matrix graphs as structures.
### 5.3 Learning paradigms

`semi-supervised/` (sklearn.semi_supervised)
built from: `learn`, `graph`, `svm`, `optim`
- `self-training.ts`: self-training, co-training, tri-training.
- `graph-based.ts`: label propagation, label spreading, harmonic functions.
- `margin-based.ts`: transductive SVMs, Laplacian regularised least squares and SVMs, entropy minimisation.

`weak-supervision/` (Snorkel, cleanlab)
built from: `learn`, `probprog` (label models as programs), `metrics`
- `label-models.ts`: majority vote, Dawid-Skene (EM, traced), data-programming label models, worker confusion
  estimates.
- `noisy-labels.ts`: confident learning, noise-transition estimation, forward and backward loss correction.
- `positive-unlabelled.ts`: Elkan-Noto, nnPU, class-prior estimation.
- `proportions.ts`: learning from label proportions.
- `multiple-instance.ts`: mi-SVM, attention-based MIL.

`active-learning/` (modAL, ALiPy)
built from: `learn/active-learning.ts`, `learn`, `neural` (for Bayesian query strategies), `numerics/spatial`
- `strategies.ts`: uncertainty sampling (least confident, margin, entropy), query by committee (vote entropy, KL),
  expected model change, expected error reduction, BALD, core-sets (k-centre greedy), density-weighted and diversity
  strategies, batch-mode selection (BADGE).
- `stream.ts`: stream-based selective sampling.
- `studies.ts`: learning-curve comparisons of strategies on generated data with known truth.

`transfer/`
built from: `learn`, `nn`, `losses`, `kernels`
- `domain-adaptation.ts`: MMD alignment, CORAL, DANN (gradient reversal), importance-weighted training.
- `label-shift.ts`: black-box shift estimation, EM prior adjustment, posterior reweighting.
- `continual.ts`: fine-tuning, elastic weight consolidation, experience replay, forgetting metrics.
- `meta-learning.ts`: MAML (second order through autodiff), Reptile, prototypical networks, few-shot episodes.

`federated/` (Flower, FedML ideas)
built from: `nn`, `optim`, `learn/privacy.ts`
- `algorithms.ts`: FedAvg, FedProx, SCAFFOLD, FedAdam, personalised federated learning.
- `simulation.ts`: client heterogeneity (non-IID partitions), stragglers, secure aggregation simulation, differentially
  private federated training.

`fairness/` (fairlearn, AIF360)
built from: `learn`, `metrics/fairness.ts`, `optim`
- `preprocessing.ts`: reweighing, disparate-impact removal, fair representations.
- `in-processing.ts`: reductions (exponentiated gradient, grid search), adversarial debiasing, constrained learning.
- `post-processing.ts`: threshold optimisation for equalised odds, reject-option classification.
- `studies.ts`: impossibility results (calibration against equalised odds) on generated data.

`privacy/` (Opacus, ML Privacy Meter)
built from: `learn/privacy.ts`, `nn`, `stats`
- `dp-training.ts`: DP-SGD (per-example clipping, noise, accounting) studies.
- `dp-statistics.ts`: differentially private means, histograms, quantiles and synthetic data.
- `attacks.ts`: membership inference (shadow models, likelihood-ratio attacks), model inversion, attribute inference.
- `anonymity.ts`: k-anonymity, l-diversity, t-closeness checks.

`adversarial/` (Foolbox, ART)
built from: `nn`, `autodiff`, `optim`, `stats`
- `attacks.ts`: FGSM, PGD, Carlini-Wagner, DeepFool, square and boundary attacks.
- `defences.ts`: adversarial training, input transformations, randomised smoothing with certified radii.
- `evaluation.ts`: robust accuracy curves, attack success against budget.

### 5.4 Probabilistic models (`models/`)

oracle: hmmlearn, sklearn-crfsuite, gensim LDA (statistical agreement), trueskill, numpyro example models

Each model is a probabilistic program plus the estimator users expect; its fast algorithm is an engine in
`probprog/engines` or an inference routine in `pgm`.

built from: `probprog`, `pgm`, `dists`, `learn`
- `hmm.ts`: hidden Markov models (categorical, Gaussian and mixture emissions): the program, Baum-Welch through the EM
  engine (traced), Viterbi, posterior states, sampling; the dishonest casino.
- `crf.ts`: linear-chain CRF estimators on template features (`text/templates.ts`): marginals, Viterbi, scores,
  posterior paths, top features and transition weights. Training with CRF++'s and CRFsuite's algorithms: L-BFGS with
  L2 (CRF++ `-a CRF-L2`, `-c C`), OWL-QN with L1 (`CRF-L1`), SGD with an L2 schedule and with L1 by cumulative penalty
  (Tsuruoka et al. 2009), averaged perceptron, passive-aggressive (PA, PA-I, PA-II), AROW, and MIRA (CRF++ `-a MIRA`);
  each a traced run (objective, active features, per-epoch accuracy). Feature counts and frequency cut-offs as CRF++.
- `crf-examples.ts`: worked taggers (toy chunking, hyphenation as character tagging) with their template files.
- `topic.ts`: LDA as a program (collapsed Gibbs and variational engines, traced), labelled LDA, perplexity, top words.
- `topic-extensions.ts`: pLSA, HDP, correlated and dynamic topic models.
- `conjugate.ts`: conjugate models with closed-form posteriors and predictives (beta-binomial, gamma-Poisson,
  normal-normal, normal-gamma, Dirichlet-multinomial, normal-inverse-Wishart), and CAVI compared with the exact answer.
- `rating.ts`: Elo (with the chess conventions), Glicko and Glicko-2, TrueSkill (EP on its factor graph) and TrueSkill
  through time, Bradley-Terry, Plackett-Luce, Massey and Colley ratings.
- `irt.ts`: item response theory (1PL, 2PL, 3PL), learner models, item information.
- `lattice.ts`: Ising and Potts lattices: exact on chains, belief propagation, Gibbs and Swendsen-Wang sampling,
  magnetisation curves.
- `hierarchical.ts`: hierarchical models (eight schools, radon, partial pooling) with centred and non-centred forms.
- `regression.ts`: Bayesian linear, logistic, Poisson and robust regression as programs.
- `bayesian-classifiers.ts`: Bayes point machines (EP), AdPredictor (online probit regression).
- `state-space.ts`: Bayesian structural time series (local level, local linear trend, seasonal, regression).
- `latent-gaussian.ts`: linear-Gaussian latent models (factor analysis, PPCA, Bayesian PCA, linear dynamical systems).

### 5.5 Causal inference and experimentation

`causal/` (DoWhy, EconML, causalml)
oracle: dowhy and econml on fixed synthetic data
built from: `pgm/causal`, `learn`, `linear`, `ensemble`, `stats`
- `estimation.ts`: average treatment effects: regression adjustment, inverse propensity weighting, AIPW, matching
  (propensity and nearest neighbour), stratification, double machine learning.
- `heterogeneous.ts`: CATE estimators (S, T, X, R and DR learners), causal forests.
- `quasi-experimental.ts`: difference in differences, event studies, synthetic control, regression discontinuity,
  instrumental variables (two-stage least squares).
- `uplift.ts`: uplift modelling, Qini and uplift curves.
- `sensitivity.ts`: E-values, placebo and refutation tests, sensitivity to unobserved confounding.

`experimentation/` (A/B testing)
oracle: statsmodels and scipy
built from: `stats`, `metrics`, `linear`, `rl/bandits.ts`
- `ab-testing.ts`: frequentist and Bayesian A/B tests, ratio metrics (delta method), sample-ratio mismatch checks.
- `variance-reduction.ts`: CUPED, stratification, regression adjustment.
- `sequential.ts`: sequential tests, always-valid inference, alpha spending in practice.
- `design.ts`: power and sample size, randomisation and blocking, factorial and fractional factorial designs, switchback
  designs.
- `adaptive.ts`: adaptive allocation with bandits compared with fixed designs.

### 5.6 Search-based discovery and classical AI

`patterns/` (pattern mining; mlxtend, SPMF, pysubgroup)
oracle: mlxtend, pysubgroup, spmf outputs recorded once
built from: `search/refinement.ts`, `stats/tests`, `information`, `learn`
- `itemsets.ts`: Apriori, FP-growth and Eclat; closed and maximal itemsets.
- `association-rules.ts`: association rules with support, confidence, lift, leverage and conviction.
- `sequences.ts`: sequential pattern mining (GSP, PrefixSpan), episodes in event sequences.
- `subgraphs.ts`: frequent subgraph mining (gSpan for small graphs).
- `subgroup-discovery.ts`: subgroup discovery for nominal and numeric targets (WRAcc, lift, binomial and z-score
  qualities), exhaustive search with optimistic estimates, beam search, SD-Map; traced searches.
- `exceptional-model-mining.ts`: exceptional model mining with correlation, regression, classifier, Bayesian-network and
  rank models, their quality measures and significance by randomisation.
- `pattern-sets.ts`: pattern-set selection (Krimp-style MDL, diverse top-k, redundancy filters).
- `interestingness.ts`: interestingness measures and statistical testing of patterns with multiple-testing control.

`gofai/` (classical AI)
oracle: hand-checked fixtures (AIMA examples)
built from: `search`, `logic`, `stats`
- `production-systems.ts`: forward and backward chaining over rules, conflict resolution strategies, a small Rete
  network, explanation traces ("why" and "how").
- `expert-systems.ts`: certainty factors (MYCIN style), rule-based diagnosis with explanations.
- `uncertainty-calculi.ts`: Dempster-Shafer belief functions and combination, possibility theory.
- `fuzzy.ts`: fuzzy sets and membership functions, operators, Mamdani and Sugeno inference, defuzzification.
- `argumentation.ts`: abstract argumentation frameworks (grounded, preferred and stable semantics).
- `knowledge-representation.ts`: semantic networks, frames with inheritance, a description logic (ALC) tableau.
- `case-based.ts`: case retrieval, adaptation and retention.
- `planning-domains.ts`: blocks world, logistics, gripper and other domains for `search/planning.ts`.
- `puzzles.ts`: 8-puzzle, n-queens, Sudoku, Towers of Hanoi, missionaries and cannibals, cryptarithms.

`evolutionary/` (DEAP ideas)
built from: `search/evolutionary.ts`, `symbolic`, `nn`, `rl`
- `genetic-algorithms.ts`: studies on OneMax, knapsack and travelling-salesman problems.
- `genetic-programming.ts`: symbolic regression by genetic programming over `symbolic` expressions.
- `neuroevolution.ts`: NEAT-style evolution of small networks on XOR and cart-pole.
- `quality-diversity.ts`: MAP-Elites and novelty search studies.

`complex-systems/`
built from: `stochastic`, `graph`, `dynamics`, `scene`
- `cellular-automata.ts`: elementary automata (Wolfram rules), Game of Life, Langton's ant.
- `agent-based.ts`: Schelling segregation, boids, predator-prey and Sugarscape-style models.
- `networks.ts`: epidemics on networks, threshold cascades, influence maximisation, small-world and scale-free studies.
- `criticality.ts`: sandpiles, percolation, forest-fire models.
### 5.7 Domains

`timeseries/` (statsmodels.tsa, sktime basics)
oracle: statsmodels, sktime, stumpy
built from: `stats`, `dynamics/filtering`, `optim`, `signal`, `metrics/forecasting.ts`
- `features.ts`: ACF, PACF, CCF, seasonal strength, differencing and lags, time-series feature sets.
- `arima.ts`: ARMA processes (roots, impulse responses, theoretical ACF), ARIMA and SARIMAX (CSS and exact likelihood by
  Kalman filtering, traced fits), forecasts with intervals, automatic order selection.
- `smoothing.ts`: simple exponential smoothing, Holt (damped), Holt-Winters, ETS state-space models.
- `decomposition.ts`: classical decomposition, STL, MSTL.
- `volatility.ts`: GARCH, EGARCH and GJR-GARCH.
- `changepoints.ts`: changepoint studies: Bayesian online changepoint detection on tunable data, PELT and binary
  segmentation compared against the true changepoints.
- `forecasting.ts`: baselines (naive, seasonal naive, drift, mean), rolling-origin backtests, forecast reconciliation.
- `classification.ts`: time-series classification (DTW k-NN, shapelets, ROCKET and MiniRocket).
- `matrix-profile.ts`: matrix-profile studies on tunable data with planted ground truth: motif discovery, discords as
  anomalies, semantic segmentation (FLUSS, FLOSS on streams), chains, snippets, multidimensional motifs with dimension
  selection, and the anytime convergence of STAMP and SCRIMP against STOMP; timing and accuracy comparisons with the
  FFT and diagonal paths. Oracle: stumpy.

`text/` (nltk, spaCy-like pipelines)
oracle: nltk, kenlm on small corpora
built from: `text` (core), `models/hmm.ts`, `models/crf.ts`, `linear`, `naive-bayes`
- `language-models.ts`: n-gram language models (add-k, Kneser-Ney, Witten-Bell), perplexity, generation.
- `tagging.ts`: part-of-speech and named-entity tagging with HMMs and CRFs, chunking.
- `classification.ts`: text-classification pipelines (TF-IDF with naive Bayes, logistic regression, linear SVM;
  fastText from `embedding/fasttext.ts`).
- `keywords.ts`: TextRank, RAKE, TF-IDF keywords, extractive summarisation.
- `hyphenation.ts`: PATGEN pattern induction, CRF and neural hyphenators.
- `tokenisers.ts`: a suite of trained tokenisers (character, byte, word, BPE, byte-level BPE, WordPiece, Unigram) with
  comparison statistics.

`llm/` (wrappers around language models)
oracle: Hugging Face transformers for scoring and watermark detection on a small model
built from: `nn/language-model.ts`, `nn/decoding.ts`, `text`, `retrieval`, `models/rating.ts`, `stats`
- `adapters.ts`: `LanguageModel` implementations: local whitebox models, a deterministic mock for tests, and HTTP
  adapters for OpenAI-compatible and Anthropic-style APIs that take caller-supplied credentials and `fetch` (the library
  stores no keys).
- `generation.ts`: sampling strategies over any language model, constrained generation (grammars, JSON schemas),
  self-consistency voting.
- `scoring.ts`: sequence log-probabilities, perplexity, calibration of answers.
- `prompting.ts`: prompt templates, few-shot assembly, chain-of-thought formatting.
- `rag.ts`: retrieval-augmented generation over a corpus (lexical, dense or hybrid retrieval).
- `evaluation.ts`: model-graded evaluation contracts, pairwise preferences, arena ratings (Elo and Bradley-Terry).
- `watermarking.ts`: green-list watermarks (embedding at generation time, z-score detection), distortion-free
  watermarks, robustness to paraphrase.
- `detection.ts`: machine-text detection (perplexity and burstiness, DetectGPT curvature, Binoculars-style ratios).
- `fingerprinting.ts`: text provenance: SimHash, MinHash and winnowing fingerprints (from `text/fingerprinting.ts`),
  stylometry (function-word profiles, Burrows' Delta), authorship attribution.
- `agents.ts`: a tool-use loop contract whose steps are trace events.

`vision/` (OpenCV, torchvision.ops)
oracle: torchvision.ops, OpenCV
built from: `signal/image`, `numerics/geometry`, `optim`, `stats/robust.ts`, `dynamics/filtering`, `metrics`
- `boxes.ts`: box formats and conversions, non-maximum suppression (hard, soft, batched).
- `detection.ts`: anchors, box encoding and decoding, anchor matching, FCOS targets, feature-pyramid level assignment.
- `classical.ts`: classical pipelines: Hough line and circle detection, lane finding, RANSAC line and circle fitting,
  Harris corners with descriptor matching, homography panoramas, template matching.
- `geometry.ts`: two-view geometry (homographies, fundamental matrices, triangulation), camera calibration, stereo
  disparity by block matching.
- `segmentation.ts`: region growing, superpixels and graph-cut segmentation studies.
- `tracking.ts`: Kalman and SORT-style tracking, mean-shift tracking, optical-flow tracking.

`audio/`
oracle: librosa
built from: `signal/audio`, `signal`, `neighbours`, `decomposition`, `models/hmm.ts`
- `speech.ts`: keyword spotting (MFCC and DTW), voice activity detection, speaker features.
- `music.ts`: pitch tracking, chord recognition from chroma templates, beat tracking.
- `enhancement.ts`: denoising and source separation (NMF and ICA through `decomposition`).

`signal/` (signal-processing studies)
built from: `signal`, `optim/sparse.ts`, `stochastic`, `metrics`
- `spectral-estimation.ts`: periodogram, Welch, multitaper, parametric and Lomb-Scargle estimates compared on known
  spectra, including uneven sampling.
- `filter-design.ts`: FIR and IIR design comparisons against specifications.
- `sampling.ts`: aliasing, reconstruction and quantisation studies.
- `time-frequency.ts`: resolution trade-offs across STFT, wavelets and Wigner-Ville.
- `compressed-sensing.ts`: sparse recovery and its phase transition.
- `denoising.ts`: wavelet, Wiener and median denoising compared.

`dynamics/` (named systems and studies)
oracle: scipy.integrate
built from: `dynamics`, `stochastic`, `systems`
- `systems.ts`: Lorenz, Rossler, Van der Pol, Duffing, simple, damped and double pendulums, Lotka-Volterra, SIR and
  SEIR, Kuramoto oscillators, the Hopf normal form, Brusselator.
- `maps.ts`: logistic, tent, sine, Henon and standard maps with bifurcation and Lyapunov studies.
- `chaos.ts`: sensitivity to initial conditions, strange attractors, Poincare sections, routes to chaos.
- `pdes.ts`: named PDE problems with exact solutions (heat, wave, Burgers, Fisher-KPP, Allen-Cahn, KdV), shared by the
  solvers and by physics-informed networks.
- `control.ts`: control studies (PID loops, inverted pendulum, LQG, MPC and iLQR on example plants).
- `solvers.ts`: solver comparisons (order verification, stiffness, energy conservation of symplectic methods,
  work-precision diagrams).

### 5.8 Scientific machine learning (`scientific/`)

Neural networks and differential equations together. Every model here integrates with the solvers of
`lib/dynamics/ode` (any explicit, adaptive, implicit or symplectic method) and differentiates through them either by
backpropagating through the solver steps or by the adjoint method, with the choice and its memory cost recorded.

oracle: torchdiffeq and torchsde (gradients and solutions), DeepXDE (PINN reference solutions), PySINDy
built from: `nn`, `dynamics`, `stochastic`, `optim`, `losses/physics.ts`, `dists`, `symbolic`

`scientific/neural-odes/`
- `node.ts`: neural ODEs for classification and regression (ODE blocks with a solver choice), function counts per step
  as trace events.
- `augmented.ts`: augmented neural ODEs (ANODE) and second-order neural ODEs, with the failure cases they fix.
- `latent.ts`: latent ODEs and ODE-RNNs for irregularly sampled series.
- `controlled.ts`: neural controlled differential equations.
- `stochastic.ts`: neural SDEs (as generative models and as models of noisy dynamics).
- `flows.ts`: continuous normalising flows and FFJORD (Hutchinson trace estimates).
- `mechanics.ts`: Hamiltonian and Lagrangian neural networks with symplectic integration.
- `studies.ts`: adjoint against backpropagation (accuracy, memory), solver choice and tolerance against accuracy and
  cost, the ODE block as an infinitely deep ResNet.

`scientific/pinns/`
- `problem.ts`: the physics-informed problem contract: PDE residual (from `dynamics/pde/operators.ts`), domain, initial
  and boundary conditions, observed data, unknown coefficients.
- `catalogue.ts`: problems with reference solutions: Burgers, heat, wave, Poisson, Helmholtz, Allen-Cahn, Schrodinger,
  advection, Navier-Stokes (lid-driven cavity at low Reynolds number), inverse problems.
- `sampling.ts`: collocation sampling (uniform, Sobol, Latin hypercube), residual-based adaptive refinement.
- `training.ts`: PINN training as a traced algorithm (Adam then L-BFGS), loss balancing (fixed, gradient-norm,
  NTK-based), causal training for time-dependent problems.
- `constraints.ts`: hard boundary and initial conditions (distance functions and lifts), Fourier feature embeddings.
- `inverse.ts`: coefficient discovery from data.
- `variants.ts`: domain decomposition (XPINN), variational PINNs, gradient-enhanced PINNs.

`scientific/operators/`
- `deeponet.ts`: DeepONet (branch and trunk networks).
- `fno.ts`: Fourier neural operators.
- `studies.ts`: operators against dense baselines and classical solvers.

`scientific/discovery/`
- `sindy.ts`: SINDy with library construction (polynomial, trigonometric), sequential thresholding, PDE-FIND.
- `symbolic-regression.ts`: symbolic regression over `symbolic` expressions (genetic programming and sparse regression).
- `universal-de.ts`: universal differential equations (known terms plus a network), followed by sparse regression on the
  learned term.

`scientific/surrogates/`
- `emulators.ts`: neural and GP emulators of simulators, multi-fidelity surrogates, active sampling of simulations.

### 5.9 Reinforcement learning (`rl/`)

oracle: gymnasium and MinAtar (step-for-step dynamics), PettingZoo, OpenSpiel; stable-baselines3 and CleanRL learning
curves as loose references
built from: `lib/rl`, `nn`, `optim`, `losses`, `search`, `games`, `dynamics`, `scene`

`rl/environments/` (each file exports `EnvDefinition`s; every environment has `render`)
- `bandits.ts`: Bernoulli, Gaussian (the 10-armed testbed), contextual linear, adversarial and non-stationary bandits,
  with oracles for regret.
- `gridworlds.ts`: grid world (AIMA 4 x 3 by default), cliff walking (and slippery), frozen lake (4 x 4, 8 x 8, random
  maps), windy grid world (king's moves, stochastic wind), mazes with keys and traps (`keyMaze`), four rooms, and
  `renderGrid`; gymnasium's `P` tables as models.
- `classic-control.ts`: CartPole, Pendulum, MountainCar (discrete and continuous), Acrobot, matching gymnasium step for
  step.
- `toy-text.ts`: Blackjack (natural and Sutton-Barto rules), Taxi (rainy, fickle passenger), NChain, RiverSwim.
- `minatar.ts`: MinAtar-style games on a 10 by 10 grid (Breakout, Asterix, Freeway, Seaquest, Space Invaders), with
  minimal and full six-action sets, sticky actions and difficulty ramping.
- `continuous-control.ts`: point mass, 2-D reacher, double integrator, planar quadrotor, cart double pendulum (analytic
  dynamics: exact zero-order-hold steps or RK4 on Lagrangian equations of motion).
- `games.ts`: tic-tac-toe, connect four, nim, Kuhn and Leduc poker, as turn-based environments with legal actions.
- `go.ts`: Go on small boards (5, 7, 9; captures, simple ko or positional superko, area scoring with komi).
- `multi-agent.ts`: matrix games (prisoner's dilemma, matching pennies, rock-paper-scissors, stag hunt), pursuit-evasion,
  cooperative navigation (MPE simple_spread).
- `procedural.ts`: procedurally generated mazes (backtracker, Prim, Kruskal, braiding) and key-door tasks with
  Procgen-style level seeds and train/test splits, for generalisation studies.
- `models.ts`: differentiable dynamics models of the control environments (`ControlModel`, matching each environment's
  step), `linearize`, trajectory derivatives and rollouts, for LQR, MPC and model-based agents.

`rl/agents/` (each file exports agents implementing the `Agent` contract)
- `random.ts`: random and constant agents.
- `bandits.ts`: epsilon-greedy, UCB1, KL-UCB, Thompson sampling (Bernoulli and Gaussian), EXP3, LinUCB, linear Thompson
  sampling, gradient bandits.
- `tabular.ts`: Q-learning, SARSA, expected SARSA, n-step SARSA, SARSA(lambda), Monte Carlo control, TD(0) and
  TD(lambda), double Q-learning, Dyna-Q (and Dyna-Q+), prioritised sweeping.
- `planning.ts`: value and policy iteration agents, MCTS agents, AlphaZero-style self-play with a small network,
  minimax agents.
- `value-based.ts`: Q networks and DQN with double, dueling, prioritised-replay, n-step and noisy-network variants.
- `distributional.ts`: distributional C51 (categorical projection), QR-DQN (quantile Huber loss) and IQN (risk
  distortions); Rainbow.
- `policy-gradient.ts`: REINFORCE (with baselines), A2C, TRPO, PPO (clipped and KL-penalised).
- `actor-critic.ts`: DDPG, TD3, SAC (continuous and discrete).
- `model-based.ts`: PETS (probabilistic ensembles with CEM), MBPO-style short rollouts, a small latent world model in
  the Dreamer style (an MLP in place of the GRU), MuZero-style planning with learned models. The Dreamer- and
  MuZero-style learners are tested for their mechanics; they do not yet solve a task within a test budget.
- `exploration.ts`: count-based bonuses, random network distillation, intrinsic curiosity, as intrinsic-reward wrappers
  around any agent.
- `offline.ts`: offline datasets and the offline training algorithm, behaviour cloning.
- `offline-q-learning.ts`: BCQ (discrete), CQL, naive offline DQN.
- `offline-actor-critic.ts`: IQL, TD3+BC.
- `decision-transformer.ts`: decision transformers.
- `imitation.ts`: DAgger, GAIL, maximum-entropy inverse reinforcement learning.
- `multi-agent.ts`: independent learners, minimax-Q (TODO: solve stage games with `linprog` once it exists), self-play,
  neural fictitious self-play (written here while `games` is not built).
- `cooperative.ts`: value decomposition (VDN, QMIX), MADDPG.
- `hierarchical.ts`: the options framework, option-critic.
- `safe.ts`: constrained policy optimisation by Lagrangian PPO, shielding.
- `control.ts`: LQR, swing-up (energy shaping plus LQR; Pendulum, TODO: the cart double pendulum), iLQR, MPC (iLQR, CEM
  or MPPI) and CEM agents.

`rl/studies.ts`: learning-curve comparisons with aggregate metrics and intervals, sensitivity to hyperparameters,
exploration and generalisation studies.
### 5.10 Generative models (`generative/`)

oracle: PyTorch (losses and single steps with fixed weights), diffusers schedules
built from: `nn`, `losses/generative.ts`, `dists`, `stochastic` (SDEs and score dynamics), `sampling`, `probprog`
- `autoencoders.ts`: autoencoders, VAEs (beta-VAE; Gaussian and Bernoulli likelihoods), conditional VAEs, ELBO terms
  traced separately.
- `vector-quantised.ts`: VQ-VAE (straight-through estimators, codebook and commitment losses, EMA codebooks), residual
  quantisation, codebook usage.
- `gan.ts`: GANs (minimax and non-saturating), WGAN-GP, hinge GANs, conditional GANs, mode-coverage diagnostics.
- `flows.ts`: planar and radial flows, RealNVP, MAF, a small Glow (actnorm, invertible 1 by 1 convolutions, coupling).
- `autoregressive.ts`: MADE, NADE, a small PixelCNN, autoregressive sampling.
- `diffusion/schedules.ts`: linear, cosine and sigmoid noise schedules; VP, sub-VP and VE SDEs; one time axis for
  discrete and continuous models.
- `diffusion/ddpm.ts`: forward noising, epsilon, x0 and v parameterisations, training, ancestral sampling, DDIM.
- `diffusion/score.ts`: denoising score matching, reverse-SDE samplers (Euler-Maruyama, predictor-corrector), the
  probability-flow ODE with exact log-likelihoods (built on `stochastic/sde.ts` and `dynamics/ode`).
- `diffusion/flow-matching.ts`: flow matching, rectified flow, consistency-style few-step sampling.
- `diffusion/guidance.ts`: classifier and classifier-free guidance, guidance-scale sweeps.
- `energy-based/energy.ts`: the `EnergyModel` contract (energy, score, sampler hooks) shared by everything in this
  folder.
- `energy-based/boltzmann.ts`: general Boltzmann machines, restricted Boltzmann machines (Bernoulli and
  Gaussian-Bernoulli), deep Boltzmann machines, deep belief networks, exact log partition for small models.
- `energy-based/hopfield.ts`: classical Hopfield networks, dense associative memories, modern Hopfield networks and
  their link to attention.
- `energy-based/training.ts`: contrastive divergence, persistent CD, score matching (denoising and sliced),
  noise-contrastive estimation, Langevin sampling with replay buffers.
- `energy-based/jem.ts`: classifiers as energy models (JEM).
- `densities.ts`: toy-density helpers (log densities on grids, mode maps, sample-quality checks against the truth).

### 5.11 Neural network models (`neural/`)

oracle: PyTorch (forward and backward with exported weights), Hugging Face for tokenizer-level checks
built from: `nn`, `optim`, `losses`, `data`, `learn`, `metrics`

Each model has `init`, `apply`, `describe` (its structure, drawn by whitebox-render), a traced training recipe, and an
estimator where that makes sense. Sizes are chosen to train in a browser.

`neural/foundations/`
- `perceptron.ts`: the perceptron and its convergence, MLPs, universal approximation in one dimension.
- `mlp.ts`: `mlpClassifier` and `mlpRegressor` (sklearn.neural_network semantics).
- `backprop.ts`: backpropagation by hand on small networks, compared step by step with `autodiff`.

`neural/vision/`
- `lenet.ts`, `alexnet.ts` (reduced), `vgg.ts` (reduced), `resnet.ts` (CIFAR-style), `mobilenet.ts`, `convnext.ts`
  (reduced): classification networks with feature-map taps.
- `unet.ts`: U-Net segmentation.
- `vit.ts`: vision transformers (patch embeddings, class tokens), attention rollout.
- `detection.ts`: a small single-stage detector (anchor-free), built on `vision/detection.ts`.
- `autoencoders.ts`: convolutional autoencoders for images.

`neural/sequence/`
- `rnn.ts`: character-level RNN, LSTM and GRU language models.
- `seq2seq.ts`: encoder-decoder models with Bahdanau and Luong attention.
- `tcn.ts`: temporal convolutional networks and WaveNet-style models.
- `transformer.ts`: the original encoder-decoder Transformer for sequence tasks, with per-head attention maps.
- `state-space.ts`: S4, Mamba and linear-attention models; associative-recall studies.

`neural/language/`
- `gpt.ts`: decoder-only GPT (learned, sinusoidal, RoPE or ALiBi positions; tied embeddings), next-token training,
  generation, KV caching.
- `bert.ts`: encoder-only masked language models and classification heads.
- `t5.ts`: a small encoder-decoder text-to-text model.
- `pipeline.ts`: corpus to trained tokenizer to trained model to generation, as one traced pipeline.

`neural/graph/`
- `node-classification.ts`: GCN, GAT and GraphSAGE models.
- `graph-classification.ts`: GIN with pooling, graph transformers.
- `link-prediction.ts`: graph autoencoders and VGAEs.
- `studies.ts`: oversmoothing, oversquashing and expressivity (WL) studies.

`neural/representation/`
- `contrastive.ts`: SimCLR, CLIP-style two-tower models, alignment and uniformity, zero-shot classification.
- `self-supervised.ts`: BYOL, DINO, VICReg, masked autoencoders, collapse diagnostics.
- `metric-learning.ts`: siamese and triplet networks, ArcFace heads.

`neural/training/`
- `semi-supervised.ts`: Pi-model, mean teacher, VAT, MixMatch, FixMatch (and threshold variants).
- `adaptation.ts`: LoRA, rsLoRA, PiSSA, adapters, prefix tuning.
- `alignment.ts`: reward models (Bradley-Terry), RLHF with PPO, GRPO and RLOO advantages, DPO, best-of-n.
- `distillation.ts`: knowledge distillation (logit and feature matching).
- `compression.ts`: pruning and lottery tickets, quantisation studies (with `nn/quantise.ts` and `nn/pruning.ts`).
- `mixture-of-experts.ts`: mixture-of-experts models (EM for linear experts; gated networks with load balancing).
- `mixture-density.ts`: mixture density networks.
- `multitask.ts`: multi-task learning with shared trunks and loss weighting.

`neural/uncertainty.ts`: deep ensembles, MC dropout, SWAG, last-layer Laplace, Bayes by backprop, evidential heads.

`neural/phenomena.ts`: grokking on modular arithmetic, double descent in width and epochs, edge of stability, neural
collapse, mode connectivity, loss-landscape slices (filter-normalised).

`neural/interpretability.ts`: linear probes, logit lens, activation patching, attention-head analysis, sparse
autoencoders, toy models of superposition, feature visualisation.

`neural/efficiency.ts`: parameter, FLOP and memory estimators, scaling-law fits, serving-memory estimates.

### 5.12 Retrieval, matching and recommendation (`retrieval/`)

oracle: rank_bm25, datasketch and rapidfuzz (matching), recordlinkage (linkage), faiss (indexes), implicit (ALS),
scikit-learn
built from: `text`, `numerics/spatial`, `embedding`, `learn`, `losses/ranking.ts`, `metrics/ranking.ts`, `nn`
- `approximate-matching.ts`: approximate search, matching and scoring with shingles. Index a corpus by shingles (from
  `text/shingles.ts`: character k-grams, word n-grams, hashed n-grams); dispatch a query by breaking it into the same
  shingles; generate candidates through an inverted shingle index or MinHash-LSH bands; score candidates by Jaccard,
  containment, overlap, Dice, weighted Jaccard, TF-IDF cosine or BM25 over shingles; rank, threshold and explain matches
  (shared shingles highlighted). Every stage (shingling, candidate generation, scoring) is a traced step, so the figure
  can show how a query flows through the index. A dense mode embeds query and corpus with fastText subword vectors
  (`embedding/fasttext.ts`, the same n-grams) and searches with HNSW; a hybrid mode fuses lexical and dense scores.
- `record-linkage.ts`: entity resolution: blocking (shingle, sorted-neighbourhood and LSH blocking), pairwise comparison
  vectors, Fellegi-Sunter probabilistic linkage (EM), match clustering (connected components, correlation clustering),
  pairwise precision and recall.
- `search.ts`: BM25 search, dense retrieval with dual encoders, hybrid fusion (reciprocal rank fusion), re-ranking.
- `indexes.ts`: approximate nearest-neighbour indexes (IVF, PQ, IVF-PQ, HNSW, LSH) with recall-against-speed benchmarks.
- `learning-to-rank.ts`: RankNet, LambdaMART, ListNet models.
- `recommenders/neighbourhood.ts`: popularity, user-based and item-based k-NN, SLIM.
- `recommenders/factorisation.ts`: matrix factorisation (SGD with biases), ALS, implicit ALS, BPR, SVD++.
- `recommenders/two-tower.ts`: two-tower retrieval models (in-batch negatives, sampled softmax with log-Q correction).
- `recommenders/neural.ts`: NCF, factorisation machines (FM, FFM), DeepFM, Wide and Deep.
- `recommenders/sequential.ts`: session and sequence recommenders (Markov chains, GRU4Rec, SASRec, BERT4Rec).
- `recommenders/graph.ts`: graph recommenders (LightGCN).
- `recommenders/bandits.ts`: contextual-bandit and slate recommenders with off-policy evaluation.
- `evaluation.ts`: offline protocols (leave-last-out, temporal splits), interleaving, counterfactual evaluation.
- `simulation.ts`: feedback-loop simulations (popularity bias, exposure, filter bubbles).

### 5.13 Information and coding studies (`information/`)

built from: `information`, `stats`, `neighbours`
- `compression.ts`: comparing codes on sources against entropy bounds; the cost of model mismatch.
- `compression-classification.ts`: classification by normalised compression distance (the gzip k-NN classifier).
- `channel-coding.ts`: bit-error-rate curves of codes over binary symmetric and Gaussian channels.
- `information-bottleneck.ts`: the iterative information-bottleneck algorithm.
- `max-entropy.ts`: maximum-entropy distributions under moment constraints, information projections.

### 5.14 Algorithms and data structures (`algorithms/`)

built from: `search`, `graph`, `random`
- `dynamic-programming.ts`: knapsack, longest common subsequence, edit distance, matrix-chain order, optimal binary search
  trees, coin change; tables and tracebacks as trace events.
- `sorting.ts`: classic sorting algorithms with comparison and swap traces.
- `strings.ts`: KMP, Boyer-Moore, Aho-Corasick, tries, suffix arrays.
- `data-structures.ts`: binary search trees, AVL and red-black trees, hash tables (chaining and probing), skip lists, with
  operation traces.
- `sketches.ts`: Bloom filters, count-min sketches, HyperLogLog, reservoir sampling, t-digest.
- `program-synthesis.ts`: FlashFill-style string programs with version-space algebra.

### 5.15 Mathematical foundations studies (`foundations/`)

built from: `foundations`, `symbolic`, `numerics`, `autodiff`, `random`, `dists`, `scene`
- `calculus.ts`: Riemann sums converging, Taylor approximations and their error, Newton's method basins (Newton
  fractals), Euler's method accuracy.
- `probability.ts`: Monty Hall, birthday problem, Buffon's needle, Galton board, law of large numbers, central limit
  theorem, Bertrand's paradox.
- `linear-algebra.ts`: linear maps transforming grids, eigenvectors as invariant directions, SVD image compression, power
  iteration.
- `fourier.ts`: Fourier series of square and sawtooth waves, the Gibbs phenomenon, epicycle drawings.
- `numerical.ts`: floating-point surprises, conditioning, catastrophic cancellation, stable and unstable algorithms.
- `geometry.ts`: conic sections, curvature of curves, rotations and gimbal lock, quaternion interpolation.

### 5.16 Optimisation and Monte Carlo studies (`optimisation/`, `monte-carlo/`)

`optimisation/`
built from: `optim`, `datasets/objectives.ts`
- `races.ts`: optimiser races on test functions (paths over contours, convergence rates).
- `conditioning.ts`: the effect of conditioning, preconditioning and momentum.
- `constraints.ts`: KKT geometry, the simplex path on a polytope, interior-point central paths.
- `landscapes.ts`: loss landscapes of classical models.

`monte-carlo/`
built from: `sampling`, `probprog`, `datasets/densities.ts`
- `races.ts`: sampler comparisons on standard targets (effective samples per gradient, mixing).
- `hmc.ts`: step size and trajectory-length effects, divergences in funnels.
- `importance.ts`: importance-sampling failure modes and Pareto diagnostics.
- `variational.ts`: variational inference against MCMC on the same models.

### 5.17 Learning theory (`theory/`)

built from: `learn`, `linear`, `kernel-methods`, `neural`, `stats`, `optim`
Each file is a reproducible study that returns data and traces for figures.
- `pac.ts`: PAC learning of finite classes, realisable and agnostic sample complexity.
- `vc.ts`: shattering, growth functions, VC dimension, VC bounds against learning curves.
- `rademacher.ts`: empirical Rademacher complexity and the bounds it gives.
- `margins.ts`: margin bounds and their behaviour for SVMs and boosting.
- `pac-bayes.ts`: PAC-Bayes bounds (McAllester, Catoni) computed for stochastic predictors.
- `stability.ts`: algorithmic stability of ridge, lasso and SGD.
- `online.ts`: regret bounds of Hedge and online gradient descent against simulation.
- `no-free-lunch.ts`: no-free-lunch demonstrations.
- `bias-variance.ts`: bias, variance and noise by resampling training sets.
- `double-descent.ts`: double descent in random-feature regression.
- `implicit-regularisation.ts`: gradient descent finding minimum-norm solutions, implicit bias in matrix factorisation.
- `ntk.ts`: lazy training and the neural tangent kernel against finite width.
- `benign-overfitting.ts`: interpolation without harm in overparameterised linear regression.
- `concentration.ts`: Markov, Chebyshev, Hoeffding, Bernstein and Chernoff bounds against simulation.
- `model-selection.ts`: selection optimism, nested cross-validation, the one-standard-error rule.
- `optimism.ts`: training-error optimism, covariance penalties, effective degrees of freedom.
- `shrinkage.ts`: James-Stein estimators and their risk.
- `adaptive-data-analysis.ts`: Thresholdout and the Ladder against leaderboard attacks.
- `algorithm-comparison.ts`: type I error of classifier-comparison tests.
- `multiple-testing.ts`: the garden of forking paths and p-hacking simulations.
- `information-theoretic.ts`: mutual-information generalisation bounds.

### 5.18 Datasets (`datasets/`)

Every dataset is a `DatasetGenerator` (knobs as a `Space`, returning `{ data, truth }`) or a static dataset (data and
provenance). Static data is embedded as generated `*.data.ts` files vendored by the oracle with source, licence and
checksum; larger data is a release asset loaded through `data/loader.ts`. Where a generator has a generating program,
`truth.program` is that program, so likelihoods and posteriors can be queried. TODO: until `lib/probprog` exists,
generators give exact `truth.logProb` and `truth.posterior` functions instead of a program.

built from: `data`, `random`, `dists`, `probprog`, `stochastic`, `graph`, `text`, `scene`
- `regression.ts`: `makeRegression` (several targets, low-rank designs with an effective rank, column shuffling),
  Friedman 1 to 3, `sparseUncorrelated`, with regression functions and joint log densities.
- `regression-curves.ts`: one-dimensional functions (`sineCurve`, `stepCurve`, `sincCurve`, `dopplerCurve`,
  `heteroscedastic` noise), additive (GAM) data (`additiveModel`, with its components), curves with known quantiles
  (`quantileCurves`: Gaussian, Laplace, Student t and Gumbel noise), `piecewiseLinear` data.
- `regression-real.ts`: `diabetes`, `linnerud` (multi-output), Anscombe's quartet (`anscombe`), `mtcars` (and
  `mtcarsRegression`), a licence-clean housing set (`californiaHousing`, a remote asset parsed from StatLib's archive
  by `parseCaliforniaHousing`), growth charts (CDC LMS tables: `growthChart`, `growthValue`, and `growthCurves` with
  exact quantiles).
- `classification.ts`: `blobs`, `moons`, `circles`, `xor`, `spirals`, `checkerboard`, `gaussianQuantiles`,
  `hastie10_2`, with exact Bayes posteriors, log densities and Bayes errors (closed form or Monte Carlo, computed on
  first access); `moonPoints`, `circlePoints`, `gaussianQuantileLabels`.
- `make-classification.ts`: `makeClassification` (informative, redundant, repeated and noise features, clusters per
  class, class separation, imbalance, label noise, shift and scale as knobs; the truth tracks every column),
  `makeMultilabelClassification`.
- `classification-real.ts`: `iris`, `wine`, `breastCancer`, 8 by 8 digits (`digits`, `digitImages`), Titanic
  (`titanic`, R's survival table expanded to rows), penguins (`penguins` as a table, `penguinsClassification`).
- `clustering.ts`: `rings`, `anisotropicBlobs`, `unequalVarianceBlobs`, `oldFaithful`, with true clusters (and
  exact posteriors for the generators).
- `manifolds.ts`: `swissRoll` (optionally with a hole), `sCurve`, `severedSphere`, `torus`, with latent coordinates;
  `swissRollEmbedding`, `sCurveEmbedding`.
- `matrices.ts`: sklearn's matrix generators with their planted truth: `makeLowRankMatrix` (`lowRankProfile`),
  `makeSparseCodedSignal`, `makeSpdMatrix`, `makeSparseSpdMatrix`, `makeBiclusters`, `makeCheckerboard`.
- `densities.ts`: two-dimensional densities with exact log densities and samplers (`Density`: `ringDensity`,
  `pinwheelDensity`, `bananaDensity`, `funnelDensity`, `gaussianMixtureDensity` on a circle or grid,
  `checkerboardDensity`) for samplers, flows and diffusion; `densitySamples` as a generator.
- `objectives.ts`: optimisation test functions with gradients, Hessians and known minima (`TestFunction`:
  `rosenbrock`, `himmelblau`, `beale`, `rastrigin`, `ackley`, `branin`, `quadraticBowl` with a set condition number),
  hyperparameter-tuning surfaces (`TuningSurface`: `learningRateSurface` with learning curves for multi-fidelity
  tuning, `braninSurface`).
- `changepoints.ts`: tunable changepoint series (`changepointSeries`: mean, variance and rate shifts, AR regime
  switches, trend breaks; hazard rate, minimum segment length and noise as knobs) with true changepoints, segment
  parameters and the exact BOCPD predictive (`exactBocpd` with conjugate `SegmentModel`s and constant or run-length
  hazards: run-length posteriors, one-step predictives, evidence).
- `timeseries.ts`: AR and ARMA processes (`armaProcess`, `arma`, with autocovariances and spectral densities),
  sinusoids in noise (`sinusoids`), uneven sampling (`sinusoids` with uneven times, `subsampleSeries`), seasonal
  series with known components (`seasonalSeries`), airline passengers (`airPassengers`), `sunspots`, coal-mining
  disasters (`coalMiningDisasters` as events, `coalMiningCounts`), an ECG record (`electrocardiogram`, remote).
- `events.ts`: Poisson and Hawkes event data with true intensities: `poissonProcess`, `inhomogeneousPoisson`
  (sinusoidal, piecewise and ramp intensities), `hawkesProcess` and `multivariateHawkes` (with each event's parent
  and the exact log-likelihood), `spatialPoisson`.
- `comparisons.ts`: games between players with latent skills (chess-like Elo data with draws and skill drift:
  `eloGames`; tournaments: `roundRobin`; TrueSkill-style `multiplayerRaces`) with true skills.
- `ranking.ts`: explicit ratings with low-rank truth (`lowRankRatings`, observed at random or not), click logs
  (`clickLogs`: position and cascade models with logging-policy propensities), implicit feedback (`implicitFeedback`,
  Poisson factorisation), Zipfian catalogues (`zipfianCatalogue`), a small MovieLens (`movielensSmall`, remote).
- `corpus.ts`: toy and topic corpora with true topics (`naiveBayesCorpus`, `topicCorpus` with Dirichlet or bars
  topics), drifting topics (`driftingTopics`), worked-example corpora (`workedExampleCorpus`: Deerwester's LSA
  titles, the IR book's Caesar passages), `nurseryRhymes`, Tiny Shakespeare (`tinyShakespeare`, remote), near-duplicate
  corpora with planted duplicates (`nearDuplicateCorpus`, for shingle matching and record linkage).
- `sequence-tasks.ts`: copy, reverse, sort, Dyck languages, addition, associative recall, parity, modular arithmetic
  (`copyTask`, `reverseTask`, `sortTask`, `dyckTask`, `additionTask`, `associativeRecall`, `parityTask`,
  `modularArithmetic`), each with an exact solver in its truth.
- `graphs.ts`: karate club (`karateClub`), Les Miserables (`lesMiserables`), contextual stochastic block models
  (`contextualSbm`), graph families (`familyGraph`, `graphFamilies`: Erdos-Renyi, Barabasi-Albert, Watts-Strogatz,
  grids, cycles, paths, stars, complete graphs, trees, geometric graphs), small molecules (`smallMolecules`, read from
  SMILES, with formulas, weights and ring counts), knowledge-graph triples (`kinshipTriples`, family trees with
  planted rules).
- `images.ts`: `shapes` (with boxes and masks), `digitGlyphs`, `barsAndStripes`, geometric scenes with true edges and
  corners (`geometricScenes`), a small Fashion-MNIST (`fashionMnistTest`, remote, parsed by `parseIdxImages`).
- `causal.ts`: synthetic causal datasets with known structural models, ATEs and CATEs: `linearScm` (random DAGs,
  Gaussian or non-Gaussian noise, total effects, interventional means), `treatmentEffects` (confounded treatment,
  potential outcomes, CATE, propensities), `instrumentalVariables`.
- `experiments.ts`: A/B test data with known effects (`abTest`: conversion, continuous or revenue metrics, several
  arms, pre-period covariates for CUPED, arrival days and novelty effects).
- `trajectories.ts`: logged bandit data (`loggedBandit`, with exact policy values), offline RL datasets from known
  behaviour policies (`offlineMdp` on random tabular MDPs or Strens' chain, `policyEvaluation`, `TabularMdp`).
- `weak-labels.ts`: labelling functions with known accuracies (`labellingFunctions`), crowd labels (`crowdLabels`,
  Dawid-Skene), positive-unlabelled data (`positiveUnlabelled`, SCAR and SAR), label-proportion bags
  (`labelProportionBags`), each with the exact posterior or mechanism.
- `operators.ts`: operator-learning pairs (`heatOperator`, `burgersOperator` solved exactly by Cole-Hopf,
  `antiderivativeOperator` with Gaussian-process inputs).
- `catalogue.ts`: `datasetGenerators` (every generator by name, for recipes and controls), `remoteDatasets`.
- `predictions.ts`: model outputs with known miscalibration (`miscalibratedPredictions`, with the population ECE),
  survival data with known hazards (`survivalData`: Weibull and exponential proportional hazards, log-normal AFT,
  censoring).

### 5.19 Hyperparameter optimisation (`tuning/`)

Model-based samplers and schedulers for `lib/tune`.

oracle: scikit-optimize and optuna (statistical agreement), closed-form acquisitions
built from: `tune`, `gp`, `ensemble`, `kernels`, `optim`
- `bayesian.ts`: Bayesian optimisation samplers (GP or random-forest surrogates, initial designs, hyperparameter refits,
  posterior and acquisition snapshots as trace events).
- `acquisitions.ts`: probability of improvement, expected improvement and log-EI, upper confidence bound, Thompson
  sampling, max-value entropy search, q-EI, local penalisation.
- `tpe.ts`: tree-structured Parzen estimators (univariate and multivariate).
- `bohb.ts`: BOHB (model-based Hyperband).
- `population.ts`: population-based training (exploit and explore events).
- `multi-objective.ts`: ParEGO, MOTPE, expected hypervolume improvement.
- `portfolios.ts`: portfolios and warm starting across tasks.
- `importance.ts`: fANOVA hyperparameter importance.
## 6. whitebox-render, `packages/render/src/`

Headless first: everything outside `react/` returns data (ECharts options, layouts, scenes, SVG strings, encoded
animation files) and runs in Node, so the slide and blog tools can render without a browser. `react/` binds the data to
React 19. The package depends on whitebox-engine for its contracts (`Scene`, `Structure`, `Timeline`, `Run`, `Space`,
`Distribution`) and never on its algorithms.

Entry points: `whitebox-render` (every headless part), `whitebox-render/<part>`, `whitebox-render/react`.

oracle: none; checked by snapshot tests of options, scenes and structures, and by browser tests in Chromium.

`theme/`
- `tokens.ts`: light and dark design tokens (`fg`, `bg`, `surface`, `muted`, `grid`, `accent`, semantic colours,
  `tab0`..`tab9`), `tokens(mode)`, `cssVariables(mode)`.
- `palettes.ts`: tab10, sequential and diverging ramps, colour interpolation, colour-blind checks.
- `echarts.ts`: `echartsTheme(mode)` (registered light and dark themes with tab10), `baseOption`.
- `format.ts`: number formatting, nice ticks, log ticks, time formatting.

`charts/` (ECharts option builders; each returns an `EChartsOption`)
- `lines.ts`: curves, steps, areas, bands, fill-between, series from tensors.
- `points.ts`: scatter plots (labels, sizes, categorical colours), bubbles, rugs.
- `bars.ts`: bar charts, histograms, Pareto charts.
- `distributions.ts`: densities from a `Distribution` or KDE, ECDFs, box and violin plots, QQ plots, PMFs, prior and
  posterior overlays.
- `matrices.ts`: heatmaps, confusion matrices, correlation matrices, images from tensors.
- `fields.ts`: contours, decision regions, vector fields, streamlines.
- `functions.ts`: function plots for the foundations (asymptotes, implicit curves, parametric and polar curves,
  tangent and secant lines, Riemann rectangles, Taylor polynomials).
- `evaluation.ts`: ROC, precision-recall, calibration, learning curves, residual plots (from `MetricResult`s).
- `multivariate.ts`: parallel coordinates, radar charts, scatter matrices.
- `tuning.ts`: optimisation histories, parameter importances, slice and contour plots of studies.
- `compose.ts`: subplot grids with shared axes, linked axes, data zoom, mark lines, annotations.
- `downsample.ts`: LTTB and min-max decimation.

`layout/` (positions only)
- `layered.ts`: layered layout through dagre (ranks, crossing reduction, edge routes, compound nodes for plates and
  repeated blocks).
- `trees.ts`: tidy trees, radial trees, dendrograms.
- `force.ts`: seeded, deterministic force-directed layout with an energy-based stopping rule; circular and spectral
  layouts.
- `grids.ts`: grid, bipartite and lattice layouts.

`structures/` (any `Structure` to a diagram)
- `diagram.ts`: `diagram(structure, layout, options)` to a `Scene`: node shapes by role (latent and observed variables,
  factors, operations, tensors, decisions, utilities, states), edge routing (straight, orthogonal, splines), arrowheads,
  edge labels, groups (plates, repeated blocks with counts, clusters).
- `styles.ts`: the visual language per structure kind: graphical models, computation graphs, neural architectures,
  decision and search trees, automata, pipelines, networks.
- `architectures.ts`: views of a network structure: compact stack, expanded block, tensor-shape flow, parameter-count
  treemap.
- `overlays.ts`: state overlays for animation (active nodes and edges, messages along edges, value badges, frontier and
  explored sets).

`trace/` (figures from runs)
- `metrics.ts`: metric curves (smoothing, log scales, runs overlaid, run trees).
- `tensors.ts`: parameter trajectories, gradient and weight histograms over time, per-layer norm heatmaps.
- `messages.ts`: message-passing timelines and convergence plots; frames for animating structures from `message` events.
- `episodes.ts`: return, length and regret curves with intervals.
- `optimisation.ts`: optimisation paths over contours, line-search trials.
- `tables.ts`: run summaries as table data.

`gym/`
- `frames.ts`: episode frames from an environment's `render`, filmstrips.
- `overlays.ts`: value heatmaps, policy arrows, visit counts and action distributions on environment scenes.

`maths/`
- `tex.ts`: typesetting through KaTeX for HTML and MathJax for SVG (used by `tex` scene nodes and SVG export).
- `derivations.ts`: step lists from `symbolic` and `foundations` as aligned equations.

`animation/`
- `timeline.ts`: building `Timeline`s from runs (any event channel), from environment episodes, from structure overlays
  and from parameter sweeps; easing and interpolation between frames.
- `svg.ts`: animated SVG (CSS keyframes) from a timeline.
- `gif.ts`: GIF encoding (gifenc) of a timeline rendered frame by frame.
- `video.ts`: MP4 and WebM encoding in the browser (WebCodecs with mp4-muxer and webm-muxer); in Node, raw frames piped
  to ffmpeg when it is installed.
- `frames.ts`: rasterising scenes to RGBA frames (canvas in the browser, resvg in Node).

`svg/`
- `svg.ts`: `sceneToSvg` (tokens resolved for a theme, 3-D layers projected).
- `export.ts`: PNG through canvas, data URLs, sizes for slides.

`react/` (React 19; React and ECharts are peer dependencies)
- `Chart.tsx`: an ECharts option as a live chart (theme from mode, resizing, disposal, events).
- `SceneView.tsx`: a `Scene` as SVG elements, with hover and click hit-testing.
- `Scene3DView.tsx`: a 3-D scene with drag-to-orbit camera, re-projected on interaction.
- `StructureView.tsx`: a `Structure` with pan, zoom, hover details and overlay state.
- `TraceView.tsx`: the standard figures of a run; `useRunStream` for live runs.
- `TimelinePlayer.tsx`: playback of any timeline (play, step, scrub, speed).
- `EpisodePlayer.tsx`: environment episodes with overlays.
- `ExportMenu.tsx`: export of the current figure as SVG, PNG, GIF, MP4 or WebM.
- `MathView.tsx`: TeX and derivation display.
- `controls/`: controls generated from a `Space` (`SpaceControls`), and `Slider`, `NumberInput`, `Select`, `Toggle`,
  `SeedInput`.
- `theme.tsx`: `ThemeProvider`, `useColourMode`.

## 7. whitebox-meta, `packages/meta/src/`

Turns the source into documentation data, runs examples and links functions to concepts. Entry points:
`whitebox-meta` (schema, runner, concepts), `whitebox-meta/extract` (Node only), `whitebox-meta/modules`,
`whitebox-meta/worker`, `whitebox-meta/react`, and the generated `whitebox-meta/catalogue.json`.

`schema/`
- `catalogue.ts`: `SymbolDoc` (id, package, module, name, kind, signature, summary, description, params, returns,
  throws, examples, concepts, references, canonical home and aliases, source location), `ModuleDoc`, `Catalogue`.
- `examples.ts`: `Example`, `Expectation` (value, figure, trace, throws), `ExampleResult`.
- `content.ts`: `HowTo`, `GalleryEntry`, `Study` (a methods study as a page: description, parameters as a `Space`,
  figures).

`concepts/`
- `taxonomy.ts`: the concept tree schema (domain, chapter, note; ids such as `signal-processing/spectral-estimation/welch`),
  loaded from `docs/concepts.yaml`, which mirrors the encyclopedia's table of contents.
- `index.ts`: concept to symbols, symbol to concepts, coverage of the taxonomy (notes without an implementation, symbols
  without a concept).

`extract/` (Node only; ts-morph)
- `project.ts`: load the packages, enumerate entry points from `exports` and barrels.
- `symbols.ts`: exported declarations to `SymbolDoc`s (signatures, overloads, classes and methods, aliases resolved to
  canonical homes).
- `tsdoc.ts`: TSDoc parsing (summary, `@param`, `@returns`, `@throws`, `@example`, `@concept`, `@see`, `@reference`).
- `built-from.ts`: the import graph between symbols and areas, for "how is this built" views.
- `content.ts`: how-to Markdown, gallery entries and studies.
- `coverage.ts`: documentation and concept coverage reports that fail the build.

`runner/` (browser and Node)
- `parse.ts`: statements and expectation markers.
- `transform.ts`: type stripping and import rewriting (sucrase), expectation instrumentation.
- `execute.ts`: execution with a module resolver, console capture, time limits, errors mapped to example lines.
- `compare.ts`: comparison to the stated precision, tensors and typed arrays as nested arrays, figure, trace and throw
  recognition.
- `format.ts`: display of values beside code.
- `run.ts`: `runExample`, `runExamples` with a report.

`modules/`
- `index.ts`: `moduleLoaders` (generated lazy imports for every public entry point).

`worker/`
- `protocol.ts`, `worker.ts`, `client.ts`: running examples and calling exports off the main thread, streaming trace
  events, latest-wins cancellation.

`react/`
- `LiveExample.tsx`: editable examples with outputs beside each expectation, figures and traces rendered with
  whitebox-render.
- `CodeEditor.tsx`: CodeMirror 6 with completion from the catalogue.
- `ValueView.tsx`: any output (numbers, tensors, distributions, structures, scenes, charts, runs, timelines).
- `ApiSymbol.tsx`, `ModuleIndex.tsx`, `Search.tsx`: browsing the catalogue.
- `ConceptPage.tsx`: the implementations of a concept with their examples, for encyclopedia notes.
- `BuiltFrom.tsx`: the "built from" graph of a symbol or area.
- `StudyView.tsx`: a study with controls generated from its parameter space.

## 8. Apps

Both apps are private Vite and React 19 applications with a navigation bar, a persistent light and dark toggle (system
default, stored choice, no flash), theme tokens from `whitebox-render/theme` and Tailwind CSS. They consume the
packages exactly as an outside project does.

`apps/docs/` (the counterpart of whitebox-engine)
- `src/main.tsx`, `src/routes.tsx`, `src/shell/` (`AppShell.tsx`, `NavBar.tsx`, `ThemeToggle.tsx`, `Sidebar.tsx`).
- `src/pages/`: `Home.tsx`, `HowTo.tsx`, `Module.tsx`, `Symbol.tsx`, `Concepts.tsx` (browse by the encyclopedia's
  taxonomy), `Studies.tsx`, `Playground.tsx`, `Traces.tsx` (open, replay, compare and export runs).
- `content/howto/`: one how-to per core domain and per major method area, with runnable code blocks checked by the build.

`apps/gallery/` (the counterpart of whitebox-render)
- `src/main.tsx`, `src/routes.tsx`, `src/shell/`.
- `src/pages/`: `Home.tsx`, `Section.tsx`, `Entry.tsx` (a figure with its editable code and export menu).
- `src/entries/`: one file per section: `charts.ts`, `distributions.ts`, `fields.ts`, `functions.ts`, `structures.ts`
  (graphical models, computation graphs, architectures, trees, automata), `networks.ts`, `traces.ts`, `animations.ts`,
  `environments.ts` (every environment, rendered and animated), `three-d.ts`, `maths.ts`, `layouts.ts`, `themes.ts`.

## 9. Python oracle, `oracle/`

A uv project (root `pyproject.toml`) with pydantic and pydantic-settings, and dependency groups split by domain so a
fixture run installs only what it needs: `oracle-core` (numpy, scipy, scikit-learn, statsmodels, sympy,
networkx, python-control with slycot), `oracle-torch` (torch, torchmetrics, torchdiffeq, torchsde), `oracle-prob` (numpyro, pyro, pgmpy, arviz),
`oracle-signal` (pywt, librosa, scikit-image, pystoi, pesq), `oracle-text` (tokenizers, tiktoken, nltk, sacrebleu,
rouge-score, jiwer, rapidfuzz, datasketch, gensim), `oracle-rl` (gymnasium, minatar, pettingzoo, open_spiel),
`oracle-misc` (lifelines, mlxtend, pysubgroup, umap-learn, hdbscan, dowhy, econml, causal-learn, pysindy, faiss-cpu,
implicit, optuna, nashpy).
- `oracle/whitebox_oracle/settings.py`: `OracleSettings` (reads the root `.env`; output paths; selection by path prefix
  or group).
- `oracle/whitebox_oracle/fixture.py`: `Case` and `Fixture` (inputs, outputs, tolerances, reference library and version,
  seed), JSON encoding of non-finite and complex numbers, deterministic writing.
- `oracle/whitebox_oracle/cli.py`: `generate`, `check` (regenerate and diff), `list`.
- `oracle/cases/<domain>/<file>.py`: one case file per engine file with a Python counterpart, at the mirrored path.
- `oracle/datasets/<name>.py`: dataset vendoring (pinned source, checksum, generated `*.data.ts` or a release asset with
  a manifest entry).
- `packages/engine/fixtures/`: generated JSON fixtures, committed.
- `packages/engine/src/testing/golden.ts` (not published): `loadFixture`, `expectClose`, the fixture schema matching
  the pydantic model.

Python never runs in `make test`. `make fixtures` regenerates; a scheduled CI job runs `make fixtures-check` group by
group.

**The fixture standard.** Fixtures are a safeguard: they show that whitebox-js agrees with the established Python
library for the same computation. They do not set a higher bar than that library.
1. Reference: the prebuilt numpy, scipy, scikit-learn, statsmodels (and so on) function itself, called as a user
   would. No arbitrary-precision references, and no reimplementations when a library function exists.
2. Inputs: the typical range of each function plus a few edges (0, infinities, NaN, domain boundaries); no extreme tails
   that the reference itself does not handle well.
3. Tolerances, by tier: exact operations `1e-12`; special functions and direct linear algebra `1e-10`; iterative or
   fitted results `1e-6` to `1e-8`, or a comparison of invariants (residuals, objective values, subspaces) where
   outputs are not unique. Where the reference is known to be less accurate, the tolerance follows it, with a comment.
4. Gradients are not fixtures: every primitive passes the conformance test (`checkPrimitive`: reverse and forward mode
   against central differences, and against each other), plus property tests. The exception is a custom rule that the
   reference library also differentiates (torch autograd for factorisations), where a small gradient fixture is kept.
5. Size: tens of points per case; every case module generates in about a second.
6. Python: every signature annotated (ruff `ANN`, pyright strict); nothing more elaborate.

## 10. Repository tooling

- `Makefile` and `mk/*.mk`: `help`, `setup`, `dev`, `dev-docs`, `dev-gallery`, `build`, `lint`, `format`, `typecheck`,
  `test`, `test-browser`, `coverage`, `docs`, `concepts`, `barrels`, `layers`, `fixtures`, `fixtures-check`,
  `datasets`, `pack`, `smoke`, `check`, `clean`, `release`.
- `mise.toml`: Node, pnpm, Python and uv versions.
- `scripts/barrels.ts`: every `index.ts` and the module map; `--check`.
- `scripts/layers.ts`: dependency-cruiser rules from `domain.json` and `area.json`.
- `scripts/docs.ts`: catalogue extraction, every example and how-to run, documentation coverage.
- `scripts/concepts.ts`: concept-taxonomy coverage report against `docs/concepts.yaml`.
- `scripts/version.ts`: `base/version.ts` and equal package versions (changesets fixed group).
- `scripts/pack.ts`: build and pack the three packages; collect release assets with a manifest.
- `scripts/smoke.ts`: install the tarballs into a throwaway Vite React app, import every entry point, run an example,
  render a scene and a chart in Node, encode a short GIF, build the app.
- `.github/workflows/`: `ci.yml` (`make check`, `make smoke`), `release.yml` (changesets; tarballs and assets attached
  to a GitHub release), `oracle.yml` (scheduled fixture checks per group), `dependabot.yml`.
- `docs/`: `structure.md`, `decisions.md`, `extending.md`, `conventions.md`, `concepts.yaml`, `history/`.
- `CLAUDE.md`, `INIT.md`, `AUDIT.md`, `PORT.md`.
## 11. Where familiar things live

### scikit-learn

| sklearn                                                     | whitebox                                              |
| ----------------------------------------------------------- | ----------------------------------------------------- |
| `base`, estimator API                                       | `lib/learn/estimator.ts`                              |
| `linear_model`                                              | `methods/linear/`                                     |
| `svm`                                                       | `methods/svm/`, `lib/optim/smo.ts`                    |
| `tree`                                                      | `methods/trees/`                                      |
| `ensemble`                                                  | `methods/ensemble/`                                   |
| `neighbors`                                                 | `methods/neighbours/`, `lib/numerics/spatial/`        |
| `naive_bayes`, `discriminant_analysis`                      | `methods/naive-bayes/`, `methods/discriminant/`       |
| `gaussian_process`                                          | `methods/gp/`, `lib/kernels/`                         |
| `kernel_approximation`, `kernel_ridge`                      | `lib/kernels/expansion.ts`, `methods/kernel-methods/` |
| `isotonic`                                                  | `methods/linear/isotonic.ts`                          |
| `multiclass`, `multioutput`                                 | `methods/multiclass/`                                 |
| `semi_supervised`                                           | `methods/semi-supervised/`                            |
| `neural_network`                                            | `methods/neural/foundations/mlp.ts`                   |
| `cluster`                                                   | `methods/cluster/`                                    |
| `mixture`                                                   | `methods/mixture/`                                    |
| `decomposition`, `cross_decomposition`, `random_projection` | `methods/decomposition/`                              |
| `manifold`                                                  | `methods/manifold/`                                   |
| `covariance`                                                | `lib/stats/covariance.ts`, `methods/covariance/`      |
| `preprocessing`, `impute`                                   | `lib/learn/preprocessing/`                            |
| `feature_extraction.text`                                   | `lib/text/features.ts`                                |
| `feature_selection`                                         | `lib/learn/feature-selection.ts`                      |
| `pipeline`, `compose`                                       | `lib/learn/compose.ts`                                |
| `model_selection` (splitters, CV)                           | `lib/learn/model-selection/`                          |
| `model_selection` (searches)                                | `lib/tune/`, `methods/tuning/`                        |
| `metrics`                                                   | `lib/metrics/`                                        |
| `calibration`                                               | `lib/learn/calibration.ts`                            |
| `inspection`                                                | `lib/learn/inspection.ts`                             |
| `datasets`                                                  | `methods/datasets/`, `lib/data/`                      |

### PyTorch and its ecosystem

| library                               | whitebox                                                                    |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `torch.Tensor`, operations            | `lib/tensor/`                                                               |
| `torch.autograd`, `torch.func`        | `lib/autodiff/`                                                             |
| `torch.nn.functional`                 | `lib/nn/functional/`                                                        |
| `torch.nn` modules                    | `lib/nn/module/`                                                            |
| `torch.nn.init`                       | `lib/nn/init.ts`                                                            |
| `torch.nn` losses                     | `lib/losses/`                                                               |
| `torch.optim`, `lr_scheduler`         | `lib/optim/`                                                                |
| `torch.utils.data`                    | `lib/data/batching.ts`                                                      |
| `torch.distributions`                 | `lib/dists/`                                                                |
| `torchvision.ops`                     | `methods/vision/boxes.ts`, `lib/metrics/detection-segmentation.ts`          |
| `torchmetrics`                        | `lib/metrics/`                                                              |
| `torchdiffeq`, `torchsde`, diffrax    | `lib/dynamics/ode/`, `lib/stochastic/sde.ts`, `lib/nn/blocks/continuous.ts` |
| DeepXDE                               | `methods/scientific/pinns/`                                                 |
| training loops (Lightning style)      | `lib/nn/training/`                                                          |
| `torchinfo.summary`                   | `lib/nn/describe.ts`                                                        |
| Hugging Face `generate`, `tokenizers` | `lib/nn/decoding.ts`, `lib/text/tokenizer.ts`                               |

### NumPy, SciPy and SymPy

| library                               | whitebox                                          |
| ------------------------------------- | ------------------------------------------------- |
| `numpy` arrays and ufuncs             | `lib/tensor/`                                     |
| `numpy.random`                        | `lib/random/`                                     |
| `numpy.linalg`, `scipy.linalg`        | `lib/numerics/linalg/`                            |
| `scipy.sparse`, `scipy.sparse.linalg` | `lib/numerics/sparse/`                            |
| `numpy.fft`, `scipy.fft`              | `lib/numerics/fourier/`                           |
| `numpy.polynomial`                    | `lib/numerics/polynomial/`                        |
| `scipy.special`                       | `lib/numerics/special/`                           |
| `scipy.integrate` (quadrature)        | `lib/numerics/quadrature/`                        |
| `scipy.integrate` (ODEs, BVPs)        | `lib/dynamics/ode/`                               |
| `scipy.optimize`                      | `lib/optim/`, `lib/numerics/roots/`               |
| `scipy.interpolate`                   | `lib/numerics/interpolate/`                       |
| `scipy.signal`                        | `lib/signal/`, `lib/systems/`                     |
| `scipy.ndimage`                       | `lib/signal/image/`, `lib/numerics/convolution/`  |
| `scipy.spatial`                       | `lib/numerics/geometry/`, `lib/numerics/spatial/` |
| `scipy.stats`                         | `lib/dists/`, `lib/stats/`                        |
| `scipy.cluster.hierarchy`             | `methods/cluster/hierarchical.ts`                 |
| SymPy                                 | `lib/symbolic/`                                   |

### Other libraries

| library                           | whitebox                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------- |
| statsmodels                       | `methods/linear/`, `methods/timeseries/`, `lib/stats/`                                            |
| mgcv, pygam, gratia               | `methods/gam/`, `lib/stats/smoothing/`                                                            |
| NumPyro, Pyro                     | `lib/probprog/`, `lib/sampling/`                                                                  |
| ArviZ                             | `lib/sampling/diagnostics.ts`, `lib/probprog/criticism.ts`                                        |
| pgmpy                             | `lib/pgm/`                                                                                        |
| DoWhy, EconML, causal-learn       | `lib/pgm/causal/`, `methods/causal/`                                                              |
| NetworkX                          | `lib/graph/`                                                                                      |
| gymnasium, MinAtar                | `lib/rl/`, `methods/rl/environments/`                                                             |
| PettingZoo                        | `lib/rl/multi-agent.ts`, `methods/rl/environments/multi-agent.ts`                                 |
| stable-baselines3, CleanRL        | `methods/rl/agents/`, `lib/rl/buffers.ts`, `lib/rl/returns.ts`                                    |
| OpenSpiel, nashpy                 | `lib/games/`, `methods/rl/environments/games.ts`                                                  |
| python-control, filterpy          | `lib/systems/`, `lib/dynamics/control/`, `lib/dynamics/filtering/`                                |
| pywt, librosa                     | `lib/signal/wavelets.ts`, `lib/signal/audio/`                                                     |
| optax                             | `lib/optim/transform.ts`, `lib/optim/adaptive.ts`                                                 |
| POT                               | `lib/optim/transport.ts`                                                                          |
| cvxpy (small problems)            | `lib/optim/linear-programming.ts`, `quadratic.ts`, `conic.ts`                                     |
| Optuna, Ray Tune, scikit-optimize | `lib/tune/`, `methods/tuning/`                                                                    |
| SWI-Prolog, Metagol, Popper       | `lib/logic/`                                                                                      |
| AIMA                              | `lib/search/`, `methods/gofai/`                                                                   |
| mlxtend, SPMF, pysubgroup         | `methods/patterns/`                                                                               |
| DEAP                              | `lib/search/evolutionary.ts`, `methods/evolutionary/`                                             |
| PySINDy                           | `methods/scientific/discovery/`                                                                   |
| faiss                             | `lib/numerics/spatial/`, `methods/retrieval/indexes.ts`                                           |
| datasketch, rapidfuzz             | `lib/text/shingles.ts`, `lib/text/fingerprinting.ts`, `methods/retrieval/approximate-matching.ts` |
| recordlinkage, Splink             | `methods/retrieval/record-linkage.ts`                                                             |
| gensim, fastText                  | `methods/embedding/`, `methods/models/topic.ts`                                                   |
| lifelines                         | `methods/survival/`                                                                               |
| fairlearn, AIF360                 | `lib/metrics/fairness.ts`, `methods/fairness/`                                                    |
| Opacus                            | `lib/learn/privacy.ts`, `methods/privacy/`                                                        |
| SHAP, Captum                      | `lib/learn/explain/`                                                                              |
| cleanlab, Snorkel                 | `methods/weak-supervision/`                                                                       |
| zlib                              | `lib/information/deflate.ts`                                                                      |

## 12. Order of construction

Bottom up, one tier at a time; a tier is finished (documented, tested against the oracle, traced) before the next relies
on it. The first pass is a thin vertical slice that proves the whole pipeline (values to traces to figures to docs):
1. `base`, `scene` (types, structure, timeline), `tensor` (primitives, core operations, CPU backend), `random`.
2. `autodiff` (reverse mode, `grad`, `valueAndGrad`, `computationGraph`, `checkPrimitive`), `trace` (schema, policy,
   algorithm, recorder, replay), `space`, `context`.
3. `numerics/special/erf.ts`, `numerics/linalg/cholesky.ts` (with its custom derivative), `optim` (`sgd`, `adam`,
   `minimizer`), `dists` (`normal`, `categorical`).
4. `pgm` (factors, Bayes networks, variable elimination, `toStructure`), `losses` (`mse`, cross-entropy), `nn` (linear
   layers, activations, `Sequential`, `fit`, `describe`), `rl` (`env.ts`, `gym.ts`, `rollout.ts`) with CartPole and its
   render, `data` (regression schema, generator and truth) with `friedman1` and `diabetes`, `metrics/regression.ts`.
5. whitebox-render: theme, `charts/lines.ts`, `layout/layered.ts`, `structures/diagram.ts`, `trace/metrics.ts`,
   `animation/gif.ts`, `svg/`, `react/` (Chart, SceneView, StructureView, TraceView, TimelinePlayer).
6. whitebox-meta: schema, extractor, runner with `// =>` checks, module map, concept taxonomy; the docs and gallery apps
   showing the slice; the oracle producing the first fixtures; CI, packing and the smoke test.

After the slice, every remaining file in this document becomes a build task, taken in tier order for the core and in
"built from" order for methods.

## 13. Questions for the reviewer

1. **Entry-point granularity.** Every folder is an entry point and subfolders are namespaces. Too many import paths, or
   the right granularity for lazy loading?
2. **Thirty-five core domains.** Are any of the new domains (`foundations`, `information`, `systems`, `games`,
   `losses`, `metrics`) better merged into neighbours, or split further?
3. **The default context.** One sanctioned module-level default (for the global seed) against strictly explicit
   contexts everywhere.
4. **Programs as the home of probabilistic models.** Mixtures, HMMs, LDA and factor analysis are programs with
   specialised engines. Does the planner's automatic choice stay predictable enough, and is the fast path through
   engines fast enough for interactive pages?
5. **Two API forms.** Functional and class forms for every estimator, layer and environment: worth the surface?
6. **Scenes and structures in the engine.** Both live in tier 1 of the engine so models and environments can describe
   themselves; the alternative is a small fifth package.
7. **Datasets in the engine package.** Embedded static data enlarges the engine tarball; the alternative is a
   `whitebox-datasets` package.
8. **Scope.** Anything missing that a comprehensive library for machine learning, statistics, numerics and their
   mathematical foundations should have.
