# mdml-js: structure (v2)

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
18. **A concept taxonomy** in `mdml-meta`, so encyclopedia notes link to the functions that implement them.

## 0.1 Review notes, answered

| note                                                     | answer                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data is not in the tiers                                 | It is: `lib/data`. The domain table in section 4 now gives every domain a one-line purpose.                                                                                                                                                                                                                                           |
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
| BOCPD with tunable changepoint data                      | `lib/dynamics/filtering/changepoint.ts`, `lib/datasets/changepoints.ts`, `methods/timeseries/changepoints.ts`.                                                                                                                                                                                                                        |
| A global seed                                            | `lib/context/`.                                                                                                                                                                                                                                                                                                                       |
| Subgroup discovery, exceptional model mining, GOFAI      | `lib/search/`, `methods/patterns/`, `methods/gofai/`, `methods/evolutionary/`.                                                                                                                                                                                                                                                        |
| Big files should be split                                | Large topics are folders; files over about 400 lines are split.                                                                                                                                                                                                                                                                       |
| Approximate search with shingles                         | `lib/text/shingles.ts`, `lib/text/approximate-search.ts`, `methods/retrieval/approximate-matching.ts`, `methods/retrieval/record-linkage.ts`.                                                                                                                                                                                         |
| PINNs, neural ODEs and their solvers                     | New area `methods/scientific/` (neural ODEs and relatives, PINNs, neural operators, equation discovery, surrogates), on `lib/dynamics/ode` solvers of every order with adjoint gradients (`lib/dynamics/ode/adjoint.ts`), PDE operators of networks (`lib/dynamics/pde/operators.ts`) and ODE blocks (`lib/nn/blocks/continuous.ts`). |

## 1. Shape of the repository

```
mdml-js/
  packages/
    engine/      mdml-engine   compute: from tensors to reinforcement learning, plus datasets       (published)
    render/      mdml-render   headless figures: ECharts options, layouts, scenes, SVG, animation;  (published)
                                   React bindings in a separate entry point
    meta/        mdml-meta     docs metadata: schema, extractor, example runner, worker, concepts   (published)
  apps/
    docs/        how to use mdml-engine: how-tos, API reference, concepts, live examples, traces (private)
    gallery/     what mdml-render can draw: charts, diagrams, scenes, animations, environments    (private)
  oracle/        Python ground truth: golden fixtures and dataset vendoring (uv, pydantic)
  scripts/       repository tooling in TypeScript
  docs/          structure.md (this file), decisions.md, extending.md, conventions.md, concepts.yaml, history/
  Makefile, mk/  the single interface for routine work
  CLAUDE.md, INIT.md, AUDIT.md, PORT.md
```

`apps/site` is the one site: the gallery shows off `render`, the hub shows off `mdml-hub` (its pages generated
from the hub catalogue), and the docs document `engine`. Outside projects (the encyclopedia,
the slide and blog tool) consume the three published packages as release tarballs.

## 2. Conventions every file follows

**Entry points.** Every folder is an entry point. `packages/engine/src/lib/numerics/linalg/` is imported as
`mdml-engine/numerics/linalg`; its parent re-exports it as the namespace `linalg`. Index files are generated from
folder contents, so adding a file never edits another file.

**Core and applications.** `packages/engine/src/lib/` is the core: general machinery. `packages/engine/src/methods/` is
applications: named models, algorithms, environments, datasets and studies, each built from the core. The core never
imports methods.

**Layering.** Core domains import one another without cycles, and so do methods areas: if `stats` imports `dists`,
nothing `dists` imports may lead back to `stats`. There is no declared order; the import graph is the order, and
`make layers` (`scripts/layers.ts`) fails with the path of any cycle. The core never imports methods; the package rules
of section 1 hold too.

**Summaries live in the code.** The source explains itself, with no metadata files beside it:
- A folder's summary is the doc comment that opens its `index.ts`: a short paragraph on what the folder is and what
  you will find in it (not an index of its files). The barrel generator keeps it and writes the exports below it. A
  subfolder re-exported flat into its parent's entry point (`tensor/ops` into `mdml-engine/tensor`) carries the
  `@inline` tag in its summary.
- A file opens with its own paragraph: what the file is and what to expect in it (for `shapley.ts`, what Shapley
  values are and which explainers it provides), as a doc comment followed by a blank line or the imports. No to-do
  lists and no lists of every export; hub entries under `collections/` can be one line.
- `make barrels` fails on a folder or a module file without one; mdml-meta reads both into the catalogue, and the
  docs show them on folder and file pages. A private file (`_tree.ts`) that defines a public symbol is shown too, so it
  opens with its paragraph as well (`make examples ARGS=--coverage` lists any that do not).

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
page imports. `mdml-meta/modules` maps every entry point to a lazy `import()`; large datasets are release assets
fetched on demand; heavy work runs in a Web Worker through `mdml-meta/worker`.

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
graphs all describe themselves as structures; mdml-render lays out and draws any structure the same way.

### 3.14 One worked path

A reader trains a classifier and sees every contract in turn:
1. `moons.generate(ctx.key('data'), { n: 400, noise: 0.2 })` returns `{ data, truth }`; its knobs are a `Space`.
2. `mlpClassifier({ hidden: [16, 16] })` is an `Estimator` with a `hyperSpace`.
3. `fit` runs a training `Algorithm`; the `TracePolicy` records loss every step and gradient norms every ten steps.
4. `predictDistribution` returns categorical distributions; `evaluate` returns `MetricResult`s with bootstrap intervals,
   and `truth.bayesError()` gives the floor.
5. `tune(estimatorObjective(est, data, { splitter: stratifiedKFold(5), metric: logLoss }), est.hyperSpace)` runs a
   study with Bayesian optimisation, each trial a child run.
6. `describe(fitted)` gives a `Structure` of the network; mdml-render draws it, plots the trace, and exports the
   training animation as a GIF.
7. Every function used has a docstring example, checked by the build and shown live in the docs app.
## 4. mdml-engine core, `packages/engine/src/lib/`

The core has 41 domains, roughly from the most basic to the most composite; a domain imports the domains it needs, and
the import graph between them has no cycles (section 2). Each domain's own summary is the doc comment of its
`index.ts`.

| domain        | purpose                                                                                                                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `base`        | errors, JSON, shared result shapes, scalar helpers, version                                                                                                                                      |
| `scene`       | drawings (`Scene`), 3-D layers, rasters and graph-shaped descriptions (`Structure`) as data                                                                                                      |
| `tensor`      | arrays, dtypes, shapes, primitives with derivative rules, backends, operations, pytrees                                                                                                          |
| `random`      | immutable counter-based keys and samplers                                                                                                                                                        |
| `symbolic`    | symbolic expressions: simplification, differentiation, series, integration, LaTeX                                                                                                                |
| `exposition`  | typed exposition records that methods return and render draws (paths, trees, stages, steps, trials, episodes)                                                                                    |
| `autodiff`    | reverse, forward and Taylor-mode differentiation, vectorisation, gradient checks                                                                                                                 |
| `trace`       | algorithms, trace schemas and policies, events, recording, replay                                                                                                                                |
| `space`       | typed spaces for parameters, knobs, observations and actions                                                                                                                                     |
| `context`     | seeds, named keys, tracer and policy defaults                                                                                                                                                    |
| `numerics`    | special functions, linear algebra, sparse, polynomials, quadrature, roots, interpolation, Fourier, convolution, geometry, nearest neighbours, approximation, inverse problems                    |
| `data`        | dataset schemas, generators and truth, loading, batching, splits, modifiers                                                                                                                      |
| `foundations` | mathematical foundations: trigonometry, vectors, rotations, curves, surfaces, calculus, series, linear algebra with steps, number theory, combinatorics, complex numbers, elementary probability |
| `graph`       | graphs and graph algorithms, graph signal processing                                                                                                                                             |
| `dists`       | distributions, bijectors, exponential families, GLM families, copulas                                                                                                                            |
| `information` | entropy, divergences, mutual information estimators, source and channel coding, compression                                                                                                      |
| `systems`     | linear time-invariant systems: representations, responses, stability, discretisation                                                                                                             |
| `worlds`      | discrete worlds: grid and graph maps, mazes and generators, terrain and rules, agents moving discretely or continuously                                                                          |
| `stats`       | descriptive statistics, tests, resampling, density estimation, smoothing bases                                                                                                                   |
| `optim`       | optimisers, minimisation, programming, sparse approximation, transport, implicit differentiation                                                                                                 |
| `kernels`     | kernel functions, compositions and explicit feature maps                                                                                                                                         |
| `signal`      | signal and image processing, spectral and time-frequency analysis, audio features                                                                                                                |
| `text`        | normalisation, tokenisation, subwords, shingles, vectorisers, similarity, parsing, fingerprinting                                                                                                |
| `search`      | state-space search, local and evolutionary search, adversarial search, constraint satisfaction, planning, refinement search                                                                      |
| `stochastic`  | stochastic processes and stochastic calculus, SDE solvers, point processes, DPPs, random measures                                                                                                |
| `sampling`    | Monte Carlo samplers and diagnostics                                                                                                                                                             |
| `graphics`    | rendering mathematics: ray casting over worlds, ray tracing, cameras, shading, textures, rasterisation                                                                                           |
| `logic`       | terms, unification, resolution, Datalog, SAT, first-order proving, ILP                                                                                                                           |
| `dynamics`    | ODE and PDE solvers, nonlinear dynamics, control, filtering, system identification                                                                                                               |
| `pgm`         | graphical models, their inference and learning, causal graphs                                                                                                                                    |
| `metrics`     | evaluation metrics in sixteen categories, with intervals and comparisons                                                                                                                         |
| `losses`      | differentiable training objectives and regularisers                                                                                                                                              |
| `rl`          | environment and agent contracts, wrappers, buffers, returns, planning, off-policy evaluation                                                                                                     |
| `games`       | normal-form, extensive-form and cooperative games, equilibria, learning in games, mechanisms                                                                                                     |
| `probprog`    | probabilistic programs, model analysis, inference engines and the planner, posteriors                                                                                                            |
| `learn`       | estimator protocol, preprocessing, composition, model selection, calibration, conformal, active learning, privacy                                                                                |
| `nn`          | neural networks: functional layers, modules, blocks, description, training, decoding                                                                                                             |
| `tune`        | studies, objectives, samplers, pruners, multi-fidelity schedules                                                                                                                                 |
| `datasets`    | dataset generators with exact truths, vendored real datasets, themed dataset collections                                                                                                         |
| `explain`     | attributions (Shapley, LIME, gradients), surrogates, counterfactuals, inspection, data valuation, explanation quality                                                                            |
| `fairness`    | group, individual and intersectional fairness, bias in data, mitigation, trade-off frontiers                                                                                                     |

### 4.1 `base/`

- `errors.ts`: `MdmlError` (codes: shape, dtype, value, domain, schema, convergence, singular, not-implemented,
  not-differentiable), `isMdmlError`, `invariant`, `assertShape`, `assertFinite`.
- `json.ts`: `Json`, `JsonSchema`, `encodeNumber`/`decodeNumber`, `toJson`/`fromJson` (non-finite numbers, -0, typed
  arrays and bigints round-trip through tagged objects).
- `results.ts`: the shared result shapes: `Estimate` (value, standard error, interval, level, method), `TestResult`
  (statistic, p-value, degrees of freedom, effect size, alternative, method, assumptions), `MetricResult` (estimate,
  breakdowns, sample size), `MetricSpec` (name, inputs, direction, range, category, concept), `formatResult`.
- `numbers.ts`: `isClose`, `allClose`, `clamp`, `kahanSum` (Neumaier), `pairwiseSum`, `maxOf`, `minOf`, `argmaxOf`,
  `argminOf` (loops, never spreading arrays into `Math.max`), `range`, `linspaceNumbers`, `popcount` (subset sizes of
  bitmasks).
- `heap.ts`: `heap`, a binary heap with a comparator (the priority queue of neighbour searches, shortest paths and
  best-first search).
- `options.ts`: shared option shapes (`Tolerance`, `Reduction`, `Alternative`, `ConfidenceLevel`) and their defaults
  (`defaultTolerance`, `resolveTolerance`, `defaultReduction`, `defaultAlternative`, `defaultConfidenceLevel`,
  `resolveConfidenceLevel`).
- `version.ts`: `VERSION`, generated from `package.json`.

### 4.2 `scene/`: pictures and structures as data

It uses `base` (errors, `clamp`, `maxOf`) and nothing else, so every other domain can describe pictures.
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

### 4.3 `tensor/`: arrays and differentiable primitives (numpy, torch.Tensor, jax.numpy)

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

### 4.4 `random/`: keys and samplers (jax.random, numpy.random)

oracle: Random123 known-answer vectors; scipy.stats.qmc for the sequences (unscrambled Sobol' and Halton exactly; mean
discrepancy and fill counts for the randomised ones); scipy.stats quantiles for distributional checks of every sampler
- `philox.ts`: `philox4x32`.
- `key.ts`: `Key`, `key(seed)`, `split`, `foldIn`, `keyFromPath(seed, ...names)` (named derivation), `randomBits`,
  `isKey`.
- `continuous.ts`: `uniform`, `uniformStream` (an endless stream of uniforms from one key in blocks
  `foldIn(key, b)`, the one stream every rejection sampler and procedural generator draws from), `normal`, `truncatedNormal` (Robert's samplers, any interval), `exponential`, `laplace`,
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
- Private: `_unit.ts` (53-bit uniform doubles, the polar normal and a small log-gamma), `_variates.ts` (scalar truncated normal, gamma, Poisson and
  binomial kernels), `_sobol.ts` (direction numbers, scramble, Gray-code generation) and `sobol-directions.data.ts`
  (the packed Joe-Kuo table for 1111 dimensions, vendored by `oracle/datasets/sobol_directions.py`).

### 4.5 `symbolic/`: a small computer algebra system (SymPy)

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
  `sub`, `mul`, `div`, `pow`, `neg`, `sqrt`), the constants, `free`, `dependsOn`, `substitute`, structural `equals`,
  `WorkBudget` and `withWorkBudget` (every operation does a bounded, deterministic amount of work, counted in work
  units rather than time: one per exact rational operation, expression node or term of polynomial arithmetic, with a
  cap on the terms of intermediate expansions; defaults 2,000,000 units, a fraction of a second, and 2,000 terms.
  When the work runs out, `integrate` keeps the terms it finished and leaves the rest unevaluated, `definiteIntegral`
  falls back to quadrature, `simplify` returns the best form found so far, `factor` and `trigSimplify` the input,
  and the others, `series` included, throw a MdmlError with code `convergence`; `integrate`, `series`,
  `simplify`, `factor` and `definiteIntegral` take a budget, `withWorkBudget` sets one for everything it calls).
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
  the search is bounded in attempts, depth, expression size and work (each method and level of the search gets half
  of the work left), so any integrand returns within a fraction of a second;
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
  and complex evaluation, adaptive Gauss-Kronrod: kept here so symbolic does not depend on numerics), `_measure.ts` (operation counts and
  numeric equivalence), `_expand.ts`, `_rational-functions.ts` (together, cancel, apart), `_trigsimp.ts`, `_poly.ts`
  (sparse polynomials, dense univariate helpers, generators with SymPy's `decompose_power`, so exp(2*x) is exp(x)^2;
  gcds with modular coprimality checks and primitive remainder sequences), `_zpoly.ts` (dense polynomials over Z and
  Z/p), `_factor.ts` (with an irreducibility test by evaluation before Kronecker's substitution), `_series.ts`
  (truncated Laurent series engine), `_integrals.ts` (tables, rational and trigonometric rules), `_latex-parse.ts`,
  `_budget.ts` (work budgets: frames, charges and fallbacks).

### 4.6 `autodiff/`: differentiation transforms (jax, torch.func)

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

### 4.7 `trace/`: recording what algorithms do

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
- `contract.ts`: the log contract (`docs/methods.md`): sources by path (`scalar`, `tensor`, `message`, `frame`,
  `derived`, `event`), subscriptions (`Log`, `On`, `off`/`defaults`/`all`), path patterns, `resolve` (contract and
  subscription to a policy, per-path schedules and hooks), `contractSchema`.
- `logging.ts`: the scheduler (every n steps, log-spaced, or on the nth firing of an event), the log context
  algorithms write to (`log`, `wants`, `emit`, `taps`), `defaultLogContext` for runs outside a session, and `gated`
  (the tracer that renames native channels to paths and passes only what is subscribed and due).
- `session.ts`: `Method` and `defineMethod` (config space, `make`, contract, defaults, named hooks, projections),
  `Session` (`advance`, `stateAt`, `fork`, `probe`, `project`, `run`, `stepsOf`) and `session` with its modes (eager,
  live, eager to a point).
- `serialise.ts`: `stateToJson` and `stateFromJson`, generic checkpoint serialisation of states made of tensors, typed
  arrays, keys and plain trees.
- `replay.ts`: `exportRun`, `parseRun`, `replay`, `mergeRuns`, `isRun`.
- `query.ts`: `metricSeries`, `tensorSeries`, `messages`, `episodes`, `frames`, `runTree`, `summary`, `diffRuns`.

oracle: numpy for `summaries.ts` (moments, norms, `np.histogram`); the rest by unit and property tests

### 4.8 `space/`: typed spaces (gymnasium.spaces, Optuna distributions, ConfigSpace)

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

### 4.9 `context/`: reproducibility and defaults

- `context.ts`: `Context` (seed, key derivation, tracer, trace policy, backend, default dtype, tolerances),
  `createContext`, `ctx.key(...names)`, `derive(ctx, name)` (child contexts with a namespaced key path).
- `default.ts`: `defaultContext`, `setDefaultContext` (the one documented piece of module state; read only by
  convenience layers).
- `resolve.ts`: `resolveKey(options, name)`, `resolveTracer(options)`, `resolvePolicy(options)`: the uniform rules every
  high-level function uses.
### 4.10 `numerics/`: numerical methods (scipy, numpy.linalg)

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
  `symmetricPart` (`(A + A^T) / 2`), `isPositiveDefinite`.
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
  `bsplineBasisElement`, `makeInterpSpline`, `makeLsqSpline`, `notAKnot`.
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
- `segments.ts`: segments as obstacles: `closestPointOnSegment` (point-segment distance), `raySegmentHit` (ray
  parameter and facing normal), `sweptCircleHit` (first contact of a moving circle with a segment's side or ends),
  `segmentsTouch` (robust closed-segment test with relative collinearity tolerance), `SegmentContact`. worlds'
  continuous motion, ray casts and line of sight are built on them.
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

### 4.11 `data/`: datasets, generators and truth (torch.utils.data, sklearn.datasets)

One schema per task kind. Each schema declares its per-row fields, so one `take` and one `split` work for all of them.
Every schema file also has a validating constructor that fills defaults (names, metadata). `schema/` and `modifiers/`
are inlined into `mdml-engine/data`. Each modifier also comes as a recipe `Modifier` (`labelNoiseModifier`,
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
  `bayesPredict`, `clusterAgreement`, `excessRisk`. `probprog` imports `data`, so `GeneratingProgram` is the
  structural contract a program over rows `{ x, y, n }` satisfies (its feature and target sites, per-row `logProb`,
  `posterior` for classification programs, `regressionFunction`, `sample`); `probprog.generatingProgram` builds one, and
  `completeTruth` derives the log density, posterior and regression function from it. Modifiers that change how the
  data arise drop the program (their own densities and posteriors remain).
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

### 4.12 `foundations/`: mathematical foundations

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

### 4.13 `graph/`: graphs and their algorithms (NetworkX, scipy.sparse.csgraph)

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

The graph machinery in `pgm/graphs.ts` and `pgm/triangulation.ts` predates this domain and works on
string-named adjacency maps; `pgm.connectedComponents`, `topologicalOrder` and `maximalCliques` can later delegate to
`graph.connectedComponents`, `graph.topologicalSort` and `graph.maximalCliques`.

### 4.14 `dists/`: distributions (scipy.stats, torch.distributions, NumPyro)

oracle: scipy.stats, torch.distributions

`continuous/`, `discrete/` and `multivariate/` are inlined into the entry point (`import { normal, gamma } from
'mdml-engine/dists'`); the other subfolders are namespaces (`bijectors`, `families`, `compose`). Sampling is a
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
- `continuous/sampling-distributions.ts`: `fDistribution`, `noncentralChi2`, `noncentralT`, `noncentralF`
  (Poisson-weighted series).
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
  tau). Its tau-b stays private: stats imports dists, so dists cannot use stats' `kendallTau`.
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

### 4.15 `information/`: information theory and coding

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
  (traced tree construction, D-ary; `huffmanTree`, the same tree built directly in O(n log n) from raw weights, which
  the hierarchical softmax in `losses` uses), canonical codes, Shannon-Fano (Fano's splits and Shannon's cumulative code),
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

### 4.16 `systems/`: linear time-invariant systems (scipy.signal LTI, python-control)

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

### 4.17 `stats/`: statistics (scipy.stats, statsmodels, pingouin)

oracle: scipy.stats, statsmodels, pingouin, patsy (smoothing bases), pandas (running windows)

Every test in `tests/` returns a `TestResult` with the statistic, p-value, effect size and interval, the alternative,
the method name and the assumptions it checked. Iterative procedures (permutation and bootstrap resampling, Newton
maximum likelihood, RANSAC, SPRT) are traced `Algorithm`s; randomness comes from explicit `Key`s. Private helpers:
`_util.ts` (input coercion, normal/t/chi-square/F tails, quadrature), `_linear.ts` (least squares and small symmetric
linear algebra over LAPACK), `_optimise.ts` (bounded one-dimensional maximisation); `tests/_shapiro.ts`,
`tests/_kolmogorov.ts`, `tests/_cvm.ts` (null distributions).
- `descriptive.ts`: `mean`, `variance` (two-pass, `ddof`), `std`, `skewness`, `kurtosis`, `moment`, weighted versions,
  `describe`, `zscore`, `sem`, `mode`, `geometricMean`, `harmonicMean`.
- `running.ts`: Welford moments (`momentState`, `push`, `pushAll`, `merge`, `momentsOf`), P-squared quantiles
  (`p2Quantile`, `p2Push`, `p2Estimate`), exponentially weighted moments (`ewmState`, `ewmPush`, `ewmMean`, `ewmVar`,
  `ewmStd`).
- `quantiles.ts`: `quantile` (the nine Hyndman-Fan methods), `median`, `iqr`, `percentileOfScore`, `ecdf` (with the
  Greenwood interval), `medianAbsoluteDeviation`.
- `histograms.ts`: `histogram`, `histogramBinEdges` (auto, Freedman-Diaconis, Scott, Sturges, Doane, Rice, sqrt,
  Knuth), `histogram2d`, `binnedStatistic`.
- `correlation.ts`: `pearson`, `spearman`, `kendallTau` (tau-b and tau-c, O(n log n), exact null without ties), `cov`
  (frequency and analytic weights), `corrcoef`, `partialCorrelation`, `pointBiserial`, `distanceCorrelation`, each with
  intervals. metrics' `pearsonCorrelation`, `spearmanCorrelation` and `kendallTau` delegate here. dists'
  `kendallTauMatrix` (copula fitting, in `dists`) keeps a private tau-b because dists cannot import stats.
- `covariance.ts`: `empiricalCovariance`, `ledoitWolf`, `oas`, `shrunkCovariance`.
- `density.ts`: `gaussianKde` (univariate and multivariate, weights, boundary reflection, resampling), `scottFactor`,
  `silvermanFactor`, `bandwidthScott`, `bandwidthSilverman`, `bandwidthCriterion`, `bandwidthCrossValidation`.
- `ranks.ts`: `rankdata`, `ranksOf`, `tieCounts`, `tieCorrection`, `tieTerm`.
- `transforms.ts`: `boxcox`, `yeojohnson` (maximum-likelihood lambda: `boxcoxNormmax`, `yeojohnsonNormmax`, log
  likelihoods), inverses, `standardise`.
- `tests/result.ts`: `TestResult` helpers (`testResult`, `resolveAlternative`, `pValueFromTails`, `intervalEstimate`,
  `waldInterval`), `formatTest`, `rejects`, assumption checks shared by tests (`levene`, `bartlett`, `fligner`,
  `checkNormality`, `checkEqualVariances`, `checkExpectedCounts`).
- `tests/means.ts`: `ttest1samp`, `ttestInd` (pooled and Welch), `ttestRel`, `ztest`, `fOneway`, `welchAnova`,
  `tukeyHsd`, `rmAnova`.
- `tests/contingency.ts`: `chi2Contingency`, `gTest`, `fisherExact` (2 x 2 exact, r x c by permutation of the
  margins), `mcnemar`, `cochranQ`, `oddsRatio`, `relativeRisk`, `expectedFrequencies`.
- `tests/nonparametric.ts`: `mannWhitneyU`, `wilcoxon`, `kruskal`, `friedman`, `nemenyi`, `nemenyiFriedman`,
  `signTest`, `brunnerMunzel`.
- `tests/distribution.ts`: `kstest`, `andersonDarling` (norm, expon, logistic, Gumbel), `shapiro`, `jarqueBera`,
  `chisquare` and `powerDivergence`, `lilliefors` (normal and exponential), `cramervonmises`.
- `tests/two-sample.ts`: `ks2samp`, `cramervonmises2samp`, `andersonKsamp`, `energyDistance`, `wassersteinDistance`,
  `energyTest`, `wassersteinTest` (permutation), `eppsSingleton`, `classifierTwoSampleTest` (any classifier passed in).
- `tests/kernel.ts`: `mmd2`, `mmdTest` (permutation and asymptotic nulls), `hsic`, `hsicTest`; they take any kernel and
  default to `gaussianKernel` with `medianHeuristic`.
- `tests/independence.ts`: `distanceCovarianceTest`, `hoeffdingD`, `mutualInformation` and `mutualInformationTest`,
  conditional independence tests (`partialCorrelationTest`, `gTestConditional`, `kernelCITest`) used by causal
  discovery.
- `tests/multivariate.ts`: `hotellingT2`, `manova`, `boxM`, `mardia`.
- `tests/proportions.ts`: `binomTest`, `proportionsZtest`, `proportionConfint` (Wald, Wilson, Agresti-Coull,
  Clopper-Pearson, Jeffreys), `proportionsDifference`.
- `tests/timeseries.ts`: `ljungBox`, `durbinWatson`, `adfuller` (with `mackinnonp`, `mackinnoncrit`), `kpss`,
  `runsTest`.
- `tests/equivalence.ts`: `tost`, `tostProportion`, `tostProportions`.
- `tests/bayesian.ts`: `bayesFactorTTest` (JZS), `bayesFactorBinomial`, `bayesFactorPearson`, `bayesianAbTest`,
  `highestDensityInterval`, `rope`.
- `tests/sequential.ts`: `sprt` (traced `sprtAlgorithm`, `sprtBernoulli`, `sprtNormal`), `alphaSpending` with
  `spendingFunction`, `alwaysValidPValues`, `msprt`, `confidenceSequence`.
- `tests/multiple.ts`: `adjustPValues`/`multipleTests`, `bonferroni`, `holm`, `hochberg`, `benjaminiHochberg`,
  `benjaminiYekutieli`, `combinePValues`.
- `effect-size.ts`: `cohensD`, `hedgesG`, `glassDelta`, `cohensH`, `cramersV`, `etaSquared`, `omegaSquared`,
  `cliffsDelta`, with intervals.
- `power.ts`: `powerTTest`/`sampleSizeTTest`, `powerProportions`/`sampleSizeProportions`, `powerAnova`/
  `sampleSizeAnova`, `powerChisquare`/`sampleSizeChisquare`, `simulatePower`.
- `resampling.ts`: `bootstrap` (percentile, basic, BCa, studentised; traced `bootstrapAlgorithm`), `permutationTest`
  (exact enumeration when small; traced `permutationTestAlgorithm`), `jackknife`, `monteCarloTest`.
- `extremes.ts`: `fitGpd` and `fitGev` (`ExtremeFit`), `peaksOverThreshold` (runs declustering), `returnLevel` (delta
  method interval), `meanExcess`.
- `robust.ts`: `trimboth`, `trimMean`, `trimmedVariance`, `winsorize`, `winsorizedMean`, `winsorizedVariance`,
  `mEstimateLocation`, `huber`, `tukeyBiweight`, `ransac` (traced `ransacAlgorithm`, generic in points and models).
- `smoothing/bases.ts`: `SmoothBasis`; `bs`, `ns`, `ps`, `cr`, `cc`, `tp`, `ts`, `duchon`, `gp`, `re`, `fs`, `te`,
  `ti`, with `placeKnots` (quantile and uniform rules).
- `smoothing/penalties.ts`: `differencePenalty`, `derivativePenalty`, `nullSpace`, `shrinkagePenalty`,
  `shapeConstraint` (increasing, decreasing, convex, concave reparameterisations), `sumToZeroConstraint`,
  `constraintNullSpace`.
- `smoothing/smoothers.ts`: `loess`, `lowess`, `nadarayaWatson`, `localLinear`, `runningStatistic`, `runningMean`,
  `runningMedian`, `smoothingSpline` (lambda, degrees of freedom or GCV over numerics' `makeSmoothingSpline`).
- `fit.ts`: `fitDistribution` (closed form or Newton for seventeen families) with standard errors from the
  `observedInformation`; `newtonMle` (traced `newtonMleAlgorithm`).

TODO (deferred, esoteric):
- `tests/bayesian.ts`: the Gauss hypergeometric series behind `bayesFactorPearson` is private; move it to
  `numerics/special` when that domain gains hypergeometric functions.
- `tests/bayesian.ts`: `highestDensityInterval` should alias `sampling`'s HDI once that domain has one
  (stats may import sampling, which does not import stats).
- `tests/distribution.ts`: `andersonDarling` lacks scipy's `weibull_min` (its critical values depend on a fitted
  shape).
- `smoothing/bases.ts`: `tp`, `duchon` and `gp` take the full eigen-decomposition of the kernel matrix (mgcv
  subsamples beyond 2000 points; add that when large data needs it).

### 4.18 `optim/`: optimisation (optax, torch.optim, scipy.optimize)

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

### 4.19 `kernels/`: kernels and feature maps (sklearn kernels, GPyTorch)

oracle: scikit-learn, gpytorch, scipy, networkx (numpy closed forms for arc-cosine and NTK: neural-tangents does not
import with the installed jax)
- `kernel.ts`: `Kernel` (`k(x, y)`, `diag`, `withParams`, hyperparameters as a differentiable pytree with per-leaf
  constraints), `defineKernel`, `asRows`, `gram`, `crossGram`, `sumKernel`, `productKernel`, `scaleKernel`,
  `powerKernel`, `onDimensions` (active dimensions), `ard`, log-parameterisation (`theta`, `withTheta`,
  `hyperparameters`, `fixHyperparameters`), `gramGradient` (scikit-learn's eval_gradient), `formatKernel`.
- `stationary.ts`: `rbfKernel`, `maternKernel` (1/2, 3/2, 5/2, infinity and general through K_nu),
  `rationalQuadraticKernel`, `exponentialKernel` (gamma-exponential), `laplacianKernel`, `periodicKernel`,
  `locallyPeriodicKernel`, `spectralMixtureKernel`, `cosineKernel`.
- `dot-product.ts`: `linearKernel`, `polynomialKernel`, `sigmoidKernel`, `cosineSimilarityKernel`, `arcCosineKernel`
  (orders 0-2, composed to any depth), `neuralTangentKernel` (NTK and NNGP of a ReLU or erf MLP), `empiricalNtk`.
- `structured.ts`: `whiteNoiseKernel`, `constantKernel`, string kernels (`spectrumKernel`, `subsequenceKernel`), set
  kernels (`setKernel` mean embedding, `jaccardKernel`), graph kernels (`diffusionKernel`, `regularisedLaplacianKernel`,
  `wlSubtreeFeatures`, `wlSubtreeKernel`), chi-square (`chi2Distance`, `chi2Kernel`, `additiveChi2Kernel`,
  `skewedChi2Kernel`), `histogramIntersectionKernel`.
- `expansion.ts`: `FeatureMap`; `randomFourierFeatures` (spectral samplers for RBF, Matérn, exponential, Laplacian,
  rational quadratic, scaled kernels, or given draws), `orthogonalRandomFeatures`, `nystroem`, `additiveChi2Sampler`,
  `skewedChi2Sampler`, `polynomialCountSketch`, `tensorSketch`, approximation error helpers (`approximateGram`,
  `approximationError`).
- `matrices.ts`: `centerKernel`, `kernelAlignment`, `targetAlignment`, `hsic` (biased, unbiased, normalised),
  `mmdSquared` (biased, unbiased, linear-time), `effectiveDimension`, `kernelPca` projections, `pairwiseKernels`
  (scikit-learn's metric names and parameters).

### 4.20 `signal/`: signal and image processing (scipy.signal, scipy.ndimage, pywt, librosa)

oracle: scipy.signal, scipy.ndimage, pywt, librosa, scikit-image, stumpy

The files follow the encyclopedia's signal-processing chapters. LTI system representations are in `systems`; sparse
representations are in `optim/sparse.ts`.
- `signal.ts`: `Signal` (samples, sampling rate, start, units, channels), `Spectrum`, `TimeFrequency`, `sampleTimes`,
  `magnitudeDb`, `unwrap`.
- `sampling.ts`: sampling and reconstruction (sinc interpolation), aliasing, quantisation (uniform, mu-law, dither),
  zero-order hold, signal-to-quantisation-noise.
- `fourier.ts`: DTFT evaluation, zero padding and frequency resolution, spectral leakage, Parseval checks, the analytic
  signal and Hilbert transform.
- `windows/` (inline): `cosine.ts` (general cosine and Hamming, boxcar, Hann, Hamming, Blackman, Blackman-Harris,
  Nuttall, flat top), `shapes.ts` (Bartlett, triangular, Kaiser, Kaiser-Bessel-derived, Gaussian and general
  Gaussian, Tukey, cosine, Parzen, Bohman, Bartlett-Hann, exponential, Lanczos), `dpss.ts` (Slepian tapers by Sturm
  bisection and inverse iteration of the tridiagonal problem, concentration ratios), `get-window.ts` (`getWindow` by
  name and parameters, as scipy), `metrics.ts` (ENBW, coherent gain, main-lobe width, sidelobe level).
- `filters/` (inline): one filtering primitive (`_iir.ts`: A(z) y = B(z) x from rest, its jvp and vjp written in terms
  of itself), and on top of it:
  - `apply.ts`: `lfilter`, `lfilterZi`, `lfiltic`, `sosfilt`, `sosfiltZi`, `filtfilt` (pad and Gustafsson),
    `sosfiltfilt`, `medfilt`, `savgolCoeffs`, `savgolFilter`, moving and exponential moving averages.
  - `fir.ts`: `firwin`, `firwin2`, `remez` (Parks-McClellan), `kaiserord`, `kaiserBeta`, `kaiserAtten`,
    `minimumPhase`.
  - `prototypes.ts`: analogue prototypes (`buttap`, `cheb1ap`, `cheb2ap`, `ellipap`, `besselap`), frequency
    transformations of zeros, poles and gain, the bilinear transform.
  - `iir.ts`: `iirfilter`, Butterworth, Chebyshev I and II, elliptic, Bessel, notch, peak and comb filters; `ba`, `zpk`
    and `sos` outputs.
  - `order.ts`: order selection (`buttord`, `cheb1ord`, `cheb2ord`, `ellipord`) and `iirdesign` from a specification.
  - `response.ts`: `sosfreqz`, `freqzZpk`, group delay, phase delay, frequency responses, specification checks.
- `multirate.ts`: `upfirdn` (every extension mode), `resamplePoly`, `resample`, `decimate`, interpolation, down- and
  upsampling, polyphase decomposition and recombination, filter expansion (noble identities).
- `filter-banks.ts`: DFT analysis and synthesis banks, quadrature-mirror and perfect-reconstruction checks, power
  complementarity, wavelet filter banks, wavelet packets.
- `spectral.ts`: detrending, periodogram, Welch, Bartlett, Blackman-Tukey, cross-spectral density, coherence, adaptive
  multitaper, Welch degrees of freedom, spectral confidence intervals.
- `lomb-scargle.ts`: the Lomb-Scargle periodogram (as astropy) with amplitudes, false-alarm probabilities and levels.
- `parametric.ts`: Levinson-Durbin, Yule-Walker, Burg, AR spectra, MUSIC, ESPRIT, Prony.
- `time-frequency.ts`: STFT (scipy and librosa framing) and inverse, spectrogram, instantaneous phase and frequency,
  time and frequency spreads, the time-bandwidth product.
- `wigner.ts`: Wigner-Ville, pseudo and smoothed pseudo Wigner-Ville distributions.
- `reassignment.ts`: reassigned spectrograms, synchrosqueezing and its inverse.
- `constant-q.ts`: constant-Q frequencies, kernels and the transform (as librosa).
- `decompositions.ts`: EMD, EEMD, CEEMDAN, VMD (a traced `Algorithm`), singular spectrum analysis.
- `wavelets/` (inline): `families.ts` (Haar, Daubechies, Symlets, Coiflets, biorthogonal, reverse biorthogonal, discrete
  Meyer and the continuous families, generated from pywt), `dwt.ts` (DWT and inverse, multilevel, 2-D), `swt.ts`
  (stationary transform and inverse), `cwt.ts` (continuous transform, centre frequencies), `denoise.ts` (thresholds,
  noise estimation, wavelet denoising as scikit-image).
- `estimation.ts`: Wiener filtering (causal and non-causal), linear prediction and LPC, the matched filter.
- `detection.ts`: energy and matched-filter detectors, CFAR, detector ROC curves.
- `adaptive.ts`: LMS, NLMS, RLS, adaptive noise cancellation, traced tap weights.
- `cepstrum.ts`: real and complex cepstra, liftering, cepstral pitch.
- `peaks.ts`: `findPeaks` (height, prominence, distance, width), prominences, widths, zero crossings.
- `similarity.ts`: dynamic time warping (with path and lower bounds), PAA, SAX, cross-correlation lag.
- `sliding.ts`: the sliding-window building blocks of matrix profiles (as STUMPY): rolling means and standard
  deviations by Welford updates with periodic exact recomputation, constancy (monotone deques) and finiteness of
  windows, subsequence statistics, sliding dot products (direct for small products, FFT or overlap-add otherwise) and
  the O(1) diagonal update, z-normalised distances from dot products (STUMPY's constant and NaN conventions),
  non-normalised distance profiles, exclusion zones, z-normalisation. The matrix-profile methods are in
  `methods/timeseries/matrix-profile/`.
- `mass.ts`: MASS distance profiles (v2 by FFT, v3 chunked), the non-normalised variant, `distanceProfile` with the
  trivial-match exclusion zone.
- `image/correlate.ts`: `correlate1d`, `convolve1d`, `correlateNd`, `convolveNd` with scipy.ndimage boundary modes and
  origins (the kernel of the image filters below).
- `audio/scales.ts`: mel (HTK and Slaney), Bark (Traunmüller, Zwicker-Terhardt, Schroeder) and ERB scales, mel and
  FFT frequencies, ERB-spaced centre frequencies; differentiable.
- `audio/filterbanks.ts`: mel and chroma filter banks (librosa.filters), gammatone filters (FIR and Slaney's IIR, as
  scipy) and gammatone banks.
- `audio/features.ts`: framing, librosa-convention spectrograms, mel spectrograms, power and amplitude decibels, MFCCs
  (liftering), spectral centroid, bandwidth, roll-off, flatness and flux, chroma, zero-crossing rate, RMS energy (from
  signals and spectrograms); the spectrogram-based ones are differentiable.
- `audio/pitch.ts`: autocorrelation, YIN and probabilistic YIN (HMM Viterbi decoding of pitch and voicing), autocorrelation
  pitch, harmonic product spectrum (cepstral pitch is `cepstrum.ts`).
- `audio/rhythm.ts`: onset strength, greedy peak picking, onset detection, the autocorrelation tempogram, tempo
  estimation under a log-normal prior, Ellis's dynamic-programming beat tracker.
- `audio/enhancement.ts`: power spectral subtraction, decision-directed Wiener denoising, energy voice activity
  detection with hangover.
- `audio/separation.ts`: ideal binary, ratio and phase-sensitive masks, mask application, oracle mask separation
  (separation scores are `metrics/audio.ts`; NMF and ICA separation are in `methods/audio`).
- `audio/coding.ts`: the absolute threshold of hearing, Schroeder's spreading function, a Bark-domain masking threshold,
  perceptual MDCT transform coding with an entropy estimate. TODO: the full MPEG psychoacoustic model (tonal and
  noise maskers, decimation of maskers).
- `audio/augmentation.ts`: SpecAugment masking (TODO: time warping), the phase vocoder, time stretching and pitch
  shifting (as librosa with FFT resampling), mixing at an SNR, additive noise.
- `image/filters.ts`: Gaussian filters and derivative kernels, uniform, rank, median, minimum, maximum and percentile
  filters, Sobel, Scharr and Prewitt, Laplace, Laplacian of Gaussian, Gaussian gradient magnitude, bilateral filtering,
  Gabor kernels, filters and banks; the linear ones differentiable.
- `image/features.ts`: structure tensors, Harris and Shi-Tomasi corners, `peakLocalMax`, Canny, local binary patterns,
  HOG, Hough lines and circles (Gabor features are `gaborBank`). TODO: LoG and DoG blob detection, Hough peak
  extraction.
- `image/morphology.ts`: binary and grey erosion, dilation, opening and closing (structures, origins, iterations),
  morphological gradient, top-hats, connected-component labelling, the exact Euclidean distance transform, thinning and
  Zhang-Suen skeletonisation.
- `image/geometry.ts`: `mapCoordinates`, affine transforms, shifts, zooms and rotations (orders 0 and 1, every ndimage
  mode), scikit-image resizing and projective warps, Gaussian pyramids, integral images. TODO: spline orders 2 to 5.
- `image/segmentation.ts`: Otsu and Li thresholds, marker watershed, Felzenszwalb's graph-based segmentation, exact
  binary graph cuts (by `graph` minimum cut). TODO: SLIC superpixels.
- `image/motion.ts`: template matching (normalised cross-correlation), phase correlation, block matching, dense
  Lucas-Kanade and traced Horn-Schunck optical flow. TODO: pyramidal iterative Lucas-Kanade and TV-L1.

### 4.21 `text/`: text processing (HF tokenizers, nltk, sklearn.feature_extraction.text)

oracle: HF tokenizers, tiktoken's patterns (with the `regex` module), nltk, scikit-learn, rapidfuzz, datasketch, gensim,
jellyfish, Biopython, pyphen
- `normalise.ts`: Unicode normalisation, case folding, accent stripping, whitespace collapsing, `AlignedText` (offsets
  back to the original).
- `tokenise.ts`: regex (nltk RegexpTokenizer, with gaps), whitespace, word (sklearn pattern), word-punct, character
  and grapheme tokenisers, pre-tokeniser patterns (GPT-2, cl100k, o200k, HF Whitespace), HF-style splitting with
  delimiter behaviours, BERT pre-tokenisation.
- `treebank.ts`: the Penn Treebank tokeniser, nltk's improved word tokeniser, Treebank spans and detokenisation.
- `tweet.ts`: nltk's TweetTokenizer (URLs, phone numbers, emoticons, handles, hashtags, emoji sequences), HTML entity
  replacement, handle removal, lengthening reduction.
- `sentences.ts`: sentence splitting: a rule-based splitter and Punkt (Kiss and Strunk 2006) training and tokenisation.
- `stem.ts`: Porter (nltk's three modes, with a rule trace) and Snowball English stemmers.
- `lemmatise.ts`: rule-based lemmatisation (WordNet's morphy detachment rules over a lexicon with exception lists, or
  longest-suffix rules with common irregular forms).
- `stopwords.ts`: stop-word lists (scikit-learn's, gensim's) and removal.
- `hashing.ts`: MurmurHash3 (scikit-learn's), FNV-1a (with fastText's sign-extended variant), UTF-8 bytes.
- `vocabulary.ts`: `Vocabulary`, building (minimum count, maximum size, specials, ordering), encoding and decoding.
- `subword/bpe.ts`: BPE training (character and byte level, traced merges), encoding with dropout, the GPT-2 byte
  alphabet.
- `subword/wordpiece.ts`: WordPiece training and greedy longest-match encoding.
- `subword/unigram.ts`: Unigram language-model training (suffix-array seeds, EM with the digamma M step and pruning,
  as HF's trainer, traced), Viterbi encoding, n-best and sampling.
- `tokenizer.ts`: an HF-tokenizers-style pipeline (added tokens with single-word, lstrip, rstrip and normalized
  matching; normaliser, pre-tokeniser, model, post-processor, decoder), encodings with offsets, type ids, masks, word
  and sequence ids, batch encoding with padding and truncation (strategies, stride, overflowing windows), decoding.
- `tokenizer-components.ts`: the pipeline's normalisers (Unicode forms, Lowercase, StripAccents, Strip, Replace,
  Prepend, BertNormalizer, NMT, ByteLevel, Sequence), pre-tokenisers (Whitespace, WhitespaceSplit, BERT, ByteLevel,
  Metaspace, Punctuation, Digits, Split, CharDelimiterSplit, FixedLength, Sequence), decoders (ByteLevel, WordPiece,
  Metaspace, BPE, Replace, Strip, Fuse, ByteFallback, CTC, Sequence) and HF regex translation.
- `tokenizer-json.ts`: loading HF `tokenizer.json` files. TODO: the Precompiled (SentencePiece charsmap) normaliser
  and the UnicodeScripts pre-tokeniser.
- `shingles.ts`: character k-shingles, word n-gram shingles, w-shingling, skip-grams, the hashed character n-grams of
  fastText (bucketed with FNV hashing), weighted shingles; one extractor shared by matching, fingerprinting and fastText.
- `features.ts`: count, TF-IDF and hashing vectorisers, word and character n-grams, BM25 and BM25+ weighting.
- `templates.ts`: feature-template languages for sequence labellers. CRF++ templates (Kudo 2005): `U` (unigram, conjoined
  with the label) and `B` (bigram, conjoined with the label pair) lines, `%x[row,col]` macros over a token-by-attribute
  table, boundary padding (`_B-1`, `_B+1`), comments, and syntax errors with line and column. Wapiti patterns (`u`,
  `b`, `*` lines; `%x`, `%t`, `%m` commands with regular expressions, case-insensitive and absolute variants, `_x-1`
  boundaries). Expansion over a sequence, a `FeatureIndex` (string to id with counts and a minimum frequency, as
  CRF++'s `-f`), CRF++ weight offsets, feature firing for inspection.
- `attribute-templates.ts`: CRFsuite-style attribute templates (windows of `(attribute, offset)` tuples joined into
  attribute strings, `w[-1]|w[0]`, skipped or padded boundaries, bias, `__BOS__`/`__EOS__`; parsing from text) and
  MALLET's offset conjunctions.
- `representations.ts`: co-occurrence counts (windows, harmonic weighting), PMI, PPMI and shifted PPMI with context
  smoothing, LSA, SVD word vectors, random indexing, analogies (3CosAdd, 3CosMul), nearest words.
- `collocations.ts`: nltk's bigram and trigram association measures (raw frequency, Student's t, chi-square, PMI,
  likelihood ratio, Poisson-Stirling, Jaccard, MI-like, Dice, phi squared, Fisher) and windowed collocation finding
  with frequency and word filters.
- `similarity.ts`: Levenshtein (weighted), the edit-alignment counts behind WER (rapidfuzz's tie-breaking),
  Damerau-Levenshtein (OSA and unrestricted), Hamming, Jaro and Jaro-Winkler, longest common subsequence (table,
  backtrace) and substring, Indel, Jaccard (set and weighted) and containment over shingles, clipped count overlaps,
  ratio, token-set and token-sort ratios (rapidfuzz). The canonical home of the sequence helpers of `metrics/text.ts`.
- `alignment.ts`: Needleman-Wunsch and Smith-Waterman with linear or affine (Gotoh) gaps, alignment scoring and
  formatting.
- `phonetic.ts`: Soundex and Metaphone (jellyfish's).
- `approximate-search.ts`: approximate string search: q-gram indexes with count filtering, BK-trees, tries with fuzzy
  lookup, Sellers' approximate substring matching, top-k fuzzy matches with traced candidate pruning.
- `fingerprinting.ts`: Rabin-Karp rolling hashes and search, winnowing (MOSS), SimHash, MinHash signatures (datasketch's
  scheme), LSH banding with its optimal parameters, near-duplicate detection.
- `parsing.ts`: context-free grammars (nltk's notation, probabilities), parse trees, CYK and Earley parsing, conversion
  to Chomsky normal form.
- `pcfg.ts`: probabilistic CFGs: the Viterbi parse, sentence probabilities, inside-outside expected counts, EM
  training as a traced run.
- `dependency-parsing.ts`: Eisner's projective parser (optionally single-rooted), projectivity, arc-standard
  transitions with the static oracle and greedy transition-based parsing.
- `hyphenation.ts`: Liang pattern hyphenation with matching traces, exceptions, and Hunspell/pyphen dictionaries.
  TODO: Németh's non-standard (`/`) patterns.
- `statistics.ts`: Zipf (log-log and maximum likelihood) and Heaps fits, type-token ratios (root, log, moving average,
  hapax ratio), tokeniser fertility and compression ratio, topic coherence (UMass, NPMI as gensim).

### 4.22 `search/`: search, constraint satisfaction and planning (AIMA)

oracle: aima-python (typed ports of its search, game and AC-3 code in `oracle/mdml_oracle/aima_*.py`, since it is
not packaged; expansion, goal-test, generation and node counts match exactly under its tie-break by state order) and
hand-checked textbook fixtures; python-constraint (CSP solution sets); pyperplan (grounding, optimal plan lengths,
h-add and h-max); pysubgroup (exhaustive WRAcc subgroup discovery); NetworkX (path costs)

One `SearchProblem` contract (initial state, actions, result, goal test, step cost, heuristic) and traced search
algorithms that record the frontier and explored set, so every search can be animated. Every search is an `Algorithm`
(one expansion, move, generation or level per step) with an expansion, node or iteration budget, so it terminates on
unsolvable and infinite spaces. Searches over explicit graphs are `graph`'s; `graphProblem` adapts a `Graph`, and the
frontiers reuse `graph`'s `BinaryHeap`.
- `problem.ts`: `SearchProblem`, `SearchNode`, `rootNode`, `childNode`, `expand`, path reconstruction (`pathNodes`,
  `solutionPath`, `solutionActions`, `replay`), `SearchResult` and `SearchStats` (expanded, generated, goal tests,
  largest frontier, explored), `effectiveBranchingFactor`, `stateKeyOf`, `graphProblem`, `searchTree` (a `Structure`).
- `uninformed.ts`: breadth-first, depth-first, depth-limited, iterative deepening, uniform cost, bidirectional search
  (each with an `...Algorithm` form).
- `informed.ts`: best-first on any `f`, greedy best-first, A*, weighted A*, IDA*, RBFS, beam search; heuristic
  admissibility and consistency checks (`isAdmissible`, `isConsistent`, via `graph`'s `dijkstra` on the reversed
  reachable space).
- `local.ts`: hill climbing (steepest, first choice, stochastic, sideways moves; random restarts as child runs),
  simulated annealing (exponential, geometric and logarithmic schedules), tabu search, local beam search, late
  acceptance; `spaceProblem` over a `Space`, permutation and bit-flip neighbourhoods.
- `evolutionary/` (inlined into `search`):
  - `operators.ts`: selection (tournament, roulette, stochastic universal sampling, linear rank, truncation),
    crossover and mutation for bit strings, permutations (order, PMX, cycle; swap, inversion, scramble) and real
    vectors (blend, SBX, arithmetic; Gaussian, polynomial).
  - `genetic.ts`: genetic algorithms over representations (`bitStrings`, `permutations`, `realVectors`),
    self-adaptive evolution strategies ((mu, lambda) and (mu + lambda); CMA-ES is `optim`'s).
  - `programming.ts`: genetic programming over expression trees (primitive sets, grow, full and ramped half-and-half,
    subtree crossover and mutation, point mutation, symbolic-regression fitness). `methods/evolutionary` maps trees to
    `symbolic` expressions.
  - `quality-diversity.ts`: MAP-Elites, novelty search (`numerics`' `kNearest`).
- `swarm.ts`: ant colony optimisation for the travelling salesman (Ant System, elitist, MAX-MIN), discrete (binary)
  particle swarm (continuous swarms are `optim`'s).
- `adversarial.ts`: minimax, alpha-beta with move ordering, negamax (with or without pruning), expectimax, Monte Carlo
  tree search (UCT and PUCT with priors and value estimates), traced node by node with visit, value and prune events.
- `csp.ts`: constraint satisfaction problems with constraints of any arity: backtracking with MRV, degree and LCV
  heuristics, forward checking, AC-3 (generalised to n-ary constraints), maintaining arc consistency, min-conflicts;
  all-solution enumeration and the incremental search formulation (`cspProblem`).
- `planning/` (inlined into `search`):
  - `strips.ts`: STRIPS actions and problems, progression and regression, plan validation, forward and regression
    planning (A*, greedy, weighted A*, breadth-first) with h-add, h-max, FF and goal-count heuristics; regression
    prunes subgoal sets with unreachable facts or invariant mutexes.
  - `pddl.ts`: a PDDL subset (`:strips`, `:typing`, `:equality`): s-expressions, domains, problems, grounding with
    static-predicate and reachability pruning.
  - `heuristics.ts`: h-add, h-max, h-FF and FF's relaxed plan (`_relaxed.ts` holds the shared relaxed reachability).
  - `graphplan.ts`: GraphPlan (mutexes, memoised backward extraction, termination on levelling off), the planning
    graph's levels, invariant mutex pairs.
  - `partial-order.ts`: partial-order planning (causal links, threats by promotion and demotion) as a search problem,
    linearisation (`graph`'s `topologicalSort`).
  - `htn.ts`: simple total-order HTN decomposition (methods with preconditions, backtracking), the decomposition tree
    as a `Structure`.
- `refinement.ts`: refinement-operator search over pattern and rule languages (beam, best-first, exhaustive with
  optimistic-estimate pruning), top-k result sets with redundancy filtering (`TopK`, `redundancyFilter`, coverage
  similarity), the conjunctive selector language (`conjunctions`); used by rule learning, subgroup discovery,
  exceptional model mining and ILP.
- Internal: `_engine.ts` (counters, trace schema and observer of path searches), `_best-first.ts` (the best-first
  frontier with lazy deletion and tie-breaking), `_csp-model.ts` (compiled CSPs), `_draw.ts` (buffered draws from a
  key). Textbook instances for tests (Romania, eight puzzle, n-queens, tic-tac-toe and connect four, Australia,
  Sudoku, SEND+MORE=MONEY) are in `packages/engine/src/testing/search.ts`; the named puzzles and domains for users are
  `methods/gofai`.

### 4.23 `stochastic/`: stochastic processes and stochastic calculus

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

### 4.24 `sampling/`: Monte Carlo (NumPyro and BlackJAX kernels, ArviZ diagnostics)

oracle: ArviZ (arviz-stats) for every diagnostic and for PSIS on fixed chains; BlackJAX for the mechanics that are
deterministic given the same inputs (integrator trajectories, the U-turn criterion, dual averaging, the warm-up schedule,
Welford mass matrices, one SVGD step, resampling from the same uniforms, the adaptive-tempering solver); statistical
agreement (moments within a few MCSE, invariance of exact draws under one transition, KS tests, detailed balance on small
discrete targets) for the kernels themselves. Every kernel is a BlackJAX-style `MCMCKernel` (`init`, `step(key, state)`)
traced through `samplerAlgorithm`; every loop has an iteration or depth cap and degenerate targets (NaN densities, zero
gradients, infinite step sizes) end in rejections or divergences, never hangs.
- `target.ts`: `LogDensity` (log density, gradient by autodiff or by hand, dimension, names, constraints,
  `constrain`/`unconstrain`), `target`, `numericTarget`, `targetFromDistribution` (unconstrained through
  `bijectorForSupport`), `unconstrainedTarget` (per-coordinate supports or a bijector), `Proposal`, `proposal`,
  `proposalFromDistribution`, and the standard targets `nealsFunnel` (centred and non-centred) and `banana` (Haario).
- `chains.ts`: the kernel protocol (`MCMCKernel`, `MCMCState`, `TransitionInfo`, `Adaptation`, `Warmup`),
  `samplerSchema`, `samplerAlgorithm` (per-iteration position, log density, acceptance, step size, tree depth,
  integration steps, energy, divergences), `sampleChain`, `sampleChains` (independent keys per chain, warm-up or
  adaptation, thinning, parent run with a child per chain), `Chains`, `parameterDraws`, `flattenChains`, `thin`,
  `discardWarmup`, `chainsFromDraws`, `constrainChains`.
- `metropolis.ts`: `metropolisAcceptance`, `metropolisHastings` (any proposal, discrete targets too),
  `randomWalkMetropolis` (scales or a full covariance), `independenceMetropolis`, `adaptiveMetropolis` (Haario et al.),
  `randomWalkAdaptation` (Robbins-Monro scale and learned shape during warm-up).
- `hamiltonian/`: `integrators.ts` (`euclideanMetric` with the generalised U-turn criterion, palindromic `integrator`s
  velocity Verlet, McLachlan, Yoshida and Omelyan, `leapfrog`, `integratorState`, `hamiltonian`,
  `hamiltonianTrajectory`); `hmc.ts` (`hmcKernel`, `hmcAdaptation`, `hmc`); `nuts.ts` (multinomial NUTS with biased
  progressive sampling, the U-turn checks across subtrees, divergences, `nutsKernel`, `nutsAdaptation`, `nuts`);
  `adaptation.ts` (dual averaging, `findReasonableStepSize`, Welford mass-matrix adaptation, diagonal or dense,
  `windowSchedule`, `windowAdaptation`); `riemannian.ts` (`riemannianHmcKernel` with the generalised leapfrog and a
  given or SoftAbs metric, `riemannianTrajectory`), for small problems.
- `langevin.ts`: `malaKernel` (optionally preconditioned), `ulaKernel`, `sgldKernel` with `minibatchGradient` and
  `polynomialStepSize`, SVGD (`svgdDirection`, `medianBandwidth`, `svgdAlgorithm`, `svgd`).
- `gibbs.ts`: `blockedGibbs` (systematic or random scan), `sliceSampler` (Neal's stepping-out and shrinkage, coordinate
  by coordinate), `ellipticalSlice`.
- `importance.ts`: `importanceSampling`, `normaliseLogWeights`, `importanceEss`, `selfNormalisedEstimate`,
  `logEvidence`, `paretoSmoothedWeights` (PSIS), `paretoKHat`, `weightDiagnostics`.
- `smc.ts`: resampling schemes (`systematicResample`, `stratifiedResample`, `multinomialResample`, `residualResample`,
  `resample`), `nextTemperature` (adaptive tempering), `adaptiveRandomWalkMove`, `smcAlgorithm` and `smcSampler`
  (adaptive-tempering SMC with MCMC rejuvenation and the evidence), `annealedImportanceSampling`.
- `rejection.ts`: `rejectionSample` (with envelope-violation counts), `adaptiveRejectionSample` (Gilks and Wild, tangent
  hull and chord squeeze), `ratioOfUniforms` (Kinderman and Monahan).
- `exact.ts`: `inverseTransformSample`, `inverseTransformFromCdf`, `discreteInverseTransform`, `boxMuller`,
  `marsagliaPolar`, `aliasMethod` (Vose), each with optional step records for teaching.
- `diagnostics/`: `convergence.ts` (`rhat`: rank-normalised split, split, folded, z-scale and identity; `ess`: bulk, tail,
  mean, sd, quantile, median, MAD, local, folded, z-scale, identity; `mcse`: mean, sd, median, quantile);
  `autocorrelation.ts` (`autocovariance`, `autocorrelation` by FFT); `intervals.ts` (`hdi`, also circular, `hdiModes` for
  discrete draws, `eti`; `stats`'s `highestDensityInterval` aliases `hdi`); `reports.ts` (`summary`, `bfmi`,
  `divergenceReport`, `treeDepthReport`, `diagnose`). TODO: multimodal HDI of continuous draws needs a density estimate
  (ArviZ uses an ISJ-bandwidth KDE, which lives in `stats`), and circular MCSE.
### 4.25 `logic/`: logic programming, satisfiability and ILP

oracle: python-sat (Glucose satisfiability, model counts by enumeration, MUSX cores, RC2 MaxSAT optima); clingo
(least and perfect models of Datalog programs, answer sets); sympy.logic (models, validity, entailment, minimal sum and
product forms); typed reference ports in `oracle/cases/logic/` for what has no packaged reference under uv (a
generator-based Prolog interpreter standing in for SWI-Prolog, a naive Datalog evaluator for round counts,
anti-unification, brute-force version spaces, FOIL, bottom-clause saturation); textbook ground truth (Pelletier's
problems and AIMA's examples for first-order proving; Quinlan's can-reach, a Michalski-style trains task, the daughter
task, and Metagol's and Popper's family tasks for ILP, each checked with the reference interpreter). NLTK's resolution
prover is not used: it lacks factoring and misreads single-letter names.

Every solver, prover and learner is an `Algorithm` (resolution steps in Byrd's ports, Martelli-Montanari rules,
DPLL/CDCL decisions, propagations, conflicts and learned clauses, WalkSAT flips, Datalog rounds, guesses of answer sets,
given clauses, FOIL literals, Progol candidates, Metagol programs, tested hypotheses) with step, conflict, depth or
size budgets and a clear status on unsatisfiable or infinite inputs.
- `terms.ts`: atoms, variables, compounds, lists, numbers, groundness, term variables, equality, the standard order,
  printing (operator-aware `writeq`, `write_canonical`) that reads back.
- `parse.ts`: Prolog syntax for clauses and queries (operators and `op/3`, lists, curly terms, quoted atoms, character
  codes, comments), DCG translation.
- `unify.ts`: unification (iterative, optional occurs check), substitutions, composition, renaming apart, matching,
  variants, a trailing binding store, Martelli-Montanari step traces.
- `resolution.ts`: SLD resolution with backtracking and depth limits (traced), built-ins, negation as failure, a small
  standard library, SLD trees as structures.
- `datalog.ts`: naive and semi-naive bottom-up evaluation, least Herbrand models, stratified negation, derivation trees.
- `propositional.ts`: formulas, evaluation, truth tables, models, entailment, CNF (Tseitin) and DNF, simplification,
  prime implicants and exact minimal DNF and CNF.
- `sat.ts`: DPLL, CDCL (clause learning, watched literals, traced decisions and conflicts), WalkSAT, DIMACS.
- `maxsat.ts`: weighted partial MaxSAT (branch and bound, traced), model counting and enumeration, minimal
  unsatisfiable cores (selector assumptions with deletion-based minimisation).
- `first-order.ts`: clausal form (Skolemisation), given-clause resolution proving, subsumption, proof checking.
- `answer-sets.ts`: stable models of small normal programs (guess and check), for comparison with Datalog semantics.
- `ilp/generalisation.ts`: theta-subsumption, least general generalisation and relative LGG, clause reduction.
- `ilp/version-spaces.ts`: Find-S, candidate elimination, boundaries and voting.
- `ilp/foil.ts`: FOIL (gain, refinement, coverage), traced.
- `ilp/progol.ts`: mode declarations, bottom clauses, inverse entailment.
- `ilp/meta-interpretive.ts`: Metagol-style learning with metarules; learning from failures (Popper-style).
- Internal: `_errors.ts` (ISO error terms), `_arith.ts` (arithmetic with SWI-Prolog's defaults), `_builtins.ts` (the
  built-in predicates), `_library.ts` (the standard library in Prolog), `_cdcl.ts` (the CDCL core shared by `sat` and
  `maxsat`), `_relations.ts` (fact stores and joins shared by Datalog and answer-set grounding). Test generators of
  terms, formulas and CNF are in `packages/engine/src/testing/logic.ts`.

### 4.26 `dynamics/`: differential equations, nonlinear dynamics, control and filtering (scipy.integrate, diffrax, python-control, filterpy)

oracle: scipy.integrate and scipy.signal, torchdiffeq (adjoint gradients), python-control, filterpy, ruptures, pysindy,
nolds; numpy for DMD and the PDE references. Tests never run Python.

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
- Private helpers: `ode/_vec.ts` (typed-array fast path that falls back to tensor ops on traced values), `ode/_step.ts`
  (step bookkeeping shared by the steppers), `ode/_newton.ts` (simplified Newton, Jacobians, the implicit-function
  correction that makes implicit steps differentiable), `ode/_radau.ts`, `ode/_bdf.ts`, `ode/_bvp.ts`,
  `ode/_dop853.ts` (scipy's coefficients), `ode/_sparse-lu.ts`, `pde/_linear.ts`, `control/_model.ts`.
- Notes: `odeSolver(method, options)` satisfies `nn/blocks/continuous.ts`'s `OdeSolver`; the `SdeSolver` belongs to
  `stochastic`. `unscentedKalmanFilter` reuses the propagated sigma points as filterpy does (`redraw` gives the
  additive-noise UKF, exact on linear models). `methods/rl/agents/control.ts` wraps `dlqr` and re-exports iLQR from
  `control/trajectory.ts`.
- TODO: move `ode/_sparse-lu.ts` into `numerics/sparse` as a general sparse direct solver.
- TODO: Kautsky-Nichols-Van Dooren iterations for complex pole pairs in `place` (they keep their initial vectors, so
  the poles are exact but the gain can differ from scipy's for complex targets).
- TODO: missing observations in `kalmanEm`, and multi-input multi-output ARX/ARMAX (both are single-output now).

### 4.27 `pgm/`: probabilistic graphical models (pgmpy, libDAI)

oracle: pgmpy, networkx, pyAgrum (loopy BP, influence diagrams), hmmlearn, sklearn-crfsuite and pytorch-crf (CRFs),
causal-learn (GES, FCI, DirectLiNGAM), numpy and scipy (Gaussian models), brute-force enumeration

A model is plain data. Every model type describes itself as a `Structure` (roles: latent, observed, factor,
deterministic, decision, utility; plates and slices), and inference algorithms emit `message` events, so the drawing of
a model can animate its inference. The `probprog` engines reuse these algorithms. Every discrete model (Bayes network,
Markov network, factor graph) is accepted by every discrete inference routine. `models` and `inference` are inlined
into `mdml-engine/pgm`; `learning` and `causal` are namespaces.
- `factor.ts`: discrete factors in log space (`factor`, `logFactor`, `unitFactor`, `DiscreteVariable`): product
  (`factorProduct`, `factorProductAll`), sum and max marginalisation (`marginalise`, `maxMarginalise`,
  `factorMarginal`), reduction by evidence (`reduceFactor`), normalisation, division (`factorDivide`, 0/0 = 0), powers,
  permutation, lookups, argmax, entropy, `allAssignments`.
- `gaussian-factor.ts`: canonical-form Gaussian factors: product, division, marginalisation (Schur complement),
  conditioning (`gaussianReduce`), conversion to and from moment form, log mass and log value, permutation, and the
  factor of a linear-Gaussian conditional.
- `graphs.ts`: DAGs (`dag`, `topologicalOrder`, `dagEdges`, `ancestors`, `descendants`), undirected graphs, moral
  graphs (`moralGraph`), interaction graphs of factors, connected components, graph separation. Topological orders,
  components and (in `triangulation.ts`) maximal cliques are `graph`'s algorithms on string-labelled vertices
  (private `_graph.ts` indexes the names in sorted order), with pgm's sorted, name-based outputs.
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
- `models/crf.ts`: conditional random fields: linear-chain CRFs with state, label-pair transition and
  observation-dependent bigram features (potentials as chains, scores, conditional log-likelihoods, marginals, Viterbi
  decoding, the penalised objective with its gradient, first-order training as a traced run), CRF++-form models and
  inputs from parsed CRF++ templates (`crfppCrf`, `crfppObservation`); general CRFs over arbitrary cliques with
  per-assignment feature tables (conditional factor graphs, objective with exact or loopy-BP marginals, training). The
  template languages and feature indexes live in `text/templates.ts` and `text/attribute-templates.ts`, the
  estimators' training algorithms in `methods/models/crf.ts`.
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

### 4.28 `metrics/`: evaluation metrics (scikit-learn, torchmetrics)

oracle: scikit-learn, torchmetrics, scipy, statsmodels, fairlearn, pingouin, krippendorff, properscoring, sacrebleu,
rouge-score, jiwer, pycocotools, torchvision, MONAI, scikit-image, sewar, prdc, pystoi, dtw-python

The files follow the encyclopedia's evaluation chapters, in its order. Every metric has a `MetricSpec` and returns a
`MetricResult` (a few return richer records: curves, reports, PRDC, permutations); per-sample metrics accept sample
weights where the reference does, and metrics over rows of samples (targets, predictions) can attach a confidence
interval (`withInterval`). Metrics that need a pretrained network (LPIPS, DISTS, FID, KID, CLIPScore, BERTScore,
Inception score) take its features, embeddings or outputs, and DNSMOS takes the networks as functions; mdml ships
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
- `contingency.ts`: `confusionMargins` (row and column totals, recall and miss rate per row, precision and false
  discovery rate per column, prevalence, accuracy) and `thresholdConfusion` (the 2 by 2 table of scores cut at a
  threshold), what a contingency table writes around its cells; `tableCost` and `thresholdCosts` (misclassification
  cost of a table and at every threshold); `analyseThresholds(yTrue, scores, { costs, posLabel, sampleWeight })`, the
  shared threshold layer: one sorted pass over every distinct threshold giving a `thresholdSweep` exposition record
  (`lib/exposition/thresholds.ts`, with `sweepIndex` and `sweepAt` to read any threshold back): TP/FP/TN/FN, TPR, FPR,
  precision, selection rate and accuracy at each, total cost and the cheapest when costs are given, ROC and PR points
  with their areas (golden against scikit-learn). Built.
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
- Fairness measures are their own domain, `lib/fairness/` (4.39).
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
  RASE, UIQI; images `[H, W]` or `[H, W, C]`, channels last; filtering by `signal/image` (uniform and Gaussian
  filters, ndimage correlation) and `numerics/convolution` (valid correlation).
- `generative.ts`: Inception score (from class probabilities), FID (`frechetDistance`) and KID
  (`kernelInceptionDistance`) from features, precision, recall, density and coverage
  (`precisionRecallDensityCoverage`), CLIPScore from embeddings (`clipScore`), Vendi score (any order).
- `text.ts`: WER, CER, MER, WIL and WIP (jiwer's alignment); BLEU (corpus and sentence, smoothing, the 13a tokeniser
  `tokenize13a`); chrF and chrF++ (`chrfScore`, `wordOrder: 2`); TER (Tercom shifts and normalisation flags); ROUGE-N,
  ROUGE-L and ROUGE-Lsum (rouge-score; `stem: true` is rouge-score's `use_stemmer`, nltk's Porter from `text/stem.ts`,
  or a stemmer can be supplied); SQuAD exact match and token F1; BERTScore from embeddings (idf weights, baseline
  rescaling); perplexity; distinct-n. The edit alignment, LCS and n-gram counts come from `text/similarity.ts` and
  `text/shingles.ts`; private `_ter.ts`.
- `audio.ts`: SNR, segmental SNR, SDR (BSS Eval with a distortion filter), SI-SDR and SI-SNR, permutation-invariant
  matching (`permutationInvariant`, Hungarian algorithm via `optim` `linearSumAssignment`), STOI and ESTOI (`shortTimeObjectiveIntelligibility`, pystoi with its
  resampling), log-spectral distance, mel-cepstral distortion from cepstra (optionally DTW-aligned), DNSMOS from a
  supplied model (torchmetrics' windows and calibration, 16 kHz); windows, FIR design, polyphase resampling and
  framed spectra come from `signal`. TODO: PESQ (ITU-T P.862), whose reference implementation is ITU-copyrighted C
  that cannot be ported under the licence.

### 4.29 `losses/`: differentiable objectives (torch.nn losses, optax losses)

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
  proposals). The Huffman tree comes from `information/source-coding.ts` (`huffmanTree`).
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

### 4.30 `rl/`: reinforcement learning infrastructure (gymnasium, PettingZoo, stable-baselines3 building blocks)

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
  game search with negamax backups; rollout planners, CEM over action sequences, MPPI). `gameMcts` stays separate from
  `search.mcts`: it searches environments with rewards on any move, and its PUCT counts a node's visits as AlphaZero
  does (the sum of its children's), where `search.mcts` also counts the expansion visit, which changes AlphaZero's
  training; `search.mcts` also labels every path for its trace.
- `models.ts`: learned-dynamics interfaces (deterministic and probabilistic ensembles) for model-based agents.
- `exploration.ts`: count-based bonuses, pseudo-counts, random network distillation and curiosity interfaces.
- `off-policy.ts`: IPS, SNIPS, clipped IPS, direct method, doubly robust, fitted Q evaluation.
- `bandits.ts`: bandit problem interface, regret, Lai-Robbins bounds, KL helpers.
- `evaluation.ts`: aggregate metrics across runs (interquartile mean, optimality gap, probability of improvement) with
  bootstrap intervals, learning-curve aggregation.
- `training.ts`: `trainAgent` (episodes or steps as a traced algorithm, checkpoints with doubling spacing, greedy
  evaluation), `replayEpisode` (rebuild any episode, including its rendered frames, from the nearest checkpoint).

### 4.31 `games/`: game theory (nashpy, OpenSpiel)

oracle: nashpy (support and vertex enumeration, Lemke-Howson from every label, fictitious play, replicator dynamics),
OpenSpiel (CFR, CFR+, discounted CFR, regret matching, projected replicator dynamics, best responses and exploitability
on Kuhn and Leduc poker, the sequence-form LP), scipy.optimize.linprog (zero-sum values, maxmin levels, correlated
equilibria), textbook cooperative games (Shapley values, the core, the nucleolus of bankruptcy games). Built on
`optim/linear-programming.ts` (every LP), `foundations/combinatorics.ts` and `numerics/linalg`. `methods/rl` takes
its zero-sum stage games (minimax-Q), matrix-game payoffs and Kuhn/Leduc rules from here.

`normal-form/` (inlined into `games`):
- `game.ts`: `NormalFormGame` (any number of players, payoffs row-major over profiles), `normalFormGame`,
  `bimatrixGame`, `payoffMatrices`, `profilePayoffs`, `expectedPayoffs`, `actionValues`, `bestResponses`,
  `deviationGains`, `nashConv`, `isNash`, `pureNashEquilibria`, `isConstantSum`, `uniformProfile`, `Equilibrium`,
  `GameRunOptions`.
- `classic.ts`: the textbook games: prisoner's dilemma, matching pennies, rock-paper-scissors, battle of the sexes,
  stag hunt, Aumann's chicken, hawk-dove, Shapley's game, n-player coordination, `randomGame`.
- `dominance.ts`: strict and weak dominance by pure or mixed strategies (mixed by LP), `dominatedActions`,
  `iteratedElimination`, `restrictGame`, `pureMaxmin`.
- `enumeration.ts`: `supportEnumeration`, `vertexEnumeration` (best-response polytopes, labels, degenerate games),
  `distinctEquilibria`.
- `lemke-howson.ts`: Lemke-Howson with lexicographic pivoting as a traced algorithm (each pivot a step),
  `lemkeHowsonEnumeration` (every initial label).
- `zero-sum.ts`: `solveZeroSum` (both LPs, value and duality gap; the shape `methods/rl` uses), `zeroSumEquilibrium`,
  `maximin` (security strategies in any game).
- `correlated.ts`: correlated and coarse correlated equilibria by LP (welfare, egalitarian, weighted or feasible),
  `correlatedGains` (swap and external regret), `isCorrelatedEquilibrium`, `jointDistribution`.

Learning in normal-form games (every routine is a traced algorithm with an iteration cap):
- `learning.ts`: fictitious play (simultaneous or alternating, nashpy's counts), stochastic fictitious play (logit
  responses), best-response dynamics (stops at a pure equilibrium or a detected cycle).
- `no-regret.ts`: regret matching (OpenSpiel's, with regret matching+ and linear averaging) and multiplicative weights
  (Hedge) in self-play; average strategies, average joint play, NashConv and external regret per iteration.
- `evolutionary.ts`: the replicator(-mutator) field and dynamics, two-population replicator dynamics (Runge-Kutta to
  given time points), projected replicator dynamics (OpenSpiel's, n players).

`extensive-form/` (inlined into `games`):
- `tree.ts`: `ExtensiveGame` (OpenSpiel-style states: current player, `CHANCE` or `TERMINAL`, legal actions, chance
  outcomes, information-state keys, returns), `treeGame` (games from literal trees), `gameTree` and `treeOf` (the
  materialised tree with information sets), `TabularPolicy`, `uniformPolicy`, `expectedReturns`, `backwardInduction`
  (subgame-perfect play of perfect-information games), `toNormalForm` (pure strategies of small games).
- `best-response.ts`: `bestResponse`, `extensiveNashConv`, `exploitability` (OpenSpiel's definitions).
- `sequence-form.ts`: sequences, realisation-plan constraints and payoff matrices; the two-player zero-sum solution
  by one LP (Koller, Megiddo and von Stengel), realisation plans back to behaviour policies.
- `cfr.ts`: counterfactual regret minimisation as a traced algorithm: vanilla CFR, CFR+, linear and discounted CFR
  (alternating or simultaneous updates, OpenSpiel's conventions); exploitability, average regret, regrets and the
  average policy per iteration.
- `mccfr.ts`: Monte Carlo CFR with external and outcome sampling, keys per iteration.
- `poker.ts`: Kuhn poker and Leduc poker (OpenSpiel's rules and information states), `kuhnEquilibrium` (the
  one-parameter family of equilibria).

Cooperative games and mechanisms:
- `cooperative.ts`: coalitional games (characteristic functions over bitmasks, `coalitionValue`, `membersOf`),
  weighted voting, glove, bankruptcy and unanimity games, Shapley values (exact, and by permutation sampling with
  standard errors for worths given as functions), the Shapley interaction index (`shapleyInteractionIndex`), Banzhaf
  indices (raw and normalised), superadditivity and convexity; `explain/shapley.ts` computes its exact Shapley
  and interaction values with them.
- `core.ts`: excesses, core membership, a core point or emptiness, the least core, the nucleolus and prenucleolus
  (sequential LPs).
- `auctions.ts`: first-price, second-price (Vickrey) and all-pay auctions with reserves, VCG with Clarke pivot payments
  and combinatorial (XOR bundle) auctions run by it, symmetric equilibrium bids and expected revenue with independent
  uniform values (revenue equivalence, optimal reserve).
- `voting.ts`: preference profiles, plurality, Borda, Copeland, instant runoff, the pairwise majority matrix,
  Condorcet winners and losers, the Smith set, and a report of the social choice paradoxes (Condorcet's paradox,
  Condorcet winners who lose or losers who win, spoilers); the Tennessee and Condorcet-paradox profiles.

### 4.32 `probprog/`: probabilistic programming and automatic inference (NumPyro, Pyro)

oracle: NumPyro (log densities, potentials, transforms, handler semantics, trace shapes, ELBOs at fixed guide draws,
Laplace modes and covariances by scipy on NumPyro's potential), pgmpy and brute-force sums for discrete programs,
scikit-learn (`GaussianMixture`, `FactorAnalysis`) for EM, ArviZ for PSIS-LOO and model comparison; closed-form
posteriors (conjugate, Gaussian, grids, brute force) within a few Monte Carlo standard errors for the stochastic
engines. Shared golden programs: `oracle/cases/probprog/_models.py` and `src/testing/probprog.ts`.

A model is a function of its data that calls the primitives below inside an effect-handler context. `analyse` traces it
once and describes it; every engine says whether it applies and at what cost; `infer` picks one, records the reason,
runs it as a traced algorithm and returns a `Posterior`. Adding an engine is adding a file to `engines/` (inlined into
`mdml-engine/probprog`). The handler stack (`_stack.ts`) is internal bookkeeping like the transform depth of
`autodiff`: pushed and popped by try/finally around one call, empty whenever no program runs.
- `primitives.ts`: `sample` (observed with `obs`, sample shapes, inference hints), `observe`, `deterministic`, `param`
  (with a support), `plate` (dimensions allocated as NumPyro, subsampling with scaling, param slicing), `factor`,
  `markov` (sequential dependence, with a history), the `Model` type, and the extension points `Message`, `Messenger`,
  `message`, `send` and `withMessenger`.
- `handlers.ts`: `trace` (and `traceOf`), `seed`, `condition`, `substitute`, `replay`, `block`, `scale`, `mask`,
  `doIntervention`, `enumerate` (mark discrete sites for exact summation), `uncondition` (for predictives) and `handle`
  (custom handlers); the recorded site types.
- `density.ts`: `logJoint` (enumerated sites summed out by plated sum-product contraction, `_contract.ts`, with
  dimension recycling in markov loops), `traceLogJoint`, `siteLogProbs`, `potentialEnergy` (to an unconstrained
  `LogDensity` with Jacobian corrections, supports re-evaluated per run, discrete sites summed out), `latentLayout`,
  initialisation strategies (`initToSample`, `initToMedian`, `initToMean`, `initToUniform`, `initToFeasible`,
  `initToValue`, `findInitialParams`), constrain and unconstrain maps (`constrainSites`, `unconstrainSites`,
  `constrainDraws`, `packLatents`).
- `analysis.ts`: `analyse(model, data)`: sites with supports, shapes and observed status; plates; the dependency graph
  (provenance tracking of tagged values, `_provenance.ts`; direct edges, seen through deterministic sites where
  needed); discrete cardinalities and the size of their joint space; conjugate prior-likelihood pairs (from
  `dists/conjugacy.ts`, with exponential families flagged by `dists/exponential-family.ts`); linear-Gaussian structure
  (affine means and constant scales, checked numerically); chain, tree or loopy moral graphs; markov loops; whether a
  likelihood exists (or only a simulator); differentiability.
- `graph.ts`: `toStructure` for programs (plates as nested groups, markov loops as repeats, roles, dependencies), so any
  program draws like a graphical model.
- `truth.ts`: `generatingProgram`, a dataset's generating program as `data`'s `Truth.program` (per-row densities with
  discrete latents summed out, class posteriors, E[y | x], draws).
- `engines/engine.ts`: the `InferenceEngine` contract (`name`, `applies(analysis)` returning feasibility, estimated cost,
  quality and a reason; `run`; the posterior form it produces), `QUALITY_RANK`.
- `engines/enumeration.ts`: exact inference for discrete programs grounded into `pgm` factor graphs (`groundDiscrete`):
  junction-tree marginals, evidence and exact joint draws, the MAP assignment by max-product variable elimination.
- `engines/conjugate.ts`: closed-form posteriors (`conjugatePosteriors`, `conjugateData`) and predictives
  (`conjugatePredictive`) for conjugate programs, with the evidence by the candidate's formula.
- `engines/gaussian.ts`: linear-Gaussian programs as one linear system (`linearGaussianForm`): exact Gaussian
  conditioning (`gaussianCondition`), Kalman filtering and RTS smoothing for chains (`kalmanSmoother`, exact joint
  draws by backward sampling), Gaussian belief propagation through `pgm`.
- `engines/chain.ts`: discrete-state chains (HMM-like programs, `chainForm`) by forward-backward, FFBS and Viterbi.
- `engines/em.ts`: EM and MAP-EM for programs whose latents are discrete (exact E-steps by differentiating the
  enumerated density) or linear-Gaussian and plated (Gaussian conditioning and sigma points: factor analysis);
  closed-form M-steps where a param is used directly or indexed by one enumerated site (normal locations and scales,
  Bernoulli and categorical probabilities, Poisson and exponential rates: mixtures, HMMs), L-BFGS M-steps otherwise
  (mixtures of experts), SQUAREM acceleration.
- `engines/gibbs.ts`: Gibbs sampling with discrete conditionals (enumeration), conjugate conditionals given the rest,
  mixture components group by group, slice steps otherwise, and collapsed Dirichlet and beta sites (topic models).
- `engines/nuts.ts`: NUTS and HMC on the unconstrained density (`mcmc`), with enumerated discrete sites summed out and
  recovered by `inferDiscrete`.
- `engines/svi.ts`: stochastic variational inference (`sviAlgorithm`, `svi`): automatic or custom guides, any ELBO
  estimator, optimisers on unconstrained parameters, subsampling over plates.
- `engines/laplace.ts`: Laplace approximation at the MAP estimate (`laplaceFit`), with the Laplace evidence.
- `engines/smc.ts`: sequential Monte Carlo: a bootstrap particle filter for programs with `markov` structure, and
  adaptive tempering for static programs, with evidence estimates.
- `engines/importance.ts`: likelihood weighting and importance sampling with a guide, marginal-likelihood estimates,
  ESS and Pareto k.
- `engines/abc.ts`: approximate Bayesian computation for simulator programs without likelihoods (`simulator`, rejection
  ABC, ABC-SMC, summary statistics).
- `infer.ts`: `defaultEngines`, `plan(model, data)` (ranked engines with reasons: quality first, then cost, within a
  budget), `infer(model, data, { engine: 'auto' | name | engine, budget })` with fallback when an engine fails,
  `explainPlan`, `withDiagnostics`.
- `guides.ts`: automatic guides (`autoDelta`, `autoNormal`, `autoMultivariateNormal`, `autoLowRankMultivariateNormal`,
  `autoLaplace`, and the normalising-flow guide `autoIafNormal`), `guideSites`.
- `elbo.ts`: ELBO estimators (`traceElbo`: reparameterised with score-function terms and baselines;
  `traceMeanFieldElbo`: analytic KL; `renyiElbo`; `importanceWeightedElbo`), `particle`, `elboAt` (fixed draws),
  `renyiBound`.
- `posterior.ts`: `Posterior` (exact distributions, variational approximations, point estimates, equally weighted or
  weighted draws behind one interface: `marginal(site)`, `sample`, `mean`, `variance`, `quantiles`, `logProb` where
  available, `summary` with ESS and R-hat for chains), `definePosterior`, `samplePosterior`, `weightedPosterior`,
  `pointPosterior`.
- `predictive.ts`: prior and posterior predictive draws (at new data too), per-observation log-likelihoods
  (`logLikelihood`, `pointwiseLogLikelihood`).
- `criticism.ts`: WAIC, PSIS-LOO (with Pareto k diagnostics), posterior predictive checks, model comparison
  (`compareModels`) and stacking and pseudo-BMA weights.

TODO: Gaussian belief propagation for programs with correlated (multivariate normal) sites, which use exact
conditioning meanwhile; EM with discrete and Gaussian E-steps together (mixtures of factor analysers); Pyro as a second
oracle once the oracle environment has it.

### 4.33 `learn/`: the toolkit around models (scikit-learn's non-model modules)

- `bases/` (inlined; `docs/methods_port/bases.md`): the bases methods are built on. `iterative.ts`: the iterative base
  (`iterative(particular)`: a library `Algorithm` over the data run as a session behind scikit-learn's `fit`, with the
  solver's channels as `solver/...` paths, `fit/iteration/end`, `model/params`, restarts, the records' rows, and
  `method(data)` and `session(data)` to drive the same fit live); `kinds.ts`: the estimator kinds as add-ons
  (`kindProbes`: the standard derived paths of classifiers and regressors, clusterers, transformers and outlier
  detectors). The one-shot, trainer, agent, sweep and procedure bases follow.

oracle: scikit-learn, statsmodels, imbalanced-learn, mapie for conformal, captum and shap for explanations, lime,
modAL for active learning, diffprivlib and opacus for privacy
- `estimator.ts`: the estimator contract (`name`, `hyper`, `hyperSpace`, `fit`, optional `fitTransform` for cross-fitted
  outputs), `defineEstimator`, `Fitted` (parameters as data, verbs as methods, `revive` after deserialisation,
  `serialiseFitted`), capability guards (`can`, `requireVerb`, `isClassifier`, `isEstimator`), `cloneEstimator`,
  `EstimatorClass` (the base of sklearn-style class wrappers).
- `validation.ts`: input checks with clear shape errors, sample weights, `asFloatMatrix`, matrix views, label
  encoding, `takeData`.
- `scoring.ts`: scorers (`makeScorer`, `getScorer` by sklearn name with the right response method, losses negated),
  the shared `scoring` option of model selection, inspection and feature selection.
- `preprocessing/scalers.ts`: standard, min-max, max-abs and robust scalers, normaliser, quantile and power
  transformers.
- `preprocessing/encoders.ts`: one-hot, ordinal, label, target (cross-fitted), label and multi-label binarisers.
- `preprocessing/features.ts`: polynomial features, spline features, binning, binariser, function transformers,
  interaction features.
- `preprocessing/imputation.ts`: simple, k-NN and iterative imputers, missing indicators.
- `preprocessing/resampling.ts`: random over- and under-sampling, SMOTE, borderline SMOTE, ADASYN, Tomek links,
  NearMiss (resamplers apply only in fit, also inside pipelines).
- `feature-selection.ts`: variance threshold, univariate scores (F tests, chi-square, mutual information) and
  selection (k best, percentile, FPR/FDR/FWE), selection from model importances.
- `feature-selection-wrappers.ts`: RFE and RFECV, sequential selection, stability selection (traced).
- `compose.ts`: `pipeline`, `featureUnion`, `columnTransformer`, `transformedTargetRegressor`, step inspection
  (intermediate outputs recorded by policy), caching of fitted steps.
- `model-selection/splitters.ts`: the splitter contract; holdout and train/validation/test splits; k-fold,
  stratified, group, stratified group, shuffle, stratified shuffle, group shuffle, leave-one-out, leave-p-out,
  leave-one-group-out and leave-p-groups-out, repeated variants; predefined splits.
- `model-selection/blocked-splitters.ts`: time-series splits (expanding window, sliding window, blocked, purged and
  embargoed); spatial and cluster blocks; index-pair adapters.
- `model-selection/cross-validation.ts`: `crossValidate` (traced, one child run per fold), `crossValPredict`, learning
  and validation curves, permutation-test scores.
- `model-selection/search.ts`: parameter grids and samplers, `gridSearchCV` and `randomizedSearchCV` (traced, one
  child run per candidate), nested cross-validation. The general studies (samplers, pruners, multi-fidelity) are
  `lib/tune`; these are the sklearn-shaped searches it can generalise.
- `model-selection/criteria.ts`: AIC, AICc, BIC, Mallows' Cp, GCV, leave-one-out shortcuts for linear smoothers, the
  one-standard-error rule, the .632 and .632+ bootstrap.
- `calibration.ts`: Platt scaling, isotonic calibration (pool adjacent violators, the home that
  `methods/linear/isotonic.ts` reuses), temperature and vector scaling, beta calibration, histogram binning,
  cross-fitted calibrated classifiers.
- `conformal.ts`: split conformal regression, conformalised quantile regression, conformal classification (LAC, APS,
  RAPS with tuning), Mondrian conformal, jackknife+ and CV+, conformal for time series (adaptive conformal inference).
- Explanation, inspection and data valuation are their own domain, `lib/explain/` (4.38).
- `active-learning.ts`: the pool-based and stream-based active-learning loop as a traced algorithm (query, label,
  refit), query-strategy contract (uncertainty, margin, entropy, random, ranked batch, expected error reduction, query
  by committee), budget handling.
- `privacy.ts`: Laplace, Gaussian (classic and analytic) and exponential mechanisms, randomised response, composition,
  RDP accounting for the subsampled Gaussian.

### 4.34 `nn/`: neural networks (torch.nn, torch.nn.functional, Flax, Haiku)

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

### 4.35 `tune/`: parameter selection for anything (Optuna, Ray Tune ideas)

oracle: optuna (TPE Parzen densities and splits, QMC sampler orders, every pruner's decisions on fixed learning curves,
Hyperband brackets, Pareto fronts and hypervolume, PED-ANOVA importance, GP posterior and log EI), scikit-learn
(`ParameterGrid` order, `GridSearchCV` results, the halving searches' schedules, GP
posteriors with fixed kernels) and scipy (acquisition closed forms); stochastic samplers are checked by optimisation
performance on Branin and Hartmann functions within budgets, with explicit keys

A study runs trials over a `Space` against an `Objective`. Objectives can be cross-validated estimators, training runs
with epochs as fidelity, agents in environments, probabilistic programs (evidence, ELBO, LOO) or any function. The same
samplers, pruners and schedules work for all of them. A study is a traced algorithm whose steps are trials, each a
child run. Samplers and pruners are pure functions of the study (what they remember lives on the trials), so a study
resumed at trial t draws exactly what an uninterrupted run would.
- `study.ts`: `Study`, `Trial` (params, values, intermediate values, state, source, budget, bracket and rung),
  `createStudy`, `bestTrial`, `bestTrials` (Pareto front), `studyHypervolume`, `exportTrials` (trials as rows),
  `studyToJson` and `studyFromJson` (store and resume).
- `tuning.ts`: `studyAlgorithm` (one trial per step; trial child runs record reports and pruning decisions; stops on the
  trial or resource budget, an exhausted sampler or schedule, repeated failures, a constant objective or lack of
  improvement) and `tune(objective, space, options)` (enqueued configurations, resuming a study).
- `objectives.ts`: the `Objective` and `TrialContext` contracts (`report` returns whether to stop), `objectiveBase`,
  `functionObjective`, `multiObjective`, `trainingObjective` (a traced algorithm as a child run, reporting per step for
  pruning, with the schedule's budget as its steps), `agentObjective`, `programObjective`. TODO(probprog):
  `programObjective` takes a fit function returning the criteria until `probprog` lands; then a model and data.
- `estimators.ts`: estimator objectives on `learn`'s contract: `estimatorFor` (`estimator.with(params)` or a factory),
  `estimatorObjective` (a `learn` splitter or `cv` count, scored by a metric, a `learn` scorer or the estimator's own
  score; the running mean reported per fold; each fit a child run of the trial) and `holdoutObjective`.
- `samplers.ts`: the `Sampler` contract; grid (scikit-learn's order), random, Sobol (plain or scrambled), Halton and
  Latin hypercube samplers; CMA-ES (IPOP restarts) and Nelder-Mead samplers from `optim`, asked and told by replaying the
  optimiser (`_replay.ts`); an evolutionary sampler (NSGA-II for several objectives) from the operators of `search`.
- `parzen.ts`: Optuna's Parzen estimators (bandwidths, prior, weights; truncated normals, stepped and log leaves,
  options), their densities and draws, gamma and weight functions, the TPE acquisition log l(x) - log g(x).
- `tpe.ts`: the tree-structured Parzen estimator sampler as Optuna's: the good/bad split (with pruned trials, and for
  several objectives non-dominated ranks with hypervolume subset selection), multivariate and independent modes.
- `surrogate.ts`: the Gaussian-process surrogate of Bayesian optimisation (Matern 5/2 with ARD and Hamming distance
  for options, MAP hyperparameters with Optuna's priors; from `kernels`, `stochastic` and `optim`), `gpPosterior`.
- `bayesian.ts`: acquisitions (EI, log EI, PI, log PI, confidence bounds), acquisition maximisation (Sobol candidates
  refined by L-BFGS-B) and `gpSampler`. TODO(tune): multi-objective acquisition (log expected hypervolume improvement).
- `pruners.ts`: median, percentile, threshold, patient, successive-halving (ASHA) and Hyperband pruners, and the no-op
  pruner, decision for decision as Optuna's.
- `schedules.ts`: multi-fidelity schedules as data (successive halving, Hyperband brackets, ASHA rungs),
  `nextAssignment` (samples and promotions that a study runs, with budgets), and `halvingPlan` (scikit-learn's halving
  iterations).
- `wrappers.ts`: familiar entry points built on studies: `gridSearchCv`, `randomizedSearchCv`, `halvingGridSearchCv`,
  `halvingRandomSearchCv`. Each is a `learn` estimator (fitted values revive and carry the best model's verbs), so
  searches nest inside cross-validation; grids are `learn`'s `parameterGrid`. (`learn/model-selection/search.ts` keeps
  the candidate-list `gridSearchCV` and `randomizedSearchCV`; these generalise them with samplers, pruners and
  multi-fidelity schedules.)
- `analysis.ts`: trial tables (`TableData`), optimisation history, intermediate values, parallel-coordinate, slice and
  contour data (a GP surrogate over two parameters), Pareto fronts, hypervolume history and the empirical distribution.
- `importance.ts`: parameter importance: correlation based, permutation based (on a GP surrogate), functional ANOVA of a
  GP surrogate, and PED-ANOVA as Optuna's evaluator. TODO(methods/trees): fANOVA on random forests as Optuna's default
  evaluator, once regression forests exist.
### 4.36 `worlds/`: discrete worlds shared by search, reinforcement learning and graphics

One representation of a map, used by every application of it: a maze solved by breadth-first search, A* or a learned
policy, a grid world in reinforcement learning, and the level a ray caster walks through are the same `World`. Movement
can be discrete (cells and actions) or continuous (positions and headings over the same cells); each application takes
the world and adds its own view (search problems in `search`, environments in `methods/rl`, renderers in `graphics`
and mdml-render).

oracle: none needed for most of it (exact constructions); networkx for connectivity, articulation points, bridges and
shortest paths on the cell graph and for spanning-tree counts (the matrix-tree theorem behind the uniformity tests of
Wilson's and the Aldous-Broder algorithms); gymnasium for the FrozenLake, CliffWalking and Taxi transition tables;
scipy.ndimage for distance transforms; shapely for continuous distances and sight. MiniGrid is not an oracle
dependency, so its layouts are checked structurally (TODO: golden-test them against minigrid if it joins `oracle-rl`).
- `world.ts`: `World` (cells on a square or hexagonal grid, or nodes of a graph; walls between cells or blocking cells;
  per-cell terrain with costs; items, doors, keys, goals, hazards; start and goal sets; landmarks), `world`,
  `setWalls`, `setTerrain`, cell and coordinate conversions (`cellIndex`, `cellCoords`, `cellCentre`, `cellAtPoint`,
  `cellCorners`, `worldSize`), moves and neighbours under 4- and 8-connectivity (`destination`, `exits`, `neighbours`,
  with corner rules and one-way terrain), passability, and `wallsToBlocks` (thin walls to blocking cells).
- `ascii.ts`: `fromAscii` and `toAscii` (the usual ASCII map conventions: one character per cell, or thin walls drawn
  between cells; custom legends and terrains; square or hexagonal rows), `fromRows`.
- `drawing.ts`: `toStructure` (the cell graph), `toScene` (a top-down drawing with theme tokens: terrain, walls, items,
  starts and goals, per-cell values, a path, the agent as a cell or a pose).
- `rules.ts`: what entering a cell does (cost, reward, terminate, teleport, slip probabilities, wind, one-way cells,
  pickups, doors opened by keys), action sets (moves, king's moves, hexagonal moves, turning), composed rule sets and
  `applyRules`.
- `models.ts`: deterministic and stochastic transition models over cells (states are cell, heading and held items),
  with sampling, so the same world feeds tabular MDPs (gymnasium-style outcome tables), search problems (a
  `SearchProblem`-shaped deterministic model) and simulators.
- `generators.ts`: keyed maze generators as traced algorithms (recursive backtracker, Prim, Kruskal, Wilson, Eller,
  Aldous-Broder, binary tree, sidewinder, recursive division; the first five on any topology), `generateMaze`,
  braiding (removing dead ends), `isPerfectMaze`.
- `levels.ts`: rooms and corridors (BSP dungeons), cellular-automaton caves (traced), noise (value and Perlin, with
  octaves) and noise terrain.
- `textbook.ts`: textbook worlds with their rules (FrozenLake, CliffWalking, four rooms, the AIMA grid, the windy grid
  world, a key maze, Taxi's map and moves) and MiniGrid-style layouts (empty rooms, door-key, four rooms, lava gap,
  river crossings).
- `continuous.ts`: continuous agents over a world: positions and headings, motion with collision against walls (swept
  circles with sliding, swept segments for point agents), ray casts, line-of-sight queries, field-of-view cones,
  distance fields from walls (exact segment distances and the Euclidean, taxicab and chessboard transforms). The
  segment predicates come from `numerics/geometry/segments.ts`; private `_geometry.ts` keeps only the bucket index
  over wall segments.
- `analysis.ts`: reachability, connected regions, shortest-path distances (as a field over cells, unweighted or by
  terrain cost) and paths, dead ends, choke points and bridges, cell distances and heuristics, and conversion to
  `graph` adjacency for graph algorithms (worlds does not import `graph`, so it returns edges `graph()` accepts).

### 4.37 `graphics/`: rendering mathematics

The mathematics of drawing worlds and scenes, as data (float images that `imageToPixels` turns into `scene` `Pixels`,
scene nodes, and the steps behind them: DDA crossings, ray trees, path vertices, per-triangle records), so figures and
the slide and blog tools can show how a renderer works step by step. Everything is deterministic (randomness only from
keys) and traced where a step is worth showing: column ray casting steps per column, Whitted tracing per generation of
rays, path tracing per sample. Every loop is capped (DDA steps, ray depth and weight, bounces). Shading terms,
intersection distances, projection and tone mapping are tensor compositions, vectorised and differentiable (checked
by `checkPrimitive`); renderers run per-ray loops on numbers and shade hits in batches through the same terms. It
builds on `scene` (`Vec3`, `Pixels`, `toPixels`, the orbit `Camera`, paints), `worlds` (maps, `Pose`, `toScene`, lattice
noise), `foundations` (vector products, homogeneous coordinates) and `numerics/linalg/blas.ts`.

oracle: closed forms (intersection roots and normals by numpy, the exact Fresnel equations, GGX, Smith through its
Lambda form), numpy for the camera matrices, unprojected pixel rays and frustum planes from the view volume's corners,
scipy.integrate.dblquad for directional albedos, scipy.ndimage.map_coordinates for texture sampling, and 16 x 12
reference renders by independent straightforward implementations (a brute-force Whitted tracer, a ray caster that
intersects every wall face, a per-pixel rasteriser solving for barycentrics). The path tracer is checked by the white
furnace test (exact for a convex object), agreement with Whitted where there is no indirect light, and convergence to
stratified area-light estimates.
- `camera.ts`: pinhole and orthographic cameras (`pinholeCamera`, `orthographicCamera`, `fromSceneCamera` for the
  orbit camera of 3-D scene layers, `cameraBasis`), view and projection matrices (`lookAt`, `perspective`,
  `orthographic`, `viewMatrix`, `projectionMatrix`, `viewProjection`, `transformPoint`), projection (`projectToPixel`,
  differentiable `projectPoints`), ray generation per pixel (`pixelRay`, `cameraRays`, jittered by a key), and frustum
  tests (`frustumPlanes`, `cameraFrustum`, `pointInFrustum`, `sphereInFrustum`, `boxInFrustum`).
- `raycasting.ts`: grid ray casting over a `World` (`ddaCast`: the Amanatides-Woo DDA with wall hits by side, face,
  distance and texture column, stopped by opaque cells, thin walls and the map edge; `fisheyeCorrect`), column casting
  as a traced algorithm (`columnAngles`, `raycastAlgorithm`, `castColumns`, with a `fisheye` switch for comparison),
  first-person frames with textured walls, floors and ceilings, fog and sprites with depth sorting against the
  column depth buffer (`renderRaycast`), and the traced rays of a top-down view-cone figure (`viewConeRays`,
  `viewConeScene`).
- `intersect.ts`: shapes (`sphere`, `plane`, `triangle` with vertex normals and texture coordinates, `box`),
  ray-primitive intersections (`intersectSphere`, `intersectPlane`, `intersectTriangle` by Moller-Trumbore,
  `intersectBox` and `slabs`, `intersect`, brute-force `nearestHit`, `bounds`) and their differentiable distances
  (`sphereDistance`, `planeDistance`, `triangleIntersection`, `boxDistances`).
- `bvh.ts`: bounding-volume hierarchies (`buildBvh` by the binned surface-area heuristic or median splits, iterative),
  nearest-hit and any-hit traversal with counts (`intersectBvh`, `occludedBvh`), and `bvhStructure` for drawing.
- `raytracing.ts`: scenes of objects with materials and albedo maps (`rayScene`, `sceneHit`, `sceneOccluded`,
  `albedoAt`), reflection and refraction (`refract` by Snell's law, `schlick` for dielectrics, `splitWeights`), and
  recursive (Whitted) ray tracing with shadow rays, mirrors and glass as a traced algorithm (`whittedAlgorithm`,
  `renderWhitted`, `traceRay` returning the ray tree).
- `pathtracing.ts`: a small path tracer (cosine-weighted hemisphere sampling with explicit keys: `cosineHemisphere`,
  `sampleCosineHemisphere`; next-event estimation of lights; Lambertian, mirror and Fresnel-chosen glass events) as a
  traced algorithm per sample (`pathTracerAlgorithm`, `pathTrace`), and single paths with their vertices (`tracePath`).
- `shading.ts`: Lambert, Phong and Blinn-Phong terms and their normalised BRDFs (`reflect`, `halfVector`, `lambert`,
  `phong`, `blinnPhong`, `lambertBrdf`, `phongBrdf`, `blinnPhongBrdf`, `blinnPhongDistribution`), physically based
  microfacet shading (`ggxDistribution`, `smithG1`, `smithG`, `fresnelSchlick`, `fresnelDielectric`, `cookTorrance`,
  `metallicRoughness`), and materials (`material`, `brdf`, `shade`).
- `lights.ts`: lights (`pointLight`, `directionalLight`, rectangular `areaLight` with stratified or keyed samples),
  `lightSamples`, and batched shading of surface points with shadow tests (`shadePoints`).
- `tonemapping.ts`: tone mapping and gamma (`exposure`, `reinhard` plain and extended, `acesFilm`, `gammaEncode`,
  `gammaDecode`, `linearToSrgb`, `srgbToLinear`, `luminance`).
- `textures.ts`: texture images (`FloatImage`, `floatImage`, `imageFromPixels`, `imageToPixels` with exposure, tone
  mapping and encoding), UV mapping (`sphereUv`, `planarUv`, `cylinderUv`, `cubeUv`), nearest, bilinear and mipmapped
  sampling (`wrapIndex`, `sampleNearest`, `sampleBilinear`, `mipmaps`, `mipLevel`, `sampleMipmap`), and procedural
  textures (`checkerboard`, `bricks`, `noiseTexture` from `worlds`' lattice noise, `bakeTexture`).
- `rasterise.ts`: triangle rasterisation (`edgeFunction`, `rasteriseTriangle` with the top-left rule on a subpixel
  grid and barycentric weights), clipping in homogeneous clip space (`clipPolygon`), back-face culling
  (`isBackFacing`), depth buffers (`depthBuffer`, `depthTest`), and the whole pipeline with perspective-correct
  interpolation, textures and deferred shading (`mesh`, `rasteriseMesh`): the counterpart of ray casting, for
  comparison figures.

### 4.38 `explain/`: explaining models and data (shap, captum, alibi, lime, interpret, quantus)

A complete domain, organised as the field is: attributions (local and global, gradient and perturbation),
example-based and contrastive explanations, concepts, glass-box models, data attribution, and the evaluation of
explanations. Every method works on a prediction function (or a differentiable model, for gradient methods) and data,
returns plain values, and is wrapped by `methods/explain` into exposition records. Round one built `shapley`,
`surrogates`, `gradients`, `counterfactuals`, `inspection` and `valuation`; round two completes the domain to the
coverage of shap, Captum, Alibi, LIME, InterpretML and Quantus together.

oracle: shap (exact, kernel, tree, linear, permutation, partition), captum (gradient, perturbation, layer and neuron
methods), alibi (anchors, CEM, counterfactuals, ALE, trust scores), lime, interpret (EBM), sklearn.inspection, quantus
(metrics on fixed attributions); goldens where cheap
- Attributions, perturbation-based (`perturbation.ts`): occlusion and feature ablation (with feature masks and
  baselines), RISE for images, Morris elementary effects, LIME (tabular, text tokens, image superpixels;
  `surrogates.ts`).
  Built: `featureAblation`, `occlusion` (captum's window count, edge windows clipped), `rise`, `morrisEffects`;
  `limeMasked` in `surrogates.ts` (lime's text and image explainers over on/off units). Goldens: captum
  FeatureAblation and Occlusion.
- Shapley family (`shapley.ts`): exact, kernel, sampling and permutation SHAP, tree SHAP (path-dependent and
  interventional) and interactions, linear SHAP (independent and correlated), partition SHAP and Owen values over
  feature hierarchies, conditional versus interventional value functions, SAGE (global importance), Shapley-based
  interaction indices.
  Built: `shapleyValues`, `shapleyInteractionValues`, `interventionalValue`, `conditionalValue` (Gaussian and
  empirical, Aas et al.), `kernelShap`, `samplingShapley`, `treeShap`, `treeShapInteractions`, `ensembleTreeShap`,
  `linearShap` (interventional, and correlation-dependent by the Gaussian conditional mean), `owenValues` (nested
  hierarchies), `hierarchyFromLinkage`, `correlationHierarchy`, `partitionShap`, `sage` (permutation-sampled or exact).
  Goldens: shap Kernel, Exact (independent masker, and partition masker = Owen values), Linear and Tree explainers.
  TODO: interventional Tree SHAP, Shapley-Taylor and Faith-SHAP interaction indices.
- Gradient-based (`gradients.ts`): saliency, input times gradient, integrated gradients (baselines, Riemann and
  Gauss-Legendre paths), SmoothGrad, VarGrad and SmoothGrad squared (noise tunnels), expected gradients and
  GradientSHAP, DeepLIFT and DeepLIFT SHAP, guided backpropagation and deconvolution, Grad-CAM and guided Grad-CAM,
  layer-wise relevance propagation (epsilon, gamma, alpha-beta rules).
  Built: `saliency`, `inputXGradient`, `integratedGradients` (`pathRule`), `smoothGrad`, `noiseTunnel` (SmoothGrad,
  SmoothGrad squared, VarGrad over any method), `expectedGradients`, `gradientShap`, `deepLift`, `guidedBackprop`,
  `deconvolution`, `gradCam` with `interpolateMap` (guided Grad-CAM is their product, in `methods/explain`), `lrp`
  (epsilon, gamma, alpha-beta). Goldens: captum Saliency, InputXGradient, IntegratedGradients, DeepLift,
  GuidedBackprop, Deconvolution, LRP (epsilon, gamma), LayerGradCam. TODO: DeepLIFT SHAP, LRP for convolutions.
- Layer and neuron attribution (`internal.ts`): layer conductance, internal influence, layer activation times
  gradient, neuron conductance and integrated gradients, attention rollout and attention flow for transformers.
  Built: `layerGradientXActivation`, `internalInfluence`, `layerConductance`, `neuronConductance`,
  `neuronIntegratedGradients`, `attentionRollout`, `attentionFlow` (max-flow through the attention graph). Goldens:
  captum LayerConductance, InternalInfluence, LayerGradientXActivation, NeuronConductance.
- Global effects and importance (`inspection.ts`): partial dependence and ICE (centred, derivative), ALE (first and
  second order), permutation, drop-column and conditional permutation importance, H-statistics and functional ANOVA,
  SHAP summaries (mean |value|, beeswarm data, dependence and interaction data).
  Built: `partialDependence` (ICE kept), `centredIce`, `iceDerivative`, `accumulatedLocalEffects`,
  `accumulatedLocalEffects2D` (ALEPlot's second order), `permutationImportance`, `dropColumnImportance`,
  `conditionalPermutationImportance` (Strobl's strata of correlated features), `hStatistic`, `functionalAnova`; SHAP
  summaries are `methods/explain`'s `explainDataset`. Goldens: sklearn PD/ICE and permutation importance, numpy ALE
  (both orders) and drop-column refits.
- Surrogates and rules (`surrogates.ts`, `rules.ts`): global surrogate trees and fidelity, anchors (precision and
  coverage), RuleFit-style rule ensembles, rule lists and Boolean rules (BRCG-style), linearity measure.
  Built: `treeRules`, `ruleMask`, `ruleFit` and `ruleFitPredict` (lasso from `optim`), `linearityMeasure` in
  `rules.ts`; `anchors`, `surrogateTree`, `fidelity`, `globalSurrogate`, `lime`, `limeMasked` in `surrogates.ts`.
  TODO: rule lists and Boolean rules (BRCG).
- Example-based (`examples.ts`): prototypes and criticisms (MMD-critic, ProtoSelect), k-nearest explanations, gradient
  similarity explanations, trust scores.
  Built: `mmdCritic` (greedy MMD prototypes, witness-function criticisms with a log-det spread), `nearestExamples`,
  `gradientSimilarity` (dot, cosine, alibi's grad-cos), `trustScores` (alibi's alpha-filtered class cores). TODO:
  ProtoSelect.
- Contrastive and counterfactual (`counterfactuals.ts`): Wachter, diverse (DiCE-style), FACE, CEM (pertinent positives
  and negatives), prototype-guided counterfactuals, growing spheres, MACE-style for tree ensembles; actionability,
  immutability and plausibility constraints; recourse (linear, cost, feasibility) and its robustness to model change.
  Built: `wachterCounterfactual`, `diverseCounterfactuals`, `faceCounterfactual`, `growingSpheres`,
  `pertinentNegative` and `pertinentPositive` (CEM by projected FISTA), `prototypeCounterfactual` (class prototypes
  from nearest neighbours, no encoder), `linearRecourse`, `actionViolations` and `projectActions` (immutable,
  monotone, bounded, integer), `recourseRobustness`. TODO: MACE for tree ensembles.
- Concepts (`concepts.ts`): concept activation vectors and TCAV scores with significance, probing classifiers,
  concept bottleneck analysis.
  Built: `linearProbe` (IRLS logistic probe), `conceptActivationVector`, `directionalDerivatives`, `tcav` (runs
  against random sets, random-vs-random null, Welch t test). TODO: concept bottleneck analysis.
- Glass-box models (`glassbox.ts`): explainable boosting (GA2M: shape functions and pairwise terms), scorecards from
  logistic models, sparse linear contributions; the shape-function data render draws.
  Built: `explainableBoosting` (cyclic boosting over quantile bins, FAST-ranked pairs, centred terms),
  `glassBoxContributions`, `glassBoxPredict`, `termImportances`, `scorecard` (points per bin, pdo scaling). Sparse
  linear contributions are `linearShap`. TODO: a golden against interpret (not in the oracle).
- Data attribution (`valuation.ts`): influence functions, TracIn, data Shapley, k-NN Shapley, Beta Shapley and data
  Banzhaf, representer points; detecting mislabelled and harmful training rows.
  Built: `influenceFunctions`, `tracIn`, `dataShapley`, `knnShapley`, `subsetUtility`, `betaShapley`,
  `dataBanzhaf` (maximum sample reuse), `representerValues`; the mislabelled-row study is in `methods/explain`.
- Modality helpers (`modalities.ts`): token, superpixel and time-window feature masks and baselines, so every
  attribution applies to text, images and series.
  Built: `tokenise`, `tokenPredictor`, `gridSuperpixels`, `slicSuperpixels`, `timeWindows`, `maskGroups`,
  `maskedPredictor` (a predictor over group presence, so Kernel SHAP, exact Shapley and LIME run over units),
  `expandGroups`.
- Evaluation of explanations (`evaluation.ts`, Quantus's categories): faithfulness (deletion and insertion curves,
  ROAR, faithfulness correlation, monotonicity), robustness (max- and average-sensitivity, local Lipschitz estimate),
  localisation (pointing game, relevance mass and rank against a ground-truth mask), complexity (sparseness, entropy),
  randomisation sanity checks (model parameter and label randomisation), axiomatic properties (completeness, input
  invariance, symmetry), agreement between explainers.
  Built: `deletionCurve` (deletion and insertion, with areas), `faithfulnessCorrelation`, `monotonicity`,
  `removeAndRetrain` (ROAR, and least-first), `sensitivity` (maximum and average), `localLipschitz`, `localisation`
  (pointing game, relevance mass and rank), `complexity` (Gini sparseness, entropy), `explanationSimilarity`
  (Spearman, Pearson, top-k, sign), `explainerAgreement`, `completenessGap`. TODO: input invariance and symmetry
  checks, Quantus goldens (not in the oracle).

### 4.39 `fairness/`: fairness and bias (fairlearn, AIF360, Aequitas)

A complete domain, organised as the field is: measuring unfairness (group, intersectional, individual, causal, for
classification, regression and ranking) with uncertainty, finding bias in data, mitigating it before, during and after
training, and selecting models under fairness constraints. Every function works on predictions, scores, labels and
protected attributes (any number of groups and attributes), returns plain values, and is wrapped by `methods/fairness`
into exposition records. Round one built `group-metrics` (and the audit in `methods/fairness`); round two completes
the domain to the coverage of fairlearn, AIF360 and Aequitas together.
Round one also built the first pass of `preprocessing` (`reweighingWeights`, `repairFeatures`), `in-processing`
(`exponentiatedGradientReduction`, `gridReduction`) and `post-processing` (`optimiseThresholds`,
`thresholdDecisions`, golden against fairlearn's ThresholdOptimizer); round two extends them.

oracle: fairlearn (metrics, MetricFrame, ExponentiatedGradient, GridSearch, ThresholdOptimizer), AIF360 (metrics,
Reweighing, DisparateImpactRemover, LFR, OptimPreproc, PrejudiceRemover, CalibratedEqOdds, RejectOptionClassification),
Aequitas (group bias); goldens where cheap
- Group metrics for classification (`group-metrics.ts`): selection, true- and false-positive, true- and false-negative,
  precision, negative predictive value, false discovery and omission rates per group (`groupRates`); demographic
  parity, equal opportunity, equalised odds, predictive parity, predictive equality, treatment equality
  (`treatmentEqualityDifference`, FN / FP), accuracy parity (`accuracyDifference`, `accuracyRatio`), conditional
  statistical parity given legitimate factors (`conditionalDemographicParityDifference`, golden against MetricFrame's
  control features), balance for each class (`balanceForPositiveClass`, `balanceForNegativeClass`), disparate impact
  (the four-fifths rule); differences and ratios between groups, to the overall rate or to any reference group
  (`method: 'to_reference'`); worst-group forms for any metric (`metricFrame`: by group, min, max, difference, ratio,
  worst group, gaps to a reference; golden against fairlearn's MetricFrame); calibration within groups
  (`calibrationByGroup`) and multicalibration over attributes and their intersections (`multicalibration`). Built.
- Uncertainty of metrics (`inference.ts`): bootstrap intervals for any fairness statistic (`bootstrapDisparity`) and
  for many at once from one set of resamples (`bootstrapStatistics`), permutation tests for gaps
  (`permutationTestDisparity`), Wilson intervals per group rate (`groupRateIntervals`), minimum detectable differences
  (`minimumDetectableDifference`, golden against statsmodels' NormalIndPower) and small-group warnings
  (`smallGroupWarnings`). The audit's `groupReport` carries `intervals` (bootstrap intervals of every rate and
  disparity, permutation p-values) and `warnings`, drawn as interval bars and an interval column. Built.
- Intersectional and subgroup fairness (`intersectional.ts`): metrics over combinations of attributes
  (`intersectionalGroups`, `subgroupRates`: every subgroup's table and rates, gaps between subgroups at least `minSize`
  large, the worst subgroup), rich-subgroup auditing (`richSubgroupAudit`, golden against GerryFair's Auditor in
  AIF360), generalised entropy and Theil indices with their between- and within-group parts
  (`generalisedEntropyIndex`, `theilIndex`, `entropyDecomposition`, golden against AIF360's ClassificationMetric).
  Built; record `subgroupReport` (`lib/exposition/subgroups.ts`, `subgroupGrid`), views `subgroupView` (heatmap over
  the attributes beside sorted subgroup bars) and the React `SubgroupView`.
- Individual fairness (`individual.ts`): consistency among nearest neighbours (`consistency`, golden against AIF360's),
  Lipschitz fairness under a task metric with every violating pair (`lipschitzFairness`), fairness through awareness as
  a linear programme (`fairnessThroughAwareness`). Built; method `individualFairnessAudit` (`methods/fairness`).
- Causal fairness (`causal.ts`): counterfactual fairness (`counterfactualFairness`) and total, natural direct and
  indirect effects (`pathSpecificEffects`) against a structural causal model written in `probprog`, by simulation with
  common random numbers (`seed` and `doIntervention` keep each individual's exogenous noise). Built (core; tests on
  linear SCMs whose effects are known exactly). TODO: path-specific effects along chosen paths beyond direct/indirect.
- Regression and scores (`regression.ts`): bounded group loss (`boundedGroupLoss`, golden against fairlearn's
  MetricFrame), statistical parity and equal opportunity in distribution (`distributionParity`,
  `equalOpportunityInDistribution`, Kolmogorov-Smirnov and Wasserstein, golden against scipy), mean-difference and
  calibration gaps (`regressionGaps`). Built (core).
- Rankings and recommendation (`ranking.ts`): exposure and attention parity with disparate treatment
  (`rankingExposure`, `positionWeight`), FA*IR (`fairMinimumTable`, golden against scipy's binomial quantiles;
  `fairTopK`), deterministic re-ranking to minimum shares (`detGreedyRerank`). Built (core). TODO: FA*IR's multiple-test
  adjustment of alpha.
- Bias in data (`data-bias.ts`): representation and sampling bias against population shares (`representation`),
  base-rate gaps in the labels (`baseRates`, golden against AIF360's BinaryLabelDatasetMetric), label bias against clean
  labels (`labelBias`), proxy detection by dependence (normalised mutual information) and predictability (per-feature
  and held-out combined AUC) of the protected attribute (`proxyScores`), bias amplification by a model
  (`biasAmplification`, Wang and Russakovsky's BiasAmp). Built; record `dataBias` (`lib/exposition/data-bias.ts`),
  views `dataBiasView` and the React `DataBiasView`.
- Pre-processing (`preprocessing.ts`): reweighing (golden against AIF360's Reweighing), massaging (`massageLabels`),
  preferential sampling (`preferentialSample`), disparate-impact removal with a repair level (`repairFeatures`, Feldman
  et al.'s geometric repair; AIF360's remover runs BlackBoxAuditing's combinatorial repairer, which moves values to
  other observed values, so no golden), optimised pre-processing of the labels (`optimisedLabelMapping`, Calmon et
  al.'s programme on the label alone, a linear programme), learning fair representations (`learnFairRepresentation`,
  `fairRepresentationMemberships`; random starts, so no golden against AIF360's LFR). Built.
- In-processing (`in-processing.ts`): reductions, ports of fairlearn's ExponentiatedGradient (with its duality-gap
  stopping, linear-programme mixtures and eta shrinking; golden against fairlearn's final mixture and decisions for
  demographic parity and equalised odds, with a deterministic weighted least-squares learner in both) and GridSearch
  (fairlearn's grid generator and tradeoff rule; golden against its errors, violations and choice). Moments:
  demographic parity, equalised odds, true- and false-positive rate parity (all four golden). TODO: error-rate parity
  and bounded group loss (the regression reduction).
  `fair-classifiers.ts`: the prejudice remover (`prejudiceRemover`), adversarial debiasing (`adversarialDebiasing`,
  Zhang et al.'s projected update), fairness-constrained logistic regression (`fairConstrainedLogistic`, Zafar et al.,
  by SLSQP), the meta fair classifier for the statistical rate (`metaFairClassifier`, Celis et al.), GerryFair
  (`gerryFairClassifier`, fictitious play against `richSubgroupAudit`; AIF360's `update_costs` charges every row the
  first member's indicator, a bug this does not copy, so no golden). Built.
- Post-processing (`post-processing.ts`): threshold optimisation per group (demographic parity, equal opportunity,
  equalised odds; randomised thresholds; golden against fairlearn's ThresholdOptimizer), calibrated equalised odds
  (`calibratedEqualisedOdds`, golden against AIF360's mixing rates for each cost), reject-option classification
  (`rejectOptionClassification`, golden against AIF360's threshold and margin for each metric). Built.
- Trade-offs and selection (`frontiers.ts`): accuracy–fairness Pareto frontiers (`paretoFrontier`), fairness-aware
  model selection (`selectFairModel`: within a violation budget, or priced), the frontier of one group's threshold
  (`thresholdFrontier`), the impossibility results on given scores (`impossibilityAt`). Built; record `paretoFront`
  (`lib/exposition/frontiers.ts`) and view `paretoFrontView`; `methods/fairness/studies.ts` is built on it.
- Dynamics (`dynamics.ts`): delayed impact of decisions (`outcomeCurve`, `delayedImpact` under maximum-utility,
  demographic-parity and equal-opportunity policies, Liu et al.), feedback loops over repeated rounds
  (`repeatedDecisions`), fairness under distribution shift (`shiftedDisparities` by importance weights,
  `labelShiftSweep` over a group's base rate). Built; method `delayedImpactStudy` (`curves` records).
  Round two's step-four domains reuse the `curves` and `coefficientTable` records; dedicated views (exposure in
  rankings, causal decompositions, dynamics of repeated decisions) are TODO while render work is paused.

Left in 4.39 after round two (one line each):
- TODO: error-rate parity moment for the reductions (fairlearn's ErrorRateParity).
- TODO: bounded group loss reduction for regression (fairlearn's BoundedGroupLoss with exponentiated gradient).
- TODO: FA*IR's multiple-test adjustment of alpha (`fairMinimumTable` uses the unadjusted test).
- TODO: path-specific effects along arbitrary chosen paths (`pathSpecificEffects` gives total, direct, indirect).
- TODO: dedicated records and views for exposure in rankings, causal decompositions and repeated decisions.
- TODO: thin methods for causal, regression and ranking fairness (the core is built).
- TODO: hub entries for the round-two mitigations, and the parked gallery-to-hub move (5.20).

### 4.40 `cluster/`: clustering algorithms (scikit-learn cluster)

Clustering as traced `Algorithm`s, moved out of the parked methods layer. `kmeans.ts` holds Lloyd's algorithm and
mini-batch k-means, the initial centres (k-means++, random rows or given) and the nearest-centre assignment. A state
keeps no past: `observe` exposes the metrics `inertia` and `moved` and the `state` tensors `centres` `[k, d]` and
`labels` `[n]`, declared in the schema, and the tape records the history. The other clusterers (bisecting, kernel,
density, hierarchical, spectral, ...) are still parked and arrive here as they are needed.

oracle: scikit-learn KMeans with fixed initial centres (`methods/cluster/kmeans` fixture, kept at its old path)
||||||| 9c6bb451
### 4.41 `linear/`: linear and generalised linear models (statsmodels, scikit-learn linear_model)

The algorithms behind linear models, moved from the parked methods layer. `design.ts` (private) builds the design
matrix from `LearnData` (intercept, centring, weights); `solvers.ts` holds least squares, ridge by SVD, IRLS for GLMs
and coordinate descent for the lasso and elastic net; `paths.ts` the alpha grid and the regularisation path over it
(with cross-validation); `model.ts` the linear model's parameters and prediction (`linearModel`); `inference.ts`
covariances, Wald tables, standard errors and intervals. The estimator factories (lasso, ridge, logistic, GLM, robust,
Bayesian, online, LARS, isotonic) stay parked until the new methods base. Goldens keep their fixtures under
`methods/linear/`.

## 5. Methods, exposition records, views and the hub

> **Status, 2026-10-26.** The methods layer is being rebuilt on the log contract and sessions (`docs/methods.md`,
> `docs/methods_port/`): one base per way of running, particulars of a few lines each, and a minimal hub that is the
> pattern rather than a catalogue. The methods described below, the `mdml-hub` package and the site's hub section are
> parked, untouched, under `_parked/` (same paths), and come back piece by piece: their algorithms move home to `lib/`
> (`docs/methods_port/inventory.md`), and the constructables already have (`lib/environments`, `lib/agents`,
> `lib/problems`, `lib/graphics/rendering.ts`). Sections 5.x below describe what is parked; the exposition records of
> `lib/exposition` and the views of render are unchanged.

Four layers, each usable without the ones after it:

| layer   | where                          | what it holds                                                                                                                                                  | draws?        |
| ------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| core    | `packages/engine/src/lib/`     | the primitives of section 4 (a library, as PyTorch or Pyro are)                                                                                                | no            |
| methods | `packages/engine/src/methods/` | named algorithms and estimators (lasso, trees, k-means, UMAP, agents), headless, built from the core; each returns typed exposition records beside its results | no            |
| views   | `packages/render/`             | reusable components, including one view per kind of exposition record                                                                                          | yes           |
| hub     | `packages/hub/` (`mdml-hub`)   | declarative entries (a method, the dataset collections it suits, its controls, which views to show) and the generated catalogue the hub app is built from      | composes only |

**Why four.** The core does not decide how it is used; the encyclopedia, a blog or a slide deck define their own uses.
Methods are worked algorithms anyone can call. Views make the exposition reusable: a lasso, an elastic net and a GAM's
smoothness sweep all return a `RegularisationPath`, which `render` draws one way everywhere. The hub never grows
bespoke capabilities: an entry only declares which views to arrange; a figure that does not exist yet becomes a new
render view, available to every consumer.

**Exposition records.** `lib/exposition/` (a core domain low in the import graph, types and small constructors only) defines the
record kinds methods return, each a plain, serialisable value with a `kind` tag:
- `RegularisationPath` (penalty grid, coefficients per penalty, active sets, duality gaps), `CoefficientTable`
  (estimates with errors and intervals), `CrossValidationCurve` (scores per parameter with standard-error bands);
- `TreeRecord` (per node: every candidate split evaluated with impurities, the chosen split, the training-row indices
  reaching the node, counts or values, growth order) and `PruningPath`;
- `EnsembleStages` (members, staged predictions, out-of-bag curves, per-round weights or residuals);
- `MarginRecord` (support vectors, dual coefficients, margins, SMO working-set steps, dual objective);
- `NeighbourRecord` (indices and distances per query) and `LikelihoodTable` (per-class, per-feature contributions);
- `PosteriorGrid` (means, variances and samples on a grid) and `HyperparameterSurface` (an objective over a
  hyperparameter grid with the optimiser's path);
- `ClusteringSteps` (assignments, centres and inertia per iteration; merges with heights; core, border and noise
  labels; reachability), `MixtureSteps` (responsibilities and log likelihood per EM step);
- `ComponentsRecord` (components, explained variance, scores, reconstruction error per rank), `EmbeddingSteps`
  (embedding per iteration, neighbour graph, calibrated bandwidths, stress);
- `AnomalyScores` (scores, thresholds, per-point evidence);
- `DecisionSurface` (predictions and probabilities on a grid over two features);
- `PipelineStages` (each stage's input and output data, fitted parameters and shape changes; per stage up to a few
  hundred output rows with their classes and per-column statistics, the parameters in full with the names of their
  entries, a projection's explained variance, and on the final estimator the whole pipeline's `DecisionSurface` over
  the raw inputs with its training loss per row and the loss it expects over the grid);
- `SearchTrials` (every trial of a grid, random or Bayesian search with values and pruning decisions) and
  `LearningCurve`;
- `EpisodeRecord` (per episode returns; per step state, action, reward and frame; value and policy snapshots).
- `BoardSteps` (a search, solver or simulation played on a grid or a graph: per step every cell's category, its text
  and a caption, with metrics per step; mazes and frontiers, game trees with pruned branches, CSP boards, itemset
  lattices, cellular automata) and `PopulationSteps` (generations of a population: fitness, positions over the
  objective's surface with the search distribution, or the best program's curve over its data);
- `ImageSteps` (small grey images per recorded step in named groups: a generative model's data, reconstructions and
  samples from fixed noise, with the objective per step);
- `StateSequence` (a hidden-state model's posteriors over time, the decoded path and the true states);
- `CurveSet` (`curves`: several series on shared axes, each with an optional band, a group sharing its colour and an
  optional right-hand axis; optimiser races and studies of schemes or strengths), `DiagramRecord` (`diagram`: a
  `Structure`, such as `describe(net)`, with optional overlays and captions to play over it), `ValueGrids` (feature
  maps, filters, hidden states as cells, with steps through training) and `ConfusionMatrix` (counts per true and
  predicted class, with steps through training; `confusionCounts` builds them).
  Records that index the training rows (clusterings, mixtures, margins, neighbours, anomaly scores, decision surfaces)
  carry those rows (`InputRows`, two coordinates each, for data up to a few thousand rows), so a view draws from the
  record alone; iterative trainers' decision surfaces carry their `steps` through training, which the views play.
  Methods return `{ ...result, exposition: Exposition[] }`; long-running ones also emit the records' steps as trace
  events, so views can play them.

**Methods.** The folders of 5.1 to 5.19 (named after the familiar libraries' groupings), each with its summary in its `index.ts`.
Methods are thin over the core, document themselves lightly (summary and parameters; examples optional) and follow the
exposition contract of 5.0. Every estimator comes as a factory (`pca({ nComponents: 2 })`) and as a class
(`new PCA({ nComponents: 2 })`), declares its `hyperSpace`, and offers `predictDistribution` where it predicts.

**Views.** `render` gains `views/` (headless: a record to chart options, scenes or structures) and
`react/views/` (`PathView`, `TreeView`, `PipelineView`, `EmbeddingPlayer`, `DecisionSurfaceView`, ... one per record
kind, each usable on its own in any page).

**The hub.** `packages/hub/src/`:
- `entry/entry.ts`: `HubEntry` (id, title, one-line summary, collection, tags, the method it runs, its `hyperSpace` with
  demo defaults, the dataset collections it suits, and a layout: which record kinds to show, with which views, in which
  slots, and optionally the dataset and knobs to start from) and `defineEntry`; `entry/run.ts`: `fitEstimator` and
  `fitEither` (an entry's controls and run from an estimator, or from a classifier and regressor pair chosen by the
  dataset's kind), the demo state and `runDemo`; `entry/slots.ts`: helpers for slot props made on the page (`rowColours`:
  an embedding's rows coloured by class or by banded regression target); `entry/url.ts`: a state in the URL;
  `entry/descriptions.ts`: a line per collection and folder of collections for their index pages.
- `collections/<collection>/<entry>.ts`: semi-organised folders, for example `sklearn/` (mirroring scikit-learn's modules,
  kebab-cased: `linear-model`, `isotonic`, `tree`, `ensemble`, `svm`, `neighbors`, `naive-bayes`,
  `discriminant-analysis`, `gaussian-process`, `cluster`, `mixture`, `decomposition`, `random-projection`, `manifold`,
  `covariance`, `semi-supervised`, `pipeline`, `model-selection`), `pyod/` (anomaly detectors outside scikit-learn,
  after PyOD), `neural/` (by `methods/neural` group: `foundations/` perceptrons, MLPs and backpropagation, `training/` optimiser,
  initialisation and regularisation studies, `vision/` convolutions, `sequence/` recurrence; later attention and
  autoencoders), `probabilistic/`
  (`density/` and the probabilistic models: `hmm/`, `topic/`, `conjugate/`, `rating/`, `latent-gaussian/`),
  `online/` (SGD, passive-aggressive and perceptron learners with their learning curves), `active-learning/` (query
  strategies by label budget), `rl/` (the playground), `timeseries/`, `tuning/`.
- `catalogue.ts`: generated by `make barrels` from every `defineEntry`, so the hub app imports new entries
  automatically.
- `lazy/index.ts` (`mdml-hub/lazy`): generated beside the catalogue, `entryModules`: each entry module as a lazy
  `import()` by its path under `collections/` (`sklearn/cluster/kmeans`), so a page or the worker loads the one entry
  it shows without the whole catalogue.
- `react/HubEntryView.tsx`: lays out an entry from its declaration (dataset picker, method controls, the declared
  views, the trace player) and re-runs it on every change, debounced, in a Web Worker (`react/_worker.ts`, by catalogue
  id, loading only the entry's module when the view is given it (`module`), else the catalogue when a run starts) or on
  the main thread for entries outside the catalogue, with a size limit (`maxRows`) for slow methods. A slot
  names a record kind and optionally a `mdml-render/react` component with props made on the page (an
  `EpisodePlayer`'s environment and policy); a run may report progress with the records so far, which the widget
  draws live. The only React in the package, and composition only.
  Entries are data: `make layers` lets the hub import the engine and render's public views, but no chart library, and
  no drawing of its own.

**Datasets live in the core.** The generators, vendored real datasets and themed dataset collections of 5.18 are the
core domain `lib/datasets/` (at the top of the core's import graph, so exact truths can come from any core domain, such as
`probprog` generating programs or `dynamics` changepoint posteriors); `lib/data` keeps the schemas, loaders, splits and
modifiers.

**Testing, by layer.** The core keeps the full standard (oracle goldens, property tests, `checkPrimitive`). Methods are
tested lightly: a smoke test per method that runs it on a small dataset and checks its exposition records' shapes,
and a golden against the familiar library only where it is cheap and matters (a lasso path, a tree's splits). The hub
is tested lightly: the catalogue builds and every entry's demo runs within its budget.

**Layering.** Methods import the core and other method areas, without cycles between areas; the core never imports methods; render
imports engine contracts (including `lib/exposition`), never algorithms; the hub imports engine and render.

### 5.0 First pass: scope, phases and the exposition contract (methods, with their views and hub entries)

The methods are a classical-AI toolkit in the spirit of scikit-learn, with a little more on the probabilistic side, and a
reinforcement-learning playground. They exist to drive small live demos and gallery entries, not as an encyclopedia: a
method wraps the core (it never re-implements what a core domain already has) and documents itself lightly, but it
always hands back the data an explanation needs.

**Exposition contract.** Every fitted method returns, beside its predictions, the intermediate quantities a figure or a
sentence of explanation would use, as plain data (and as trace events where they evolve), for example:
- linear models: coefficients, the whole regularisation path over a lambda grid (coefficients, active set, duality
  gap per lambda), cross-validation curves with their standard-error bands, solver iterations (IRLS, coordinate steps),
  residuals and leverage;
- trees: for every node, every candidate split evaluated (feature, threshold, impurity before and after, decrease), the
  chosen split, the indices of the training rows that reach it, its class counts or value, the order of growth, and the
  cost-complexity pruning path;
- ensembles: every member, staged predictions, out-of-bag curves, per-round sample weights (AdaBoost) and residuals
  (gradient boosting);
- SVMs: support vectors and their dual coefficients, margins, the decision function on a grid, the working-set trace
  of SMO and the dual objective;
- neighbours, naive Bayes, discriminants and GPs: neighbour indices and distances per query, per-class and per-feature
  likelihood contributions, posterior means, variances and samples on a grid, the marginal-likelihood surface over
  hyperparameters and the optimisation path across it;
- clustering, mixtures and decompositions: assignments, centres and inertia per iteration, merges with heights,
  core/border/noise labels and reachability, responsibilities and log likelihood per EM step, components, explained
  variance per component and reconstruction error per rank;
- manifolds and anomalies: the embedding per iteration (t-SNE, UMAP), stress, neighbour graphs and calibrated
  bandwidths, anomaly scores with their thresholds and per-point evidence (isolation path lengths, reachability);
- searches and agents: every trial of a hyperparameter search with its values and pruning decisions; every episode's
  return and every step's state, action, reward and frame for an agent, with value and policy snapshots.

**Documentation.** Methods carry a summary and parameter docs; `@example` blocks are optional (the core is where the
exposition of the mathematics lives).

**Dataset collections.** `lib/datasets/collections.ts` groups the generators and real datasets into themed lists
for pickers: `classification2d`, `regression1d`, `clustering2d`, `density2d`, `anomaly2d`, `manifold3d`,
`timeSeries`, `comparisons` (games and tournaments for rating systems), `textCorpora` and `environments` (the rl
environments by family), each entry with its name, its generator, the `Space` of its knobs with demo defaults, and
the methods it suits. mdml-render turns a collection
and a method's `hyperSpace` into a dataset picker and parameter controls (see `react/controls/`).

**Phases** (first pass; later phases follow the same contract):
1. Supervised: `linear/` (model, solvers, regression with paths and CV, classification, the GLM families, robust
   regression, Bayesian ridge and ARD), `trees/` (tree, criteria, growth, pruning), `ensemble/` (bagging, forests,
   AdaBoost, gradient and histogram boosting, stacking), `svm/` (SVC, SVR, linear, one-class SVM and SVDD),
   `neighbours/` (k-NN, centroids), `naive-bayes/`, `discriminant/`, `gp/` (regression and classification, dense and
   sparse inducing-point versions), `kernel-methods/kernel-ridge.ts`, `multiclass/reductions.ts`.
2. Unsupervised: `cluster/` (k-means family, medoids, hierarchical, DBSCAN/HDBSCAN/OPTICS/mean shift, spectral,
   affinity propagation and BIRCH), `mixture/`, density estimation (KDE with bandwidth selection, histograms; thin
   wrappers over `stats/density.ts` and the mixtures), `decomposition/` (PCA and its variants, probabilistic PCA and
   factor analysis, kernel PCA, ICA, NMF, truncated SVD, CCA/PLS, random projections), `manifold/` (MDS, Isomap, LLE,
   spectral, t-SNE, UMAP, SOM, quality measures), `anomaly/` (isolation forests, LOF, statistical, combination),
   `covariance/`.
3. Paradigms and probabilistic models: `semi-supervised/` (self-training, label propagation and spreading), online
   learning (the SGD and passive-aggressive models of `linear/`, mini-batch k-means, incremental PCA, with
   learning-curve studies), `active-learning/` studies over `learn`, and from `models/`: HMMs, the CRF estimator, LDA,
   conjugate models, hierarchical models, Bayesian regression programs, linear-Gaussian latent models.
4. Showcases: an rl playground study (`rl/playground.ts`: choose an environment, from classic control, toy text, grid
   worlds built on `worlds`, MinAtar, and an agent, tabular Q-learning and SARSA, DQN, REINFORCE and PPO, LQR for
   control, with defaults that learn visibly within a short budget, live frames and learning curves) and
   hyperparameter optimisation (`tuning/`: a study over a method and a dataset collection, with the analysis data of
   `tune`).
5. Later (deprioritised): causal inference (5.5), search-based discovery and classical AI (5.6), the remaining domain
   areas (5.7), scientific machine learning (5.8), generative models (5.10), neural network models (5.11), retrieval,
   matching and recommendation (5.12), information and coding (5.13), algorithms and data structures (5.14),
   foundations studies (5.15), optimisation and Monte Carlo studies (5.16), learning theory (5.17), further datasets
   (5.18), and the rest of 5.1 to 5.4 not named above (GAMs, survival, rule learners, ordinal and mixed models,
   multi-output, embeddings, weak supervision, transfer, federated learning, fairness, privacy, adversarial examples,
   ratings, IRT, lattices, topic-model extensions).

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
- `paths.ts`: the regularisation paths and cross-validation along them that `regression.ts` uses (`enetPath`,
  `lassoPath`, `ridgePath`, `cvPath`, the one-standard-error rule).
- `lars.ts`: least angle regression and lasso-LARS (`larsPath`, `lars`, `lassoLars`), traced one knot per step.
- `classification.ts`: `logisticRegression` (binary and multinomial, L1, L2 and elastic net), `probitRegression`,
  `ridgeClassifier`, `perceptron`.
- `online.ts`: online learning: scikit-learn's plain SGD loop (traced, one epoch per step; its losses, penalties with
  the cumulative L1 penalty, and schedules) behind `sgdClassifier`, `sgdRegressor`, `onlinePerceptron` and the
  passive-aggressive classifier and regressor (PA-I, PA-II), each with a learning curve of progressive and training
  error against samples seen.
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
- `gaussian.ts`: `gaussianMixture` (the familiar estimator; traced EM with closed-form M-steps for full, tied, diagonal
  and spherical covariances, densities from `dists.multivariateNormal`; the route through `probprog/engines/em.ts`
  with VI, Gibbs or NUTS by option is a TODO with `programs.ts`), BIC and AIC.
- `bayesian.ts`: Bayesian and Dirichlet-process mixtures (variational and collapsed Gibbs).
- `selection.ts`: choosing the number of components (BIC, ICL, held-out likelihood, stability).
- `density.ts`: density estimation: the Gaussian KDE of `stats/density.ts` as an estimator with cross-validated
  bandwidth factors, and histogram densities for one or two features (from `stats/histograms.ts`).

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
- `shrinkage.ts`: the empirical, shrunk, Ledoit-Wolf and OAS estimators of `stats/covariance.ts` as estimators
  (Mahalanobis distances, Gaussian log likelihood).
- `robust.ts`: minimum covariance determinant (fast MCD, traced).
- `graphical-lasso.ts`: graphical lasso (traced), cross-validated selection, precision-matrix graphs as structures.
### 5.3 Learning paradigms

`semi-supervised/` (sklearn.semi_supervised)
built from: `learn`, `graph`, `svm`, `optim`
- `self-training.ts`: self-training, co-training, tri-training.
- `graph-based.ts`: label propagation, label spreading, harmonic functions.
- `labels.ts`: the unlabelled marker (-1), stratified label hiding for demos, semi-supervised label codes.
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
- `strategies.ts`: the core's strategies by name (`queryStrategy`: uncertainty sampling by least confidence, margin
  and entropy; query by committee by vote entropy, consensus entropy and maximum disagreement; expected error
  reduction; ranked batch; random) and `subsampled` (a strategy over a random subset of the pool, for expected error
  reduction lite); later expected model change, BALD, core-sets (k-centre greedy), density-weighted and diversity
  strategies, batch-mode selection (BADGE).
- `stream.ts`: stream-based selective sampling.
- `studies.ts`: `activeLearningStudy`: a strategy on a labelled dataset from a few labels per class, with the learning
  curve by labels bought against random sampling, the labelled rows per round and the final decision surface.

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
built from: `learn`, `fairness`, `optim`
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
- `hmm.ts`: hidden Markov models (`categoricalHMM`, `gaussianHMM`; mixture emissions later): Baum-Welch over the pgm
  forward-backward (traced; through the EM engine on the program later), Viterbi, posterior states; the dishonest
  casino (`dishonestCasino`, `casinoStudy`).
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
- `rating.ts`: `rateGames` over a `RatingSystem` (each game predicted, then rated; races split into pairs): Elo (with
  the chess conventions), Glicko, TrueSkill for two players (the exact EP update, with draws), Bradley-Terry by MM;
  later Glicko-2, TrueSkill on its factor graph for teams and races, TrueSkill through time, Plackett-Luce, Massey and
  Colley ratings.
- `irt.ts`: item response theory (1PL, 2PL, 3PL), learner models, item information.
- `lattice.ts`: Ising and Potts lattices: exact on chains, belief propagation, Gibbs and Swendsen-Wang sampling,
  magnetisation curves.
- `hierarchical.ts`: hierarchical models (eight schools, radon, partial pooling) with centred and non-centred forms.
- `regression.ts`: Bayesian linear, logistic, Poisson and robust regression as programs.
- `bayesian-classifiers.ts`: Bayes point machines (EP), AdPredictor (online probit regression).
- `state-space.ts`: Bayesian structural time series (local level, local linear trend, seasonal, regression).
- `latent-gaussian.ts`: linear-Gaussian latent models: `linearGaussianStateSpace` (local level and local linear trend,
  noise variances by the Kalman EM of `dynamics`, smoothed and filtered levels); later factor analysis, PPCA and
  Bayesian PCA as programs.

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
- `itemsets.ts`: Apriori, FP-growth and Eclat; closed and maximal itemsets (so far `apriori`, played level by level
  over the candidate lattice, and `fpGrowth`, played as its FP-tree grows and each item is mined, both as
  `BoardSteps`; `TEXTBOOK_BASKETS`, `marketBaskets`; TODO Eclat, closed and maximal itemsets).
- `association-rules.ts`: association rules with support, confidence, lift, leverage and conviction
  (`associationRules`, the top rules' lifts as a `CoefficientTable`).
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
- `planning-domains.ts`: blocks world, logistics, gripper and other domains for `search/planning/` (so far the
  four-operator `blocksWorld` with `SUSSMAN`, `randomBlocks` and `towersOf`).
- `planning.ts`: `planBlocks`: forward STRIPS planning played twice as `BoardSteps` (the states the search expands,
  then the plan carried out), blocks drawn on a table under the gripper.
- `puzzles.ts`: 8-puzzle, n-queens, Sudoku, Towers of Hanoi, missionaries and cannibals, cryptarithms (so far
  `nQueens`, `sudoku` with `SUDOKUS`, map colouring of `AUSTRALIA` and `randomMap`s).
- `csp.ts`: CSPs played on their boards (`playCsp` with a `CspPicture`; `solveQueens`, `solveSudoku`,
  `solveMapColouring`): backtracking with ordering and inference, or min-conflicts, step by step as `BoardSteps`.
- `adversarial.ts`: small games (`explicitTree`, `textbookTree`, `randomTree`, `ticTacToe` from a position) and
  `playGameSearch`: minimax or alpha-beta over the whole tree, played node by node (values backed up, pruned subtrees,
  the principal variation) as a `BoardSteps` record on the tree's layout.
- `maps.ts`: search worlds: hand-drawn maps (open, detour, river, rooms) and generated mazes as blocked-cell worlds.
- `search.ts`: path search on grid worlds (`gridProblem`, `gridSearch`: breadth-first, depth-first, uniform-cost,
  greedy, A*, weighted A*, IDA*, bidirectional) played expansion by expansion as a `BoardSteps` record.

`evolutionary/` (DEAP ideas)
built from: `search/evolutionary/`, `symbolic`, `nn`, `rl`
- `genetic-algorithms.ts`: studies on OneMax, knapsack and travelling-salesman problems (`oneMax`, `knapsack`,
  `evolveBitStrings` with the population as `BoardSteps`; `evolveOnPlane`, a real-coded GA over an objective's
  contours as `PopulationSteps`; TODO the travelling salesman).
- `evolution-strategies.ts`: `evolutionStrategyOnPlane` ((mu, lambda) and (mu + lambda)) and `cmaesOnPlane` (the core's
  CMA-ES with its sampled offspring and search distribution) as `PopulationSteps`; `_plane.ts` holds the plane
  objectives (the core's test functions with their surface on a grid).
- `genetic-programming.ts`: symbolic regression by genetic programming over `symbolic` expressions
  (`symbolicRegression`, `treeToExpr`: the best program per generation simplified, with its curve over the data).
- `neuroevolution.ts`: NEAT-style evolution of small networks on XOR and cart-pole.
- `quality-diversity.ts`: MAP-Elites and novelty search studies.

`complex-systems/`
built from: `stochastic`, `graph`, `dynamics`, `scene`
- `cellular-automata.ts`: elementary automata (Wolfram rules), Game of Life, Langton's ant (so far
  `elementaryAutomaton` and `gameOfLife`, played as `BoardSteps`; TODO Langton's ant).
- `agent-based.ts`: Schelling segregation, boids, predator-prey and Sugarscape-style models (so far `schelling`,
  played round by round with unhappy agents marked; TODO the rest).
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
- `matrix-profile/` (inline; built from `lib/signal/sliding.ts` and `mass.ts`; oracle: stumpy):
  - `algorithms.ts`: `stump` (diagonal traversal with the O(1) covariance update; self- and AB-joins, top-k, left and
    right profiles; non-normalised AAMP p-norms), `stomp` (row order), `stamp` (anytime, random rows), `prescrump`
    and `scrump` (SCRIMP++: PreSCRIMP seeding, random diagonals), `stumpiInit`/`stumpiUpdate` (incremental, with or
    without egress); each iterative one also as an `Algorithm` with traced progress.
  - `multidimensional.ts`: `mstump` (k-dimensional profiles by sorted per-dimension distances, included dimensions,
    discords), `multiDistanceProfile`, `subspace`, `mdl` (MDL-based dimension selection).
  - `discovery.ts`: `match`, `motifs`, `discords` (top-k with exclusion zones), time-series chains (`atsc`, `allc`).
  - `segmentation.ts`: arc counts, the idealised arc curve (the exact expectation of STUMPY's random model), the
    corrected arc curve, regime extraction, `fluss`, and streaming `flossInit`/`flossUpdate`/`flossAlgorithm`.
  - `snippets.ts`: `mpdist`, `snippets` (MPdist-based), `ostinato` (consensus motifs).
  - `pan.ts`: pan matrix profiles over many window lengths (`stimp`, breadth-first window order, display transforms).
  - `studies.ts`: planted ground truth (`plantedSeries`), a discovery study (motif recall, discord and boundary
    recovery), and the anytime convergence of STAMP and SCRIMP++ against the exact profile.
    TODO: GPU and distributed variants (gpu_stump, stumped), `stimped`, non-normalised streaming (aampi) and
    multidimensional non-normalised profiles (maamp), mSTUMP's excluded dimensions beyond include/discords.

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
- `adapters.ts`: `LanguageModel` implementations: local mdml models, a deterministic mock for tests, and HTTP
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
  maps), windy grid world (king's moves, stochastic wind), mazes with keys and traps (`keyMaze`), four rooms;
  gymnasium's `P` tables as models. All but the key maze are `worlds`' textbook tasks (`frozenLakeWorld`,
  `cliffWalkingWorld`, `aimaGridWorld`, `windyGridWorld`, `fourRoomsWorld`) through `transitionModel` into
  `tabularEnv`, drawn by `worlds.toScene`; the maps (`FROZEN_LAKE_MAPS`, `AIMA_GRID`, `FOUR_ROOMS`, `KEY_MAZE`) are
  worlds'. The key maze keeps its own single has-a-key flag (`2 nCells` states for any map; `keyMazeWorld` tracks
  each key present).
- `classic-control.ts`: CartPole, Pendulum, MountainCar (discrete and continuous), Acrobot, matching gymnasium step for
  step.
- `toy-text.ts`: Blackjack (natural and Sutton-Barto rules), Taxi (rainy, fickle passenger; its map, walls, moves and
  stands are `worlds.taxiWorld`), NChain, RiverSwim.
- `minatar.ts`: MinAtar-style games on a 10 by 10 grid (Breakout, Asterix, Freeway, Seaquest, Space Invaders), with
  minimal and full six-action sets, sticky actions and difficulty ramping.
- `continuous-control.ts`: point mass, 2-D reacher, double integrator, planar quadrotor, cart double pendulum (analytic
  dynamics: exact zero-order-hold steps or RK4 on Lagrangian equations of motion). `doubleIntegratorMatrices` is
  `systems.c2dMatrices` (zero-order hold); the per-step damped-mass coefficients in `_mechanics.ts` stay closed-form
  scalars (one exponential per step, not a matrix exponential).
- `games.ts`: tic-tac-toe, connect four, nim, Kuhn and Leduc poker, as turn-based environments with legal actions
  (the poker rules are `games`' `kuhnPoker` and `leducPoker`, with the cards dealt at reset).
- `go.ts`: Go on small boards (5, 7, 9; captures, simple ko or positional superko, area scoring with komi).
- `multi-agent.ts`: matrix games (prisoner's dilemma, matching pennies, rock-paper-scissors, stag hunt, with `games`'
  payoffs), pursuit-evasion,
  cooperative navigation (MPE simple_spread).
- `procedural.ts`: procedurally generated mazes (backtracker, Prim, Kruskal, braiding) and key-door tasks with
  Procgen-style level seeds and train/test splits, for generalisation studies. Distances and dead ends are worlds'
  (`distanceField`, `deadEnds` on the character grid read as a world); the carving and braiding stay here, since
  worlds' `generateMaze` draws other random streams (every level would change) and thin-walled worlds, where these
  levels place keys and doors in the passages of a character grid.
- `models.ts`: differentiable dynamics models of the control environments (`ControlModel`, matching each environment's
  step), `linearize`, trajectory derivatives and rollouts, for LQR, MPC and model-based agents.

`rl/agents/` (each file exports agents implementing the `Agent` contract)
- `random.ts`: random and constant agents.
- `bandits.ts`: epsilon-greedy, UCB1, KL-UCB, Thompson sampling (Bernoulli and Gaussian), EXP3, LinUCB, linear Thompson
  sampling, gradient bandits.
- `tabular.ts`: Q-learning, SARSA, expected SARSA, n-step SARSA, SARSA(lambda), Monte Carlo control, TD(0) and
  TD(lambda), double Q-learning, Dyna-Q (and Dyna-Q+), prioritised sweeping.
- `planning.ts`: value and policy iteration agents (solving the model at once, or a sweep at a time between
  episodes: `sweepingPlannerAgent`, which puts dynamic programming on a learning curve), MCTS agents, AlphaZero-style self-play with a small network,
  minimax agents. Its `alphaBeta` (negamax with a transposition table, bound flags and rewards on any move, untraced
  for speed inside agents) is not `search.alphaBeta`, a traced node-by-node engine over terminal utilities without a
  table.
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
- `multi-agent.ts`: independent learners, minimax-Q (stage games solved exactly by `games.solveZeroSum`), self-play,
  neural fictitious self-play.
- `cooperative.ts`: value decomposition (VDN, QMIX), MADDPG.
- `hierarchical.ts`: the options framework, option-critic.
- `safe.ts`: constrained policy optimisation by Lagrangian PPO, shielding.
- `control.ts`: LQR, swing-up (energy shaping plus LQR; Pendulum, TODO: the cart double pendulum), iLQR, MPC (iLQR, CEM
  or MPPI) and CEM agents.

`rl/studies.ts`: learning-curve comparisons with aggregate metrics and intervals, sensitivity to hyperparameters,
exploration and generalisation studies.

`rl/playground.ts`: the playground of phase 4. `playgroundPairings` lists environment and agent pairings whose demo
defaults (a `hyperSpace` each, with the budget) learn visibly in about a second in a browser: Bernoulli bandits
(epsilon-greedy, UCB1, Thompson sampling; an episode is a block of pulls), cliff walking (Q-learning, SARSA, expected
SARSA), frozen lake (Q-learning, value and policy iteration a sweep at a time), CartPole (REINFORCE, A2C) and
MountainCar (SARSA(lambda) on a grid of the state; DQN needs too many steps there). `playground(options)` trains one
and returns the returns and lengths per episode with an `EpisodeRecord` (chosen episodes recorded step by step, the
last always; value and policy snapshots of tabular agents; frames beside the record on request);
`playgroundSession` is the same run as a steppable `Algorithm` (over `lib/rl`'s `trainingAlgorithm`) with the record
of any intermediate state, and `onEpisode` reports progress for live pages.
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

Each model has `init`, `apply`, `describe` (its structure, drawn by mdml-render), a traced training recipe, and an
estimator where that makes sense. Sizes are chosen to train in a browser.

Every trainer records how training went, so a page can play it: the parameters at up to 40 geometrically spaced epochs
(or minibatch steps, for short runs), and from them the decision surface (or regression curve) at each, with the
misclassified rows highlighted; the loss and score per epoch on training and held-out rows; and where useful a hidden
layer's representation of the rows, feature maps, filters or hidden states at the same snapshots (maps are recorded as
playable value surfaces, their channels tiled in a mosaic). Trainers take an `onProgress` callback that reports the
records so far, which hub pages draw live. Private plumbing: `_train.ts` (lib/nn's `fitAlgorithm` run with per-epoch
curves, snapshots and scikit-learn's stopping rules), `_nets.ts` (the task of a dataset, losses and scores, MLP builders
with initialisation schemes, normalisation and dropout, snapshot predictors and surfaces, optimisers by name) and
`_records.ts` (curves, representation steps, mosaics, progress).

`neural/foundations/`
- `perceptron.ts`: Rosenblatt's perceptron as a study of its rule (every update, the surface after geometric updates
  with the row that triggered each, Novikoff's bound); the multiclass estimator is `linear/`'s `perceptron`.
- `approximation.ts`: universal approximation in one dimension (one hidden layer of growing width on one curve, the
  widest network's units as scaled, shifted activations).
- `mlp.ts`: `mlpClassifier` and `mlpRegressor` (sklearn.neural_network semantics: hidden sizes, activation, Adam or
  SGD with momentum, alpha, minibatches, the constant, invscaling and adaptive rates, `tol`/`nIterNoChange` and early
  stopping; TODO the 'lbfgs' solver), with the surface, curves, the last hidden layer's representation and the first
  layer's units over the input plane per snapshot.
- `backprop.ts`: backpropagation by hand on small networks, compared step by step with `autodiff` (the network as a
  structure with forward-value and delta overlays per step).

`neural/vision/`
- `lenet.ts`: a LeNet for small images given as rows (the 8 by 8 digits trains in seconds), with the feature maps of
  each convolution and the first filters at geometric steps.
- `alexnet.ts` (reduced), `vgg.ts` (reduced), `resnet.ts` (CIFAR-style), `mobilenet.ts`, `convnext.ts`
  (reduced): classification networks with feature-map taps.
- `unet.ts`: U-Net segmentation.
- `vit.ts`: vision transformers (patch embeddings, class tokens), attention rollout.
- `detection.ts`: a small single-stage detector (anchor-free), built on `vision/detection.ts`.
- `autoencoders.ts`: convolutional autoencoders for images.

`neural/sequence/`
- `rnn.ts`: `recurrentTagger`, an Elman RNN, GRU or LSTM on the toy sequence tasks of `lib/datasets` (a prediction per
  position, or from the last one), with the hidden states and predictions of one sequence at geometric steps and the
  accuracy by position. TODO: character-level RNN, LSTM and GRU language models.
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
- `optimisers.ts`: `optimiserRace` (SGD, momentum, Nesterov, RMSprop, Adagrad, Adam, AdamW from the same weights on the
  same minibatches) and `learningRateFinder` (Smith's range test).
- `initialisation.ts`: `initialisationStudy` (activation spread and gradient size by layer at initialisation, then
  training, per scheme) and `normalisationStudy` (none, batch and layer normalisation).
- `regularisation.ts`: `regularisationStudy` (dropout rates or weight decay against overfitting: the validation curve,
  the fits and the train-validation gap).
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

### 5.18 Datasets (moving to the core as `lib/datasets/`, see the section introduction)

Every dataset is a `DatasetGenerator` (knobs as a `Space`, returning `{ data, truth }`) or a static dataset (data and
provenance). Static data is embedded as generated `*.data.ts` files vendored by the oracle with source, licence and
checksum; larger data is a release asset loaded through `data/loader.ts`. Where a generator has a generating program,
`truth.program` is that program, so likelihoods and posteriors can be queried: `friedman1` and every isotropic
Gaussian-mixture class generator (`blobs`, `moons`, `circles`, `spirals` and `unequalVarianceBlobs`, through
`_mixtures.ts`'s `mixtureProgram`) carry one, and agree with their exact functions. TODO: programs for the remaining
generators, which give exact `truth.logProb` and `truth.posterior` functions.

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
- `fairness.ts`: data with protected attributes (named with their coded values in `meta.sensitive`): `biasedCredit`
  (a group with knobs for representation, a real base-rate gap, label bias, noisier measurements and a proxy feature;
  exact posterior, Bayes error, clean labels and base rates), the vendored Statlog German credit data
  (`germanCredit`, UCI, CC BY 4.0; sex derived from the personal status) and `titanicSurvival` (sex protected); the
  collection `fairnessTabular` gathers presets of the generator (biased labels, an under-represented group, a real
  gap, a strong proxy, no bias) and the two real datasets.
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
  hazards: run-length posteriors, one-step predictives, evidence; the mean, variance and rate models run
  `dynamics`' `filtering.bocpd`, and the AR and trend regressions, which `dynamics` lacks, run the recursion here).
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
- `search.ts`: `tuneEstimator` (phase 4): cross-validated search over an estimator's hyperparameters with `tune`'s
  grid, random, TPE or Gaussian-process samplers and its median or successive-halving pruners (fold by fold), as
  `studyAlgorithm` through `runAlgorithm` with a per-trial callback for live pages; it returns the study and best
  configuration with `SearchTrials` (every trial, its running fold means and fate), a `HyperparameterSurface` over two
  numeric parameters (a GP surrogate's prediction, `tune.contourData`, with the visited path), a
  `CrossValidationCurve` for one-parameter searches, and a `LearningCurve` of each trial's score beside the best so
  far.
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
### 5.20 Explainability, fairness and trustworthiness (`explain/`, `fairness/`, `trust/`, `privacy/`, `adversarial/`)

The core domains `lib/explain/` (4.38) and `lib/fairness/` (4.39) compute the quantities (`lib/explain/`: exact, kernel, sampling and tree SHAP and interaction
values, LIME, anchors, surrogate trees and fidelity, saliency, integrated gradients, SmoothGrad, expected gradients,
DeepLIFT, Wachter, DiCE and FACE counterfactuals, linear recourse; `lib/explain/inspection.ts`: permutation importance,
partial dependence and ICE, ALE, the H-statistic, functional ANOVA; `lib/learn/valuation.ts`: influence functions,
TracIn, data Shapley, k-NN Shapley; `lib/fairness/group-metrics.ts`: group rates, demographic parity, equal opportunity,
equalised odds, predictive parity, disparate impact, calibration by group). These areas turn them into methods over
any fitted estimator, each returning exposition records that views draw and the hub and gallery show; process shown
over time where there is one (sampling SHAP converging, mitigation trading accuracy for parity, conformal coverage
as data arrive).

oracle: shap (exact and tree explainers), scikit-learn inspection (permutation importance, partial dependence),
fairlearn (metrics, ExponentiatedGradient, ThresholdOptimizer), MAPIE (conformal), netcal (calibration); goldens only
where cheap and important.

Exposition records added to `lib/exposition/` (`attributions.ts`: `featureAttributions`, `importances`):
`featureAttributions` (per row: base value, one value per feature,
the feature values, the prediction; optional interaction values), `dependence` (PDP with ICE curves or ALE per
feature, with the data's rug), `importances` (per feature: mean and spread across repeats, the metric and its
direction), `counterfactuals` (original row, counterfactuals, changed features, distances, validity), `groupReport`
(per group: rates, counts, metrics and gaps against a reference group, at one or many thresholds), `calibration`
(reliability bins, ECE, MCE, per group optionally), `conformal` (prediction sets or intervals with nominal and
empirical coverage). Views in render (`views/`, `react/views/`): waterfall and force plots, beeswarm summary, bar of
mean |SHAP|, dependence scatter coloured by an interacting feature, PDP with ICE, ALE, importance bars with intervals,
counterfactual tables with the changes highlighted, group comparison panels (rates side by side, gap bars, per-group
ROC with thresholds), reliability diagrams, conformal bands and set sizes.

`explain/` (shap, lime, alibi, interpret, sklearn.inspection)
built from: `explain`, `learn`, `metrics`
- `attributions.ts`: `explainPrediction(model, data, row, { method: 'tree' | 'kernel' | 'exact' | 'sampling' |
  'lime' | 'linear' | 'partition' | 'conditional' })` and `explainDataset` (attributions for many rows: summary and beeswarm, with each feature's strongest
  interaction partner), with the sampling estimate's convergence recorded (and as a `curves` record); `_models.ts`
  picks the output explained (prediction, probability, a booster's log odds) and reads trees, forests and gradient
  boosting as Tree SHAP models.
- `inspection.ts`: `featureImportance` (permutation, with every repeat; conditional permutation; or impurity),
  `sageImportance` (SAGE's loss shares),
  `dropColumnImportances` (refits), `dependencePlot` (PDP with ICE curves raw, centred or derivative, the rug, two-way
  surfaces) and `aleEffects` (first and second order) for any fitted estimator: `importances` and `dependence`
  records (`lib/exposition/dependence.ts`), drawn by `DependenceView` and `DependenceSurfaceView`. Built.
- `counterfactuals.ts`: `explainCounterfactuals` (growing spheres and FACE for any classifier; Wachter, DiCE,
  prototype-guided and CEM for models linear in their inputs, read off the decision function), with actionability
  constraints and robustness to bootstrap refits: a `counterfactuals` record (`lib/exposition/examples.ts`) drawn as a
  table by `CounterfactualView`; `prototypesAndCriticisms` (MMD-critic on standardised rows: a `prototypes` record,
  `PrototypeView` on principal components or as images); `trustScoreStudy` (accuracy as fewer rows are trusted, by
  trust score and by confidence, as `curves`). Built.
- `glassbox.ts`: `explainableBoostingMachine` (shape functions and pairs as a `dependence` record of method
  `shape function`, drawn as steps and heat maps; term importances), `logisticScorecard`, `ruleFitExplanation` and
  `anchorExplanation` (each a `rules` record, `lib/exposition/rules.ts`, drawn by `RuleListView`). Built.
- `concepts.ts`: `tcavStudy` (TCAV at the LeNet's hidden layer, concepts as rows of the data): a `concepts` record
  (`lib/exposition/concepts.ts`) drawn by `ConceptView`. Built.
- `valuation.ts`: `mislabelledRowStudy` (flipped labels found by k-NN Shapley, data Banzhaf and Beta Shapley, as
  `curves`). Built.
- `evaluation.ts`: `evaluateExplanations` (explainers compared on deletion and insertion areas, faithfulness
  correlation, sensitivity, sparseness and completeness, with their agreement: an `explanationQuality` record,
  `lib/exposition/explanation-quality.ts`, drawn by `ExplanationQualityView`; mean deletion and insertion curves as
  `curves`) and `sanityChecks` (cascading layer randomisation and label randomisation of a LeNet: `curves` and the maps
  as `valueGrids`). Built.
- `surrogates.ts`: global surrogate trees with fidelity, LIME neighbourhoods, anchors with precision and coverage.
- `gradients.ts`: `explainNetwork` (saliency, input times gradient, integrated gradients, SmoothGrad, VarGrad,
  GradientSHAP, guided backpropagation, deconvolution, Grad-CAM, guided Grad-CAM) for the LeNet of `methods/neural`,
  as image-laid attribution records, and `explainLayer` (the last convolution's conductance per channel, as value
  grids). Built.
- `valuation.ts`: which training rows matter (influence, data Shapley, k-NN Shapley), with mislabelled-row studies.
- `perturbation.ts`: `explainImage` (occlusion, superpixel ablation, RISE, superpixel SHAP and LIME), `explainText`
  (word ablation, SHAP, LIME over any text scorer), `explainSeries` (window occlusion, ablation, SHAP) and
  `morrisScreening` (an `importances` record). The `featureAttributions` record gains `layout` (image, tokens,
  series) and `segments`; render draws them with `AttributionImageView`, `TokenStripView` and
  `AttributionSeriesView` (`views/modalities.ts`), which `AttributionView` picks by layout. Built. Render also
  draws one row as shap's force plot (`forceView`, `ForcePlotView`).

`fairness/` (fairlearn, AIF360)
built from: `fairness`, `learn`, `metrics`, `optim`, `datasets` (with protected attributes)
- `groups.ts`: `sensitiveGroups` (a column by name or index, labels per row, or the dataset's `meta.sensitive`),
  `withoutColumns` (fit unaware), `positiveScores`, `binaryLabels`.
- `audit.ts`: `fairnessAudit(model, data, { sensitive, threshold })`: a `groupReport` with the scored rows (so views
  move each group's threshold), per-group ROC curves and a sweep; `buildGroupReport` for scores, randomised decisions
  and per-group thresholds.
- `preprocessing.ts`: `reweighing` and `reweighedFit` (weights and the group reports with and without them),
  `disparateImpactRemover` (repaired data and each group's distribution of the most group-dependent feature); round
  two: `massaging`, `preferentialSampling`, `optimisedPreprocessing`, `fairRepresentation`.
- `in-processing.ts`: `exponentiatedGradient` (the trade-off moving over iterations, the path from the unconstrained
  model, group reports before and after) and `gridSearchReduction`; round two: `prejudiceRemoverFit`,
  `adversarialDebiasingFit` (with the losses over training), `fairConstrainedFit`, `metaFairFit`, `gerryFairFit` (with
  error and violation over the rounds), each with the group reports before and after (`_mitigation.ts`).
- `post-processing.ts`: `thresholdOptimizer` (group reports before and after, the trade-off curves with the choice,
  `rulesAt` for other shared rates); round two: `calibratedEqOdds`, `rejectOption` (with balanced accuracy and the
  metric over thresholds).
- `intersectional.ts`: `intersectionalAudit(model, data, { attributes })`: a `subgroupReport` (rates over the
  attributes' combinations, entropy decomposition, GerryFair audit over indicators of the attributes). Built.
- `data-bias.ts`: `dataBiasAudit(data, { cleanLabels, populationShares, model })`: a `dataBias` record. Built.
- `individual.ts`: `individualFairnessAudit(model, data)` (consistency and Lipschitz pairs). `dynamics.ts`:
  `delayedImpactStudy(model, groups)` (outcome curves with the policies, mean scores over rounds). Built.
- `studies.ts`: `fairnessFrontier` (a `paretoFront` of accuracy against violation, in training and after it, with the
  model selection's choice) and `impossibilityStudy`
  (calibrated scores against equalised odds and predictive parity as the base-rate gap grows).

`trust/` (MAPIE, netcal, uncertainty and robustness)
built from: `fairness`, `learn` (calibration, conformal), `metrics`, `stats`
- `calibration.ts`: `reliabilityCurve`, `calibrateClassifier` (Platt, isotonic and temperature scaling fitted on
  calibration rows, measured on held-out ones; per group optionally): a `calibration` record.
- `conformal.ts`: `conformalRegression` (split, normalised, CV+), `conformalClassification` (LAC, APS, by class),
  `onlineConformal` (adaptive conformal inference against a fixed width as rows arrive): `conformal` records.
- `uncertainty.ts`: `ensembleUncertainty` (bootstrap ensemble: epistemic spread and out-of-bag aleatoric noise).
  TODO: MC dropout, GP and Bayesian posteriors, the entropy decomposition for classifiers.
- `shift.ts`: `shiftStudy` (covariate or concept shift; KS tests, a domain classifier and the model's accuracy).
- `robustness.ts`: `robustnessStudy` (accuracy under noise, in all features or one, and missing values). TODO:
  stability of explanations.
- `model-cards.ts`: `modelCard` (a card as data, with the group report, reliability and robustness as evidence).

`privacy/` and `adversarial/`: as in 5.3 (DP training and statistics, membership inference; FGSM, PGD, defences),
each with the same records and views.

Datasets: `lib/datasets` gains tabular datasets with protected attributes (a credit or income-like generator with a
controllable dependence on the attribute, and vendored public ones where licences allow) for audits and mitigation.

Where they are shown: these are model demonstrations, so they live in the hub (collections `explainability/`,
`fairness/`, `trust/`: SHAP on fitted models, counterfactuals for a credit model, audits with per-group thresholds,
mitigation trading accuracy for parity, conformal coverage). The gallery's exposition section keeps one small recipe
per new view kind on simple data, showing what the view can draw.

Round two adds the records and views the completed domains need: perturbation and occlusion maps over images and text
(value grids and token strips), layer and neuron attributions, concept (TCAV) scores with significance, rule and
anchor lists, prototypes and criticisms, contrastive (pertinent positive and negative) explanations, glass-box shape
functions and pairwise heatmaps, explanation-evaluation curves (deletion and insertion, sanity checks); and for
fairness, metric gaps with intervals, intersectional subgroup heatmaps, Pareto frontiers, exposure in rankings, data-bias
summaries and the dynamics of repeated decisions.

Built in round one. Records built: `groupReport` (`lib/exposition/fairness.ts`, with `groupReportFromTables`, `groupReportAt`,
`tableRates`, `groupDisparities`), `calibration` and `conformal` (`lib/exposition/trust.ts`, with `setSizes`); views
`groupReportView`, `groupSweepView`, `reliabilityView`, `conformalView`/`conformalFrames` and the React
`GroupReportView` (with per-group `ContingencyTable`s and the disparities), `GroupSweepView`, `CalibrationView`,
`ConformalView`.

Hub: collections `explainability/`, `trust/` (calibration, conformal intervals, sets and streams, uncertainty, shift,
robustness, model card), `fairness/` (audit, threshold optimiser, exponentiated gradient, reweighing,
disparate-impact remover, frontier, impossibility; round two: intersectional audit, bias in data).

Round two built so far: the shared threshold layer (`thresholdSweep` record, `ThresholdExplorer`, section 6), group
report intervals and warnings, `subgroupReport` and `dataBias` records with their views, gallery recipes
`exposition/43-subgroup-report` and `44-data-bias`, and the `paretoFront` record (view `paretoFrontView`, React
`ParetoFrontView`). Round two's new methods have smoke tests but no hub entries (the hub and gallery are paused).
TODO (parked by the user): move the gallery's model-heavy `fairness/` and `trust/` recipes to hub entries, keeping as
small exposition recipes only the views not otherwise shown (conformal sets and the online stream). Gallery: the exposition section gains a recipe per new
view, driven by real methods on small data, with live controls (`exposition/40-group-report`: per-group thresholds
dragged on each group's scores); the `fairness/` section holds the mitigation recipes (threshold optimiser, exponentiated
gradient, reweighing and repair, the frontier, the impossibility); `exposition/41-reliability-diagram` and
`42-conformal-intervals`; the `trust/` section holds conformal sets, coverage as data arrive, uncertainty, shift,
robustness and the model card.

## 6. mdml-render, `packages/render/src/`

Headless first: everything outside `react/` returns data (ECharts options, layouts, scenes, SVG strings, encoded
animation files) and runs in Node, so the slide and blog tools can render without a browser. `react/` binds the data to
React 19. The package depends on mdml-engine for its contracts (`Scene`, `Structure`, `Timeline`, `Run`, `Space`,
`Distribution`) and never on its algorithms.

Entry points: `mdml-render` (every headless part), `mdml-render/<part>`, `mdml-render/react`.

oracle: none; checked by snapshot tests of options, scenes and structures, and by browser tests in Chromium.

Three consumers shape it: the machine-learning encyclopedia, a blog, and slide decks whose slides hold live toy data.
The reference implementation is aifn-engine's `packages/render` (read-only): reuse what works there (the numeric
field in `controls/numeric/NumberField.tsx` with `useNumberDraft`, the MathJax macros in `layout/math-macros.tsx`,
`layout/Tex.tsx` and `equation-tex.ts`, the drag-a-point and live-equation recipes in `examples/src/recipes/`) and
improve it as below; its player (`controls/playback/Player.tsx`) is replaced, not ported.

Design rules:
- **Figures have a default size and aspect**, as a matplotlib figure does: every figure is a fixed area with a default
  width and aspect ratio, and only changes shape when its properties say so. Content never resizes the frame.
- **Layouts are grids.** Axes are placed on a grid of rows and columns with spans (gridspec), may share x or y axes and
  ranges, may sit with almost no padding between them (a scatter plot with marginal histograms on its x and y axes),
  and grid cells may hold text or equations instead of axes.
- **Two aspects.** The box aspect (the figure's shape, e.g. 2:1) is independent of the data aspect (`equal` makes a
  circle look like a circle); with `equal` data aspect in a 2:1 box, the axis ranges are adjusted to fit.
- **Limits have policies** for reactive figures: `auto` (fit the data, honouring the aspect), `fixed` (a stated range,
  so moving a Gaussian's mean visibly moves it), `expand` (grow but never shrink) and `follow` (track a window).
- **Grids are smooth by default.** Heatmaps of gridded fields (a classifier's probability on a 30x30 grid) render
  interpolated (bilinear or bicubic, or filled contours) rather than as blocks, with a predictable `smooth` option
  and an efficient path (one upsampled image, not thousands of cells); blocky cells remain available.
- **Zoom is a modal.** A figure's expand button opens it full screen in a modal; the in-page area never changes size.
- **Mechanisms, not layouts.** Pages own their design; the package provides the pieces to specify it.

`theme/`
- `tokens.ts`: light and dark design tokens (`fg`, `bg`, `surface`, `muted`, `grid`, `accent`, semantic colours,
  `tab0`..`tab9`), `tokens(mode)`, `cssVariables(mode)`.
- `palettes.ts`: tab10, sequential and diverging ramps, colour interpolation, colour-blind checks.
- `echarts.ts`: `echartsTheme(mode)` (registered light and dark themes with tab10), `baseOption`.
- `format.ts`: number formatting, nice ticks, log ticks, time formatting.

`charts/` (ECharts option builders; each returns an `EChartsOption`, built from layers)
- `plot.ts`: `Layer` (series with their data extent, colour bar and categories; every builder returns layers so they
  combine), `cartesian(layers, options)` (axes from limit policies and the data aspect, legend, tooltip, zoom),
  `panelParts` and `chartFrame` (shared with `compose.ts`). A colour bar is calculable (draggable handles, clickable
  bands) only when its layer answers a selection (`ColourBar.select`); otherwise it is a plain legend.
- `selection.ts`: colour bar range selections for layers ECharts cannot recolour (heatmap images, matrix cells):
  `ColourSelect`, `selectionFilter` (a `datarangeselected` event as soft value weights), `dimColour` (values outside
  faded towards the background) and `colourSelection` (the option update the React `Chart` merges, once per frame).
- `echarts-core.ts`: `echartsCore`, the tree-shaken ECharts core with every chart type the builders use, both
  renderers and the registered themes (shared by the React bindings and server-side export).
- `lines.ts`: curves, steps, areas, bands, fill-between, series from tensors.
- `points.ts`: scatter plots (labels, sizes, categorical colours), bubbles, rugs.
- `bars.ts`: bar charts, histograms, Pareto charts.
- `distributions.ts`: densities from a `Distribution` or KDE, ECDFs, box and violin plots, QQ plots, PMFs, prior and
  posterior overlays.
- `matrices.ts`: heatmaps, confusion matrices, correlation matrices, images from tensors; gridded fields smooth by
  default (bilinear or bicubic upsampling to one image, or filled contours), blocky cells on request.
- `fields.ts`: contours, decision regions, vector fields, streamlines.
- `functions.ts`: function plots for the foundations (asymptotes, implicit curves, parametric and polar curves,
  tangent and secant lines, Riemann rectangles, Taylor polynomials).
- `evaluation.ts`: ROC, precision-recall, calibration, learning curves, residual plots (from `MetricResult`s).
- `multivariate.ts`: parallel coordinates with brushing (`parallelCoordinates`, `brushFromAxisEvent`, `brushRows`,
  `parallelLayout`), radar charts, scatter matrices with a linked selection (`scatterMatrix`, `scatterMatrixCells` for
  overlays).
- `tuning.ts`: figures of a `Study` (through `tune`'s analysis helpers): optimisation histories, parameter importances,
  slice plots, surrogate contour plots, intermediate values with pruned trials.
- `compose.ts`: `subplots` (a gridspec as one option: spans, shared and linked axes, zero gaps, marginals, per-panel
  legends and titles, text cells), `subplotRects`.
- `downsample.ts`: LTTB and min-max decimation.

`figure/` (the figure model; headless)
- `figure.ts`: `Figure` (default size and box aspect, padding, title, theme), defaults per medium (page, blog, slide).
- `grid.ts`: grid specifications (rows, columns, spans, ratios, gaps down to zero), shared and linked axes, text and
  equation cells, marginal-axes helpers (`withMarginals`), bento layouts from named grid areas.
- `aspect.ts`: box aspect versus data aspect (`equal`, numeric), and how ranges adjust to satisfy both.
- `limits.ts`: limit policies (`auto`, `fixed`, `expand`, `follow`) and nice-range rounding.
- `linking.ts`: linked brushing and hover across figures, shared selections.

`layout/` (positions only; no layout dependency)
- `graph.ts`: the shared shapes: a graph to lay out (nodes with sizes and groups, edges) and a layout (node boxes, edge
  routes, group boxes), with boundary clipping, straight routes, group boxes and normalisation.
- `layered.ts`: a Sugiyama-style layered layout, as dagre draws (cycle breaking, longest-path ranks, virtual points on
  long edges, barycentre crossing reduction keeping groups together, spacing-preserving coordinates; nodes of other groups kept clear of a group's box,
  nested groups from `LayoutGraph.groups`).
- `trees.ts`: tidy trees, radial trees, dendrograms.
- `force.ts`: seeded, deterministic force-directed layout with an energy-based stopping rule; circular and spectral
  layouts.
- `grids.ts`: grid, bipartite and lattice layouts.

`structures/` (any `Structure` to a diagram)
- `diagram.ts`: `diagram(structure, layout, options)` to a `Scene`: node shapes by role (latent and observed variables,
  factors, operations, tensors, decisions, utilities, states), edge routing (straight, orthogonal, splines), arrowheads,
  edge labels, self-loops, groups (plates, repeated blocks with counts, clusters), layered, tree, circular, force
  and bipartite (factor graphs) layouts, node sizes held steady across an animation's `frames`.
- `styles.ts`: the visual language per structure kind: graphical models, computation graphs, neural architectures,
  decision and search trees, automata, pipelines, networks.
- `architectures.ts`: views of a network structure: compact stack, expanded block, tensor-shape flow, parameter-count
  treemap.
- `overlays.ts`: state overlays for animation: traversal frames (active node, the edge it was reached by, visit order
  as badges) and breadth- and depth-first search orders (messages from traces are `messageFrames` in `trace/`).
- `models.ts`: presentations of parameterised models beyond what `diagram` and `architectures.ts` draw: `uNet` (U-Nets
  and encoder–decoders, a diffusion time embedding, `uNetOrder` for an animated pass), `bayesNetCards` (a Bayes network
  with its CPTs, evidence and marginals), `nestPlates` (plate-model plates nested). Decision trees with class ratios are
  the `TreeRecord` view in `views/`.

`trace/` (figures from runs)
- `metrics.ts`: metric curves (smoothing, log scales, runs overlaid, run trees).
- `tensors.ts`: parameter trajectories, gradient and weight histograms over time, per-layer norm heatmaps.
- `messages.ts`: message-passing timelines and convergence plots; frames for animating structures from `message` events.
- `episodes.ts`: return, length and regret curves with intervals.
- `optimisation.ts`: optimisation paths over contours, line-search trials.
- `tables.ts`: run summaries as table data.

`gym/`
- `frames.ts`: episode frames from an environment's `render`, filmstrips.
- `worlds.ts`: one way to draw any `worlds` world (`worldView`: the engine's top-down map with search frontiers and
  explored cells, policies as arrows sized by probability, values and numbers on cells, agent paths), and the
  first-person ray-cast view from `graphics` with its top-down view cone (`raycastViews`).

`maths/`
- `tex.ts`: typesetting through KaTeX for HTML and MathJax for SVG (used by `tex` scene nodes and SVG export).
- `macros.ts`: the shared macro registry (vectors, matrices, sets, distributions, operators, as in aifn-engine's
  `math-macros.tsx`), exported so consumers and their own packages get the same symbols with one import, and
  extendable with their own.
- `live.ts`: equations bound to parameters: values substituted and highlighted as controls move, so a slider changes
  the equation and the figure together.
- `derivations.ts`: step lists from `symbolic` and `foundations` as aligned equations.

`animation/`
- `timeline.ts`: building `Timeline`s from runs (any event channel), from environment episodes, from structure overlays
  and from parameter sweeps; easing and interpolation between frames.
- `playback.ts`: the two playback modes: precomputed (train up front, play a timeline) and live (step an `Algorithm`
  frame by frame through `iterate`, with a frame budget), behind one interface.
- `svg.ts`: `animatedSvg`, every frame of a timeline in one SVG shown in turn by CSS keyframes.
- `gif.ts`: `encodeGif` (dependency-free: a median-cut palette from `quantise`, LZW, delays, looping).
- `video.ts`: `muxWebm` (a pure WebM muxer), `encodeWebm` (WebCodecs VP8 in the browser, else a `MediaRecorder`),
  `canEncodeVideo`. TODO: MP4, and raw frames piped to ffmpeg in Node.
- `frames.ts`: `timelineScenes` (structure frames on one steady layout), `rasteriseScene` (canvas in the browser, the
  engine's rasteriser in Node, which leaves out text) and `timelineFrames`.

`svg/`
- `svg.ts`: `sceneToSvg` (tokens resolved for a theme, 3-D layers projected).
- `paths.ts`: SVG path data (straight and smooth) and arrow heads, shared by `sceneToSvg` and `SceneView`.
- `png.ts`: `encodePng` (dependency-free PNG: filtered rows, a small synchronous deflate in `_deflate.ts`),
  `encodePngAsync` (the platform's `CompressionStream`), `pngDataUrl`, `base64`.
- `export.ts`: whole-figure export: `chartToSvg` (ECharts server-side rendering), `composeSvg` (charts and overlay
  cells as one SVG), `svgDataUrl`, `svgToPng` (canvas), `exportSize` for slides and blogs.

`views/` (headless: one builder per exposition record kind, from `lib/exposition`)
- `paths.ts` (`RegularisationPath`, `CrossValidationCurve`, `LearningCurve`, `PruningPath`), `trees.ts` (`TreeRecord`
  as a structure with split candidates and row counts), `stages.ts` (`EnsembleStages`, `PipelineStages`: the steps
  of a pipeline, the panel each step's output suits (`pipelinePanelOf`: scatter, table, projection or the final
  surface) and their figures),
  `surfaces.ts` (`DecisionSurface`, `PosteriorGrid`, `HyperparameterSurface`), `steps.ts` (`ClusteringSteps`,
  `MixtureSteps`, `EmbeddingSteps` as timelines), `tables.ts` (`CoefficientTable`, `LikelihoodTable`,
  `NeighbourRecord`, `SearchTrials`), `margins.ts`, `anomaly.ts`, `components.ts`, `episodes.ts`, `boards.ts`
  (`BoardSteps` and `PopulationSteps` as players), `images.ts` (`ImageSteps` as a mosaic of image rows per step),
  `curves.ts` (`CurveSet`, revealed along x as a player), `cells.ts` (`ValueGrids` as blocky small multiples on one
  colour range, `ConfusionMatrix`, both played through training), `diagrams.ts` (`DiagramRecord` as a scene with a
  step's overlay) and `attributions.ts` (`FeatureAttributions` as a waterfall, played through a sampled estimate's
  convergence, a beeswarm summary coloured by feature value, mean |attribution| bars and dependence scatters coloured
  by the interacting feature; `Importances` as bars with spread whiskers and every repeat).
- `registry.ts`: the standard view of each record kind (`viewFor(kind)`: its React component, title, whether it plays,
  what it accepts beside the record) and `stillOf(record)`, a headless still (a chart option, or a scene for trees and
  pipelines) for slides, blogs and server-side export; the registry is exhaustive, so a new record kind does not compile
  without a view.
- Views that take rows in input space (clusters, surfaces, margins, neighbours, anomalies) accept `points` and `labels`
  beside the record; views that play return `ViewFrames` (a frame count, a label and the option of each frame).

`react/` (React 19; React and ECharts are peer dependencies)
- `Chart.tsx`: an ECharts option as a live chart (theme from mode, resizing, disposal, events).
- `SceneView.tsx`: a `Scene` as SVG elements, with hover and click hit-testing.
- `Scene3DView.tsx`: a 3-D scene with drag-to-orbit camera, re-projected on interaction.
- `StructureView.tsx`: a `Structure` in a frame (by default) with pan, zoom, hover details and overlay state (`frames`
  sizes nodes once for a whole animation).
- `TraceView.tsx`: the standard figures of a run; `useRunStream` for live runs.
- `Player.tsx`: playback of a precomputed timeline or a live algorithm, redesigned and simpler than aifn-engine's:
  play and pause with a caret menu for the target frames per second, step back and forward, a draggable scrubber, and
  an optional time stamp. `usePlayback.ts` (`usePlayback`, `useTimelinePlayback`, `useLiveRun`: one controller over
  precomputed frames and live algorithm steps) and `playhead.ts` (frame timing, the pure part).
- `ContingencyTable.tsx`: a contingency table (confusion matrix) with full margins: named cells (TP, FN, FP, TN),
  row and column totals with their rates (TPR, FNR, TNR, FPR; PPV, FDR, NPV, FOR; recall and precision for more
  classes), prevalence and accuracy; hovering a rate outlines its numerator and denominator. Static from any table,
  reactive by recomputing the table (e.g. `thresholdConfusion` at a threshold from a control). Ported from ml-concepts'
  `ContingencyTableView`.
- `ThresholdExplorer.tsx`: the shared threshold layer's figure (`yTrue`, `scores`, optional `costs`, or a
  `thresholdSweep` record; `threshold`/`onThreshold` controlled or uncontrolled; optional follow-the-cheapest): each
  class's scores and the cost against the threshold (each with a `DraggableLine`), ROC and precision-recall with the
  operating point, and the `ContingencyTable` at the threshold, one threshold driving all of them. Headless panels in
  `views/thresholds.ts` (`thresholdScoresPanel`, `thresholdCostPanel`, `thresholdRocPanel`, `thresholdPrPanel`,
  `thresholdSweepView`); `ThresholdSweepView` is the record's standard view and `GroupThresholds` (react/views/fairness)
  gives each group of a `groupReport` its own explorer. The gallery's `evaluation/threshold` and `evaluation/costs` and
  the group report are built on it. Built.
- `FigureFrame.tsx`: the fixed-size figure area with its expand-to-modal button and export menu; one frame per figure
  (a grid of panels is one figure), exporting the whole figure (charts, scenes, overlay cells) as SVG or PNG, and given
  a `playback`, recording it as GIF or WebM. Fixed content is magnified in the modal; content that is a function of the
  size redraws at it. `SceneFrame.tsx`: a scene as one framed figure (`fitScene`).
- `FigureGrid.tsx`: a gridspec drawn as one subplot chart with text, TeX or React content in the other cells, scaled to
  fit their cells.
- `echarts.ts`: `loadEcharts` for the React bindings.
- `interaction/`: overlays laid over any figure through a projection (`projection.ts`: linear, or a live chart's via
  `useChartProjection`): `DraggablePoints` (drag, click to add, remove, keyboard nudges) with `constraints.ts` (on a
  graph, segment, polyline or circle, along an axis, on a grid, in a box, chained), `Brush`, `Probe`, `NearestHover`,
  `selection.ts`, `useLinkState` (a figure `LinkStore` in React) and `keys.ts` (`useHeldKeys`: keys captured while a
  figure has focus, per press and per frame, for playing worlds).
  `DraggableLine` (a vertical or horizontal line the reader drags, bound two-way to a control's value: thresholds,
  cut-offs, levels; snaps to a step, keyboard-nudgeable).
- `EpisodePlayer.tsx`: environment episodes (run once from a seed, rendered, played with captions) with overlays;
  `useEnvStepper.ts`: an environment stepped by hand from a seed.
- `ExportMenu.tsx`: export of the current figure as SVG, PNG, GIF, MP4 or WebM.
- `MathView.tsx`: `Tex`, `Equation` (live), `Derivation`, `MathProvider` (KaTeX HTML with its stylesheet, MathML for
  assistive technology; live values highlighted).
- `controls/` (shadcn/ui components): controls generated from a `Space` (`SpaceControls`); `NumberField` (type a
  value, slide it, step it, or pick from a dropdown of 11 increments between its bounds, as aifn-engine's); `Select`
  and `Choice` with previous and next arrows to cycle through options (datasets, models) without opening the menu;
  `Toggle`, `SeedInput`; `DatasetPicker` (a dataset collection from `lib/datasets/collections.ts`, with the
  chosen generator's knobs as controls) and `MethodControls` (controls from a method's `hyperSpace`, demo defaults
  first); and `ControlPanel`: collapsible, laid out in a given number of rows and columns, stacked in a
  left sidebar beside the figure or in a bar above it, or placed in named bento-grid areas (`ControlLayout` places it).
  Plain React and Tailwind over the `--wb-*` tokens; the rules are pure modules (`number.ts`, `options.ts`, `seed.ts`,
  `space-controls.ts`, `datasets.ts`) and menus open in a `Popover` portal so no container clips them.
- `views/`: a React component per exposition view, one file per headless module (`paths.tsx`: `PathView`,
  `CrossValidationView`, `LearningCurveView`, `PruningPathView`; `trees.tsx`: `TreeView`; `steps.tsx`:
  `ClusteringPlayer`, `MixturePlayer`, `EmbeddingPlayer`, `DendrogramView`, `ReachabilityView`; `surfaces.tsx`,
  `stages.tsx` (`EnsembleView`; `PipelineView`: a strip of stage cards and the selected stage's panel, with its
  data table and fitted parameters beside the figure), `components.tsx`, `tables.tsx`, `margins.tsx`, `anomaly.tsx`, `episodes.tsx`, `boards.tsx`:
  `BoardPlayer`, `PopulationPlayer`; `curves.tsx`: `CurvesView`; `densities.tsx`: `DensityView`; `sequences.tsx`:
  `StateSequenceView`; `cells.tsx`: `ValueGridView`, `ConfusionMatrixView`; `diagrams.tsx`: `DiagramPlayer`, a
  `StructureView` with the record's steps played beneath; `topics.tsx`: `TopicWordsView`), each one figure in one
  `FigureFrame` with any player beneath it, and `ExpositionView`, which picks the standard view of any record.
- `EnvView.tsx`: one consistent view of any environment's render, decorated, with room for controls.
- `theme.tsx`: `ThemeProvider`, `useColourMode`: the mode comes from the page (the `dark` class on the root element,
  or a prop), dark by default; no system-preference detection.

## 7. mdml-meta, `packages/meta/src/`

Turns the source into documentation data, runs examples and links functions to concepts. Entry points:
`mdml-meta` (schema, runner, concepts), `mdml-meta/extract` (Node only), `mdml-meta/modules`,
`mdml-meta/worker`, `mdml-meta/react`, and the generated `mdml-meta/catalogue.json`.

`schema/`
- `catalogue.ts`: `SymbolDoc` (id, package, module, name, kind, signature, summary, description, params, returns,
  throws, examples, concepts, references, canonical home and aliases, source location), `ModuleDoc` (with its folder's
  summary), `FolderDoc` (each folder with the summary that opens its `index.ts`, flat subfolders included), `FileDoc`
  (each source file with its opening paragraph), `Catalogue`.
- `examples.ts`: `Example`, `Expectation` (value, figure, trace, throws), `ExampleResult`.
- `content.ts`: `HowTo`, `GalleryEntry`, `Study` (a methods study as a page: description, parameters as a `Space`,
  figures).

`concepts/`
- `taxonomy.ts`: the concept tree schema (domain, chapter, note; ids such as `signal-processing/spectral-estimation/welch`),
  loaded from `docs/concepts.yaml`, which mirrors the encyclopedia's table of contents.
- `concepts.ts`: `conceptIndex`, concept to symbols and symbol to concepts from the `@concept` tags; coverage of the
  taxonomy (notes without an implementation, symbols without a concept) once the taxonomy exists.

`extract/` (Node only; ts-morph)
- `catalogue.ts`: `extractCatalogue({ root, packages })`, the whole catalogue in one call (about two seconds for the
  engine: syntax trees only, no type checker).
- `project.ts`: load the packages, enumerate entry points from `exports` and barrels (a namespaced subfolder is its own
  entry point, a flat one belongs to its parent).
- `symbols.ts`: exported declarations to `SymbolDoc`s (signatures, overloads, classes and methods, aliases resolved to
  canonical homes).
- `tsdoc.ts`: TSDoc parsing (summary, `@param`, `@returns`, `@throws`, `@example`, `@concept`, `@see`, `@reference`).
- `built-from.ts`: the import graph between symbols and areas, for "how is this built" views.
- `content.ts`: how-to Markdown, gallery entries and studies.
- `coverage.ts`: documentation coverage (summary, every parameter described, `@returns`, an `@example` with a `// =>`
  for every core function, class and constant; a summary for every folder and file of every package);
  `make examples ARGS=--coverage` reports it.

`runner/` (browser and Node)
- `parse.ts`: statements and expectation markers.
- `transform.ts`: type stripping and import rewriting (sucrase), expectation instrumentation.
- `execute.ts`: execution with a module resolver, console capture, time limits, errors mapped to example lines.
- `compare.ts`: comparison to the stated precision (a literal with d decimals matches what rounds to it; integers and
  computed values to about twelve digits), tensors and typed arrays as nested arrays (`Float64Array [1, 2]` too),
  figure, trace and throw recognition.
- `format.ts`: display of values beside code.
- `run.ts`: `runExample`, `runExamples` with a report.

`modules/`
- `index.ts`: `moduleLoaders` (generated lazy imports for every public entry point).

`worker/`
- `protocol.ts`, `worker.ts`, `client.ts`: running examples and calling exports off the main thread, streaming trace
  events, latest-wins cancellation.

`react/`
- `live-example.ts`: `useLiveExample`, the state of an editable example (code, latest result, run on demand and on
  first view); the site draws it (`apps/site/src/docs/LiveExample.tsx`: highlighted editor, outcomes beside each
  `// =>` line). Figures and traces rendered with mdml-render come later.
- `CodeEditor.tsx`: CodeMirror 6 with completion from the catalogue.
- `ValueView.tsx`: any output (numbers, tensors, distributions, structures, scenes, charts, runs, timelines).
- `ApiSymbol.tsx`, `ModuleIndex.tsx`, `Search.tsx`: browsing the catalogue.
- `ConceptPage.tsx`: the implementations of a concept with their examples, for encyclopedia notes.
- `BuiltFrom.tsx`: the "built from" graph of a symbol or area.
- `StudyView.tsx`: a study with controls generated from its parameter space.

## 8. The site, `apps/site/`

One private Vite and React 19 app, served by `make dev` (http://localhost:5300) and built by `make site`, with three
sections on one shell, in priority order: the gallery shows off mdml-render, the hub shows off mdml-hub, and
the docs document the engine. It consumes the packages exactly as an outside project does. aifn-engine's
`examples/` app is the template: its shell, recipe registry, live-example pages, Markdown and docstring-driven docs
pages are recycled rather than reinvented.

`src/shell/`: the top bar with the three sections, a dark and light toggle at the top right (dark by default,
remembered, set before first paint; no system-preference detection),
theme tokens from `mdml-render/theme`, Tailwind CSS, a dependency-free path router (`usePath`, `navigate`,
`Link`), the contents sidebar (`TocSidebar`: any tree of `TocNode`s, a filter, folders closed but for the way to
the current page, folder names linking their pages) with previous and next links over the same tree (`PrevNext`, arrow
keys too), and the recipe and code-view components (a live figure beside its own source, `ErrorBoundary` keeping a
failing example to itself).

`src/gallery/` (hand-written: what mdml-render can show and how its pieces interact)
- Pages: the index of sections, a section, an entry (a figure with its source and export menu).
- `registry.ts`: the recipes from their `recipe` metadata alone (an eager glob with `?recipe`, which
  `plugins/recipe-meta.ts` serves as just the file's `recipe` literal), each with a loader for its code and source (lazy
  globs) that its page calls (`shell/LazyRecipe.tsx`), so the index and section pages import no recipe.
- `recipes/<section>/NN-name.tsx`, gathered by `import.meta.glob`: charts, layouts (grids, shared axes, marginals,
  aspects, limit policies), colour and themes, statistical, fields (smooth grids, contours, regions), text and maths
  (macros, live equations), controls (number fields, cycling selects, control panels), interaction (adding and
  dragging points, probes, brushes, linked hover), animation (the player, live and precomputed), diagrams and
  structures, multivariate (parallel coordinates, radar, scatter matrices), tuning studies, environments and worlds
  (maps, mazes, the ray-cast view), three-d, evaluation (contingency tables, ROC, precision-recall and PR-gain,
  reliability, confusion matrices), hierarchies and flows (treemaps, sunbursts, Sankey), time (calendars, candles),
  parts of a whole (pies, funnels, gauges). Every figure has one frame (expand, export) and a live control. Toy data
  only, no model training: model demos belong to the hub.

  The gallery is for general visualisation types only; on 2026-10-25 its model demonstrations (the `exposition/`,
  `fairness/` and `trust/` sections and the RL episodes) moved to the hub as standalone pages, and `moved.ts` sends
  their old paths to the new ones.

`src/hub/` (generated from `mdml-hub`'s catalogue, plus standalone pages)
- Pages: collections and tags, a collection, an entry through `HubEntryView` (dataset picker, method controls, its
  views, its trace, its source). Adding a `defineEntry` adds its page.
- `pages/<collection>/NN-name.tsx` (`standalone.ts`): one-off live demos in the recipe format, listed in their
  collections beside the catalogue's entries; code of their own, not necessarily a method. Loaded as the gallery's
  recipes are: metadata with the hub, code with the page.
- The index pages and contents come from the hub index, `virtual:mdml-hub/index` (`plugins/hub-index.ts`, read in a
  child Node process by `plugins/hub-index-data.ts` and read again after an edit to the hub's source): each entry's
  card and header (title, summary, collection, tags, method links) with its module, and the collection descriptions.
  An entry's page shows its header at once and loads its widget lazily (`live.tsx`: the entry's module through
  `mdml-hub/lazy`, `HubEntryView`, the state in the URL), so no page imports the whole catalogue.
- TODO (consolidate when the hub is rethought around small, particular tasks): pages that duplicate catalogue entries,
  suffixed `-demo` (explainability counterfactuals, prototypes, rules, TCAV, explanation quality; fairness exponentiated
  gradient, frontier, impossibility, data bias; trust conformal sets, online conformal, uncertainty, shift, robustness,
  model card), and the other moved pages that overlap entries (feature importances, tree SHAP views, threshold
  optimiser, reweighing and repair).

`src/docs/` (generated from `mdml-meta`'s catalogue of the engine's TSDoc)
- Navigation mirrors the source tree exactly (`lib/<domain>/<subfolder>/<file>`, `methods/<area>/...`, then the
  symbols of each file), as aifn-engine's docs: collapsed by default, only the current path expanded, a filter on
  top. Folder pages show the folder's summary (from its `index.ts`) and file cards each file's opening paragraph.
- Pages: home, folder (its subfolders and files), file (its exports), symbol (signature, TSDoc, live examples with
  their `// =>` values), concepts (by the encyclopedia's taxonomy), how-tos (`content/howto/`, runnable code blocks checked by the
  build), traces (open, replay, compare and export runs).
- The catalogue is precompiled, never extracted on a request: `make docs-catalogue` (run first by `make dev` and
  `make site`, a no-op when its output is newer than the engine's source) writes static files named by content hash
  to `apps/site/.generated/docs/` (gitignored): a small index for the tree (folders with summaries, files with their
  exports' names and kinds: about 0.5 MB where the whole catalogue is 7 MB), one data file per source file (its
  comment, full `SymbolDoc`s and `refs`: the type names in its signatures resolved through its imports to
  `[module, name]`), a types file per module (each exported type's declaration and lead sentence, named in the index
  and fetched on the first hover) and a search index of every export's summary (fetched when the search box is used,
  or once the page is idle), then `manifest.json` naming the current version.
- Type names in signatures, parameter types and prose (`{@link}`, inline code) link their page and, on hover or
  keyboard focus, show the type's declaration (highlighted) and lead sentence in mdml-render's `Popover`
  (`src/docs/TypeHover.tsx`); Escape or moving away closes it. Names in code resolve only through the file's `refs`,
  so globals and private types stay plain.
- `plugins/`: `docs-data.ts` (writing the files, the staleness check), `build-docs.ts` (the command behind
  `make docs-catalogue`) and the Vite plugin `docs-catalogue.ts`. In dev it serves `<base>docs-data/*` from the
  folder and `virtual:mdml-docs/index` (the current index's URL); after an edit to the engine it regenerates in a
  background process (debounced, low priority) while still serving the previous version, then sends
  `mdml-docs:update` and open pages fetch the new index. In a build it copies the files into `dist/docs-data/`.
- Each section's code loads when the section is first visited (`React.lazy` in `main.tsx`), so the docs load none
  of the gallery's recipes or the hub's collections; the first page's section loads before the first render, so it
  appears without a Suspense fallback (which React shows for at least 300 ms).

## 9. Python oracle, `oracle/`

A uv project (root `pyproject.toml`) with pydantic and pydantic-settings, and dependency groups split by domain so a
fixture run installs only what it needs: `oracle-core` (numpy, scipy, scikit-learn, statsmodels, sympy, python-sat, clingo,
networkx, python-control with slycot), `oracle-torch` (torch, torchmetrics, torchdiffeq, torchsde), `oracle-prob` (numpyro, pyro, pgmpy, arviz),
`oracle-signal` (pywt, librosa, scikit-image, pystoi, pesq), `oracle-text` (tokenizers, tiktoken, nltk, sacrebleu,
rouge-score, jiwer, rapidfuzz, datasketch, gensim), `oracle-rl` (gymnasium, minatar, pettingzoo, open_spiel with ecos for its LPs, nashpy),
`oracle-misc` (lifelines, mlxtend, pysubgroup, umap-learn, hdbscan, dowhy, econml, causal-learn, pysindy, faiss-cpu,
implicit, optuna).
- `oracle/mdml_oracle/settings.py`: `OracleSettings` (reads the root `.env`; output paths; selection by path prefix
  or group).
- `oracle/mdml_oracle/fixture.py`: `Case` and `Fixture` (inputs, outputs, tolerances, reference library and version,
  seed), JSON encoding of non-finite and complex numbers, deterministic writing.
- `oracle/mdml_oracle/cli.py`: `generate`, `check` (regenerate and diff), `list`.
- `oracle/cases/<domain>/<file>.py`: one case file per engine file with a Python counterpart, at the mirrored path.
- `oracle/datasets/<name>.py`: dataset vendoring (pinned source, checksum, generated `*.data.ts` or a release asset with
  a manifest entry).
- `packages/engine/fixtures/`: generated JSON fixtures, committed.
- `packages/engine/src/testing/golden.ts` (not published): `loadFixture`, `expectClose`, the fixture schema matching
  the pydantic model.

Python never runs in `make test`. `make fixtures` regenerates; a scheduled CI job runs `make fixtures-check` group by
group.

**The fixture standard.** Fixtures are a safeguard: they show that mdml-js agrees with the established Python
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

- `Makefile` and `mk/*.mk`: `help`, `setup`, `dev`, `dev-docs`, `docs-catalogue`, `build`, `lint`, `format`, `typecheck`,
  `test`, `test-browser`, `coverage`, `docs`, `concepts`, `barrels`, `layers`, `fixtures`, `fixtures-check`,
  `manifest`, `datasets`, `pack`, `smoke`, `verify`, `check`, `clean`, `release`.
- Two gates: `make verify` for every commit and merge (whole-repo checks that take seconds, then only the tests,
  examples, type checks and fixtures related to the change; everything when shared infrastructure changes) and
  `make check` (which includes `make examples`) with `make fixtures-check` as the periodic full run after a batch of
  merges and before a release.
- `scripts/verify.ts`: the scoped gate (changed files against a base, `vitest related`, fixture-to-test lookup, the
  examples of the changed engine source files through `scripts/examples.ts --only <files>` (about 3 seconds for a
  typical change, most of it building the catalogue; all examples when `scripts/` or mdml-meta's `extract/`,
  `runner/` or `modules/` change), scoped pyright and fixture checks).
- `mise.toml`: Node, pnpm, Python and uv versions.
- `scripts/barrels.ts`: every `index.ts` below its hand-written summary, and the module map; `--check`; fails on a
  folder without a summary or a module file without an opening paragraph (section 2).
- `scripts/layers.ts`: no import cycles between core domains or between methods areas (a cycle is reported as its
  path, one import per step), the core never imports methods, and the package rules (section 1).
- `scripts/examples.ts` (`make examples`): every `@example` of the engine core run in worker threads with its `// =>`
  lines checked (about 6,000 in under 20 seconds; `--methods` adds the methods' optional examples, `--only` narrows to module or file prefixes,
  `--coverage` reports documentation coverage). How-tos join it when they exist.
- `apps/site/plugins/build-docs.ts` (`make docs-catalogue`): the docs catalogue extracted once (about 3 seconds) and
  written as the site's static docs files (section 8); `--if-stale` returns at once when they are newer than the
  engine's source, mdml-meta's `extract/` and `schema/`, and the generator.
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

| sklearn                                                     | mdml                                                  |
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
| `model_selection` (searches)                                | `lib/learn/model-selection/search.ts`, `lib/tune/`    |
| `metrics`                                                   | `lib/metrics/`                                        |
| `calibration`                                               | `lib/learn/calibration.ts`                            |
| `inspection`                                                | `lib/explain/inspection.ts`                           |
| `datasets`                                                  | `lib/datasets/`, `lib/data/`                          |

### PyTorch and its ecosystem

| library                               | mdml                                                                        |
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

| library                               | mdml                                              |
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

| library                           | mdml                                                                                              |
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
| pysat, clingo, pyDatalog          | `lib/logic/sat.ts`, `maxsat.ts`, `answer-sets.ts`, `datalog.ts`                                   |
| AIMA                              | `lib/search/`, `methods/gofai/`                                                                   |
| mlxtend, SPMF, pysubgroup         | `methods/patterns/`                                                                               |
| DEAP                              | `lib/search/evolutionary/`, `methods/evolutionary/`                                               |
| PySINDy                           | `methods/scientific/discovery/`                                                                   |
| faiss                             | `lib/numerics/spatial/`, `methods/retrieval/indexes.ts`                                           |
| datasketch, rapidfuzz             | `lib/text/shingles.ts`, `lib/text/fingerprinting.ts`, `methods/retrieval/approximate-matching.ts` |
| recordlinkage, Splink             | `methods/retrieval/record-linkage.ts`                                                             |
| gensim, fastText                  | `methods/embedding/`, `methods/models/topic.ts`                                                   |
| lifelines                         | `methods/survival/`                                                                               |
| fairlearn, AIF360                 | `lib/fairness/`, `methods/fairness/`                                                              |
| Opacus                            | `lib/learn/privacy.ts`, `methods/privacy/`                                                        |
| SHAP, Captum                      | `lib/explain/`                                                                                    |
| cleanlab, Snorkel                 | `methods/weak-supervision/`                                                                       |
| zlib                              | `lib/information/deflate.ts`                                                                      |

## 12. Order of construction

Bottom up, following the import graph; a domain is finished (documented, tested against the oracle, traced) before the
ones that import it rely on it. The first pass is a thin vertical slice that proves the whole pipeline (values to traces to figures to docs):
1. `base`, `scene` (types, structure, timeline), `tensor` (primitives, core operations, CPU backend), `random`.
2. `autodiff` (reverse mode, `grad`, `valueAndGrad`, `computationGraph`, `checkPrimitive`), `trace` (schema, policy,
   algorithm, recorder, replay), `space`, `context`.
3. `numerics/special/erf.ts`, `numerics/linalg/cholesky.ts` (with its custom derivative), `optim` (`sgd`, `adam`,
   `minimizer`), `dists` (`normal`, `categorical`).
4. `pgm` (factors, Bayes networks, variable elimination, `toStructure`), `losses` (`mse`, cross-entropy), `nn` (linear
   layers, activations, `Sequential`, `fit`, `describe`), `rl` (`env.ts`, `gym.ts`, `rollout.ts`) with CartPole and its
   render, `data` (regression schema, generator and truth) with `friedman1` and `diabetes`, `metrics/regression.ts`.
5. mdml-render: theme, `charts/lines.ts`, `layout/layered.ts`, `structures/diagram.ts`, `trace/metrics.ts`,
   `animation/gif.ts`, `svg/`, `react/` (Chart, SceneView, StructureView, TraceView, Player).
6. mdml-meta: schema, extractor, runner with `// =>` checks, module map, concept taxonomy; the docs app (with its gallery)
   showing the slice; the oracle producing the first fixtures; CI, packing and the smoke test.

After the slice, every remaining file in this document becomes a build task, taken in import order for the core and in
"built from" order for methods.

Effort: the engine (core, then methods) is built at high effort, each domain to its full plan line before the next
one. mdml-render, mdml-meta and the docs app follow at normal effort, guided by the design rules of section 6
and reusing aifn-engine's render and docs code where it already works.

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
6. **Scenes and structures in the engine.** Both live low in the engine (`lib/scene`) so models and environments can describe
   themselves; the alternative is a small fifth package.
7. **Datasets in the engine package.** Embedded static data enlarges the engine tarball; the alternative is a
   `mdml-datasets` package.
8. **Scope.** Anything missing that a comprehensive library for machine learning, statistics, numerics and their
   mathematical foundations should have.
