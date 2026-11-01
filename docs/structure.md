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
| Classical vision (Hough)                                 | `lib/signal/image/features.ts` (Hough) and `lib/vision/` (4.43).                                                                                                                                                                                                                                                                      |
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

`apps/site` is the one site: the gallery shows off `render` and the docs document `engine`. Outside projects (the
encyclopedia, the slide and blog tool) consume the three published packages as release tarballs; pages that explain
a method are the encyclopedia's, composed from the packages, and the site has no hub (removed 2026-10-28).

## 2. Conventions every file follows

**Entry points.** Every folder is an entry point. `packages/engine/src/lib/numerics/linalg/` is imported as
`mdml-engine/numerics/linalg`; its parent re-exports it as the namespace `linalg`. Index files are generated from
folder contents, so adding a file never edits another file.

**The library.** `packages/engine/src/lib/` is all of the engine: general machinery, and the named models, algorithms,
environments, datasets and studies built from it. There is no methods layer (retired 2026-11-01): wherever this
document says `methods/<area>/...`, read it as an entry of the methods catalogue (`docs/methods-catalog/`) that a
page of the site composes from library estimators; the code of the old layer is in `docs/methods-catalog/archive/`.

**Layering.** Core domains import one another without cycles: if `stats` imports `dists`, nothing `dists` imports may
lead back to `stats`. There is no declared order; the import graph is the order, and `make layers`
(`scripts/layers.ts`) fails with the path of any cycle; the package rules of section 1 hold too.

**Summaries live in the code.** The source explains itself, with no metadata files beside it:
- A folder's summary is the doc comment that opens its `index.ts`: a short paragraph on what the folder is and what
  you will find in it (not an index of its files). The barrel generator keeps it and writes the exports below it. A
  subfolder re-exported flat into its parent's entry point (`tensor/ops` into `mdml-engine/tensor`) carries the
  `@inline` tag in its summary.
- A file opens with its own paragraph: what the file is and what to expect in it (for `shapley.ts`, what Shapley
  values are and which explainers it provides), as a doc comment followed by a blank line or the imports. No to-do
  lists and no lists of every export.
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

The core has 59 domains, roughly from the most basic to the most composite; a domain imports the domains it needs, and
the import graph between them has no cycles (section 2). Each domain's own summary is the doc comment of its
`index.ts`.

| domain           | purpose                                                                                                                                                                                                                                                                                                                       |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `base`           | errors, JSON, shared result shapes, scalar helpers, version                                                                                                                                                                                                                                                                   |
| `scene`          | drawings (`Scene`), 3-D layers, rasters and graph-shaped descriptions (`Structure`) as data                                                                                                                                                                                                                                   |
| `tensor`         | arrays, dtypes, shapes, primitives with derivative rules, backends, operations, pytrees                                                                                                                                                                                                                                       |
| `random`         | immutable counter-based keys and samplers                                                                                                                                                                                                                                                                                     |
| `symbolic`       | symbolic expressions: simplification, differentiation, series, integration, LaTeX                                                                                                                                                                                                                                             |
| `exposition`     | typed exposition records that methods return and render draws (paths, trees, stages, steps, trials, episodes)                                                                                                                                                                                                                 |
| `autodiff`       | reverse, forward and Taylor-mode differentiation, vectorisation, gradient checks                                                                                                                                                                                                                                              |
| `trace`          | algorithms, trace schemas and policies, events, recording, replay                                                                                                                                                                                                                                                             |
| `space`          | typed spaces for parameters, knobs, observations and actions                                                                                                                                                                                                                                                                  |
| `context`        | seeds, named keys, tracer and policy defaults                                                                                                                                                                                                                                                                                 |
| `numerics`       | special functions, linear algebra, sparse, polynomials, quadrature, roots, interpolation, Fourier, convolution, geometry, nearest neighbours, approximation, inverse problems                                                                                                                                                 |
| `data`           | dataset schemas, generators and truth, loading, batching, splits, modifiers                                                                                                                                                                                                                                                   |
| `foundations`    | mathematical foundations: trigonometry, vectors, rotations, curves, surfaces, calculus, series, linear algebra with steps, number theory, combinatorics, complex numbers, elementary probability                                                                                                                              |
| `graph`          | graphs and graph algorithms, graph signal processing                                                                                                                                                                                                                                                                          |
| `dists`          | distributions, bijectors, exponential families, GLM families, copulas                                                                                                                                                                                                                                                         |
| `information`    | entropy, divergences, mutual information estimators, source and channel coding, compression                                                                                                                                                                                                                                   |
| `systems`        | linear time-invariant systems: representations, responses, stability, discretisation                                                                                                                                                                                                                                          |
| `worlds`         | discrete worlds: grid and graph maps, mazes and generators, terrain and rules, agents moving discretely or continuously                                                                                                                                                                                                       |
| `stats`          | descriptive statistics, tests, resampling, density estimation, smoothing bases                                                                                                                                                                                                                                                |
| `optim`          | optimisers, minimisation, programming, sparse approximation, transport, implicit differentiation                                                                                                                                                                                                                              |
| `kernels`        | kernel functions, compositions and explicit feature maps                                                                                                                                                                                                                                                                      |
| `distances`      | named distances and similarities between points, each with its parameters as a space, evaluated by name between rows or all pairs                                                                                                                                                                                             |
| `signal`         | signal and image processing, spectral and time-frequency analysis, audio features                                                                                                                                                                                                                                             |
| `text`           | normalisation, tokenisation, subwords, shingles, vectorisers, similarity, parsing, fingerprinting                                                                                                                                                                                                                             |
| `search`         | state-space search, local and evolutionary search, adversarial search, constraint satisfaction, planning, refinement search                                                                                                                                                                                                   |
| `stochastic`     | stochastic processes and stochastic calculus, SDE solvers, point processes, DPPs, random measures                                                                                                                                                                                                                             |
| `sampling`       | Monte Carlo samplers and diagnostics                                                                                                                                                                                                                                                                                          |
| `graphics`       | rendering mathematics: ray casting over worlds, ray tracing, cameras, shading, textures, rasterisation                                                                                                                                                                                                                        |
| `logic`          | terms, unification, resolution, Datalog, SAT, first-order proving, ILP                                                                                                                                                                                                                                                        |
| `dynamics`       | ODE and PDE solvers, nonlinear dynamics, control, filtering, system identification                                                                                                                                                                                                                                            |
| `pgm`            | graphical models, their inference and learning, causal graphs                                                                                                                                                                                                                                                                 |
| `metrics`        | evaluation metrics in sixteen categories, with intervals and comparisons                                                                                                                                                                                                                                                      |
| `losses`         | differentiable training objectives and regularisers                                                                                                                                                                                                                                                                           |
| `rl`             | environment and agent contracts, wrappers, buffers, returns, planning, off-policy evaluation                                                                                                                                                                                                                                  |
| `games`          | normal-form, extensive-form and cooperative games, equilibria, learning in games, mechanisms                                                                                                                                                                                                                                  |
| `probprog`       | probabilistic programs, model analysis, inference engines and the planner, posteriors                                                                                                                                                                                                                                         |
| `learn`          | estimator protocol, preprocessing, composition, model selection, calibration, conformal, active learning, privacy                                                                                                                                                                                                             |
| `nn`             | neural networks: functional layers, modules, blocks, description, training, decoding                                                                                                                                                                                                                                          |
| `tune`           | studies, objectives, samplers, pruners, multi-fidelity schedules                                                                                                                                                                                                                                                              |
| `datasets`       | dataset generators with exact truths, vendored real datasets, themed dataset collections                                                                                                                                                                                                                                      |
| `explain`        | attributions (Shapley, LIME, gradients), surrogates, counterfactuals, inspection, data valuation, explanation quality                                                                                                                                                                                                         |
| `fairness`       | group, individual and intersectional fairness, bias in data, mitigation, trade-off frontiers                                                                                                                                                                                                                                  |
| `vision`         | classical computer vision: colour, histograms, contours, features and matching, multi-view geometry, calibration, stereo, tracking, denoising, segmentation, detection                                                                                                                                                        |
| `timeseries`     | ARMA and ARIMA theory, estimators and forecasts, exponential smoothing, decomposition, the matrix profile, online detectors, vector autoregression, volatility and reconciliation                                                                                                                                             |
| `svm`            | support vector classification and regression (C and nu forms, one-versus-one, Platt probabilities) and novelty detection (one-class, SVDD) as stepped estimators over the SMO duals                                                                                                                                           |
| `gp`             | Gaussian-process regression and classification (Laplace, EP), sparse approximations, the relevance vector machine, as stepped estimators                                                                                                                                                                                      |
| `trees`          | decision trees: the tree structure, split criteria, the stepped grower, pruning, ID3, C4.5 and CHAID, the Hoeffding tree, CART estimators                                                                                                                                                                                     |
| `ensembles`      | bagging, random forests and extra trees, AdaBoost, gradient and histogram boosting, the cyclic booster of explainable boosting, voting and stacking                                                                                                                                                                           |
| `kernel-methods` | kernel ridge regression with exact leave-one-out, online kernel recursive least squares (ALD, coherence, sliding window) and NORMA                                                                                                                                                                                            |
| `gam`            | generalised additive models: smooth, tensor-product and parametric terms on the penalised bases of `stats/smoothing`, penalised IRLS with REML, GCV and UBRE smoothing selection, backfitting, effective degrees of freedom                                                                                                   |
| `covariance`     | covariance and precision estimation: shrunk, Ledoit-Wolf and OAS estimators with Mahalanobis distances and likelihoods, FastMCD and the elliptic envelope, the graphical lasso                                                                                                                                                |
| `anomaly`        | outlier detection: isolation forest and its extended variant, local and connectivity-based outlier factors, k-nearest-neighbour distances, score normalisation and combination, threshold rules                                                                                                                               |
| `sketches`       | probabilistic data structures for streams: Bloom, counting Bloom and cuckoo filters, HyperLogLog, Count-Min and Count sketches, Misra-Gries and Space-Saving, MinHash and LSH, reservoir samplers, GK, t-digest and KLL quantile sketches, the skip list                                                                      |
| `survival`       | survival analysis of right-censored data: Kaplan-Meier and Nelson-Aalen, weighted log-rank tests, Cox proportional hazards with Breslow and Efron ties, accelerated-failure-time models, concordance, censoring-weighted Brier scores and time-dependent AUC                                                                  |
| `causal`         | causal inference from data: treatment-effect estimators (regression adjustment, propensity scores, matching, entropy balancing, AIPW, TMLE, double machine learning), quasi-experimental designs, heterogeneous effects and uplift, sensitivity analysis                                                                      |
| `embedding`      | embeddings learned from data: word2vec (skip-gram, CBOW, negative sampling, hierarchical softmax), fastText subwords, GloVe, PPMI with SVD, knowledge-graph embeddings (TransE, DistMult, ComplEx, RotatE) with filtered ranking, and analogy and similarity evaluation, each loss a smooth problem with the optimiser a slot |
| `patterns`       | pattern mining: Apriori, FP-growth and Eclat (identical itemsets and counts), closed and maximal itemsets, association rules with their measures, PrefixSpan and GSP with gap constraints, subgroup discovery by quality function with beam, exhaustive, CN2-SD and Apriori-SD search                                         |
| `rules`          | rule learning on the `learn` contract: CN2 (ordered and unordered), PRISM, RIPPER and IREP, OneR and decision tables, CORELS-style certifiably optimal and Bayesian rule lists, the rule, coverage and quality-measure representation, rules from tree paths                                                                  |

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
- `registry.ts`: the shape the dataset and environment registries share: `Registry<E>` (named entries in order, a
  plain value), `registry`, `register` (a new registry with more entries, replacing by name; nothing registers itself
  on import), `lookup`, `find`, `matchesText` (the search-box test of the registries' queries).
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
- `tracer.ts`: `Tracer`, `nullTracer`, `recorder` (with `forget`, so a kept window can let its oldest event go),
  `tee`, `filtered`, `startRun`, `RunHandle` (applies the policy).
- `algorithm.ts`: `Algorithm<S, K>` (`init`, `step`, `done`, `observe`, `schema`, `serialise`), `StepContext<K>`,
  `runAlgorithm`, `iterate` (a generator for live views), `runFor` (budgeted slices that continue exactly; the budget
  is a step count or milliseconds of an injected clock, since published code reads no clocks).
- `nesting.ts`: child runs (`subRun`, keyed from the outer step, with the policy `childPolicy` derives); run trees are
  read back with `runTree` in `query.ts`; `mountContext` (the step context of an inner method mounted under a prefix,
  typed `StepContext<K>` over an outer `StepContext<Prefixed<P, K>>`).
- `taps.ts`: `tapContext` (`taps.tap(name, value)` inside model code, `TapContext`), `tapsFor(run, step)` (records
  activations when the policy asks).
- `instrument.ts`: `instrument(fn, options)` (trace calls of any function: numeric inputs and outputs as metrics or
  tensor summaries, with a generated schema; timing excluded unless a clock is injected), `traceCall` (step records
  for direct methods).
- `checkpoint.ts`: `checkpointEvery` (a policy storing checkpoints every n steps), `resume`, `seek`.
- `contract.ts`: the log contract (`docs/methods.md`): sources by path (`scalar`, `tensor`, `message`, `frame`,
  `derived`, `event`), subscriptions (`Log` with `every`, `detail`, `args`, `budget` and `keep`: `all`, `last` or the
  most recent n events of the path, checkpoints never among them; `On`; `off`/`defaults`/`all`), path patterns,
  `resolve` (contract and subscription to a policy, per-path schedules and hooks; it rejects a bad `keep`),
  `contractSchema`. **Typed paths:** a path is a `string` at run time (tapes, saved panel configs, JSON and the render
  layer name paths as data, subscriptions are patterns) and a type parameter `K` (default `string`) at compile time:
  `Contract<S, K>`, `Subscription<K>`, `Method<S, C, K>`, `Algorithm<S, K>`, `StepContext<K>`, `Session<S, C, K>`. With
  `K` a union of literals, `ctx.log`, `ctx.emit` and `ctx.wants` take only those paths (a contract key that is a
  pattern is logged by any path under it, `Concrete<K>`), `defaults` and a session's `logs` take those paths or any
  pattern (a `*` in it, `Pattern`; patterns are checked by `resolve` at run time), and `defineMethod` infers `K` from the
  contract. `defineContract` keeps a literal contract's keys (and checks them), `PathsOf<typeof contract>` reads them
  back, fragments compose by spreading (`{ ...solverContract, ...own }`), `Prefixed<P, K>` and `prefixContract` mount a
  contract under a prefix. A path built at run time stays `string`, or a template literal for the part that varies
  (`layer${number}/weights`); the type cannot see a pattern's matches. Because `isolatedDeclarations` wants an exported
  constant annotated, an exported contract (a fragment) writes its path union and annotates `Contract<S, FooPath>`; the
  annotation is the check that the two agree. The default `K = string` leaves every untyped method as it was; a typed
  `Algorithm<S, 'a'>` is accepted wherever an `Algorithm<S>` is wanted (the context's methods are bivariant). The
  migration recipe is in `defineContract`'s TSDoc. A path union may hold template literals (`solver/${string}`,
  `layer${number}/weights`): `SubscriptionMap` keeps the literal members as optional keys and the template members as
  index signatures, so that a method or session typed with them is still assignable to the untyped `Method<S, C>` and
  `Session<S, C>` that the render layer takes. The estimator bases (`learn/bases`) carry the same parameter: see 4.33.
  **Backstop:** a session's log context throws
  `MdmlError('value', "log: the contract has no path '...'")` when an algorithm logs to or emits a
  path the resolved contract (`Resolved.paths`) does not hold, as opposed to one that is merely unsubscribed (`wants`
  is a question and answers `false`; checked once per path, remembered; not in a run with no tape, nor in a bare
  `runAlgorithm`, which has no contract).
- `logging.ts`: the scheduler (every n steps, log-spaced, or on the nth firing of an event), the log context
  algorithms write to (`log`, `wants`, `emit`, `taps`), `defaultLogContext` for runs outside a session, and `gated`
  (the tracer that renames native channels to paths, passes only what is subscribed and due, and holds a path's
  `keep` window by having the recorder forget its oldest events as new ones arrive, the budget counting what is held).
- `session.ts`: `Method<S, C, K>` and `defineMethod` (config space, `make`, contract, defaults, named hooks,
  projections; `K` inferred from the contract), `Session` (`advance`, `stateAt`, `fork`, `probe`, `project`, `run`,
  `stepsOf`) and `session` with its modes (eager, live, eager to a point).
- `serialise.ts`: `stateToJson` and `stateFromJson`, generic checkpoint serialisation of states made of tensors, typed
  arrays, keys and plain trees.
- `replay.ts`: `exportRun`, `parseRun`, `replay`, `mergeRuns`, `isRun`.
- `query.ts`: `metricSeries`, `tensorSeries`, `messages`, `episodes`, `frames`, `runTree`, `summary`, `diffRuns`.

oracle: numpy for `summaries.ts` (moments, norms, `np.histogram`); the rest by unit and property tests

### 4.8 `space/`: typed spaces (gymnasium.spaces, Optuna distributions, ConfigSpace)

- `types.ts`: `Space` union (`real` with bounds, log scale and step; `int`; `choice`; `bool`; `box`; `discrete`;
  `multiDiscrete`; `multiBinary`; `tuple`; `dict`; conditional dimensions through `when` on a dict child, which may
  chain; `union`, a tagged union of named dict options, the value naming its option under `tag` (default `kind`) and
  carrying that option's knobs beside it, so a mode and its settings are one space and unions nest into a tree of
  modes), metadata (label, description, unit, and a `default` on every kind), with a builder per kind.
- `operations.ts`: `sample(space, key)`, `contains`, `clip`, `defaults`, `flatDim`, `flatten`, `unflatten`,
  `activeDims` (dotted paths of the active dimensions of a value).
- `encoding.ts`: `encode`/`decode` to the unit cube (every kind, inactive keys at 0.5; a union is its tag then every
  option's coordinates, the unchosen options at 0.5, the same layout as `flatten`), `encodedDim`, `grid` enumeration
  (tuples, nested and conditional dicts, a union option by option), `neighbours` (one-move neighbourhoods for local
  search over configurations, repairing conditional dimensions; a union's tag moves to each other option at its
  defaults), `distance`.
- `_active.ts` (internal): resolution of conditional dict keys shared by the operations; `_union.ts` (internal): the
  tag, option and rest of a union value.

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
- `subspace.ts`: `subspaceIterationAlgorithm` (a block power iteration, one block product and QR per step, with the
  Rayleigh-Ritz values and residuals), `subspaceIteration`.
- `svd-steps.ts`: `jacobiSvdAlgorithm` (one-sided Jacobi, a step is a rotation or a sweep; the off-diagonal measure is
  the largest column cosine), `jacobiSvd`, `jacobiSingularValues`.
- `completion.ts`: matrix completion from observed entries (`ObservedEntries`): `alsCompletionAlgorithm` (one
  half-sweep of ridge solves per step), `sgdCompletionAlgorithm` (biased factorisation, one epoch per step),
  `implicitAlsAlgorithm` (confidence-weighted implicit feedback), `softImputeAlgorithm` and `softImpute` (iterated
  soft-thresholded SVD), `softImputeLambdaMax`, `predictEntry`, `factorsToMatrix`.

`sparse/` (scipy.sparse)
- `csr.ts`: `CsrMatrix`, `csr`, `fromCoo`, `toDense`, `spmv`, `spmm`, `transposeCsr`, `diagCsr`, `csrAdd`,
  `csrDiagonal` (differentiable in the values), `LinearOperator` and `asOperator`.
- `iterative.ts`: `cg`, `bicgstab`, `gmres` (restarted), `minres`, each also an `Algorithm` (`cgAlgorithm`, ...) with
  traced residuals; preconditioners (`jacobiPreconditioner`, `incompleteCholesky`); `sparseSolve` (custom rules by
  adjoint solves).
- `eigen.ts`: `lanczos` (full reorthogonalisation, deflation), `arnoldi`, `eigsh` (top k, with locking for repeated
  eigenvalues), `svds`.
- `lanczos-steps.ts`: `lanczosAlgorithm` (one Krylov vector per step, with the Ritz values and residuals, and a switch
  for reorthogonalisation to watch the basis lose orthogonality), `lanczosSteps`.
- `stationary.ts`: `stationaryAlgorithm` (Jacobi with damping, Gauss-Seidel, SOR; one sweep per step), `stationarySolve`,
  `jacobi`, `gaussSeidel`, `sor`, `iterationMatrix`, `stationarySpectralRadius`, `sorOptimalOmega`.
- `multigrid.ts`: `poisson1d`, `poisson2d`, `prolongation` (linear and bilinear), `multigridAlgorithm` (two-grid, V and W
  cycles with Galerkin coarse operators and a stationary smoother; the residual after each stage of a cycle),
  `multigridCycle` (the iterate after each stage, for the error by frequency), `multigridSolve`.

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
- `floating-point.ts`: `ulp`, `nextAfter`, `machineEpsilon` (for float64, float32, float16, bfloat16), `FLOAT_FORMATS`, `toFloat32`, `toFloat16`, `toBfloat16`,
  `roundToFormat` (emulated rounding with subnormals and overflow), `twoSum`, `twoProduct`, `compensatedSum`
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
- `distances.ts`: `distanceBetween` (any of the metrics over the coordinate axis of broadcast operands: rows against
  rows, two points, or `[m, 1, d]` against `[1, n, d]`), `cdist`, `pdist`, `squareform` built on it, and
  `resolveMetricOptions` (scipy's data-dependent defaults for seuclidean and mahalanobis), with the metrics euclidean,
  sqeuclidean, cityblock, chebyshev, minkowski (with weights), seuclidean, cosine, correlation, hamming, jaccard,
  mahalanobis, canberra, braycurtis and haversine (continuous ones differentiable); `metricFunction` is the scalar form
  on number arrays. `lib/distances/` names these under one vocabulary with their parameters as spaces.
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
- `streams.ts`: the stream base: data arriving one batch per step with planted drift. `stream(key, options)` is a
  `DataStream` (a classification or a regression generator; the concept, covariate or label `shift`; `drift` abrupt,
  gradual, incremental or recurring, with `points` or an even `count`, a `width`, a `period`, `concepts` and a
  `severity`) whose `next(t)` is a pure function of the key and the step (random access), and whose `StreamTruth`
  carries the drift points (`changepoints`, as `Truth` names them), the `PlantedDrift`s, the active concept per step
  (`conceptAt`, `transitionAt`), each concept's `parameters` and the rule it gives a point (`predict`). `streamKnobs`
  is the stream's knobs as a `Space` (for a later registry entry); `streamAlgorithm(options)` runs the same stream as
  a serialisable `Algorithm` (`StreamState`: key, step, last batch) that logs `stream/concept`, `stream/mix`,
  `stream/drift`, `stream/x` and `stream/y` (`streamContract`).
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
- `modifiers/shift.ts` also has `withSubpopulationShift` (a group under-represented in the source, with the density ratio) and
  `withShortcut` (a feature tracking the label at the source and independent of it at the target).
- `modifiers/leaks.ts`: `withLeak` (a noisy copy of the target, or a target encoding fitted on all rows) and
  `withDuplicates` (exact or jittered copies shuffled in, with the source row of each as truth).
- `modifiers/images.ts`: image augmentations with their draws kept as truth: `withRandomCrop` (padding, offsets), `withRandomFlip`,
  `withCutout`, `withMixup` and `withCutmix` (weights, partners and soft labels), each also a recipe modifier.

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
- `spreading.ts`: things that spread over a network: `networkEpidemicAlgorithm` (SIR, SIS or SEIR by Gillespie's exact
  method, one event per step, with vaccinated nodes and a horizon), `independentCascadeAlgorithm` and
  `linearThresholdAlgorithm` (one wave per step), and the results known about them: `sirTransmissibility`,
  `epidemicFinalSize` (Newman's generating-function relation for any degree sequence), `homogeneousFinalSize`,
  `epidemicThresholds` (spectral, heterogeneous mean-field and percolation), `wattsCascadeCondition`, `vaccinationSet`
  (random, targeted, acquaintance), `herdImmunityThreshold`.
- `influence.ts`: influence maximisation: `liveEdgeSamples` (independent cascade or linear threshold as reachability in
  sampled live-edge graphs), `liveEdgeReach`, `estimateSpread`, `exactIndependentCascadeSpread` (enumeration, the
  truth for small graphs) and `influenceMaximisationAlgorithm` (greedy, CELF or highest degree, one seed per step with
  its marginal gain and the evaluation count on the tape).
- `opinion.ts`: opinion dynamics: `voterModelAlgorithm` with `voterFixationProbability` (the degree-weighted share),
  `degrootAlgorithm` (DeGroot and Friedkin-Johnsen with stubbornness) with `degrootConsensus` and
  `friedkinJohnsenEquilibrium`, and `boundedConfidenceAlgorithm` (Hegselmann-Krause on a graph).
- `_run.ts`: the state codec, seed resolution and argument checks the stepped processes share.
- `percolation.ts`: `percolationAlgorithm` (site or bond, any graph, one site or bond per step, clusters by union-find,
  a spanning cluster between two sides), `sitePercolationAlgorithm`/`bondPercolationAlgorithm` on a square lattice,
  `latticeSides`, `spanningCluster`, `percolationSweep`, `spanningThresholds` (finite-size scaling),
  `PERCOLATION_THRESHOLDS` (square, triangular, honeycomb, cubic), `erdosRenyiGiantFraction`, `giantComponentFraction`.
  `structures.ts`'s `UnionFind` gained `clone`, `toJSON` and `fromJSON` for this.
- `growth.ts`: networks grown a step at a time, with their known results: `edgeByEdgeAlgorithm` (Erdos-Renyi, with
  `erdosRenyiConnectivityEdges`), `preferentialAttachmentAlgorithm` (preferential or uniform, with
  `barabasiAlbertDegreeLaw`, `uniformAttachmentDegreeLaw`), `rewiringAlgorithm` (Watts-Strogatz, with
  `wattsStrogatzClustering`, `ringLatticePathLength`, `smallWorldSweep`), `blockGrowthAlgorithm` (with
  `blockModelExpectedEdges`), `configurationModelGraph`, `growthGraph`, `unrankPair`. `measures.ts` gained
  `averageShortestPathLength`.
- `resistance.ts`: electrical flow on a graph: `effectiveResistance` (all pairs, from the Laplacian pseudoinverse per
  component), `effectiveResistanceBetween` (grounded solve), `commuteTime`, `kirchhoffIndex`,
  `edgeEffectiveResistances` (Foster's theorem, spanning-tree probabilities), `spanningTreeCount` (matrix-tree).
- `sampling.ts`: sampling from a graph: `neighbourSample` (GraphSAGE fan-out sampling with unbiasedness weights, the
  layers, per-hop edges and the sampled subgraph) with `neighbourSampleVariance`, `layerwiseImportanceSample`
  (FastGCN and LADIES) with `sampledAggregate`, `randomWalks`, `node2vecWalks` with `node2vecTransition` (the exact
  second-order probabilities), `randomWalkSubgraph` (GraphSAINT).

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
- `bijectors/elementwise.ts`: `identity`, `exp`, `softplus`, `sigmoid`, `tanh`, `affine`, `power`, `normalCdf`,
  `sinhArcsinh` (Jones-Pewsey skew and tail weight), `kumaraswamyCdf`.
- `bijectors/structured.ts`: `ordered`, `stickBreaking`, `correlationCholesky`, `lowerCholesky`,
  `choleskyOuterProduct`, `affineCoupling`, `permute`, `autoregressive` (MAF-style), `madeMasks`.
- `bijectors/flows.ts`: the layers of normalising flows beyond coupling: `planar` and `radial` (inverted by bisection with a
  Newton step for the gradients), `actNorm` and `actNormInitialise`, Glow's `invertibleLinear` and `luLinear` (with
  `luParameters`: O(d) log-determinant), `inverseAutoregressive` (IAF), `madeParameters` and `madeConditioner` (the MADE
  network for `autoregressive` and IAF), the rational-quadratic spline of neural spline flows (`rationalQuadraticSplineMap`,
  `rationalQuadraticSpline`, `rationalQuadraticCoupling`, `splineParametersFromVector`), and `residualFlow` (fixed-point
  inverse; log-determinant by the power series of traces, exact or Hutchinson) with `lipschitzNormalise`.
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
- `families/extended.ts`: `ExtendedFamily`, a response given by the derivatives of its log-likelihood in one or several
  linear predictors (`derivatives`: log-likelihood, gradient, packed Hessian by `hessIndex`, optional third
  derivatives) with `aux` parameters (beta precision, negative binomial dispersion, ordinal cut points, Tweedie power) on an
  unconstrained scale; `tensorExtendedFamily` builds one from a per-observation log-likelihood over tensor operations
  (gradient by reverse mode, the Hessian blocks by K Hessian-vector products), `gaussianLocationScale` writes its
  derivatives out with the third ones, and `betaRegression`, `negativeBinomialTheta`, `zeroInflatedPoissonLogit`,
  `orderedCategorical`, `multinomialLogit` and `tweedieEstimated` use the tensor route.
  `families/extended-distributional.ts`: `gamlssExtended` (normal, gamma, negative binomial, Student-t, log-normal) with
  starting values from the moments.

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
  adaptive frequency models; each coder also as a traced algorithm, one symbol per step (`arithmeticEncoderAlgorithm`,
  `arithmeticDecoderAlgorithm`, `rangeEncoderAlgorithm`, `rangeDecoderAlgorithm`) with the working interval, the bits
  against the information content and the adaptive counts on the tape, the output identical to the whole-message
  functions. `source-coding.ts` gained `prefixEncoderAlgorithm`.
- `sources.ts`: sources with their truth: `memorylessSource` (entropy, the sample's information content),
  `markovSource` (stationary law, entropy rate).
- `bottleneck.ts`: the information bottleneck: `informationBottleneckAlgorithm` (Tishby's self-consistent iteration at one
  beta), `bottleneckCurveAlgorithm` (one beta per step, the information curve), `agglomerativeBottleneckAlgorithm`
  (merges by relevance cost), `bottleneckCriticalBeta` (1 / lambda_2), `plantedBottleneck` (a joint with the optimal
  clustering, I(X; Y) and H(G) as truth).
- `mdl.ts`: minimum description length: `multinomialParametricComplexity`, `nmlRegret`, `nmlRegretAsymptotic`,
  `nmlCodeLength`, `dirichletCodeLength` (KT and Laplace), `parameterBits`, `gaussianDataBits`, `polynomialMdl` (total =
  BIC in bits), `histogramMdl`, `bestMdl`, and `mdlSelectionAlgorithm` (one candidate per step).
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
  `isDiscrete`; minimal realisation (`minreal`: staircase reductions, pole-zero cancellation); `pade` (the rational
  approximation of a delay, python-control's).
- `conversions.ts`: scipy's pairwise conversions: `normalize`, `abcdNormalize`, `tf2ss` (controller canonical form),
  `ss2tf` (differentiable through `charpoly`), `tf2zpk`, `zpk2tf`, `zpk2ss`, `ss2zpk`.
- `sections.ts`: second-order sections: `zpk2sos` (nearest, keep-odd and minimal pairing; analog), `sos2zpk`,
  `sos2tf`, `tf2sos`.
- `interconnect.ts`: `series`, `parallel`, `feedback`, `append`, `connect` (python-control semantics; static gains
  accepted; transfer functions combine as polynomials, multivariable systems in state space).
- `responses.ts`: impulse, step, initial-condition and arbitrary-input responses (`lsim` and the traced
  `lsimAlgorithm`, zero- and first-order hold, continuous and discrete), `defaultResponseTimes`; `stepMeasures` (on
  sampled data) and `stepInfo` (on a system): rise time, settling time, overshoot, undershoot and peak, as
  python-control's `step_info`.
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
  `describe`, `zscore`, `sem`, `mode`, `geometricMean`, `harmonicMean`, `gini`.
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
- `tests/timeseries.ts`: `ljungBox`, `durbinWatson`, `adfuller` (with `mackinnonp`, `mackinnoncrit`, which take the number
  of integrated variables N for the Engle-Granger tables; the private `tests/_mackinnon.ts` holds the 1994 surfaces and
  2010 critical values), `kpss`, `runsTest`.
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
  `ti`, with `placeKnots` (quantile and uniform rules). `SmoothBasis.derivative(x, order)` (optional) is the exact first
  and second derivative of the basis functions in the covariate for the one-covariate bases with a closed form: `bs` and
  `ps` through `smoothing/basis-derivatives.ts` (`bsplineDerivativeDesign`, the exact piecewise polynomials of the
  identity-coefficient B-spline), `cr` and `cc` from the cubic-spline formulas, `tp`, `ts`, `duchon` and `gp` from the
  kernels' derivatives in the distance; `ns`, `re`, `fs`, `te` and `ti` have none.
- `smoothing/penalties.ts`: `differencePenalty`, `derivativePenalty`, `nullSpace`, `shrinkagePenalty`,
  `shapeConstraint` (increasing, decreasing, convex, concave reparameterisations), `sumToZeroConstraint`,
  `constraintNullSpace`.
- `smoothing/smoothers.ts`: `loess`, `lowess`, `nadarayaWatson`, `localLinear`, `runningStatistic`, `runningMean`,
  `runningMedian`, `smoothingSpline` (lambda, degrees of freedom or GCV over numerics' `makeSmoothingSpline`).
- `fit.ts`: `fitDistribution` (closed form or Newton for seventeen families) with standard errors from the
  `observedInformation`; `newtonMle` (traced `newtonMleAlgorithm`).
- `experiments.ts`: online-experiment estimators: `cuped` (pre-experiment covariate adjustment), `postStratify`,
  `regressionAdjustment` (Lin's demeaned interaction estimator with HC2 errors), `deltaMethodRatio` (user-level ratio
  metrics, one and two sample), `clusterRandomisedTest` (ICC, design effect, CR1 errors) and `switchbackAnalysis`
  (period means, Newey-West errors, optional randomisation p-value). Results carry an `Estimate` and a `TestResult`.
- `design.ts`: randomisation (`simpleRandomisation`, `permutedBlockRandomisation`, `stratifiedRandomisation`,
  Pocock-Simon `minimizationAlgorithm`/`minimizationAllocation` as a traced `Algorithm` whose draws come from the step
  key), `matchedPairsDesign` (exact matching up to 16 units), `fullFactorialDesign`, `fractionalFactorialDesign` and
  `aliasStructure` (defining relation, resolution, alias chains), `latinSquareDesign`, `oneWayAnova` and `twoWayAnova`
  (Type I, II and III sums of squares; golden against statsmodels `anova_lm`).
- `shrinkage.ts`: `jamesStein`, `positivePartJamesStein` (targets zero, a vector or the grand mean; SURE) and
  `empiricalBayesNormalMeans` (homoscedastic Efron-Morris, heteroscedastic DerSimonian-Laird moments; the
  heteroscedastic fit is golden-tested against statsmodels `combine_effects`).
- `missing.ts`: `missingnessProfile` (columns, rows, patterns) and `littleMcarTest` (Gaussian EM moments, chi-square
  over the pattern means; tested against the closed-form monotone-pattern maximum likelihood, no reference library
  implements it).
- `concentration.ts`: `markovBound`, `chebyshevBound`, `chernoffBound`/`chernoffGaussianBound`, `chernoffKlBound`,
  `hoeffdingBound` with `hoeffdingSampleSize`, `bernsteinBound`, `bennettBound`, `empiricalBernsteinInterval`,
  `mcDiarmidBound` and the `dkwBand`.

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
- `schedule-table.ts`: `schedules`, the schedules as an option table (constant, step, exponential, inverse-time,
  cosine, restarts, linear, polynomial, warm-ups), each taking the optimiser's learning rate as its peak, with
  properties (warm-up, decays, restarts) for validity rules.
- `optimiser-table.ts`: `optimisers`, the optimisers as an option table (SGD with momentum and Nesterov, RMSprop,
  Adagrad, Adadelta, Adam, AdamW, AMSGrad, Adamax, NAdam, RAdam, Lion), each a shape that takes a learning rate or a
  schedule, with its own knobs and properties (adaptive, momentum, decoupled decay).
- `stochastic.ts`: `gradientAlgorithm`, a transformation-driven descent over an objective that may be summed over a
  batch (`BatchObjective`, `Batches`): the optimiser at a scheduled rate, a batch of `batch` examples drawn without
  replacement from each step's key (null: the whole set), a serialisable `GradientState` (iterate, value, gradient,
  optimiser state, step count, last rate); the engine of the `gradient` mode.
- `slot.ts`: `familySolvers` (a model's own modes beside every generic solver, over the model's `SmoothProblem`),
  `slotAlgorithm` (a mode of such a table as the model's `Algorithm`, state serialisable, `solver/gradNorm` and
  `solver/step` beside the model's channels) and `newtonAt`; used by `timeseries` and `pgm` models (the `linear` and
  `kernel-methods` folders keep private copies of the wrapper).
- `solvers.ts`: `smoothSolvers`, the smooth solvers as one slot of modes (gradient descent with its step rules,
  conjugate gradient, Newton, BFGS, L-BFGS, coordinate descent and `gradient`, whose knobs are `optimiser`
  (`optimisers`), `learningRate`, `schedule` (`schedules`), `batch` (0: the whole set) and `gradTol`), over a
  `SmoothProblem` (an objective that may take a batch index set, `batches`, a start); `loggedSolver` (logs
  `solver/objective`, `solver/gradNorm`, `solver/step` and, for `gradient`, `solver/lr`; serialises the state),
  `solverContract`, `gradientContract`, `solverDefaults`, `gradientDefaults`; `loggedSolverOf` logs by the same
  convention for any table whose states carry `x`, `value`, `grad` and `step`.
- `glm.ts`: the smooth penalised GLM as a problem and a slot. `GlmProblem` (design, family, link, prior weights,
  offset, intercept, `GlmPenalty`: none, ridge, or a general quadratic penalty matrix as an additive model needs) and
  `glmSolvers`, the table of its solvers: `irls` (Fisher scoring with step halving, the weighted problem solved by
  Cholesky) and, from `smoothSolvers` on the GLM objective (`glmSmoothProblem`), `newton` (observed information),
  `bfgs`, `lbfgs`, `conjugateGradient`, `gradientDescent` and `gradient` (the stochastic mode with its optimiser,
  schedule and batch). `fitGlm` runs a mode to its fit; `glmValue`, `glmGradient` evaluate. Non-smooth penalties
  (lasso, elastic net, group lasso) are deliberately not in the table: they are particular algorithms (`linear/lars.ts`,
  `paths.ts`, `group-lasso.ts`, `optim/sparse.ts`, `proximal.ts`). Home: `optim`, not `linear`, because `optim` may
  import `dists/families` without a cycle (`dists` imports neither `optim` nor `linear`) and the table serves `linear`,
  `gam` and any future GLM-shaped model alike. A model whose objective is not GLM-shaped builds a `SmoothProblem` and
  takes `smoothSolvers`; a model that is a GLM takes `glmSolvers`; the model never carries a solver of its own.
- `analytic.ts`: `analyticObjective`, an objective written over plain numbers with its analytic gradient (and Hessian)
  as a differentiable function the solvers accept (the gradient is its reverse rule, the Hessian its forward rule), and
  `finiteDifferenceHessian` for the Newton mode of a model that has a gradient only.
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
  (C-SVC with per-sample C) and `svrDual` (epsilon-SVR over 2n variables), `oneClassDual` and `svddDual`; `smoRho`.
- `dual-cd.ts`: dual coordinate descent for the linear SVM (`linearSvmDualCdAlgorithm`, `linearSvmDualCd`; Hsieh et al.,
  liblinear's algorithm) for the hinge and squared-hinge losses, random sweeps with shrinking, the primal, dual and
  duality gap on the tape. It sits here beside SMO because `svm` imports `learn`; an SVM estimator takes it as a mode of its
  solver table. oracle: `optim/dual_cd` against `sklearn.svm.LinearSVC(dual=True)` (cases where liblinear hits its
  iteration cap are refused); also cross-checked in a test against `svcDual` on a symmetrised dataset (primal objective).
- `online-experts.ts`: prediction with expert advice as rules run by `expertsAlgorithm` with the regret on the tape (and
  against the best k-switching sequence by `bestSwitchingLoss`): weighted majority (deterministic and randomised, with the
  mistake bounds), follow the leader, follow the perturbed leader, Hedge, fixed share, variable share; `exp4Algorithm`
  (bandit feedback, Exp3 as identity advice, `exp4RegretBound`); the p-norm mirror descent and AdaGrad (box constraints,
  `adagradRegretBound`) as `OnlineLearner`s of `online.ts`; `hindsightComparator`.
- `smo-nu.ts`: libsvm's Solver_NU (`nuSmoAlgorithm`: two equalities, pairs within a class), the feasible starts
  (`nuSvcStart`, `nuSvrStart`), `nuRho`, and the `nuSvcDual` and `nuSvrDual` duals.
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
- `landscape.ts`: loss landscapes after Li et al.: `filterNormalise` and `randomDirection` (per-filter, per-layer or no
  normalisation; biases zeroed or normalised; any filter axis), `slice1d`, `slice2d` and `randomSlice` (losses along one
  or two directions, with the directions returned), `interpolationSlice` and `bezierSlice` (the line and a quadratic
  Bezier path between two minima) with `barrier`, and `sharpness` (the largest Hessian eigenvalue by Lanczos through
  `eigsh` or by power iteration, on Hessian-vector products from `flatObjective`) with `stabilityThreshold` (2 / eta).
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
- `named.ts`: kernels chosen by scikit-learn's names (`KernelChoice`: a metric name or a `Kernel`, `gamma` of `'scale'`
  or `'auto'` or a number, `degree`, `coef0`), `resolveKernel` against the training rows into plain JSON,
  `kernelMatrix`, and `kernelSpace` (the choice as knobs).
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
- `language-models.ts`: n-gram language models (Chen and Goodman 1998): counts of padded n-grams, the smoothers as a
  mode (maximum likelihood, add-k, Witten-Bell, absolute discounting, interpolated and modified Kneser-Ney with
  continuation counts, Katz backoff over Good-Turing discounts), `<unk>` for rare words, probabilities, whole
  next-word distributions, perplexity and samples; `ngramTrainingAlgorithm` counts one sentence per step with the
  held-out perplexity so far. Oracle: Chen and Goodman's definitions written directly in Python (NLTK's `lm` models
  differ in convention, see `oracle/cases/text/language_models.py`).
- `keywords.ts`: RAKE (reproduces the scores of the RAKE paper's worked abstract), the TextRank word graph and keywords
  with phrase merging (PageRank from `graph`), the sentence graph with Mihalcea and Tarau's overlap similarity and
  extractive summaries.
- `linkage.ts`: record linkage: standard and sorted-neighbourhood blocking, field comparison into agreement levels,
  the Fellegi-Sunter model fitted by EM on the comparison patterns (`fellegiSunterAlgorithm`, one iteration per step,
  serialisable), match weights and posteriors, pairwise precision and recall, entity clusters from the matched pairs.
  Tested on `linkageTables`, where the EM recovers the planted agreement rates.
- `spelling.ts`: the noisy-channel spelling corrector: candidates from a BK-tree, a channel learned from (typed,
  intended) pairs (weighted Damerau-Levenshtein alignment with counted operation probabilities) or uniform, the prior a
  word-frequency table or an n-gram language model reading the corrected context, whole sentences left to right.
- `retrieval.ts`: a positional inverted index (`invertedIndex` with `add` per document, `indexingAlgorithm` one
  document per step), the scoring models as a mode over its postings (TF-IDF cosine, BM25 with Robertson or Lucene idf,
  the binary independence model with optional relevance judgements, query likelihood under Dirichlet, Jelinek-Mercer or
  absolute-discount smoothing), each query term's contribution to a score, `search` (top k), phrase and proximity
  queries from the positions, boolean queries. BM25 over the postings equals `features.ts`'s rank_bm25 scores in the
  tests; the other models are checked against their hand formulas and the smoothed models against normalisation.

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
- `diffusion-models.ts`: diffusion models as generative models (the SDE solvers are `sde.ts`): noise schedules with alpha,
  sigma, log-SNR and the forward SDE's drift and squared diffusion (`linearSchedule`, `cosineSchedule`,
  `sigmoidSchedule`, `discreteSchedule` with `linearBetas` and `cosineBetas`, `veSchedule`, `edmSchedule`,
  `defineSchedule`), `vpSde` and `veSde` (closed-form marginals, as `Sde`s for `reverseSde` and `probabilityFlow`),
  `samplerTimes` (uniform, quadratic, log-SNR, Karras spacing), `forwardSample` (q(x_t | x_0)), predictions and their
  conversions (`toX0AndNoise`: epsilon, x0, v, score), training losses with weightings (`x0ErrorWeight`,
  `weightedDiffusionLoss`: simple, SNR-plus-one, truncated and min-SNR, ELBO, EDM; `diffusionBatch`,
  `edmPreconditioning`), guidance (`classifierFreeGuidance`, `guidedScore`, `guidedNoise`, `guidedModel`), and
  `gaussianMixtureTruth`, the exact noised score of a Gaussian mixture, a planted truth for samplers and networks.
- `diffusion-samplers.ts`: reverse samplers as stepped `Algorithm`s with typed paths (`sampler/xt`, `x0Hat`,
  `noiseLevel`, `time`, `step`, `cloud`): `diffusionSampler` (DDPM ancestral, DDIM with eta, probability-flow Euler and
  Heun, DPM-Solver-2), `ddpmSampler`, `ddimSampler`, `sampleDiffusion`, `samplerProperties`.
- `diffusion-training.ts`: `scoreNetwork` (an MLP over x_t and Fourier time features, from `nn`), `networkModel`,
  `diffusionTrainer` (the training loop as an `Algorithm` over `nn`'s `TrainState`), so a toy mixture is trained and
  sampled end to end.
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
- `variance-reduction.ts`: `monteCarloEstimate` (the baseline), `controlVariateEstimate` (one or several controls, fitted
  coefficients and multiple correlation), `antitheticEstimate` (uniform and normal reflection, pair correlation),
  `stratifiedEstimate` with `uniformStrata` (proportional and Neyman allocation, pilot or known spreads),
  `raoBlackwellEstimate` (with the law-of-total-variance split), `quasiMonteCarloEstimate` (scrambled Sobol or shifted
  Halton replicates), each an `Estimate` with the plain standard error for the same budget and the reduction factor;
  `varianceReductionAlgorithm` runs plain, antithetic, control-variate and stratified side by side, one batch per step
  (`varianceReductionEstimates` reads the state).
- `gradient-estimators.ts`: gradients of `E_{x ~ p_theta} f(x)` in the parameters: `scoreFunctionGradient` (baselines:
  none, batch mean, leave-one-out, constant), `pathwiseGradient` (through `rsample`), `straightThroughGradient` (over
  `dists`' `straightThrough`), `gumbelSoftmaxGradient` (over `relaxedOneHotCategorical`, optionally hard), each returning
  the mean, the single-sample variance and the standard error per parameter; `gradientError` against an exact gradient.
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
- `production.ts`: production systems: rules over a working memory of ground facts (`productionRule`, `not(...)`
  conditions, facts added and removed, `halt`), conflict resolution by rule order, priority, recency (OPS5's LEX) or
  specificity (`resolveConflict`, ties broken by the matched facts so every matcher fires the same instantiation), a
  naive matcher and `ReteNetwork` (alpha memories and beta memories of partial matches, updated on `add` and `remove`; a
  property test checks that both matchers give identical runs on random rule bases), and `productionSystemAlgorithm`
  (one firing per step, the working memory, the conflict-set size and the rule fired with its bindings in the state).
- `uncertain.ts`: MYCIN certainty factors (`combineCertainty`, `certaintyInference`), Dempster-Shafer belief functions
  (`massFunction`, `belief`, `plausibility`, `combineDempster` with its conflict, `combineYager`, `pignistic`,
  `evidenceAlgorithm` adding one body of evidence per step with the belief intervals), fuzzy logic (membership functions,
  t-norms and t-conorms, `mamdani` and `sugeno` inference, `defuzzify` with centroid by exact integration of the
  piecewise-linear curve, bisector and the means/smallest/largest of the maxima) and `possibilityMeasures`.
- `argumentation.ts`: Dung frameworks: conflict-free and admissible sets, the characteristic function, grounded,
  complete, preferred and stable extensions (`extensions`, `accepted` sceptically or credulously), and
  `groundedAlgorithm`, the grounded labelling one round per step (in, out, undecided).
- `description.ts`: ALC: `parseConcept`, `nnf`, `alcTableauAlgorithm` (one rule per step: and, all, some, or with
  backtracking, clashes, subset blocking, a TBox of general concept inclusions internalised), `conceptSatisfiable`,
  `conceptSubsumes` (unsatisfiability of the difference) and `classifyConcepts` (the direct-parent hierarchy of named
  concepts).
- `problog.ts`: ProbLog-style probabilistic logic programs: `groundProbabilistic` (forward-chaining grounding of
  function-free definite rules over probabilistic facts `p::f`), `proofSets` (minimal proofs of every atom to a fixed
  point, so cycles terminate), `successProbability` (exact weighted model counting by memoised Shannon expansion, with
  the proofs and their probabilities exposed), `conditionalProbability` (evidence true or false) and `probLogEmAlgorithm`
  (EM for the facts' probabilities from observed interpretations, one iteration per step). Checked against enumeration of
  all possible worlds on random cyclic graphs.
- `markov-logic.ts`: Markov logic networks: `groundMln` (weighted first-order formulas over a finite domain into ground
  atoms and ground formulas; free variables universally quantified, quantifiers expanded, `=` decided; a hard formula has
  infinite weight), `mlnExact` (the exact distribution by enumeration: log partition, marginals, query probabilities,
  expected counts; the truth the samplers are tested against), `mlnSampleWorlds` (exact samples of complete worlds, the
  generator with its planted weights as truth), `mlnGibbsAlgorithm` (one sweep per step), `mlnMcSatAlgorithm` (slice
  sampling; the slice by `sampleSat`, Wei-Erenrich-Selman's near-uniform SampleSAT, or exactly by enumeration) with
  marginals, query probabilities and energy on the tape (`MlnSamplerPath`, `mlnSamplerContract`), and `mlnMap` (MAP as
  weighted partial MaxSAT through `maxsat.ts`).
- `mln-learning.ts`: weight learning: `mlnPseudoLikelihoodProblem` (a convex logistic regression on the changes in each
  formula's true-grounding count, as a `SmoothProblem` with value, gradient and Hessian), `mlnLikelihoodProblem` and
  `mlnConditionalLikelihoodProblem` (exact, by enumeration), `learnMlnWeights` and `mlnPseudoLikelihoodAlgorithm` (the
  solver is a slot of `smoothSolvers`, L-BFGS by default; recovers planted weights), and the discriminative voted
  perceptron `mlnPerceptronAlgorithm` (MAP counts from `mlnMap`, or exact expected counts; the weights averaged over
  the passes).
- `fuzzy-logic.ts`: fuzzy logic over tensors in the manner of Logic Tensor Networks: t-norm families as differentiable
  tensor ops (`normFamily`: product, Lukasiewicz, Goedel, Hamacher with their t-conorms, standard negation, residuated
  and material implications, folds along an axis), aggregators for the quantifiers (`pMean`, `pMeanError`, the extremes,
  a t-norm), `formulaSatisfaction` (propositional formulas over a network's outputs), learnable predicates
  (`tablePredicate` over constants, `networkPredicate` on features), and `fuzzyTheory` (first-order axioms evaluated to
  tensors with one axis per free variable, satisfaction as the objective, a `SmoothProblem` for the solver slot) with
  `fuzzyTheoryAlgorithm` and `trainFuzzyTheory`.
- `neural-theorem-prover.ts`: the neural theorem prover of Rocktaschel and Riedel: symbol embeddings, soft unification by
  a Gaussian kernel, backward chaining to a depth over facts and rules (including `#template` predicates whose vectors
  are learned), proof score the maximum over proofs of the minimum of the unification scores, all proofs batched as
  tensors (`neuralTheoremProver`, `ntpTrainingSet` with leave-one-out positives and corrupted negatives, `trainNtp` with
  restarts, `ntpAlgorithm` over the solver slot, `decode` of a template to the nearest predicate). With crisp embeddings
  the network is classical backward chaining (tested to the proof depth). Goals with an unbound variable are proved
  against facts only; a template's rule is not reliably induced from scratch at this scale (see the test notes).
- `diff-ilp.ts`: differentiable ILP (Evans and Grefenstette): clause templates over a language, candidate clauses with
  precomputed groundings, a softmax over each template's candidates, forward-chained valuation (product conjunction,
  maximum over existential groundings, previous valuation amalgamated by maximum) as tensor ops, the example
  cross-entropy as a `SmoothProblem` (`diffIlp`, `diffIlpAlgorithm`, `learnProgram` with restarts), `extract` of the
  program by taking each template's most probable clause and a crisp forward chainer (`derive`, `accuracy`) to check it.
  Learns the predecessor, graph reachability (recursive) and, with restarts, the even numbers with an invented auxiliary.
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
  order; energy-drift diagnostics; `separableField` (the Hamiltonian vector field on stacked positions and momenta, so
  a Runge-Kutta method can run a Hamiltonian system beside the splittings).
- `ode/flow.ts`: `flowAlgorithm`, a field integrated at a fixed step by an explicit Runge-Kutta method as an
  `Algorithm` whose state is plain numbers (serialises, seeks, resumes), with an observer hook for the paths a system
  logs and a stopping rule (the base of the celestial, named-system and geodesic algorithms below).
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
- `named-systems.ts`: named systems as vector fields in tensor operations (so they integrate and differentiate):
  `lotkaVolterraField`, `lorenzField`, `vanDerPolField`, `dampedPendulumField`, `sirField`, `rosslerField` (with
  `rosslerFixedPoints`), `thomasField` (constant divergence -3b) and the double pendulum from `mechanics.ts`, and
  `namedSystems`, each with a start, a time span that shows its behaviour and its component names (what
  `methods/dynamics/odeSystems.ts` runs).
- `mechanics.ts`: the Euler-Lagrange equations derived by autodiff from a Lagrangian (`lagrangianField`,
  `lagrangianEnergy`), the Legendre transform (`hamiltonianOfLagrangian`) and Hamilton's equations (`hamiltonianField`)
  for non-separable systems; the double pendulum worked through (`doublePendulumLagrangian`, the closed-form
  `doublePendulumField` the derivation must reproduce, `doublePendulumEnergy`, `doublePendulumHamiltonian`,
  `doublePendulumFlipTime` for the flip map). Oracle: scipy.integrate (`dynamics/mechanics`).
- `synchronisation.ts`: Kuramoto oscillators (`kuramotoField` all-to-all in mean-field form or on a graph,
  `orderParameter`, `kuramotoFrequencies` quantile or sampled populations, `kuramotoCriticalCoupling` and
  `kuramotoTheoreticalOrder` as the closed-form truth, `kuramotoAlgorithm` logging the order parameter and the locked
  share) and the consensus flow (`consensusField`, whose rate is the algebraic connectivity).
- `celestial/` (the celestial-mechanics namespace, `mdml-engine/dynamics/celestial`; the chapter tables call it
  `dynamics/celestial.ts`, it is a folder because it holds five concerns; oracle: scipy.optimize and scipy.integrate):
  - `kepler.ts`: the anomalies (mean, eccentric, hyperbolic, parabolic, true) and their conversions; Kepler's equation
    solved by Newton, Halley, bisection, secant and fixed-point iteration as a stepped `Algorithm` (`keplerAlgorithm`:
    iterate, residual, error against the root and correct digits per step, serialisable) and as a table of modes
    (`keplerSolvers`, the slot of the `keplers-equation` method); the hyperbolic form M = e sinh H - H; Barker's
    equation in closed form; the Fourier-Bessel and Lagrange (through e^5, Laplace's limit) series; `solveKepler` on
    any conic.
  - `elements.ts`: classical elements to and from state vectors (ellipse, parabola, hyperbola; the circular and
    equatorial degenerate conventions documented), the analytic propagation `propagateKepler`/`keplerState` (the truth
    for the integrators), vis-viva, period, escape speed, the Laplace-Runge-Lenz invariants.
  - `nbody.ts`: the n-body Hamiltonian as a `SeparableSystem` (vectorised force over tensor operations, softening,
    momentum, angular momentum, centre of mass, closest pair), `nBodyStep` and `nBodyAlgorithm` with the integrator as
    a choice (the symplectic splittings of `ode/symplectic.ts` against Runge-Kutta) logging energy and its drift,
    angular momentum and the closest approach, and a `bodies/close` event.
  - `configurations.ts`: `twoBody` (launch speed to circular, elliptic, parabolic or hyperbolic, with period and energy),
    `lagrangeTriangle`, `FIGURE_EIGHT`, `PYTHAGOREAN`, and `solarSystem` from the planets' mean elements (`PLANETS`,
    `GAUSS_K`), each carrying its truth.
  - `three-body.ts`: the circular restricted three-body field (planar and spatial), pseudo- and effective potential,
    the Jacobi constant, `lagrangePoints` and their stability (`classifyLagrangePoints`, Routh's critical mass ratio),
    `zeroVelocityCurves`, and `restrictedThreeBodyAlgorithm` logging the Jacobi constant and its drift.
  - `geodesics.ts`: null geodesics of the Schwarzschild metric in the Binet form (`schwarzschildNullGeodesicField`,
    `nullInvariant`), the exact `deflectionAngle` by quadrature with the weak-field `weakFieldDeflection`, the photon sphere
    and `criticalImpactParameter`, `flammParaboloid`, and `lightRayAlgorithm` (one ray stepped in the angle, logging
    radius, position, bend so far, with a `ray/captured` event and the exit angle refined; `measuredDeflection`).
- `maps.ts`: iterated maps, orbits, cobwebs, bifurcation diagrams, Lyapunov exponents.
- `nonlinear.ts`: Poincare sections, limit cycles by shooting, Floquet multipliers, Lyapunov spectra of flows,
  correlation dimension, recurrence quantification, bifurcation continuation (fold and Hopf detection).
- `control/optimal.ts`: LQR, discrete LQR, LQE, LQG, finite-horizon Riccati recursions.
- `control/placement.ts`: pole placement (Kautsky-Nichols-Van Dooren, Ackermann).
- `control/compensators.ts`: lead and lag compensators from their defining numbers (`leadCompensator`: phase and centre
  to zero and pole; `lagCompensator`: zero and ratio), `leadLag`, and the textbook design rules from a plant's margins
  (`leadDesign`: centre where the plant's gain is 1/sqrt(alpha); `lagDesign`: zero a decade below the crossover).
- `control/indirect.ts`: Pontryagin's maximum principle: `ControlProblem`, the Hamiltonian and the state-costate system
  (`hamiltonianSystem`), `bangBang`, and `pontryaginAlgorithm` (Newton shooting for the initial costate and, when free,
  the horizon: one shot per step, the miss, the trajectory with its costate and control, and the switches as events);
  `doubleIntegratorEnergy` and `minimumTimeDoubleIntegrator` with `minimumTimeShooting` (continuation in the smoothing
  of the bang-bang rule) carry their closed-form truth.
- `control/robust.ts`: H-infinity synthesis by the two Riccati equations: `hinfStateFeedback`, `hinfSynthesis` (the
  central output-feedback controller, with the three feasibility conditions reported), `closedLoopFromDisturbance`
  (w to z, to check the norm with `systems/norms.ts`), and `hinfBisectionAlgorithm` (the smallest achievable gamma, one
  level per step). Oracle: SLICOT SB10FD through slycot (python-control's engine).
- `control/adaptive.ts`: model-reference adaptive control of a first-order plant (`mracAlgorithm`: the MIT rule and the
  Lyapunov law; logs output, model output, parameter, error, gain and the distance from the matching parameters,
  `matchingParameters`), and `gainSchedule`.
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
- `identification.ts`: ARX, ARMAX, prediction-error methods, N4SID, dynamic mode decomposition, SINDy, extended DMD, and
  recursive least squares (`rlsAlgorithm`, one sample per step with a forgetting factor, Joseph-form covariance;
  `arxRegression`, `simulateArx` with a mid-run change of plant as the planted truth).
- `automata/` (namespace `dynamics.automata`; no oracle, since no library has these: the tests check hand-worked rows,
  known periods and displacements, exact densities and conserved quantities): every system is an `Algorithm` whose state
  (a grid or an agent list in typed arrays) serialises, whose step is a pure function of the state and the step's key,
  with an exported path union and log contract (`XxxPath`, `xxxContract`) that the matching method adopts.
  - `elementary.ts`: Wolfram's 256 rules (`ruleTable`, `elementaryStep`, `elementaryBoard`, `elementaryAutomaton`), the
    mirror and complement symmetries, `wolframClass` (by simulation; class 4 by the rule 110 family), `rule90Cells`
    (the exact Sierpinski cell count), `bitEntropy`.
  - `life.ts`: life-like rules by B/S string (`parseLifeRule`), RLE patterns (`parseRle`, `LIFE_PATTERNS` with known
    periods and drifts: blinker, glider, lightweight spaceship, pentadecathlon, R-pentomino, Gosper's gun), `detectPeriod`,
    `lifeAutomaton` (births, deaths, population, the period and a `life/stabilised` event).
  - `ant.ts`: Langton's ant for any rule string and several ants (`langtonsAnt`), `highwayDetected`.
  - `sandpile.ts`: the Abelian sandpile (`stabilise`, `sandpileAdd`, `sandpileIdentity`, `sandpileAlgorithm` with the
    avalanche size, duration and sites).
  - `forest-fire.ts`: Drossel-Schwabl (`forestFire`, with `treeDensityWithoutFire` as its exact check).
  - `schelling.ts`: Schelling segregation (`schelling`, `segregationIndex`, random or satisfying destinations).
  - `traffic.ts`: Nagel-Schreckenberg (`nagelSchreckenberg`, `trafficStep`, `deterministicFlow` as the exact check for
    no dawdling, `trafficDiagram` as the fundamental-diagram sweep).
  - `wa-tor.ts`: Wa-Tor fish and sharks (`waTor`, with an extinction event).
  - `sugarscape.ts`: Sugarscape (`sugarLandscape`, `sugarscape` with `stats.gini` of the wealth on the tape; trade and
    reproduction are not built).
  - `lattice-gas.ts`: HPP and FHP-I lattice gases (`latticeGas`, `gasMassMomentum`, `coarseGrainGas`, `curl2d`,
    `gasNeighbour`; exact mass and momentum conservation, bounce-back solids).
  - `boltzmann.ts`: D2Q9 BGK lattice Boltzmann with Guo forcing, bounce-back walls and obstacles, inlet and outlet
    (`latticeBoltzmann`, `d2q9Equilibrium`, `d2q9Moments`, `poiseuilleProfile` as the analytic check; the shear-wave
    decay rate `nu k^2` is checked in the tests).
  - Private: `_core.ts` (state codec, torus index, board hash, checks), shared by `fractals/`.
- `fractals/` (namespace `dynamics.fractals`; checked against the dimensions and counts the constructions imply, and
  hand-known points): each is an `Algorithm` with serialisable state, a path union and a log contract.
  - `escape-time.ts`: Mandelbrot, Julia, Burning Ship and multibrot (`escapeCount`, `escapeOrbit`, `orbitPeriod`,
    `pixelGrid`/`ComplexViewport`, `escapeTimeAlgorithm`: one iteration over the whole grid per step, smooth counts).
  - `newton.ts`: Newton fractals for a polynomial given by its roots (`newtonStep`, `rootsOfUnity`, `newtonFractal`).
  - `ifs.ts`: affine IFS (`SIERPINSKI`, `SIERPINSKI_CARPET`, `BARNSLEY_FERN`, `similarityDimension`, `chaosGame`,
    `ifsDeterministic` on a pixel grid).
  - `box-counting.ts`: `boxCountingDimension` (points) and `boxCountingDimensionOfMask` (pixels), with the fit.
  - `l-systems.ts`: `rewrite`, `turtle`, `L_SYSTEMS` (Koch, dragon, Hilbert, Peano, arrowhead, plants, one stochastic),
    `lSystemAlgorithm`.
  - `attractors.ts`: `mapCloud` (the accumulating cloud of any planar map) and `lobeSwitches`; the maps themselves
    (`deJongMap`, `cliffordMap`, `ikedaMap`) are in `maps.ts` and `rosslerField`, `thomasField` in `named-systems.ts`.
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
  estimators' training algorithms in `methods/models/crf.ts`. `crfFeatureCounts` and `crfExpectedFeatureCounts` are the
  counts whose difference is the objective's gradient.
- `models/crf-training.ts`: CRF training through the solver slot: `crfSmoothProblem` (the penalised objective with
  unbiased batch estimates), `crfSolvers` (L-BFGS by default beside every generic smooth solver, whose `gradient` mode is
  stochastic gradient descent, and the online structured-prediction modes `perceptron`, `averaged`, `passive-aggressive`),
  `crfTrainAlgorithm` (objective, gradient and sequence error per step), `trainCrf`.
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
- `inference/cluster-updates.ts`: Swendsen-Wang and Wolff cluster updates and the Metropolis sweep for the q-state Potts model
  (Ising at q = 2) on a periodic square lattice (`swendsenWangUpdate`, `wolffUpdate`, `metropolisSweep`,
  `pottsSamplerAlgorithm` with energy and magnetisation per site on the tape), the exact truth (`pottsCriticalCoupling`,
  `ISING_CRITICAL_ENERGY_PER_SITE`, `onsagerMagnetisation`) and `integratedAutocorrelationTime`. Tests check detailed balance
  against exact enumeration of small lattices and the decorrelation advantage over Metropolis at the critical coupling.
- `inference/map.ts`: MAP and MPE by max-product (elimination and junction trees elsewhere; loopy BP decoding here), by
  LP relaxation (MPLP on its dual, with the integrality gap), and by local search (ICM with restarts).
- `learning/baum-welch.ts`: Baum-Welch for hidden Markov models (`baumWelchAlgorithm` as a traced algorithm with no past,
  `baumWelch` to fit, `initialHmmParameters`, `hmmDecode`, `hmmFromParameters`): categorical or Gaussian emissions
  (diagonal or full covariances with hmmlearn's prior) over `inference/chain.ts`; parameters are plain JSON
  (`HmmParameters`). The golden against hmmlearn keeps its fixture under `methods/models/hmm`. `methods/sequence/hmm.ts`
  is its particular on the iterative base.
- `ratings.ts`: rating systems as filters over games: `eloExpected`/`eloUpdate`, `glickoUpdate`/`glickoInflate`,
  `glicko2Update` (the Illinois iteration on the log volatility), `trueskillUpdate` (two players or two teams by Gaussian
  moment matching, with draws and the dynamics term), and `ratingAlgorithm` (one game, or one period for the Glicko
  systems, per step; the pre-game win probability's running log-loss on the tape). Goldens are the worked examples of
  the Glicko and Glicko-2 papers and the reference TrueSkill update (no rating package is an oracle dependency).
- `choice-models.ts`: Plackett-Luce and Bradley-Terry: `choiceSmoothProblem` (the log-likelihood in the free log strengths),
  `choiceSolvers` (Hunter's minorise-maximise as the model's own mode beside every generic solver), `choiceAlgorithm`,
  `fitPlackettLuce`, `fitBradleyTerry`, `pairedComparisons`, `winProbability`.
- `irt.ts`: item response theory for scored responses: the 1PL, 2PL and 3PL models (`irtProbability`), Bock-Aitkin marginal
  maximum likelihood on Gauss-Hermite nodes (`irtSmoothProblem` with the analytic gradient by Fisher's identity,
  `irtSolvers` with `em` as the model's own mode beside every generic solver, `irtAlgorithm`, `fitIrt`), abilities by EAP,
  MAP or MLE (`irtAbility`), the item and test information functions, and `irtSimulate` (abilities as the truth).
- `learning/hmm-variants.ts`: the left-right (Bakis) topology (`leftRightParameters`, `isLeftRight`; Baum-Welch preserves
  the structural zeros) and Viterbi training (`viterbiTrainingAlgorithm`, hard-assignment EM that never decreases the joint
  score of data and best paths and stops when the paths settle).
- `learning/hsmm.ts`: the hidden semi-Markov model with explicit durations: `hsmmPosterior` (forward-backward over
  durations in log space, the last segment right-censored; start, move, duration and occupancy expectations),
  `hsmmAlgorithm` (EM with nonparametric, shifted Poisson or shifted negative binomial durations, the last two fitted
  exactly on the truncated support by `fitPoissonDurations` and `fitNegativeBinomialDurations`), `hsmmFromHmm` (geometric
  durations: equal to the HMM), `hsmmSample`. `emissionMStep` (in `baum-welch.ts`) is the emission update shared with
  Baum-Welch. Truth: exact enumeration of the segmentations.
- `learning/io-hmm.ts`: the input-output HMM: transitions and emissions softmax (or Gaussian regressions) of the input
  features, `ioHmmChain` (per-step transitions on `inference/chain.ts`), `ioHmmAlgorithm` (EM whose softmax M-steps are
  `softmaxRegressionProblem`s minimised by the smooth solver slot, Newton by default, with a ridge so the EM objective is
  the penalised likelihood), `ioHmmSample`.
- `learning/gmm-hmm.ts`: Gaussian-mixture emissions per state (`gmmHmmAlgorithm`, diagonal or full covariances,
  `initialGmmHmmParameters`, `gmmHmmLogLikelihood`); one component per state is the Gaussian Baum-Welch exactly.
- `learning/factorial-hmm.ts`: the factorial HMM: `factorialHmmExact` (the product chain, the truth at small sizes),
  `factorialHmmVariational` (structured mean field across chains, exact within each, the ELBO per sweep),
  `factorialHmmAlgorithm` (EM with either E-step; the variational one warm-starts so the bound rises at every step),
  `factorialHmmSample`.
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
- `sequence-labelling.ts`: entity-level scores as seqeval reports them (conlleval's chunk rules): `entitySpans` (BIO,
  IOB1, BIOES, BILOU), `bioViolations` (an inside tag that continues nothing), `entityScores` (micro, macro and weighted
  precision, recall and F1 with the per-type scores), `entityF1`. Checked on the worked example of seqeval's README.
- `interleaving.ts`: team-draft interleaving of two rankers (`teamDraftInterleave`, one coin flip per round), the
  clicks credited to each (`interleavingCredit`) and the preference over many impressions (`interleavingPreference`).
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
- `latent.ts`: latent-factor data with the factors as truth: `mixedSources` (non-Gaussian sources and their mixing
  matrix, for ICA), `twoViews` (shared latent signals in two views, with the population canonical correlations, for CCA
  and PLS), `toySuperposition` (sparse features with importances), `sparseDictionary` (sparse non-negative codes through a dictionary of unit directions, the plant a sparse autoencoder must recover; truth: codes, dictionary, coherence).
- `audio.ts`: SNR, segmental SNR, SDR (BSS Eval with a distortion filter), SI-SDR and SI-SNR, permutation-invariant
  matching (`permutationInvariant`, Hungarian algorithm via `optim` `linearSumAssignment`), STOI and ESTOI (`shortTimeObjectiveIntelligibility`, pystoi with its
  resampling), log-spectral distance, mel-cepstral distortion from cepstra (optionally DTW-aligned), DNSMOS from a
  supplied model (torchmetrics' windows and calibration, 16 kHz); windows, FIR design, polyphase resampling and
  framed spectra come from `signal`. TODO: PESQ (ITU-T P.862), whose reference implementation is ITU-copyrighted C
  that cannot be ported under the licence.
- `uplift.ts`: scoring of uplift models and targeting rules on randomised data: `upliftCurve` (the Qini and uplift gain
  curves with the random line), `qiniCoefficient`, `auuc`, `upliftAtK`, `upliftDeciles` and the inverse-propensity
  `policyValue` (with the treat-all and treat-none values). The estimators that produce the scores are `lib/causal`.

### 4.29 `losses/`: differentiable objectives (torch.nn losses, optax losses)

oracle: torch.nn.functional, optax, MONAI, POT, torchvision, scikit-learn, scipy

Training objectives with a uniform `reduction` option. Evaluation metrics live in `metrics`; where a loss and a metric
share a concept they share a definition but not a name.
- `regression.ts`: `mse`, `mae`, `huber`, `smoothL1`, `logCosh`, `pinball`, `expectile`, `poissonNll`, `gaussianNll`,
  `tweedieDeviance`.
- `classification.ts`: binary and categorical cross-entropy (`binaryCrossEntropy`, `binaryCrossEntropyWithLogits`,
  `crossEntropy`: with logits, label smoothing, class weights and ignore index), `focal` (sigmoid) and
  `categoricalFocal`, `nllLoss`, `multiLabelSoftMargin`; the losses for noisy labels: `generalisedCrossEntropy`,
  `truncatedCrossEntropy`, `symmetricCrossEntropy`, `probabilityMae`, `bootstrapped` (soft and hard), `forwardCorrected`
  and `backwardCorrected` (given a transition matrix), `biTemperedLogistic` with `temperedSoftmax`, `logTemperature`
  and `expTemperature`.
- `large-output.ts`: losses for very many classes: hierarchical softmax (`huffmanCoding`, `hierarchicalSoftmax`,
  `hierarchicalSoftmaxLogProbs`), sampled softmax (`sampledSoftmax`, `sampleClasses` with uniform and log-uniform
  proposals). The Huffman tree comes from `information/source-coding.ts` (`huffmanTree`).
- `margin.ts`: `hinge`, `squaredHinge`, `logistic`, `exponential`, `modifiedHuber`, `perceptron`, `crammerSinger`,
  `westonWatkins`.
- `density.ts`: negative log-likelihood of any `Distribution` (`negativeLogLikelihood`) or conditional distribution
  (`conditionalNll`), mixture-density NLL (`mixtureDensityNll`), heteroscedastic Gaussian NLL with beta-NLL weighting
  (`heteroscedasticGaussianNll`), censored likelihoods (`censoredNll`: exact, left, right and interval); the evidential
  losses: Dirichlet (`evidentialNll`: squared error, log and digamma forms with the KL regulariser; `dirichletUncertainty`:
  vacuity, aleatoric and epistemic parts) and normal-inverse-gamma (`normalInverseGammaNll`,
  `normalInverseGammaUncertainty`).
- `sequence.ts`: CTC: `ctc` (log-semiring forward algorithm over the blank-extended target, torch's CTCLoss conventions:
  padded targets, input and target lengths, mean-by-length reduction, `zeroInfinity`), `ctcTrellis` (alpha, beta and the
  state and label alignment posteriors; the loss's gradient is minus the label posterior), `ctcForcedAlignment`
  (Viterbi), `ctcGreedyDecode`.
- `invariance.ts`: penalties for behaving the same across environments: the IRMv1 gradient penalty (`irmPenalty`, biased and
  unbiased; `irmObjective`), V-REx (`vrexPenalty`, `vrexObjective`), `coral` (Deep CORAL, optional mean matching),
  `mmdPenalty` (pairwise MMD between environments), and gradient criteria: `environmentGradient`, `gradientAlignment`
  (cosine, and Fish's inner product), `perSampleGradients` and `fishrPenalty` (per-environment gradient variances).
- `preference.ts`: preference objectives on response log-probabilities: `bradleyTerry` (reward-model pairwise loss with a
  margin), `dpo` (with conservative label smoothing and the reference-free form), `implicitRewards`, `ipo`, `kto`,
  `orpo`, `simpo`, `sequenceLogProb`; `tables.ts` has them as `preferenceLosses` (needs a reference, length-normalised,
  reward model, paired, bounded margin).
- `ranking.ts`: `marginRanking`, `rankNet`, `pairwiseHinge`, `bpr`, `lambdaRank` (NDCG@k deltas), `listNet`,
  `softmaxRanking`, `listMle`, `approxNdcg`; padded lists through a mask.
- `contrastive.ts`: `infoNce` (in-batch, shared or per-query negatives), `ntXent`, `tripletMargin`, `contrastive`,
  `cosineEmbedding`, `hingeEmbedding`, `supervisedContrastive`, `clipLoss`, ArcFace and CosFace margins
  (`arcFaceLogits`, `cosFaceLogits`, `arcFace`, `cosFace`); the self-supervised objectives `barlowTwins` and `vicreg`
  and the `alignment` and `uniformity` terms.
- `segmentation.ts`: `dice` (soft and squared), `tversky`, `focalTversky`, `lovaszHinge`, `lovaszSoftmax`; classes last,
  index or one-hot targets, activations, background, batch pooling, ignore index and class weights as MONAI.
- `generative.ts`: GAN losses (`discriminatorLoss`, `generatorLoss`: minimax, non-saturating, Wasserstein, hinge, least
  squares; `gradientPenalty` (WGAN-GP) and `r1Penalty`), ELBO terms (`gaussianKl`, `sampledKl`, `negativeElbo`,
  `bernoulliLogLikelihood`, `gaussianLogLikelihood`, `importanceWeightedBound`), `denoisingScoreMatching`, diffusion
  objectives (`diffusionTarget`, `diffusionLoss`: epsilon, x0 and v), flow matching (`flowMatchingPath`,
  `flowMatching`); `noiseContrastive` (NCE with a learned log-normaliser), `scoreMatching` (exact trace) and
  `slicedScoreMatching` (Gaussian or Rademacher directions, optional variance reduction, forward-mode derivatives),
  vector quantisation (`vectorQuantise`: nearest code, codebook and commitment terms, straight-through output, usage and
  perplexity; `vqEmaUpdate`: the EMA codebook).
- `divergence.ts`: `klDiv` (torch, with 'batchmean'), `klDivergence`, `jensenShannon`, `distillation`, `mmd` (Gaussian,
  Laplacian and energy kernels, multi-scale, median heuristic, biased and unbiased) and `sinkhorn` (log domain, weighted,
  debiased) losses, and `slicedWasserstein` (random projections, sorted).
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
  `method(data)` and `session(data)` to drive the same fit live: an `IterativeSession`, whose `fittedAt(t)` is the
  fitted value, with the kind's verbs, at any reached step through the checkpoints, and `fit`'s value at the head);
  `kinds.ts`: the estimator kinds as add-ons
  (`kindProbes`: the standard derived paths of classifiers and regressors, clusterers, transformers and outlier
  detectors; `KindPath`, their paths as a type); `trainer.ts`: the trainer base (`defineTrainer(particular)`: minibatch training of one `nn` module as an
  `Algorithm` over `fitAlgorithm`, a session step being an optimiser step or, with `unit: 'epoch'`, an epoch; the
  particular declares `model`, `objective`, `optimiser(learningRate)`, `settings` (learning rate, a `schedules`
  configuration, epochs or steps, batch size, clipping, accumulation, callbacks, a validation fraction), an optional
  `source` (rows, or a `LearnStream` drawn into the rows the budget needs), `validation` and `metrics`; the base logs
  `train/loss`, `train/gradNorm`, `train/lr`, `eval/loss` and `eval/<metric>` (on validation rows, at the schedule the
  subscription gives, by default at each `train/epoch/end`), `layer/<i>/weights`, `layer/<i>/params` and
  `layer/<i>/activations` (layers are the children of a `Sequential`, activations are computed on probe rows only when
  due), the events `train/step/end` and `train/epoch/end`, `model/params` and the kind's probes; the state serialises
  with the optimiser's moments, so `session.fittedAt(t)`, `probe` and `fork` work at any reached step, and `fit`
  returns the verbs of the kind over the trained network, the training `curves` record and the tape); `agent.ts`: the
  agent base (`defineAgentMethod(particular)`: an environment from the registry or a definition, an `Agent` from
  `agents`, an `AgentSchedule` of episodes or steps, steps per episode and evaluation, run by `trainingAlgorithm`, a
  session step being an environment step or an episode; it logs `episode/return`, `episode/length`, `eval/return`,
  `train/loss` where the agent reports one, `agent/<statistic>`, `env/frame` and the event `rl/episode/end`, and probes
  `eval/episode` (a greedy episode as an `episodes` record, with the tables of a tabular agent as snapshots) and, for
  tabular agents, `agent/value-map` and `agent/action-map` as `valueGrids`; `session(...)` adds `agentAt(t)` and
  `policyAt(t)`, `train()` runs it to completion). Private: `_session.ts` (`extendSession`, a session with extra verbs
  that its forks keep, behind `fittedSession` and the agent session), `_trainer.ts` (plan arithmetic, layer records,
  evaluation, curves and the verbs of a network) and `_agent.ts` (the value and action maps on a grid, an episode as a
  record).; **Typed paths in the bases:** each base takes the type `K` of the paths its particular adds (default
  `string`, so every estimator written before stays as it was): `iterative<H, S, P, V, K>`, `IterativeParticular<..., K>`,
  `IterativeEstimator<..., K>`, `defineTrainer<H, K>`, `TrainerEstimator<H, K>`, `defineAgentMethod<H, G, O, A, K>`,
  `AgentMethod<..., K>`, and the sessions (`IterativeSession<S, P, V, K>`, `TrainerSession<K>`, `AgentSession<G, O, A, K>`
  with `SessionWith<S, X, K>`) and their options (`IterativeSessionOptions<K>`, `AgentSessionOptions<K>`). The session
  and the method of an estimator are typed by the paths of the whole run, the particular's with the base's:
  `IterativePath<K>` (the base's `fit/iteration/end`, `model/params`, the kind's `KindPath` and the solver's channels
  `solver/${string}`), `TrainerPath<K>` (`train/...`, `eval/${string}`, `layer/${number}/...`, `model/params`, the kind's)
  and `AgentPath<K>` (`episode/...`, `eval/return`, `eval/episode`, `train/loss`, `agent/${string}`, the maps, the
  events); so the `logs:` of a session and its `probe` and `stepsOf` paths are checked, and a typed estimator is accepted
  wherever the untyped one is. `K` is inferred from the particular's annotations (no type arguments needed for the
  Algorithm, the contract); an estimator whose algorithm logs only through the solver's channels (`observe`) is
  `K = never`. To type an estimator: write `export type FooPath = ...`, return `Contract<S, FooPath>` from `contract`
  (`defineContract`), return `Algorithm<S, FooPath>` from `algorithm`, and give the estimator its `IterativeEstimator<...,
  FooPath>` type (the recipe is in `IterativeParticular`'s TSDoc). Migrated: `svm/` (`SvmPath`: `svc`, `nuSvc`, `svr`, `nuSvr`, `oneClassSvm`, `svdd`), `trees/` (`GrowPath` and `TreePath` for the
  CART family, `ClassicPath` for ID3, C4.5 and CHAID, `HoeffdingPath`, `PrunePath` for the two pruning algorithms),
  `linear/lars.ts` (`never`), in `kernel-methods/` `kernelRidge` and `norma` (`never`) and `krls` (`KrlsPath`), in
  `ensembles/` (`AveragingPath` and the path of each family: `ForestPath`, `EmbeddingPath`, `BaggingPath`, `CyclicPath`,
  `StackingPath`, `BoostingPath`, `AdaBoostPath`), in `gp/` (`GpPath`, `GpcPath`, `SparseGpPath`, `RvmPath`), in
  `rules/` (`CoveringPath`, `RipperPath`, `CorelsPath`, `BayesianRuleListPath`, `OneRPath`, `DecisionTablePath`),
  `anomaly/isolation.ts` (`IsolationForestPath`), `learn/naive-bayes.ts` (the tree-augmented naive Bayes, `never`),
  `nn/training/callbacks.ts` (`Callback<K>`, `HookContext<K>` and `CallbackPath`, with template literals for the
  parameter and tap names), `rl/training.ts` (`TrainingPath`), the methods `mixture/` (`GmmPath`, `BgmmPath`,
  `MoePath`), `linear/bayesianRegression.ts`, `cluster/kmeans.ts`, `sequence/hmm.ts`, and the `defineMethod` pages of
  `dynamics/`, `linalg/`, `probability/`, `stochastic/`, `tune/` and `vision/`. **What is left (counted by grep,
  2026-10-31, after the mechanical pass; non-test sources under `lib/` and `methods/`, `lib/trace` and `lib/learn/bases`
  excluded):** 51 files define 88 estimators on the bases (`iterative(`, `defineTrainer(`, `defineAgentMethod(`) with
  the default `K`, and none of them declares a `contract` or logs a path with `ctx.log` or `ctx.emit`, so `string`
  (unchecked) is already the right `K` and they need nothing: they report through `observe` alone (the solver's
  `solver/...` channels). By folder, as `files / estimators`: `lib/linear` 12/18, `lib/learn` 9/25, `lib/cluster` 8/13,
  `lib/decomposition` 5/8, `lib/kernel-methods` 5/8 (`online-learners`, `kernel-classifiers`, `kernel-multivariate`,
  `kernel-quantile`, `feature-ridge`), `lib/manifold` 4/4, `lib/covariance` 3/4, `methods/linear` 3/3 (`ridge`,
  `logisticRegression`, `lassoPath`), `lib/anomaly` 1/3 (`neighbours`), `lib/gam` 1/2. A file moves to a typed `K` the
  day it gains a `contract` or a `ctx.log` (then the recipe above applies, and `never` is the type of an estimator that
  declares no path of its own). Files with no contract and no log sites are not counted as work; the count above is the
  truth to compare against after the next migration. To regenerate it: take the files whose code, comments stripped,
  matches `\b(iterative|defineTrainer|defineAgentMethod)\s*[<(]`, `^\s*contract[:(]` or `ctx\.(log|emit)\(`, and drop
  those that also mention `Algorithm<_, K>`, `Contract<_, K>`, `Method<_, _, K>`, `StepContext<` or `defineContract(`
  (the file is the unit, so a file typed in part counts as typed); `callbacks.ts` and `vision/refinement.ts` match the
  log rule and are typed through `HookContext<K>` and `Algorithm<_, RefinementPath>`, which the pattern does not see.; `sweep.ts`: the sweep and repeated-fit base. `sweep(spec)` is an `Algorithm` with one trial per step
  (a configuration at a repeat) that runs an inner `Method` as an eager `session` over a `grid` or `sample` of some of
  its knobs (`knobsOf` picks them), or over explicit configurations, under paired keys (repeat `r` is the same key in
  every configuration) or fresh ones; it logs `trial/<knob>`, `trial/<metric>` (the final value of each metric path
  and of each derived `final` path), `trial/config`, `trial/repeat`, `trial/best`, and the running mean and spread of
  the configuration's repeats (`config/<metric>/mean`, `config/<metric>/std`), keeps the inner tapes by `keep`
  (`'none'` summarises each in its `TrialRecord`, `'last'`, `'all'`), serialises its `SweepState`, and has the readers
  `sweepConfigurations` and `sweepBest`; `sweepTrials` counts the steps. `sweepMethod(spec)` is the study as a
  `Method` (repeats, steps, keep, design, points and trials as knobs; derived `sweep/configurations` and
  `sweep/best`). `race(method, variants)` is the sweep over a slot in one run: a method whose state holds every
  variant's, stepped in lockstep from the same key, logging each under `race/<variant>/`, with the inner contract's
  paths prefixed (typed: `SweepPath` for the study, `RacePath<K>` = `race/${string}/${K}`, with `mountContext` giving
  each variant its corner of the context; `SweepSpec<S, C, K>` checks `metrics`, `final` and `logs` against the inner
  method's paths). The one-shot and procedure bases follow.

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
- `preprocessing/aggregates.ts`: per-row group statistics (`groupAggregates`, pandas `groupby.transform`, with a
  leave-one-out option), rolling-window statistics (`rollingFeatures`, pandas `rolling` with `min_periods` and
  `center`) and lagged copies (`lagFeatures`); NaN where there is no history.
- `preprocessing/resampling.ts`: random over- and under-sampling, SMOTE, borderline SMOTE, ADASYN, Tomek links,
  NearMiss (resamplers apply only in fit, also inside pipelines).
- `feature-selection.ts`: variance threshold, univariate scores (F tests, chi-square, mutual information) and
  selection (k best, percentile, FPR/FDR/FWE), selection from model importances.
- `feature-selection-wrappers.ts`: RFE and RFECV, sequential selection, stability selection (traced).
- `knockoffs.ts`: Gaussian model-X knockoffs (`gaussianKnockoffs`) with the equicorrelated scale and a coordinate-ascent
  scale (`knockoffScaleAlgorithm`, an SDP-free approximation), the knockoff statistics (lasso coefficient difference at a
  cross-validated penalty, lasso signed maximum along the path, marginal correlation difference) and the knockoff and
  knockoff+ thresholds (`knockoffThreshold`, `knockoffFilter`); the FDR control is tested over 100 replicates of a
  planted sparse regression with a known design covariance. The lasso is `optim/sparse.ts`'s.
- `selective-inference.ts`: post-selection inference by the polyhedral lemma: the truncated-Gaussian pivot
  (`truncatedNormalCdf` in the log scale, `polyhedralBounds`, `selectiveInference`), the lasso selection event at a fixed
  penalty (`lassoPolyhedron`, `lassoSelectiveInference`) and forward stepwise (`forwardStepwisePolyhedron`,
  `forwardStepwiseInference`): selective p-values and intervals for the coefficients of the selected model, next to the
  naive ones. The polyhedron is checked against independent refits along the contrast; uniformity and coverage in simulation.
- `mrmr.ts`: minimum-redundancy maximum-relevance selection (MID and MIQ) as a stepped algorithm, mutual information on
  equal-frequency bins or by the nearest-neighbour estimators of `information`.
- `relief.ts`: Relief and ReliefF weights as a stepped algorithm, one sampled instance per step (k neighbours,
  class-prior weighted misses, a discrete mask); finds the XOR halves univariate scores miss.
- `stability-selection.ts`: the Meinshausen-Buhlmann bound E[V] <= q^2 / ((2 pi - 1) p) (`stabilitySelectionBound`,
  `stabilityThreshold`) and lasso stability selection as an algorithm (`lassoStabilityAlgorithm`); the generic selector
  version stays in `feature-selection-wrappers.ts`.
- `feature-synthesis.ts`: deep feature synthesis over a schema of entity and child tables (`deepFeatureSynthesis`:
  aggregates of child rows per parent, arithmetic on columns, to a chosen depth, with names and definitions) and
  `selectSynthesised`, forward selection by the cross-validated score of an estimator.
- `theory.ts`: learning-theory bounds as functions with their assumptions documented: PAC sample complexity and bounds
  for finite classes (realisable, agnostic) and the union bound; VC bounds (Vapnik's form, the Blumer sample
  complexity, Sauer-Shelah) with the shattering count, growth function and VC dimension of intervals, half-lines, boxes
  and half-spaces by enumeration (`conceptClasses`, half-space separability by `linprog`); Rademacher complexity (the
  Monte Carlo estimate as `rademacherAlgorithm`, Massart's finite-class bound, the linear-class closed form, the
  generalisation bound); margin bounds; algorithmic stability (the closed-form uniform stability of ridge, the
  Bousquet-Elisseeff bound, `empiricalStability` by replacement); PAC-Bayes (McAllester, Seeger by `klInverse`, Catoni,
  `gaussianKl`, the closed-form Gibbs risk of a Gaussian linear classifier); the Hedge and online-gradient regret bounds;
  the information-theoretic bound; the bias-variance decomposition (squared loss, Domingos' 0-1 loss, the Monte Carlo
  study as `biasVarianceAlgorithm`); degrees of freedom (Efron's covariance definition by Monte Carlo, the closed forms
  for ridge, lasso and k-NN). The tests are planted-truth: each bound holds across simulated draws and is within a
  stated factor of the observed gap.
- `representation.ts`: measures of learned representations: alignment and uniformity (Wang and Isola), the effective
  rank (Roy and Vetterli, RankMe) and centred kernel alignment (`cka`, linear and RBF; invariant to rotations).
- `sequence-model.ts`: `sequenceModelTrainer`, token-sequence models on the trainer base: the layer is the `mixer` slot
  (`sequenceMixers` of `nn/blocks`), the optimiser the `optimiser` slot (`optimisers`); derived paths `mixer/spectrum`
  (eigenvalues of the state transitions), `mixer/step` (learned Delta) and `mixer/state-norm`; `tokenAccuracy`.
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
- `model-selection/compare.ts`: comparing learning algorithms from resampled scores: the plain and the corrected
  resampled t-test (Nadeau-Bengio), Dietterich's 5x2 cv t-test and Alpaydin's 5x2 cv F-test, the Bayesian correlated
  t-test with a ROPE (Benavoli), `pairedCvScores` (two estimators on the same folds), and Demsar's procedures for many
  algorithms over many data sets (`compareAlgorithms`: average ranks, Friedman and Iman-Davenport, `nemenyiQ` and
  `criticalDifference`, cliques, Bonferroni-Dunn and Holm against a control). McNemar, Wilcoxon, Friedman and the
  Nemenyi p-values are in `stats/tests`. goldens: the published Nemenyi q table (Demsar 2006, table 5); the type I
  error of each test is checked in simulation against the plain resampled t-test.
- `model-selection/adaptive.ts`: adaptive data analysis: Thresholdout (`thresholdoutAlgorithm`, one step per query,
  with an adaptive analyst), the Ladder leaderboard (`ladderAlgorithm`), and the planted problems that show them
  against naive reuse of a holdout (`adaptiveOverfittingDemo`, `leaderboardAttack`: noise labels, an analyst who
  combines whatever looks good).
- `calibration.ts`: Platt scaling, isotonic calibration (pool adjacent violators, the home that
  `methods/linear/isotonic.ts` reuses), temperature and vector scaling, beta calibration, histogram binning,
  cross-fitted calibrated classifiers; for regression, the probability integral transform (`pitValues`, `pitHistogram`)
  and Kuleshov's quantile recalibration (`quantileCalibration`, `quantileCalibrationCurve`, `recalibratedQuantiles`).
- `neighbours.ts`: `neighbourIndex` (exact k-nearest and radius search under any metric of `distances`, brute force or a KD
  or ball tree), the k-nearest classifier and regressor and their radius twins (uniform or distance weights, an outlier
  label), and the nearest centroid with Tibshirani's shrinkage and `shrunkenCentroidPath`; estimators on the iterative
  base that absorb the training rows one per step (`solver/rows`), so `fittedAt(t)` is the model on the first `t` rows.
  oracle: `learn/neighbours` against `sklearn.neighbors`.
- `discriminant.ts`: `discriminantAnalysis` (pooled or per-class Gaussian covariances, shrinkage by a fixed intensity or
  Ledoit-Wolf, Friedman's mix of the two, eigenvalue regularisation, priors, singular covariances by pseudo-inverse;
  Fisher's axes as `transform` with `explainedVarianceRatio`), its presets `linearDiscriminant` and
  `quadraticDiscriminant`, and `kernelFisherDiscriminant` (the axes in a kernel's span with a ridge on the within-class
  scatter). Iterative base, one row per step. oracle: `learn/discriminant` against `sklearn.discriminant_analysis`.
- `naive-bayes.ts`: `naiveBayes` (Gaussian, multinomial, Bernoulli, categorical and complement as one family whose fitted
  parameters are sufficient statistics; `naiveBayesPartialFit` adds one weighted row, and the fit is the algorithm that
  absorbs rows one per step logging `rows`, the prequential `accuracy` and `logLoss`), and on integer-coded features
  `treeAugmentedNaiveBayes` (`tanAlgorithm`: Prim over the conditional mutual information, one edge per step, logging
  `edges`, `weight` and the message `edge`) and `aode` (averaged one-dependence estimators, counted a row at a time).
  oracle: `learn/naive_bayes` against `sklearn.naive_bayes` and, for the last two, numpy forms of their definitions.
- `multiclass.ts`: `oneVsRest`, `oneVsOne`, `outputCode` (error-correcting output codes, random or given code book) and
  `nestedDichotomies` (random hierarchies or one given, averaged) over any base `Estimator`; iterative base, one binary
  problem per step (`solver/problems`), so `fittedAt(t)` is the classifier of the problems solved so far.
  oracle: `learn/multiclass` against `sklearn.multiclass` (the codes and hierarchies from its fitted binary models).
- `multilabel.ts`: `binaryRelevance`, `classifierChain` (order and chain method as knobs) and `labelPowerset` over a base
  classifier, on a 0/1 label matrix. `multioutput.ts`: `multiOutputRegressor`, `multiOutputClassifier`,
  `regressorChain` and the engines (`outputsEstimator`, `independentOutputs`, `outputChain`) the multilabel wrappers
  share. oracle: `learn/multilabel` and `learn/multioutput` against `sklearn.multioutput` and `sklearn.multiclass`.
- `_staged.ts` (private): the algorithm that fits one base-estimator problem per step, and readers of a fitted base.
- `metric-learning.ts`: neighbourhood components analysis (`ncaValueAndGradient`, the objective and exact gradient;
  `ncaAlgorithm`, L-BFGS on it logging `objective`, `gradNorm` and the map as `transform`;
  `neighbourhoodComponentsAnalysis` with the LDA, PCA, identity, random or given start) and large margin nearest
  neighbour (`lmnnAlgorithm`, gradient descent with the adaptive step logging `objective`, `impostors`, `step` and
  `transform`; `largeMarginNearestNeighbour`). Transformers on the iterative base. oracle: `learn/metric_learning`
  against `sklearn.neighbors.NeighborhoodComponentsAnalysis` (objective, gradient and fits scored by its own loss).
- `_classify.ts` (private): the verbs of a classifier from class scores, the default scores, `rowsAlgorithm` (the rows of
  the training set revealed one per step, which closed-form and memory-based learners use as their algorithm).
- `conformal.ts`: split conformal regression, conformalised quantile regression, conformal classification (LAC, APS,
  RAPS with tuning), Mondrian conformal, jackknife+ and CV+, conformal for time series (adaptive conformal inference),
  and weighted conformal regression under covariate shift (`weightedConformalQuantile`, `weightedConformalRegressor`, which
  takes the `predictRatio` of any density-ratio estimator).
- Explanation, inspection and data valuation are their own domain, `lib/explain/` (4.38).
- `active-learning.ts`: the pool-based and stream-based active-learning loop as a traced algorithm (query, label,
  refit), query-strategy contract (uncertainty, margin, entropy, random, ranked batch, expected error reduction, query
  by committee, expected model change, BALD and BatchBALD, density-weighted uncertainty, k-centre core sets through
  `coresets.ts`, BADGE), budget handling. `activeLearningAlgorithm` is the stepped form.
- `coresets.ts`: Gonzalez's k-centre traversal (`kCenterGreedyAlgorithm`, with fixed outside centres), kernel herding,
  leverage-score and sensitivity-sampling coresets with their importance weights, Frank-Wolfe Bayesian coresets.
- `uncertainty.ts`: deep ensembles (`deepEnsembleAlgorithm`, one member per step under its own key, optionally on
  bootstrap resamples; `deepEnsemble`, `ensembleUncertainty`), Monte Carlo dropout over any stochastic forward pass
  (`mcDropoutAlgorithm`, `mcDropoutPredict`), and the epistemic/aleatoric decomposition (`deepEnsemblePredict`,
  `disagreement`).
- `drift.ts`: stream drift detectors as algorithms (`ddmAlgorithm`, `eddmAlgorithm`, `adwinAlgorithm` with the
  exponential histogram, `kswinAlgorithm`) with batch runners, and the population stability index.
- `shift.ts`: label shift: black-box shift estimation (`blackBoxShiftEstimate`), Saerens' EM (`emPriorAdjustmentAlgorithm`)
  and `adjustPriors`.
- `density-ratio.ts`: covariate-shift density ratios: `classifierDensityRatio` (any base estimator), KLIEP and uLSIF
  (relative uLSIF with `alpha`) as problems whose solver is a slot (`kliepProblem`/`kliepSolvers`,
  `ulsifProblem`/`ulsifSolvers`: the closed form, or any smooth solver of `optim`), and the closed-form leave-pair-out
  score `ulsifLoo`.
- `privacy.ts`: Laplace, Gaussian (classic and analytic) and exponential mechanisms, randomised response, composition,
  RDP accounting for the subsampled Gaussian.
- `dp-sgd.ts`: differentially private training: `perExampleGradients` (`vmap` of `valueAndGrad`), `clipPerExample`
  (flat clipping to a norm bound, the clipped share), `privateGradient` and `dpGradientTransformation` (the Gaussian
  noise stage, a `GradientTransformation` for the sum of clipped gradients), `dpSgdStep` (a step over `TrainState`),
  `dpSgdAlgorithm` (Poisson batches, the Renyi account on the tape: `budget/epsilon`, `budget/rdp`, with
  `training/clippedShare` and `training/noiseNorm`), `dpEpsilon` and `noiseMultiplierFor` (the accountant and its
  inverse over `privacy.ts`). The accountant is checked against opacus on Abadi et al.'s, TensorFlow Privacy's and
  Opacus's MNIST settings (`oracle/cases/learn/privacy.py`).
- `privacy-attacks.ts`: attacks on a trained model, models as logits functions: membership inference by
  `lossThresholdAttack`, `shadowModelAttack` (shadows calibrate the loss threshold) and `likelihoodRatioAttack` (LiRA,
  online, per-record Gaussians of `confidenceLogit`), scored by `membershipReport` (AUC, best accuracy, advantage, TPR
  at low FPR); `attributeInference` (Bayes over candidate values of a hidden feature); `modelInversionAlgorithm`
  (gradient ascent on a class, stepped); gradient leakage by `analyticGradientLeak` (exact for a first linear layer) and
  `gradientLeakageAlgorithm` (deep leakage from gradients, stepped); `extractionAttack`.
- `semi-supervised.ts`: rows with target `-1` (NaN for regression) are unlabelled. `selfTraining`, `coTraining` (two
  feature views, shared labelled set, product of the views) and `triTraining` (Zhou and Li's error and size conditions)
  over any probabilistic base `Estimator`; `labelPropagation` and `labelSpreading` (`sklearn.semi_supervised`, rbf and
  neighbour graphs, with the algorithms of both); `laplacianRlsClassifier` and `laplacianRlsRegressor` (manifold
  regularisation, closed form); `s3vm` (Joachims' transductive SVM, one pair switch or weight doubling per step) over any
  weighted binary base (e.g. `svc` of `lib/svm`, which `learn` cannot import). Iterative base, one round per step.
  oracle: `learn/semi_supervised` against `LabelPropagation`, `LabelSpreading` and `SelfTrainingClassifier`.
- `weak-supervision.ts`: votes `[items, sources]` with `-1` for an abstention. `labellingMatrixSummary`, `majorityVote`;
  `dawidSkene` (full, one-coin and GLAD modes, EM whose log-likelihood never decreases) and `labelModel` (accuracies from
  agreements alone by the triplet method refined by coordinate descent, with dependency residuals) on the iterative base;
  `estimateNoiseTransition`, `confidentJoint`, `labelQuality`, `findLabelIssues` (confident learning); `puRisk`
  (uPU and nnPU, differentiable), `puClassifier`, `elkanNoto`; `bagData`, `labelProportionClassifier` (proportion loss, L-BFGS) and
  `multipleInstanceSvm` (mi-SVM and MI-SVM over any binary base). oracle: `learn/weak_supervision` against numpy
  references written from the papers (EM for Dawid-Skene, the confident joint and the prune rules).
- `click-models.ts`: click models for logs of ranked lists: `clickSessions` (an interaction log grouped into sessions),
  the position-based model (`positionBasedAlgorithm`, EM one iteration per step, `positionBasedModel`), the cascade
  model (`cascadeModel`, closed form) and the dynamic Bayesian network model of Chapelle and Zhang (`dbnAlgorithm`, EM
  with a forward-backward pass over the examination chain, `dbnModel`), each with click probabilities by rank. Tested
  on `clickLogs` and `dbnClickLogs`, where the examination curve, the relevance, the satisfaction and the continuation
  are recovered.
- `noisy-labels.ts`: `forwardCorrectedLoss` and `backwardCorrectedLoss` (Patrini et al.), `smallLossSelection` and
  `coTeaching` (two models exchanging their small-loss rows over any probabilistic base).

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
  `lru`, `lruScan` with a resumable state), `parallelScan` and `parallelScanComplex` (associative scans of linear
  recurrences), `ssd` (structured state-space duality: recurrent, quadratic and chunked forms of the scalar-decay
  selective SSM), `channelCausalConvolution` (one kernel per channel).
- `functional/linear-recurrent.ts`: `decayedLinearAttention` (state with no, per-head or per-key-dimension decay;
  recurrent, parallel and chunked forms) and `wkv` (the RWKV recurrence, with a stable running shift). The complex diagonal S4D is a layer in
  `blocks/state-space.ts`.
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
  - `hopfield.ts`: associative memories: `hopfieldWeights` (Hebb or pseudo-inverse rule), `hopfieldEnergy`,
    `classicalMemory` and `denseAssociativeMemory` (Krotov-Hopfield polynomial or exponential interaction) as one
    `AssociativeMemory` contract, `hopfieldRecallAlgorithm` (asynchronous updates, sequential or random, the energy,
    overlap and state on the tape; the energy never rises), and the modern continuous network:
    `modernHopfieldUpdate` (attention over the patterns), `modernHopfieldEnergy`, `modernHopfieldAlgorithm`,
    `ModernHopfield` (a layer with learnable patterns).
  - `rbm.ts`: restricted Boltzmann machines: `RBM` (binary or Gaussian visible units), `rbmFreeEnergy`,
    `rbmHiddenProbabilities`, `rbmVisibleMean`, `rbmGibbs` (block Gibbs), `rbmContrastiveLoss` (the surrogate whose
    gradient is the CD update), the exact `rbmLogPartition`, `rbmLogLikelihood` and `rbmModelStatistics` for small
    machines, and `contrastiveDivergenceAlgorithm` (CD-k or persistent CD with momentum and weight decay; the
    reconstruction error, free energy, free-energy gap and exact log partition on the tape).
  - `masked.ts`: `CausalConv1d` (left-padded, dilated, grouped), `pixelCnnMask` (masks A and B, optional colour-channel
    ordering), `MaskedConv2d` (PixelCNN) and `GatedMaskedConv` (Gated PixelCNN vertical and horizontal stacks, no blind spot).
  - `graph.ts`: `rgcnConv` and `RGCNConv` (relational GCN: one weight per edge type, basis decomposition, mean or sum per
    relation), `EGNNLayer` and `EGNN` (E(n)-equivariant graph layers: features from distances, coordinates from relative
    vectors), `GraphTransformerLayer` and `GraphTransformer` (edge-softmax attention, dense when no edges are given,
    Laplacian or random-walk positional encodings from `graph` with random sign flips in training).
  - `operators.ts`: operator learning: `DeepONet` (branch over sensors, trunk over the query point), `SpectralConv1d`
    and `SpectralConv2d` (low Fourier modes times learned complex weights, through the differentiable FFT),
    `FourierLayer` and `FNO` (lift, Fourier layers, projection, in one or two dimensions).
  - `hippo.ts`: the initialisations of state-space models: `hippoLegS`, `hippoLegT`, `hippoLagT` (the matrices of
    `x' = A x + B u` that project the input's history onto Legendre or Laguerre bases), `nplrHippo` (the
    normal-plus-low-rank form `V diag(lambda) V^H - P P^T` with its eigenbasis forms) and `nplrReconstruct`,
    `s4dEigenvalues` (S4D-Lin, -Inv, -Real).
  - `mixer.ts`: `SequenceMixer`, the interface of the interchangeable sequence layers (state in, state out; the
    parallel form on a sequence and the recurrent form on one token agree), `ownKey`.
  - `state-space.ts`: `S4D` (diagonal, convolution or recurrence), `S4` (HiPPO normal-plus-low-rank in its
    eigenbasis, bilinear), `S5` (one MIMO complex state, parallel scan) and `LRUMixer` (over `lruScan`), all as
    sequence mixers whose two forms give the same outputs and final state.
  - `mamba.ts`: `causalDepthwiseConv` (with a carried window), `MambaBlock` (selective scan; the convolution window
    and SSM state carry between calls), `Mamba2Block` (scalar decay per head, `ssd` in its recurrent, quadratic or
    chunked form), `mambaModel` (a `SequenceModel` of Mamba blocks).
  - `sequence-model.ts`: `MixerBlock` (pre-norm residual block around any `SequenceMixer`, optional gated
    feed-forward) and `SequenceModel` (input layer, a stack of mixer blocks, norm, head; tokens or features in,
    per-position logits or a pooled classification or regression out; `run` threads the per-layer states).
  - `linear-attention.ts`: matrix-state mixers over `decayedLinearAttention`: `LinearAttentionMixer` (feature maps),
    `RetNetMixer` (retention: parallel, recurrent and chunkwise forms), `GatedLinearAttentionMixer`, `DeltaNetMixer`
    (the delta rule).
  - `recurrent-mixers.ts`: `RWKVTimeMix`, `RWKVChannelMix` and `RWKVMixer`, `MinGRU`, `MinLSTM` (parallel scans),
    `Hyena` (implicit long convolutions, history as state).
  - `mixers.ts`: `AttentionMixer` (softmax attention with a KV cache, the baseline) and `sequenceMixers`, the option
    table of every mixer (`mamba`, `mamba2`, `s4d`, `s4`, `s5`, `lru`, `linear-attention`, `retnet`, `gla`, `deltanet`,
    `rwkv`, `mingru`, `minlstm`, `hyena`, `attention`) with its knobs, for the `mixer` slot of a `SequenceModel`.
  - `sequence-generation.ts`: `sequenceLanguageModel` (a `SequenceModel` as a `LanguageModel` whose cache is the
    per-layer mixer state), `generationAlgorithm` (the decoding algorithm over it, with a serialisable state),
    `idTokenizer`.
  - `forecasting.ts`: N-BEATS: `trendBasis`, `seasonalityBasis`, `NBeatsBlock` (generic, trend or seasonality basis),
    `NBeats` (doubly residual stacks returning the forecast, the final residual and each block's forecast and backcast).
  - `recommender.ts`: recommender blocks: `fmScore`, `FactorisationMachine` (with `interactions` field by field),
    `FieldAwareFm`, `DeepFm`, `WideAndDeep` over `{ids, values}` feature rows; `NeuralCF` (NeuMF: GMF and MLP branches,
    modes), `TwoTower` (with `embeddingTower`, in-batch logits, `logQCorrection`), `CrossNetwork` (DCN), `dotInteraction`
    and `Dlrm`. The sequential recommenders are the existing `GRU` and `TransformerLM` (no separate block).
  - `flow.ts`: `CNF` (FFJORD continuous normalising flow: the state and the accumulated Jacobian trace integrated
    together; exact or Hutchinson trace, Rademacher or Gaussian noise; `logProb`, `sample` by backward integration) and
    `standardNormalLogProb`.
  - `mechanics.ts`: `HamiltonianNet` (a network for H(q, p), the field its symplectic gradient, energy conserved
    along the flow) and `LagrangianNet` (a network for L(q, qdot), accelerations from the Euler-Lagrange equation).
- `receptive-field.ts`: `receptiveField` (size, jump and start per level from kernels, strides, dilations and paddings),
  `receptiveFieldSize`, `layerSpecs` and `receptiveFieldOf` (read from convolution and pooling modules, through
  containers), `effectiveReceptiveField` (the normalised absolute gradient of one output unit).
- `swag.ts`: SWAG (stochastic weight averaging Gaussian): `swagState`, `swagCollect` (running mean, mean square and the
  last deviations), `swagVariance`, `swagCovariance` (dense, small models), `swagSample` (diagonal plus low-rank, or
  diagonal), `swagAlgorithm` (collection along any constant-step optimiser, as a stepped algorithm with the mean and
  deviation matrix on the tape), `swagUnravel`, `swagPredictive`. Lives here, not in `training/callbacks.ts`, so it
  works with any trajectory.
- `describe.ts`: `describe` (a module to a `Structure`: blocks, tensor shapes, repeats, skip connections into `add`
  nodes, parallel branches, parameter counts, FLOPs, optionally composite blocks' parts), `describeFunction` (any
  functional model, by its computation graph), `layerTable`, `summary` (torchinfo-style, with trainable and frozen
  counts and sizes), FLOP estimates (`flops`, `functionFlops`, counted from traced primitives) and `memoryEstimate`.
- `module/adapters.ts`: parameter-efficient fine-tuning as module wrappers that freeze the base: `LoRA` (low-rank
  `(alpha / r) B A` updates of chosen weights, by default a `Linear`'s weight or an attention layer's query and value
  rows; `merged`, `unmerge`, `delta`, rank-stabilised scaling), `loraTargets`, `loraParameterCount` (`r (d_in + d_out)`),
  `BottleneckAdapter` and `withAdapter` (near-identity residual adapters), `PromptTuning` (learnt input embeddings
  prepended to a sequence), `PrefixAttention` (learnt key and value rows prepended in an attention layer's input space),
  `trainOnly` and `bitFit` (train selected parameters only, through the module's `trainableMask`), and
  `parameterCounts` (total, trainable and frozen).
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
- `training/alternating.ts`: several modules trained against one another: `alternatingAlgorithm` (players of a module,
  an optimiser and a loss that sees every player's parameters, stepped through a schedule with `times` per player; one
  round per step; per-player loss and gradient norm on the tape; buffers carried; frozen players skipped),
  `ganAlgorithm` (generator and discriminator or critic with `nCritic`, the objectives of `losses/generative`, the WGAN-GP
  penalty, spectrally normalised layers through `SpectralNorm`, and the tracks `gan/discriminatorAccuracy`,
  `gan/wasserstein`, `gan/penalty`, `gan/samples`) and `modeCoverage` (modes of a known mixture reached by samples).
- `training/serialise.ts`: `saveState`/`loadState` and `stateToText`/`stateFromText` (parameters and buffers as JSON
  with dtype and shape), `loadTorchStateDict` (weights exported by the oracle, with renaming).
- `adversarial.ts`: attacks and adversarial training on any logits function: `fgsm`, `pgd` (random start inside an
  l-infinity or l2 ball, targeted or untargeted, cross-entropy, margin or TRADES-KL loss, data range), `pgdAlgorithm` (one
  gradient step per step, `attack/*` paths, serialisable), `marginLoss`, `carliniWagnerL2` (Adam on the penalised margin,
  closest adversarial point kept), `deepFool`, `transferAttack`, `squareAttack` (black box: logits only, query counts),
  `accuracyOf`, `robustAccuracy` (the sweep over radii) and `adversarialTrainStep` (PGD inner maximisation, or TRADES, as a
  step over `TrainState`).
- `certify.ts`: certificates: randomised smoothing (`smoothedCounts`, `randomisedSmoothing` with the Clopper-Pearson
  bound and the radius sigma Phi^-1(p), `certifiedRadius`, `certifiedAccuracy`, `smoothingAlgorithm` stepped with
  `sample/*`, `vote/*`, `certify/*` paths, `linearSmoothedRadius` the closed form for a linear classifier), interval
  bound propagation through a `Sequential` (`intervalBoundPropagation`, `ibpCertify` with the margin bounded exactly
  through the last layer) and the Lipschitz bound from spectral norms (`lipschitzBound`, `lipschitzCertifiedRadius`).
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
  `rules.ts`; `anchors`, `surrogateTree` (the grower of `trees`), `fidelity`, `globalSurrogate`, `lime`, `limeMasked` in `surrogates.ts`.
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
- Mechanistic interpretability of transformers (`residual.ts`, `lens.ts`, `patching.ts`, `planted.ts`): the model opened
  along its residual stream, with a hook at every site where a component writes (`embed`, each head, attention, feed-forward,
  block output), so everything below is a function of one object.
  Built: `residualModel` (pre-norm `TransformerLM`, forward pass rebuilt head by head, equals `lm.forward`, differentiable
  through hooks), `headSite`, `logitLens` (the last depth is the model's logits exactly; ranks and probabilities of given
  targets), `tunedLensAlgorithm` (affine translator per depth by KL to the final layer, one Adam step per step, paths
  `lens/loss`, `lens/layerDivergence`, `lens/translators`), `lensDivergence`, `activationPatching` (denoise and noise,
  every site and position, share of the gap recovered), `causalTrace` (ROME: noised subject embeddings, restore stream,
  feed-forward or attention), `attributionPatching` (one gradient for every site), `pathPatching` (sender to chosen
  receivers or the logits, everything else frozen clean), `directLogitAttribution` (exact split of a logit difference
  including the final norm's frozen scale), the metrics `logitDifference`, `tokenLogProbability`, `tokenProbability`, and
  `plantedCopyCircuit` (a hand-built one-layer model whose head 0 copies the name tokens, with its truth). Oracle: none
  (no reference library); tested against the planted circuit and identities (lens at the last depth, self-patching,
  patching every component equals replacing the run, contributions sum to the logit difference).
- Probes and sparse autoencoders (`probing.ts`, `sae.ts`): the probe is `linear`'s multinomial (its solver slot passes
  through) trained on a split with a control task per word type (Hewitt and Liang).
  Built: `probeAccuracy` (train, test, majority baseline), `controlTask`, `probeSelectivity`, `probeLayers` (accuracy,
  control and selectivity down a stack); `saeAlgorithm` (L1 or top-k code, unit-norm dictionary rows, Adam on batches,
  optional resampling of dead features; paths `sae/loss`, `sae/reconstruction`, `sae/l0`, `sae/dead`, `sae/dictionary`),
  `saeEncode`, `saeDecode`, `dictionaryRecovery` (best cosine per planted direction). The plant is the dataset
  `sparseDictionary`. Oracle: none; tested by recovery of the planted dictionary (mean cosine above 0.97) and by the
  dead-feature count with and without resampling.
- Feature visualisation (`visualisation.ts`): activation maximisation over an image as an algorithm.
  Built: `featureVisualisation` (a sigmoid of the pixels, or of the real inverse FFT of a spectrum scaled by 1/f^decay;
  jitter by rolls, scale and rotation by a random affine grid sample, a total-variation penalty, optional blur of pixels;
  Adam or plain ascent; state is the parameters, optimiser and image; paths `vis/activation`, `vis/loss`, `vis/image`),
  the objectives `unitObjective` and `channelObjective`, `totalVariation`. Oracle: none; tested by monotone rise of the
  target channel of a tiny CNN under plain ascent, the smoothness of the Fourier start, the penalty's effect.
- Sanity checks (`sanity.ts`, Adebayo et al.): `randomisationTestAlgorithm` (one layer randomised per step, last layer
  first, cascading or independent; paths `sanity/layer`, `sanity/similarity`, `sanity/map`), `modelRandomisationTest`
  (run to the end), `dataRandomisationTest` (retrain on permuted labels); the comparison is `explanationSimilarity`.
  Tested: gradient and input-times-gradient maps decorrelate, an input-only map stays at similarity 1 and fails.
- Modality helpers (`modalities.ts`): token, superpixel and time-window feature masks and baselines, so every
  attribution applies to text, images and series.
  Built: `tokenise`, `tokenPredictor`, `gridSuperpixels`, `slicSuperpixels`, `timeWindows`, `maskGroups`,
  `maskedPredictor` (a predictor over group presence, so Kernel SHAP, exact Shapley and LIME run over units),
  `expandGroups`.
- Evaluation of explanations (`evaluation.ts`, Quantus's categories): faithfulness (deletion and insertion curves,
  ROAR, faithfulness correlation, monotonicity), robustness (max- and average-sensitivity, local Lipschitz estimate),
  localisation (pointing game, relevance mass and rank against a ground-truth mask), complexity (sparseness, entropy),
  randomisation sanity checks (model parameter and label randomisation; `sanity.ts`), axiomatic properties (completeness, input
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

### 4.40 `cluster/`: clustering algorithms (scikit-learn cluster, scipy hierarchy)

Clustering as traced `Algorithm`s with estimators on the iterative base. A state keeps no past: `observe` exposes the
metrics and the `state` tensors (`labels` `[n]`, `centres` `[k, d]`, ...) declared in the schema, and the tape records
the history; every state serialises (`stateToJson`). Estimators are `iterative(...)` values (`kind: 'clusterer'`), so
`.method(data)`, `.session(data)` and `fittedAt(t)` exist and the standard `model/assignments` probe works at any step;
the distance-based ones take a `metric` from `distances` (or `'precomputed'`, through `distanceMatrix`).
- `kmeans.ts`: Lloyd's algorithm and mini-batch k-means, the initial centres (k-means++, random rows or given) and the
  nearest-centre assignment. `kmeans-variants.ts`: Elkan's bounds (`elkanAlgorithm`, counting the distances computed and
  skipped), bisecting k-means (one split per step) and kernel k-means (`kernelKMeansAlgorithm` over a Gram matrix, with
  the spectral relaxation as an initial labelling), each with an estimator (`elkanKMeans`, `bisectingKMeans`,
  `kernelKMeans`).
- `density.ts`: `dbscanAlgorithm` (one cluster grown per step), `opticsAlgorithm` (one point of the reachability
  ordering per step) with `opticsOrdering` and `opticsClustersAt`, `hdbscan` (core distances, the single-linkage tree of
  the mutual reachability distance, `condenseTree`, `selectClusters` by excess of mass: labels, probabilities, the
  condensed tree and the stabilities), `meanShiftAlgorithm` (every seed moves each step) with `estimateBandwidth` and
  `meanShiftCentres`; estimators `dbscan`, `optics`, `hdbscanClustering`, `meanShift`.
- `hierarchical.ts`: `agglomerativeAlgorithm` (one merge per step, every linkage by Lance-Williams, the partition and the
  linkage matrix on the tape), `linkageMatrix`, `singleLinkageTree`, `cutTree`, `copheneticDistances`,
  `copheneticCorrelation`, `dendrogram`; BIRCH as a CF-tree in an arena (`birchAlgorithm` inserts one row per step,
  node splits and the chain of leaves as scikit-learn's); estimators `agglomerativeClustering` and `birch`.
- `medoids.ts`: PAM (`pamAlgorithm`: the BUILD steps, then the best swap per step), CLARA (`claraAlgorithm`: PAM on a
  random sample per step), `kMedoids`. `fuzzy.ts`: `fuzzyCMeansAlgorithm`, `fuzzyMemberships`, `fuzzyCMeans`.
- `spectral.ts`: the assembly over `graph/spectral.ts` (`affinityMatrix`, `spectralClusteringEmbedding`,
  `spectralClusteringAlgorithm`: the embedding, then Lloyd's steps on it) and `spectralClustering`. `exemplar.ts`:
  `affinityPropagationAlgorithm` (one damped round of messages per step, scikit-learn's tie-breaking noise from a key),
  `affinityPropagationFit`, `affinityPropagation`. `constrained.ts`: `copKMeansAlgorithm`, `constrainedAssignment`,
  `brokenConstraints`, `copKMeans`.
- `validity.ts`: the choice of k: `kmeansElbow`, `gapStatistic` (Tibshirani's rule over uniform references) and
  `stabilityIndex` (adjusted Rand agreement of clusterings of subsamples). `_support.ts` is private to the folder.

oracle: scikit-learn KMeans with fixed initial centres (`methods/cluster/kmeans` fixture, kept at its old path, also
checked against Elkan's), DBSCAN, OPTICS and HDBSCAN (`methods/cluster/density`), affinity propagation
(`methods/cluster/exemplar`), scipy linkage and cophenet (`methods/cluster/hierarchical`), BIRCH subcluster centres in
leaf-chain order and labels (`cluster/birch`), mean shift centres, labels and bandwidth (`cluster/mean_shift`)
### 4.40a `decomposition/`: linear and factor decompositions (scikit-learn decomposition, cross_decomposition, random_projection)

Decompositions as transformers: each iterative routine is a traced `Algorithm` with a serialising state, each model an
estimator on the iterative base (`kind: 'transformer'`, so the standard `model/embedding` probe exists at every step and
`fittedAt(t)` is the model so far). `_linear.ts` (private) holds the matrix helpers they share: column means, centring, a
thin SVD with scikit-learn's sign convention, projection and reconstruction.
- `pca.ts`: `pcaSvdAlgorithm` (one step finds every component; exact or the randomised SVD of `numerics/linalg`),
  `powerPcaAlgorithm` (power iteration, one component per step, kept orthogonal to those found), `incrementalPcaAlgorithm`
  (one batch folded into the running mean, variance and SVD per step), `sparsePcaAlgorithm` (a penalised rank-one
  decomposition, one soft-thresholded component per step), `pcaLogLikelihood`, `pcaVerbs` (transform with whitening,
  inverse transform, the probabilistic score) and the estimators `pca` (solver `full`, `randomized` or `power`),
  `incrementalPca`, `sparsePca`.
- `ica.ts`: FastICA on whitened data: `fastIcaAlgorithm` (parallel: symmetric decorrelation, one iteration per step),
  `fastIcaDeflationAlgorithm` (one component per step), the contrasts `logcosh`, `exp` and `cube`, the estimator `fastIca`.
- `nmf.ts`: `nmfAlgorithm` (multiplicative updates for the Frobenius or Kullback-Leibler loss, HALS sweeps for Frobenius;
  the stopping rule lives in the state, so a replay stops where the run did), `nndsvda`, `nmfDivergence`, the estimator
  `nmf` (transform solves W for new rows with H fixed).
- `cross-decomposition.ts`: `plsAlgorithm` (NIPALS for PLS regression, PLS canonical and CCA, one component per step, the
  score correlation on the tape), estimators `plsRegression`, `plsCanonical`, `cca`, and `plsSvd` (closed form).
- `rank-regression.ts`: `pcr` (principal-component regression, closed form, the same fit as a scikit-learn PCA plus
  `LinearRegression` pipeline) and `reducedRankRegression` (multi-output least squares constrained to a rank by the SVD of
  the fitted values; `identity` or `covariance` (canonical) weighting of the outputs, an optional ridge; the Eckart-Young
  excess error is tested). oracle: `decomposition/rank_regression`.
- `probabilistic.ts`: `probabilisticPca` (closed form from the SVD), `factorAnalysisAlgorithm` (scikit-learn's SVD-based
  maximum-likelihood iteration, one per step), `factorAnalysis` with varimax and quartimax (`rotateLoadings`),
  `latentVerbs` and `latentCovariance`. `random-projection.ts`: `randomProjection` (Gaussian and sparse, the
  Johnson-Lindenstrauss dimension, a distortion report).
- TODO: Infomax ICA, online and coordinate-descent NMF with penalties, truncated SVD and dictionary learning as
  estimators (`text/representations.ts` and `optim/dictionary.ts` have the routines), kernel PCA pre-images.

oracle: scikit-learn PCA, IncrementalPCA, FastICA (parallel, with fixed initial unmixing), NMF (`mu` from fixed factors),
PLSRegression, PLSCanonical, CCA, PLSSVD and FactorAnalysis (`methods/decomposition/*` fixtures, kept at their old
paths; PCA is also checked through power iteration), plus deflationary FastICA (`decomposition/ica_deflation`) and NMF
by coordinate descent against HALS (`decomposition/nmf_hals`)

### 4.40b `manifold/`: embeddings and manifold learning (scikit-learn manifold, umap-learn)

Embeddings as traced `Algorithm`s with estimators (`kind: 'transformer'`). The algorithms that iterate observe the
configuration at every step as the `embedding` tensor `[n, k]` (and their objective as a metric), which is what an
`embeddingSteps` record is built from. A fitted embedding has `transform`: the training rows give their fitted
embedding, other rows the inverse-distance-weighted mean of their nearest training rows' (none of these methods has a
closed-form out-of-sample map; the exact Isomap and LLE extensions are TODO). `_support.ts` is private: the embedding
verbs, `pcaStart` (the PCA start, scaled), `scaledNeighbours` (each row's nearest rows with the local scale sigma, the
mean distance to the 4th to 6th nearest, and the scaled squared distance that PaCMAP and TriMap rank neighbours by) and
`drawIndices` (a keyed stream of random rows); `_planted.ts` (private) holds the planted layouts of the tests.
- `mds.ts`: `classicalMds`, `classicalMdsAlgorithm`, `smacofAlgorithm` (a Guttman transform per step; metric, or
  non-metric with disparities by isotonic regression through `learn`'s `poolAdjacentViolators`), estimator `mds`.
- `isomap.ts`: `neighbourEdges`, `geodesicDistances` (Floyd-Warshall over the neighbour graph), `isomap`. `lle.ts`:
  `barycentreWeights`, `locallyLinearEmbedding` (standard; modified, Hessian and LTSA are TODO). `spectral.ts`:
  `spectralEmbedding` (Laplacian eigenmaps over `cluster`'s affinities and `graph/spectral.ts`) and `diffusionMap`.
- `tsne.ts`: `conditionalProbabilities` (the perplexity search), `jointProbabilities`, `sparseJointProbabilities`,
  `tsneGradient`, `barnesHutKl` (a quadtree or octree of the embedding), `tsneAlgorithm` (early exaggeration, momentum,
  gains, the two stages as scikit-learn runs them; exact for dense P, Barnes-Hut for sparse), `tsneSetup`, `tsne`.
- `umap.ts`: `findAbParams` (the curve fit by `optim`), `fuzzySimplicialSet`, `umapAlgorithm` (one epoch of edge and
  negative sampling per step), `umapSetup`, `umap`. `som.ts`: `somAlgorithm` (one epoch per step; observes the
  weights and the rows' grid positions), `uMatrix`, `selfOrganisingMap`.
- `pacmap.ts`: `pacmapPairs` (near pairs by the scaled distance, mid-near pairs as the second closest of six random rows,
  further pairs), `pacmapWeights` (the three-phase schedule 100/100/250 with the mid-near weight falling from 1000 to 3
  and then 0), `pacmapLoss` (bounded pair losses, with their gradient), `pacmapProblem` (the loss at one phase's weights
  as a `SmoothProblem`, so `smoothSolvers` can minimise it), `pacmapAlgorithm` (one iteration per step; the optimiser
  is a slot over the `optimisers` table, Adam with eps 1e-7 as `pacmapAdam` the named default), `pacmap`. `trimap.ts`:
  `trimapTriplets` (nearest neighbours with random outliers, random triplets, weights by the tempered log of the
  similarity ratio), `temperedLog`, `trimapLoss`, `trimapProblem`, `trimapAlgorithm` (delta-bar-delta gains and
  momentum with the reference's step-size adaptation, or any optimiser of the table in place of it), `trimap`. Neither
  package is in the oracle: the pair and triplet losses are checked against hand computations, the layouts by
  trustworthiness on planted manifolds and by the global structure of planted two-scale clusters against UMAP.
- `quality.ts`: `trustworthiness`, `continuity`, `neighbourhoodPreservation`, `coRankingMatrix`, `rnxCurve`,
  `embeddingStress`, `neighbourRanks`.
- TODO: Barnes-Hut error control by a tree built once per step in typed arrays, UMAP with a sparse
  graph (the memberships are dense `[n, n]`), the supervised and parametric variants.

oracle: scikit-learn trustworthiness, Isomap, LocallyLinearEmbedding, SpectralEmbedding and metric SMACOF
(`methods/manifold/*`, kept at their old paths), non-metric SMACOF (`manifold/mds_nonmetric`), and t-SNE through
scikit-learn's own `_joint_probabilities`, `_joint_probabilities_nn`, `_kl_divergence`, `_kl_divergence_bh` (angle 0) and
`_gradient_descent` (`manifold/tsne`; umap-learn is not in the oracle group, so UMAP is checked by properties)

### 4.41 `linear/`: linear and generalised linear models (statsmodels, scikit-learn linear_model)

The algorithms behind linear models, moved from the parked methods layer. `design.ts` (private) builds the design
matrix from `LearnData` (intercept, centring, weights); `solvers.ts` holds least squares, ridge by SVD, IRLS for GLMs
and coordinate descent for the lasso and elastic net; `paths.ts` the alpha grid and the regularisation path over it
(with cross-validation); `model.ts` the linear model's parameters and prediction (`linearModel`); `inference.ts`
covariances, Wald tables, standard errors and intervals. `bayesian.ts` is conjugate Bayesian linear regression (zero-mean
Gaussian prior of precision `alpha`, known noise precision `beta`): the batch posterior `bayesianLinearFit`, the same
posterior one observation per step as `bayesianLinearAlgorithm(design, prior, order)` (a rank-one update of precision and
information; `observe` exposes the metrics `logEvidence` and `meanPredictiveStd` and the `param` tensors `mean` and
`covariance`), `posteriorPredictive` (mean and variance per row), `maximiseEvidence` (MacKay's fixed point for `alpha`
and `beta`) and `polynomialBasis` (features scaled to `[-1, 1]`, then the library's `polynomialFeatures`). The estimator
factories (lasso, ridge, logistic, GLM, robust, online, LARS, isotonic) stay parked until the new methods base; the
sequential Bayesian regression is the method `methods/linear/bayesianRegression`. Goldens of the parked factories keep
their fixtures under `methods/linear/`.

`online.ts` holds the online learners that take one example per step (iterative base, logging `solver/mistakes`):
`perceptronAlgorithm` and `perceptron` (plain, averaged, voted and pocket-with-ratchet variants, passes in the data's
order or a fresh random order each), `kernelPerceptronAlgorithm` and `kernelPerceptron` (the dual), and
`passiveAggressiveAlgorithm`, `passiveAggressive` and `passiveAggressiveRegressor` (PA, PA-I, PA-II). Classifiers are
binary; `learn`'s `oneVsRest` makes them multiclass. oracle: `linear/online` against scikit-learn's SGD with the
perceptron loss (plain and averaged) and the passive-aggressive learning rates, and numpy loops for the voted, pocket
and kernel variants.

`lars.ts` is least angle regression: `larsAlgorithm` (one knot per step, plain LAR or the lasso modification where a
variable leaves when its coefficient crosses zero; logs `alpha`, `active`, the `coefficients` tensor and the message
`event`), `larsPath` (the whole path: penalties, coefficients, entry order), `coefficientsAt` (interpolation at a
penalty) and the `lars` estimator (the model at a penalty or active count, the path in `params.path`; `fittedAt(t)` is
knot `t`). oracle: `linear/lars` against `sklearn.linear_model.lars_path` and `LassoLars`.

`diagnostics.ts` is regression diagnostics from one SVD of the (weighted, intercept-last) design: `olsInfluence` (leverage,
internal and external studentised residuals, Cook's distance, DFFITS and its internal form, DFBETA(S), COVRATIO,
leave-one-out sigma), the single-measure functions (`hatMatrix`, `leverage`, `studentizedResiduals`, `cooksDistance`,
`dffits`, `dfbetas`, `covRatio`), collinearity (`varianceInflationFactors`, `conditionIndices` with variance-decomposition
proportions), the `breuschPagan` (Koenker or classical) and `whiteTest` heteroscedasticity tests, and
`regressionDiagnostics`, the report with the observations the usual cut-offs flag (it adds the Durbin-Watson test of
`stats/tests`). Weighted fits are those of the whitened data. oracle: `linear/diagnostics` against statsmodels
(`OLSInfluence`, `variance_inflation_factor`, `het_breuschpagan`, `het_white`; weighted cases use the whitened ordinary
fit because `OLSInfluence` on a WLS result mixes raw and whitened designs in its hat matrix).

`group-lasso.ts` is structured sparsity: `groupStructure` (labels to groups, with weights sqrt(size) by default),
`groupLassoAlgorithm` and the `groupLasso` estimator (block proximal-gradient sweeps or FISTA with restarts and the
library's `proxGroupLasso`; the KKT violation is the stopping measure and a logged metric), `groupLassoAlphaMax`;
`multiTaskDesign`, `multiTaskLassoAlgorithm` and the `multiTaskLasso` estimator (the l2,1 row penalty, with `l1Ratio`
the multi-task elastic net; `y` is `[n, k]`, `coef` is a row per task as in scikit-learn). Singleton groups are the
lasso, which a test checks against the GLM core. oracle: `linear/group_lasso` against `MultiTaskLasso` and
`MultiTaskElasticNet`; the group lasso itself has no scikit-learn reference and is tested by its optimality conditions.

`ordinal.ts` is ordinal regression: `ordinalLogLikelihood` (the likelihood and its gradient for the cumulative-link
model `P(Y <= k) = F(theta_k - x beta)` with the logit, probit, cloglog and loglog links, and for the sequential
(continuation-ratio) model), `ordinalProbabilities`, `ordinalSmoothProblem` (the objective with its analytic gradient, thresholds as a first cutpoint plus log
increments) and `ordinalAlgorithm` (any solver of the slot, Newton by default with a Hessian from central differences
of the gradient) and the `ordinal` classifier
(`predictProba`, the latent `decisionFunction`, `predictDistribution`, a Wald table of slopes and thresholds). oracle:
`linear/ordinal` against statsmodels' `OrderedModel` (slopes, thresholds, log-likelihood, slope standard errors and
probabilities for the four links; the sequential mode has no reference and is checked by finite differences and its
equality with logistic regression for two categories).

`mixed.ts` is the linear mixed model `y = X beta + Z b_g + e` for grouped data: `mixedProblemOf` summarises each group's
rows into sufficient statistics (the group label is a column of `x`; random intercept and slopes, unstructured or
diagonal covariance), `mixedProfile` is the profiled (restricted) deviance at a relative covariance factor (fixed effects
and residual variance profiled out, the matrix determinant lemma per group), `mixedSmoothProblem` is it as an objective
of the solver slot and `mixedAlgorithm` maximises it with any solver of `smoothSolvers` (Newton by default), and the `mixedModel`
estimator has the fixed-effects Wald table, `randomCovariance`, `sigma2`, the group effects (BLUPs), and `predict`
(group effects for groups seen in fit) and `predictFixed`. Sample weights are refused. oracle: `linear/mixed` against
statsmodels' `MixedLM` (REML and ML; random intercept, and correlated random intercept and slope); statsmodels stops
slightly short of the optimum, so the golden tolerance is 1e-3 and a dense-matrix test checks the likelihood itself.

The solver is a slot, not part of the model (docs/structure.md 4.18, `glm.ts`, `solvers.ts`, `analytic.ts`). The GLM core
takes `solver` as a name or a configuration of `glmSolvers` (IRLS, Newton on the observed information, BFGS, L-BFGS,
conjugate gradient, gradient descent, the stochastic mode with Adam and batches) when the penalty is smooth; `multinomial`,
`ordinal`, `zeroInflated`/`hurdle` and `mixedModel` each build a `SmoothProblem` (`multinomialSmoothProblem`,
`ordinalSmoothProblem`, `countModelSmoothProblem`, `mixedSmoothProblem`, from their analytic gradients through
`analyticObjective`) and take `solver` as a configuration of `smoothSolvers`, default Newton, through `_slot.ts`
(`slotAlgorithm`: the model's state fields read off the solver's state, which is kept inside so any solver serialises
and resumes; `solver/gradNorm` and `solver/step` beside the model's channels). The lasso, elastic net, LARS, the paths and
the group lasso are particular algorithms and stay outside the slot.

`robust.ts` is regression that tolerates outliers: `theilSenAlgorithm` and the `theilSen` estimator (least squares on every
row subset, or `maxSubpopulation` random ones drawn from a key, then Weiszfeld steps to the spatial median of the
solutions, as scikit-learn), `huberAlgorithm`, `huberObjective` and the `huber` estimator (majorise-minimise steps with
a jointly estimated scale, the objective of scikit-learn's `HuberRegressor`, `params.outliers` marking the linearly
treated rows) and `quantileRegression` and the `quantileRegressor` estimator (the pinball loss with an optional L1
penalty, solved exactly by `optim`'s `linprog`). RANSAC stays `stats/robust.ts`'s generic `ransac`. oracle:
`linear/robust` against scikit-learn's `TheilSenRegressor` (all subsets), `HuberRegressor` and `QuantileRegressor`.

`zero-inflated.ts` is count regression for excess zeros: `countProblemOf` (counts, the count design and the zero-part
design, whose features are `'constant'`, `'all'` or a list), `countModelObjective` (the negative log-likelihood and its
analytic gradient for the zero-inflated and the hurdle models over Poisson or negative-binomial counts),
`countModelAlgorithm` (damped Newton steps, Hessian by central differences of the gradient) and the `zeroInflated` and
`hurdle` estimators (`predict` the mean, `predictInflation`, `predictCount`, `predictZero`; Wald tables of the count and
zero parts). oracle: `linear/zero_inflated` against statsmodels' `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomialP`
and, for the hurdle model's two independent parts, `Logit` with `TruncatedLFPoisson` and
`TruncatedLFNegativeBinomialP`.

`expectile.ts` is expectile regression (Newey and Powell): `expectileProblemOf`, `expectileObjective` (the mean
asymmetric squared loss with its gradient and generalised Hessian), `expectileSmoothProblem` for the solver slot,
`expectileAlgorithm` (Newton, which is IRLS, by default, any solver of `smoothSolvers`) and the `expectileRegressor`
estimator with the sandwich covariance. `tobit.ts` is censored regression: `tobitProblemOf` (limits, the censoring
pattern), `tobitObjective` (the concave likelihood in Olsen's parameters `(beta/sigma, 1/sigma)` with analytic gradient
and Hessian), `tobitSmoothProblem`, `tobitAlgorithm` over the slot and the `tobit` estimator (`predict` the latent mean,
`predictCensored` the expected clipped value, Wald errors of the coefficients and sigma by the delta method).
`gls.ts` is generalised least squares: `glsFit` and `gls` (known covariance as a scalar, per-row variances or a matrix,
closed form; `ar1Covariance`), `glsarAlgorithm` and `glsar` (statsmodels' `GLSAR.iterative_fit`: Yule-Walker AR
coefficients alternating with a whitened fit that drops the first p rows) and `heteroscedasticRegressor` (a log-linear
variance `exp(z gamma)`: `method: 'fgls'` iterates a weighted fit with a regression of the log squared residuals, one
round being the textbook estimator, and `method: 'ml'` maximises the joint Gaussian likelihood over the solver slot).
`errors-in-variables.ts` is total least squares and Deming regression (`totalLeastSquares`, `deming`: the smallest
singular vector of the centred, error-scaled `[X y]`). `subset.ts` is variable selection by an information criterion
(`subsetAlgorithm`, `subsetRegression`: best subset, forward, backward and stepwise by AIC, BIC, AICc, Cp or adjusted
R-squared, scored from the Gram matrix, the visited subsets and their criteria on the path). `sur.ts` is seemingly
unrelated regressions (`surAlgorithm`, `sur`: Zellner's feasible GLS over the stacked system, iterable to maximum
likelihood). `_verbs.ts` (affine-map verbs) and `_cov.ts` (inverse and sandwich helpers) are private to the folder.
oracle: `linear/expectile` against scipy BFGS on the asymmetric loss; `linear/tobit` against a statsmodels
`GenericLikelihoodModel` (coefficients, sigma, log-likelihood, standard errors); `linear/gls` against statsmodels `GLS`
and `GLSAR`, with FGLS written as statsmodels `WLS`/`OLS` rounds and the heteroscedastic likelihood by scipy BFGS;
`linear/errors_in_variables` against the orthogonal-distance objective minimised with scipy (`scipy.odr` is deprecated
and returned wrong fits in this build); `linear/subset` against statsmodels OLS criteria over the same searches
written in Python; `linear/sur` against the stacked Kronecker GLS in numpy.

`multinomial.ts` is the multinomial logit (softmax regression as a GLM): `multinomialProblemOf`, `softmaxRows`,
`multinomialAlgorithm` (Newton steps with the exact Hessian) and the `multinomial` classifier (class 0 the reference
when unpenalised, every class free with a ridge penalty; `predictProba`, logits as `decisionFunction`,
`predictDistribution`, a Wald table). oracle: `linear/multinomial` against statsmodels' `MNLogit` (coefficients,
standard errors, log-likelihood, probabilities) and scikit-learn's multinomial `LogisticRegression` (ridge; centred
intercepts, which the penalty leaves determined only up to a common shift).

oracle: `bayesian` forms the closed forms with numpy (`linear/bayesian` fixture): posterior, log evidence, predictive,
and a fixed number of evidence-maximisation updates

### 4.42 `mixture/`: mixture models (scikit-learn mixture)

Expectation-maximisation for a Gaussian mixture, moved from the parked methods layer. `gaussian.ts` holds
`gaussianEmAlgorithm` (full, tied, diagonal or spherical covariances), the k-means or random initialisation, the M-step,
and the responsibilities and per-row log density under a mixture. A state keeps no past: `observe` exposes the metrics
`logLikelihood` and `change` and the tensors `responsibilities` `[n, k]` (state), `means` `[k, d]` and `weights` `[k]`
(params), declared in the schema. `variational.ts` is the Bayesian Gaussian mixture by variational inference
(scikit-learn's `BayesianGaussianMixture`, every covariance type, Dirichlet or Dirichlet-process weight prior):
`variationalMixtureAlgorithm(x, start, priors, options)` with `variationalPriors` for the data-dependent priors,
`expectedWeights` and `variationalResponsibilities`; `observe` exposes the metrics `lowerBound` and `change` and the
tensors `responsibilities`, `means`, `weights` (expected) and `covariances`, declared in the schema; the start is
`initialResponsibilities` of `gaussian.ts` (k-means or random). The estimator factories, component selection and the
density helpers stay parked; the method is `methods/mixture/bgmm`. The EM golden keeps its fixture under
`methods/mixture/gaussian`.

oracle: `variational` starts scikit-learn's `BayesianGaussianMixture` from given responsibilities (its private
`_initialize`) and compares a fixed number of iterations, for both weight priors and all four covariance types
(`mixture/variational` fixture)
(params), declared in the schema. The estimator factories, the Bayesian mixture, component selection and the density
helpers stay parked. The golden keeps its fixture under `methods/mixture/gaussian`.
- `experts.ts`: the mixture of linear experts (Jacobs and Jordan) by EM: `mixtureOfExpertsAlgorithm(x, y, { nExperts,
  ... }, key)` has a softmax gate of linear functions with a bias (the last expert's row held at zero) over
  linear-Gaussian experts, each with its own noise variance. The E-step is the posterior responsibility of each expert
  for each row; the M-step is weighted least squares per expert (`linear`'s `leastSquaresSolve`) and a few damped,
  line-searched Newton steps on the ridge-penalised weighted multinomial logistic objective for the gate. The start is
  one random chord (an exact fit through `d + 1` random rows) per expert. The metrics are `logLikelihood` and `change`;
  the tensors `responsibilities` `[n, k]`, `gate` and `experts` `[k, d + 1]` and `variances` `[k]`. `moeGate`,
  `moeExpertMeans`, `moePredict` (the mixture mean), `moePosterior` and `moeAssignments` read parameters.

### 4.43 `vision/`: classical computer vision (OpenCV)

Classical computer vision, shaped as OpenCV's imgproc, calib, features, video and photo modules, on images as tensors
`[H, W]` or `[H, W, C]` with values in [0, 1]. It builds on `signal/image` (the scikit-image and scipy.ndimage layer:
filters, morphology, Canny, Harris, HOG, Hough, watershed) and never repeats it; where OpenCV's 8-bit semantics matter
(thresholds, FAST, LSD, CLAHE, adaptive thresholds) the functions see the image at 256 levels and emulate OpenCV's
float32 arithmetic, so they match it exactly. Point sets are `[n, 2]` as (x, y); boxes `[n, 4]` as (x1, y1, x2, y2).
Iterative pieces are traced `Algorithm`s (SORT, KLT, background models, anisotropic diffusion, inpainting, SLIC, active
contours, GrabCut, AdaBoost), each with a function that runs it. Exact outputs of SLIC, GrabCut, snakes, mean-shift
segmentation and seam carving are implementation-defined, so those are property-tested.

oracle: OpenCV 5 (opencv-python-headless, group `oracle-vision`), scikit-image, torchvision.ops, scipy (least squares
references); goldens per module under `fixtures/vision/`
built from: `signal/image`, `numerics/linalg` (blas `dgemm`, lapack), `numerics/geometry`, `optim`
(Levenberg-Marquardt, linear sum assignment), `stats` (RANSAC, quantiles), `dynamics/filtering`, `mixture`, `cluster`,
`graph` (min cut), `metrics` (box IoU), `random`, `trace`
- `_image.ts` (private): rasters, 256-level views, float32 and fused multiply-add emulation, OpenCV's fastAtan2.
- `colour.ts`: `cvtColor` and its conversions between RGB, grey (BT.601 and BT.709), HSV, XYZ, Lab (D65 or another
  white) and YCrCb, golden against OpenCV. `colour-balance.ts`: sRGB linearisation, gamma, grey-world and white-patch
  balance, channel split and merge.
- `histograms.ts`: `calcHist`, `calcBackProject`, `equalizeHist`, `clahe` (bit-exact on arm64), `matchHistograms`,
  `rescaleIntensity`, `contrastStretch`, `normalize`, `compareHist` (correlation, chi-square, intersection,
  Bhattacharyya, KL).
- `blur.ts`: OpenCV's border modes, `getGaussianKernel` (its bit-exact tables), `gaussianBlur`, `blur`, `medianBlur`,
  `filter2D`.
- `thresholds.ts`: `threshold` (binary, inverse, truncate, to zero, with Otsu, Li or triangle), `adaptiveThreshold`
  (mean and Gaussian, 8-bit semantics), multi-level Otsu, Niblack and Sauvola.
- `contours.ts`: `findContours` (Suzuki and Abe, OpenCV's hierarchy and order, every retrieval mode, chain or simple
  approximation), `approxPolyDP`, `arcLength`, `contourArea`, `boundingRect`, `isContourConvex`, `pointPolygonTest`.
  `moments.ts`: spatial, central and normalised moments, Hu moments, `matchShapes`. `hull.ts`: `convexHull`
  (Sklansky), `convexityDefects`. `fitting.ts`: `minAreaRect`, `boxPoints`, `minEnclosingCircle`, `fitEllipse`,
  `fitLine` (least squares and robust M-estimators). `components.ts`: `connectedComponentsWithStats`, `floodFill`.
- `draw.ts`: functional drawing returning new images: OpenCV's line iterator, Wu's anti-aliased lines, rectangles,
  circles, ellipses, polylines, filled polygons, and contour, keypoint and match views.
- `boxes.ts`: box formats, area, clipping, scaling, `nms`, `batchedNms`, `softNms` (as cv2.dnn), anchors, box
  encoding and decoding, anchor matching (torchvision's Matcher), golden against torchvision.ops.
- `fast.ts`: FAST-9 corners with OpenCV's score and suppression. `scale-space.ts`: Gaussian and difference-of-Gaussian
  scale spaces and SIFT-style keypoints (sub-pixel refinement, edge rejection, orientations). `descriptors.ts`:
  SIFT-like descriptors, BRIEF and ORB (keyed binary patterns). `matching.ts`: brute-force matching (L1, L2, Hamming,
  cross-check, k nearest) as BFMatcher, and the ratio test.
- `geometry.ts`: `getPerspectiveTransform`, `getAffineTransform`, `perspectiveTransform`, `findHomography`,
  `estimateAffine2D` and `findFundamentalMat` (direct, RANSAC through `stats`, with local re-scoring and
  Levenberg-Marquardt refinement), epipolar lines, Sampson distance. `epipolar.ts`: essential matrices (five-point
  free: normalised eight-point with the rank constraint), their decomposition, `triangulatePoints`, `recoverPose`.
- `camera.ts`: Rodrigues vectors, `projectPoints` (OpenCV's radial and tangential distortion, differentiable
  `projectValue`), `undistortPoints`, `initUndistortRectifyMap`, `remap`, `undistortImage`, reprojection error,
  `solvePnP` and `solvePnPRansac`. `calibration.ts`: `calibrateCamera` (Zhang's method with zero skew, then
  Levenberg-Marquardt over intrinsics, distortion and poses) and `stereoRectify` (Bouguet).
- `stereo.ts`: block matching (SAD, SSD or NCC, sub-pixel, left-right check), semi-global matching (4 or 8 paths),
  `disparityToDepth`, `reprojectImageTo3D`.
- `tracking.ts`: SORT (a Kalman filter per track from `dynamics/filtering`, Hungarian assignment on IoU from `optim`),
  `meanShift` and `camShift` as OpenCV, pyramidal Lucas-Kanade (`calcOpticalFlowPyrLK`, `kltAlgorithm`).
  `background.ts`: `accumulateWeighted`, a running-average model and Stauffer-Grimson per-pixel mixtures.
- `denoise.ts`: non-local means, `denoiseTvChambolle` (golden against scikit-image), Perona-Malik diffusion, the guided
  filter, harmonic inpainting (the adaptive Wiener filter stays `signal`'s `wiener`).
- `segmentation.ts`: SLIC superpixels, seeded region growing, mean-shift segmentation, k-means colour quantisation
  (over `cluster`). `snakes.ts`: active contours. `grabcut.ts`: GrabCut with mixture colour models and min cuts.
  `seams.ts`: seam energy, minimum seams, removal, content-aware resizing.
- `pyramids.ts`: `pyrDown`, `pyrUp` (OpenCV's kernels, borders and odd sizes), `buildPyramid`, Laplacian pyramids and
  multiresolution blending.
- `haar.ts`: Haar-like features (scikit-image's types and order), discrete AdaBoost on stumps, attentional cascades,
  `detectMultiScale`. `hog-detector.ts`: HOG features of windows and sliding-window detection with a linear classifier
  over a pyramid. `lines.ts`: line segment detection as OpenCV's LSD (exact at LSD_REFINE_NONE, scale 1), validated by
  density and the number of false alarms.
- Stepped forms for sessions (`edges.ts`, and beside the one-shot routines): `cannyAlgorithm` (a step per stage: smoothing,
  gradient, non-maximum suppression, double threshold, then one hysteresis round per step; the last image is `canny`'s;
  `cannyStages` in `signal/image/features.ts` returns every stage and `canny` is its last), `houghLineAlgorithm` (the
  accumulator filling a pixel's votes per step, equal to `houghLine` at the end) with `houghLinePeaks` (strongest cell,
  suppression window, repeat), `watershedAlgorithm` in `signal/image/segmentation.ts` (flooding by level over the same
  queue as `watershed`, so the labels are identical; the queue is serialised as arrays) and
  `meanShiftSegmentationAlgorithm` in `segmentation.ts` (one window iteration per pixel per step, the modes grouped into
  regions every step; the one-shot function is the same iteration). Paths `canny/*`, `hough/*`, `flood/*`, `shift/*`.
  Tested against the one-shot results (identical), planted lines, planted basins and planted region colours.
- Calibration and structure from motion in steps (`calibration.ts`, `refinement.ts`, `sfm.ts`): `calibrationAlgorithm`
  (the first state is Zhang's closed form, each step one Levenberg-Marquardt iteration of the joint refinement of
  intrinsics, distortion and poses, the calibration and RMS reprojection error at the iterate in the state; the same
  optimum as `calibrateCamera`, which now shares its setup), `steppedRefinement` (the wrapper that makes `optim`'s
  Levenberg-Marquardt algorithm a serialisable refinement with a decoded geometry; paths `refine/cost`,
  `refine/reprojectionError`, `refine/parameters`), `essentialRansacAlgorithm` (one eight-point hypothesis per step, the
  best inliers in the state), `twoViewReconstruction` (essential matrix refit to the inliers, cheirality pose, inliers
  triangulated, up to scale), `addView` (`solvePnP` against the points built so far), `bundleAdjustmentAlgorithm` (all
  poses but the first and all points, one iteration per step, with the scale free) and `bundleReprojectionError`. Tested
  on planted rigs: exact correspondences give the pose and structure to 1e-6, noisy ones are corrected by bundle
  adjustment with the error non-increasing, calibration recovers a planted intrinsic matrix and distortion.
- Image quality (PSNR, SSIM and the rest) is `metrics/image-quality.ts`; neural detection heads stay in `methods/vision` (5.x).
- TODO: TC89 chain approximations and RETR_FLOODFILL, `fitEllipseDirect` and AMS, Telea inpainting, the fast mode of
  non-local means, stereoRectify's alpha-dependent camera matrices, StereoBM's pre-filter and SGBM's Birchfield-Tomasi
  cost, LSD's sub-sampling and refinement, OpenCV's cascade XML and HOGDescriptor layouts.

### 4.44 `distances/`: named distances and similarities (scipy.spatial.distance, sklearn.metrics.pairwise)

One entry point for distances and similarities between points, so that a page, a clusterer or a nearest-neighbour
index picks a metric by name and a control can draw its knobs. Every computation is `numerics/geometry`'s
`distanceBetween` (section 4.10); this domain names them under one vocabulary and arranges them as registries.
- `distances.ts`: `DistanceEntry` (`{ name, description, metric, parameters, compute, scalar }`: whether the triangle
  inequality holds, the knobs as a `DictSpace`, the differentiable `compute` over the coordinate axis of broadcast
  operands, and the plain-number `scalar` form from `metricFunction`), the entries `euclideanDistance`,
  `squaredEuclideanDistance`, `manhattanDistance`, `chebyshevDistance`, `minkowskiDistance` (`p`),
  `seuclideanDistance` (`variance`), `cosineDistance`, `correlationDistance`, `mahalanobisDistance` (`covariance` or
  `inverseCovariance`), `hammingDistance`, `jaccardDistance`, `canberraDistance`, `braycurtisDistance`,
  `haversineDistance` (`degrees`, `radius`), the table `distances`, `distanceNames`, `distanceSpace` (a union over the
  names, a value being `{ name, ...knobs }`), `distanceEntry` (names and the aliases `l1`, `l2`, `cityblock`,
  `sqeuclidean`, `linf`) and `DistanceOptions` (`weights` too; tensor parameters are options, not knobs).
- `similarities.ts`: `SimilarityEntry` (`{ name, description, range, parameters, compute }`), the entries
  `cosineSimilarity`, `dotSimilarity`, `pearsonSimilarity`, `gaussianSimilarity` and `laplacianSimilarity` (`gamma`
  or a `bandwidth`), `inverseDistanceSimilarity` (1 / (1 + d / scale) over any registered `distance`, whose knobs join
  the space as conditional dimensions), the table `similarities`, `similarityNames`, `similaritySpace`,
  `similarityEntry` (aliases `rbf`, `linear`, `correlation`), and the conversions `distanceToSimilarity` (inverse,
  exponential, Gaussian, linear) and `similarityToDistance` (linear, inverse, chord: the kernel-induced distance of a
  normalised kernel).
- `evaluate.ts`: `distance(a, b, name, options)` and `similarity(a, b, name, options)` between corresponding rows, two
  points, or a point and rows; `pairwiseDistances(a, b?, name, options)` and `pairwiseSimilarities(a, b?, name,
  options)` for the `[m, n]` matrix (`b` omitted: among the rows of `a`); shapes checked once, scipy's defaults for
  seuclidean and mahalanobis filled from the points. `distanceMatrix(x, metric, options)` is the flat `n x n` form
  for algorithms that work on distances (clusterers, embeddings, outlier factors), with `'precomputed'` as a metric
  (the input is the matrix). An unknown name is a `value` error, mismatched points a `shape` error.

oracle: scipy.spatial.distance (cdist and the scalar functions row by row), scikit-learn's cosine_similarity,
linear_kernel, rbf_kernel, laplacian_kernel and haversine_distances, numpy.corrcoef (`fixtures/distances/evaluate.json`);
properties: symmetry, zero diagonal, the triangle inequality of the metric entries, ranges, elementwise equals the
diagonal of pairwise, derivative conformance of every continuous distance.
built from: `numerics/geometry`, `numerics/linalg`, `space`, `tensor`. Sequence and string distances (DTW in
`signal/similarity.ts`, Levenshtein and set Jaccard in `text/similarity.ts`, chi-square in `kernels/structured.ts`)
keep their homes, as do the `Kernel` objects of `kernels/` (`pairwiseKernels` by scikit-learn's names).

### 4.45 `timeseries/`: time-series models and mining (statsmodels, stumpy, sktime)

Series modelled and mined, moved out of the parked methods layer and extended to what `methods/time-series` asks of
the library. A domain over `signal` (sliding dot products, MASS, Yule-Walker and Burg), `dynamics/filtering` (the
Kalman filter, BOCPD), `optim` (BFGS), `stats` and `numerics`. Every iterative routine is an `Algorithm` whose state
serialises (`stateToJson`), and every fitted model is plain JSON data with pure functions over it, so it serialises.
Series are `TensorLike` or number arrays; AR coefficients follow `x_t = sum phi_j x_(t-j) + e_t + sum theta_j e_(t-j)`
throughout (as `signal/parametric.ts` and `datasets/timeseries.ts`).
- `differencing.ts`: `difference(y, order, lag)`, `differencePolynomial(d, D, s)` (the coefficients of
  `(1 - B)^d (1 - B^s)^D`), `integrate(z, initial, order, lag)`, the inverse.
- `acf.ts`: `autocovariance` (biased or adjusted), `acf` (with Bartlett bands), `pacf` (Yule-Walker adjusted or biased,
  or by regressions on lags) and `ccf`, as statsmodels.
- `arma.ts`: the theory of ARMA processes: `armaPsiWeights`, `armaPiWeights`, `armaAutocovariance`, `armaAcf`,
  `armaPacf`, `isStationary`, `isInvertible`, and the innovations algorithm `innovationsAlgorithm(gamma)` (one row, one
  lag, per step; state `InnovationsState`) with its runner `innovations`.
- `state-space.ts`: `armaForm` (the Harvey state-space form with the stationary initial covariance by doubling),
  `armaStateSpace` (as a `LinearGaussianModel`, so `kalmanLogLikelihood` applies), the O(r^2)-per-step scalar filter
  `armaFilter` and `armaLogLikelihood` (exact Gaussian likelihood, variance concentrated out, missing values skipped),
  and `forecastStateSpace` (h-step means and variances of any single-noise linear Gaussian form).
- `arima.ts`: `arimaFit(y, spec, options)` for ARIMA(p, d, q)(P, D, Q)_s with trend (`n`, `c`, `t`, `ct`) and exogenous
  regressors (regression with ARIMA errors, as statsmodels' `ARIMA`), by `mle`, `css` or `css-mle`; the fit is
  `arimaFitAlgorithm` (BFGS on the unconstrained partial autocorrelations, so the polynomials stay stationary and
  invertible; `logLikelihood` per iteration; the likelihood is that of the differenced series, the diffuse limit),
  `arimaResiduals`, `arimaForecast` (exact means and variances on the original scale by augmenting the state with the
  last observations that the differencing operator integrates; intervals by level). `ArimaFit` is plain data with
  standard errors from the numerical Hessian.
- `arma-fit.ts`: `fitAr` (Yule-Walker adjusted or biased, Burg, least squares), `selectArOrder` (AIC, BIC, HQIC),
  `fitMa` (innovations), `hannanRissanen` (with the bias-correction step), `fitArma` and `armaFitAlgorithm`.
- `smoothing.ts`: ETS models (`EtsSpec`: error, trend, damped, seasonal, period; the presets `smoothingModels`
  (`simple`, `holt`, `damped`), `holtWintersAdditive`, `holtWintersMultiplicative`): `etsFilter` (fitted values,
  residuals, states and the concentrated log-likelihood at given parameters and initial states), `etsAlgorithm` (one
  observation per step: forecast, residual, states and the running likelihood as metrics; serialisable), `etsFitAlgorithm`
  and `etsFit` (BFGS on logistic-bounded smoothing parameters and the initial states), `etsForecast` (point forecasts,
  analytic standard errors and intervals for additive models). Recursions and likelihood follow statsmodels' `ETSModel`.
- `decomposition.ts`: `seasonalDecompose` (classical, moving averages, additive or multiplicative, one- or two-sided,
  trend extrapolation) and the shared `DecompositionResult`.
- `stl.ts`: `stlAlgorithm(y, options)` (STL by loess, a port of the netlib routine as statsmodels has it; one inner
  loop per step, robustness weights after each pass), `stl`, `mstlAlgorithm` (one STL fit per step) and `mstl`.
- the matrix profile family, ported from the parked methods layer with its STUMPY goldens (fixtures kept at their old
  path `fixtures/methods/timeseries/matrix_profile/`), over the kernels of `signal/sliding.ts` and `signal/mass.ts`:
  `matrix-profile.ts` (`stumpAlgorithm`/`stump` by diagonals with the O(1) covariance update, z-normalised or raw
  p-norm distances (AAMP), AB-joins and top-k neighbours with left and right neighbours; `stompAlgorithm`/`stomp` by rows;
  `stampAlgorithm`/`stamp` by random rows through MASS), `scrimp.ts` (`prescrump`, `scrumpAlgorithm`/`scrump`: the
  anytime SCRIMP++), `stumpi.ts` (`stumpiInit`, `stumpiUpdate`, `stumpiAlgorithm`: the streaming profile, with or
  without egress), `motifs.ts` (`match`, `motifs`, `discords`, `atsc`, `allc`: anchored and all time-series chains),
  `segmentation.ts` (`arcCounts`, `idealArcCurve`, `correctedArcCurve`, `regimeLocations`, `fluss`, and the streaming
  `flossInit`, `flossUpdate`, `flossAlgorithm`), `snippets.ts` (`mpdist`, `snippets`, `ostinato`), `pan-profile.ts`
  (`stimpAlgorithm`/`stimp`, breadth-first over window lengths; `panImage`) and `multidimensional-profile.ts`
  (`mstumpAlgorithm`/`mstump`, `multiDistanceProfile`, `subspace`, `mdl`). Each algorithm's step is a batch of
  diagonals or rows (or one value, or one window length) and its state serialises.
- `online-detection.ts`: `cusumAlgorithm`/`cusum` (two-sided tabular CUSUM in standard deviations, with a known or
  warm-up-estimated target) and `pageHinkleyAlgorithm`/`pageHinkley` (cumulative deviation from the running mean above its
  minimum, with forgetting and direction), one observation per step, recording their statistics and alarms. Bayesian
  online changepoint detection stays in `dynamics/filtering/changepoint.ts` and is not repeated.
- `volatility.ts`: GARCH(p, o, q) in the `arch` convention (`GarchSpec`; ARCH, GJR through `o`; constant or zero
  mean; Gaussian or standardised Student-t): `garchVariance` (the recursion with the `arch` backcast),
  `garchLogLikelihood`, `garchFilterAlgorithm` (the recursion one observation per step with the running likelihood),
  `garchFitAlgorithm`/`garchFit` (BFGS on parameters mapped into the stationary region, on the series scaled to unit
  spread; standard errors from the numerical Hessian), `garchForecast` (variance forecasts relaxing to the unconditional
  variance) and `garchSimulate`. The fit also takes a `solver` from the slot (`garchFitSlot`, or `garchFit` with
  `options.solver`): the likelihood is a smooth objective of the unconstrained parameters (`_ml.ts`: `likelihoodSlot`,
  `finiteDifferenceProblem`), BFGS by default, and L-BFGS, conjugate gradient and the rest reach the same optimum.
- `asymmetric-volatility.ts`: EGARCH(p, o, q) (the log variance with |z| - E|z| and signed z terms, Gaussian or
  Student-t) and APARCH(p, q) (the power delta estimated): `asymmetricVariance`, `asymmetricLogLikelihood`,
  `asymmetricFitAlgorithm`/`asymmetricFit` (over the solver slot, on the unit-spread scaling with the parameters mapped
  back), `asymmetricSimulate`, `asymmetricForecast` (the one-step recursion, simulation beyond it) and
  `newsImpactCurve` (Engle-Ng, for GARCH, GJR, EGARCH and APARCH). `arch` is not an oracle dependency, so the models are
  tested by hand-computed recursions, the GARCH limit of APARCH, recovery of planted parameters and agreement of solvers.
- `ets-intervals.ts`: prediction intervals of every ETS type after Hyndman et al.: `etsClass`, `etsPredictionVariance`
  (closed form for class 1, additive error, and class 2, multiplicative error with additive or no trend and season:
  (1 + sigma2) theta_h - mu_h^2) and `etsPredictionIntervals` (closed form where there is one, else the quantiles of
  `etsSimulate` paths drawn from the final state through the model's own recursion, which also checks the closed forms).
  Multiplicative season with multiplicative error (class 3) and multiplicative trends are simulated.
- `box-cox.ts`: `guerreroLambda` (the exponent that makes the block spread independent of the level) and `boxCoxMstl`
  (MSTL on the transformed series with lambda given, by Guerrero or by profile likelihood, the fit, trend and seasonal
  factors carried back to the original scale).
- `features.ts`: `timeSeriesFeatures` (a labelled vector of about 50 features: location and shape, autocorrelation, sample,
  approximate, permutation and spectral entropy, spectral shape, trend and stability, crossings and strikes, tsfresh-style
  closed forms, and with `set: 'all'` the Hurst and detrended-fluctuation exponents), `featureMatrix` and
  `permutationEntropy`. In the spirit of catch22 and tsfresh with the definitions written out here; the values are not
  the pycatch22 numbers.
- `shapelets.ts`: `shapeletDistance` (early abandoning, optional z-normalisation), `shapeletSplit` (information gain with
  the gap tie-break), `shapeletDiscoveryAlgorithm`/`findShapelets` (one candidate per step, best non-overlapping kept,
  state serialisable) and `shapeletTransform`.
- `learned-shapelets.ts`: learned shapelets (Grabocka et al.): `softMinDistance`, `initialShapelets` (k-means centres of
  the windows), `learnedShapeletProblem` (the cross-entropy of a softmax classifier on soft-minimum distances, as a
  `SmoothProblem` with batches of series and the analytic gradient through the soft minimum), `learnedShapeletsAlgorithm`
  (a step of any solver of the slot; the stochastic `gradient` mode with Adam on mini-batches is the default
  `learnedShapeletSolver`; the shapelets are a tensor per step), `fitLearnedShapelets`, `predictLearnedShapelets`.
  Started from one k-means centre it can sit on the noise plateau where no window of the pattern has weight; a few
  shapelets (k-means++ seeding finds the far windows) avoid it. Truth: `datasets/shapelet-series.ts`.
- `var.ts`: `varFit` (least squares per equation with statsmodels' covariance, likelihood, AIC/BIC/HQIC/FPE and
  standard errors), `varSelectOrder`, `varForecast` (with intervals), `varImpulseResponse` (plain, orthogonalised,
  cumulative), `varForecastErrorVarianceDecomposition`, `varStability` (companion eigenvalues) and `grangerCausality`
  (F or Wald). Matrices are arrays of rows, via the private `_matrix.ts`.
- `hierarchy.ts`: `treeSummingMatrix`, `groupedSummingMatrix`, `reconcile` (bottom-up, OLS, structural and variance WLS,
  MinT with the sample or shrunk covariance of the base errors: y~ = S G y^), `shrinkCovariance`, `topDownProportions`
  and `isCoherent`.
- `cointegration.ts`: `engleGranger` (the two-step test: the cointegrating regression with `n`/`c`/`ct`/`ctt`, the
  residual spread tested by ADF with MacKinnon's N-variable p-values and critical values), `johansen` (the trace and
  maximum-eigenvalue statistics from the reduced-rank eigenproblem with the MacKinnon-Haug-Michelis tables of
  `_johansen-tables.ts`) and `selectCointegrationRank`.
- `vecm.ts`: the vector error-correction model. `vecmProblem` builds Johansen's matrices (deterministic terms `n`, `co`,
  `ci`, `lo`, `li` and the combinations), `vecmSmoothProblem` is the likelihood concentrated on beta = [I; B] as a
  smooth objective with its analytic gradient, `vecmSolvers` is the slot (`johansen` the closed-form eigen-solve as the
  model's own mode, beside every generic solver), `vecmAlgorithm`/`vecmFit` fit through it, `vecmForecast` iterates the
  VAR form (`vecmVarRepresentation`) with intervals from the moving-average representation.
- private: `_params.ts` (partial autocorrelations to and from coefficients, the stationary transform), `_ml.ts`
  (a number-level likelihood as an `Objective` by central differences for `bfgsAlgorithm`, least squares, a numerical
  Hessian), `_arima-model.ts` (specifications, regression columns, seasonal polynomial expansion, the CSS and exact
  objectives), `_series.ts`, `_profile.ts` (STUMPY's diagonal kernels and the `MatrixProfile` container) and
  `_matrix-profile.ts` (the setup, merge and batch helpers of the profile algorithms).

oracle: statsmodels (`acf`, `acovf`, `pacf`, `ccf`; `arma_acovf`, `arma_acf`, `arma_pacf`, `arma2ma`, `arma2ar`,
`innovations_algo`; `ARIMA` fits and forecasts with exogenous regressors and seasonal terms; `hannan_rissanen`,
`innovations`, `AutoReg`; `ETSModel` smoothing, likelihood and prediction intervals; `seasonal_decompose`, `STL`, `MSTL`;
`VAR` fits, order selection, forecasts, impulse responses, variance decompositions and Granger tests; `coint`,
`coint_johansen` and `VECM` fits with forecasts) in `fixtures/timeseries/`; CUSUM, Page-Hinkley, GARCH and the reconciliation have no available reference and are tested by
recursion identities, recovery of simulated parameters and the algebra of the solutions; properties: the likelihood by the fast filter equals the library's
Kalman filter and the innovations form, psi and pi weights multiply to one, integrate inverts difference.
built from: `signal`, `dynamics/filtering`, `optim`, `autodiff` (the custom derivative), `numerics`, `trace`.

### 4.46 `svm/`: support vector machines (sklearn.svm, libsvm)

oracle: scikit-learn (`SVC`, `NuSVC`, `SVR`, `NuSVR`, `OneClassSVM`); `optim/smo_nu` for the nu and one-class duals
Every estimator is built on the iterative base (`learn/bases/iterative.ts`), so `.fit`, `.method(data)` and
`.session(data)` all exist. The fit is the dual solve stepped: one SMO working-pair update per step, without shrinking so
the gradient (and with it the objective and the gap) is exact at every step. Every machine logs `solver/dual` (the dual
objective, to be maximised), `solver/gap` (the primal-dual gap, at least zero and closing to zero), `solver/violation`,
`solver/support`, `solver/alphas`, `solver/pair`, `smo/alphaChanged` (|da_i| + |da_j| of the step) and the derived
`model/support-vectors` (a `marginRecord` of the model at that step); the base adds `model/params`, `model/predictions`
and the decision surface. Parameters are plain JSON (the support vectors, their coefficients, the intercepts and the
resolved kernel), so `revive` rebuilds the verbs and `fittedAt(t)` gives the model at any step.
- `_dual.ts` (private): the six duals behind one interface (`Dual`: the algorithm, the kernel expansion of a state, the
  dual, the gap, the margins, the tube): `cSvcDual`, `nuSvcDual`, `svrDual`, `nuSvrDual`, `oneClassDual`, `svddDual`. The
  gap of each is its primal at the best slack threshold (`hingeMinimum`) minus the dual.
- `classifier.ts`: `svc` (C-SVC; `form: 'nu'` is NuSVC, also `nuSvc`): any `kernels` kernel (`rbf`, `linear`, `poly`,
  `sigmoid`, `laplacian` by scikit-learn's names with `gamma` of `'scale'`, `'auto'` or a number, or a `Kernel` object),
  one-versus-one for more than two classes (the pairs solved in turn, libsvm's class order and scikit-learn's
  `decisionShape` `ovr` or `ovo`), class weights (`balanced` or by label) and sample weights, and Platt probabilities
  (`probability: true`: a last step fits each pair's sigmoid on internal cross-validated decision values, coupled for more
  than two classes). The parameters hold `supportIndex` and `nSupport` (scikit-learn's `support_`, `n_support_`) and
  per pair the coefficients and intercept.
- `regressor.ts`: `svr` (epsilon-SVR; `form: 'nu'` is NuSVR, also `nuSvr`), the tube width given or found
  (`params.tube`), the probe's margins being the residuals.
- `one-class.ts`: `oneClassSvm` (nu) and `svdd` (C, at least 1/n), the signed score positive inside the learnt region;
  with a kernel of constant diagonal they are one machine (C = 1 / (nu n)).
- `probability.ts`: `pairwiseCoupling` (libsvm's second method of Wu, Lin and Weng).
  The duals themselves are in `optim/smo.ts` (`oneClassDual`, `svddDual`, `smoRho`) and `optim/smo-nu.ts` (`nuSmoAlgorithm`,
  libsvm's Solver_NU: both equalities kept by moving pairs within a class; `nuSvcDual`, `nuSvrDual`, `nuRho`). Kernel
  choice by scikit-learn's names is `kernels/named.ts` (`resolveKernel`, `kernelMatrix`, `kernelSpace`).
  tests: goldens against scikit-learn on small data (support vectors, dual coefficients, intercepts, ovo and ovr decision
  values, predictions; libsvm keeps its kernel cache in single precision, so agreement is to 1e-4, and the nu forms divide
  by the margin, so 1e-3); probabilities (scikit-learn's cross-validation is random) are checked by properties.
  built from: `optim` (SMO), `kernels`, `learn` (the iterative base, Platt scaling), `exposition`, `trace`.

### 4.47 `gp/`: Gaussian processes (sklearn.gaussian_process, GPML)

oracle: scikit-learn (`GaussianProcessRegressor`, `GaussianProcessClassifier`, `ARDRegression`); numpy forms for EP (GPML Algorithm 3.5 with loops) and the dense sparse-GP likelihoods
Estimators on the iterative base. The hyperparameter fits are one stepped L-BFGS-B search (`_search.ts`: one iteration per
step, restarts in turn, the best run kept; the objective is a `customVjp` function whose derivative is the analytic
gradient, so any `optim` minimiser could use it).
- `likelihood.ts`: `logMarginalLikelihood` (value and gradient in theta through `gramGradient`, scikit-learn's
  `log_marginal_likelihood(theta, eval_gradient=True)`), `negativeLogMarginalLikelihood`, `hyperparameterCount`.
- `regression.ts`: `gpRegression` over any `Kernel` (alpha noise, `normalizeY`, `restarts`, bounds as scikit-learn's): logs
  `gp/logMarginalLikelihood` and `gp/hyper` per step; verbs `predict`, `predictWithStd`, `predictCovariance`,
  `predictDistribution`, `sample` (posterior functions); probes `model/posterior`, `model/samples`, `solver/surface` (the
  likelihood over two hyperparameters, a `hyperparameterSurface`).
- `approximations.ts`: `laplaceAlgorithm` (one Newton iteration per step), `epAlgorithm` (one site update per step, probit),
  `laplaceMarginal` and `epMarginal` (approximate log marginal likelihood with gradient, Algorithm 5.1 and its EP form),
  `latentTerms`.
- `classification.ts`: `gpClassifier` (two classes; `method` laplace or ep, `likelihood` logit or probit): phase one the
  hyperparameter search, phase two the approximation itself stepped (`gp/latent`); `predictive` scikit-learn's five-erf
  approximation or quadrature.
- `sparse.ts`: `sparseGp` (`sor`, `dtc`, `fitc`, `vfe`; the kernel, the noise and the inducing inputs optimised together by
  autodiff of the Woodbury form; logs `gp/bound`, `gp/hyper`, `gp/inducing`), `sparseBound`.
- `relevance-vector.ts`: `relevanceVectorMachine`, the evidence updates of `ARDRegression` on the kernel columns, pruning
  at a precision threshold; logs `solver/precisions`, `solver/relevant`, `solver/logEvidence`.
  tests: goldens against scikit-learn (fixed kernels to 1e-8 for the likelihood, gradient, mean, std and covariance; the
  optimised fit and the Laplace classifier, including gradients and latent moments), a numpy GPML reference for EP, dense
  numpy forms for the sparse approximations, ARDRegression for the RVM; properties by finite differences and exact limits (one
  observation for EP, two by quadrature, m = n for the sparse bounds). Not built: multi-class and multi-output GPs, the
  stochastic variational GP, RVM classification.
  built from: `kernels`, `optim` (L-BFGS-B), `learn`, `stochastic` (posterior sampling), `numerics`, `exposition`, `trace`.

### 4.48 `trees/`: decision trees (scikit-learn tree, river)

A fitted tree is plain arrays in the layout of scikit-learn's `tree_` (`Tree`), with categorical subset splits and a
missing-value direction per split; `tree.ts` holds prediction, `applyTree`, `decisionPath`, `nodeRegions`, impurity
importances, `treeToStructure`, `exportText`, and `TreeModel` and `treePredict` (the scalar-valued view tree SHAP and rule
extraction read; moved from `explain/shapley.ts`, which re-exports them). `criteria.ts` holds Gini, entropy (bits; the same
as log loss), squared error, Friedman's improvement, absolute error, Poisson deviance and C4.5's gain ratio, and
`searchSplits`: every threshold between consecutive values (or one random threshold, for extra trees), the best subset of
the codes of a categorical feature, missing values sent to the better side, sample weights and leaf-size limits, with
every candidate evaluated and recorded. `grow.ts` is the grower as an `Algorithm`, one node decided per step, depth-first
(ids in preorder, as scikit-learn), breadth-first or best-first under a leaf limit, with depth, samples, minimum impurity
decrease and a per-split feature subset (`maxFeatures`, for forests); a state, even a partial one, gives a `Tree`
(`treeOfState`) and a `TreeRecord` with every node's candidates (`treeRecordOf`); `growTree` runs it. `prune.ts` has
minimal cost-complexity pruning as an algorithm (the weakest link per step; `costComplexityPath` is the `PruningPath`
record, `pruneTree` stops at an alpha) and reduced-error pruning on held-out rows (also an algorithm).
`classic.ts` is ID3, C4.5 (gain ratio, continuous thresholds, pessimistic error pruning) and CHAID (chi-square merging of
categories), growing multiway `ClassicTree`s one node per step; their records list each node's `branches`.
`hoeffding.ts` is the Hoeffding tree (`hoeffdingTreeAlgorithm`, one stream example per step, Gaussian and categorical leaf
statistics, the Hoeffding bound, prequential accuracy). `decision-tree.ts` holds the estimators on the iterative base,
`decisionTreeClassifier`, `decisionTreeRegressor`, `extraTreeClassifier` and `extraTreeRegressor`: `.method(data)` and
`.session(data)` beside `.fit`, `fittedAt(t)` the tree after t nodes, logging `tree/split`, `tree/candidates`,
`tree/leaves`, `tree/depth`, `tree/impurity` and `eval/accuracy` (or `eval/r2`), probes `model/structure` and
`model/importances`. `surrogateTree` in `explain` is the depth-first grower on a model's predictions.

oracle: scikit-learn trees (`trees/grow` fixture): the arrays of the grown tree (features, thresholds, children,
impurities, counts, values, missing-value directions), the pruning path and the pruned tree, importances, for Gini and
entropy, squared, absolute and Poisson error, best-first growth with a minimum decrease, sample weights, and missing
values. TODO: the Hoeffding tree has no reference (river is not an oracle group); its tests are behavioural.
built from: `learn`, `metrics`, `stats/tests` (CHAID), `exposition`, `scene`

### 4.49 `ensembles/`: ensembles of learners (scikit-learn ensemble, LightGBM)

Every ensemble is a stepped algorithm on the iterative base, one member or stage per step. `averaging.ts` is the core of
bagging and forests: members on bootstrap samples (or samples without replacement), the running sum of outputs, the
out-of-bag sum and the training and out-of-bag scores after each member; logs `ensemble/member`, `ensemble/oobError`,
`ensemble/memberError`, `ensemble/bootstrap`, `ensemble/trainScore`, `ensemble/oobScore`. `bagging.ts` takes any
estimator of the library as the member (random subspaces and pasting as modes). `forest.ts` has random forests and extra
trees (`ensemble/importances`, probes `model/proximity` and `model/importances`) and the random trees embedding.
`adaboost.ts` is AdaBoost over any base estimator (SAMME, SAMME.R, and R2 for regression) with the weighted stump search
`bestStump` that `vision/haar.ts` shares; logs `boost/error`, `boost/alpha`, `boost/trainingError`, `boost/stumps`, probes
`boost/weights`, `boost/stump`, `model/margins`. `gradient-boosting.ts` (machinery in `_boosting.ts`) has the squared,
absolute, Huber, quantile and Poisson losses, the logistic and exponential losses, Friedman's leaf line search or Newton
leaves with an L2 penalty, subsampling and early stopping on a held-out loss (event `boost/stopped`); logs `boost/round`,
`boost/trainLoss`, `boost/residuals`, `boost/leafValues`, `eval/loss`, probe `model/stagedPrediction`.
`histogram-boosting.ts` plugs a histogram grower into the same loop: features binned as scikit-learn (`binFeatures`),
leaf-wise growth, the subtraction trick, missing values to the better side (`boost/bins`). `cyclic.ts` is the cyclic
booster of explainable boosting machines (moved from `explain/glassbox.ts`, which re-exports it): `cyclicBoosting` is an
algorithm with one cycle over the features per step, `explainableBoosting` runs it, and the estimators
`explainableBoostingRegressor` and `explainableBoostingClassifier` have the probe `model/shapeFunctions`.
`stacking.ts` has voting (hard, soft, weighted) and stacking (the final estimator on out-of-fold predictions), one base
estimator per step, logging `ensemble/baseScores`, `ensemble/weights`, `ensemble/diversity`. The fit of each records the
stages (`ensemble/stages`, an `EnsembleStages` projection of the tape).

oracle: scikit-learn `ensemble`: AdaBoost SAMME (weights, errors, decision function, stage by stage), gradient boosting
(squared, Huber and quantile losses; binary and multinomial log loss; exponential loss; every stage) and histogram
gradient boosting (squared error, L2, missing values, Poisson, coarse bins, binary and multinomial). TODO: the absolute
error loss (tied splits), SAMME.R and R2 (scikit-learn dropped the first and resamples the second), categorical features
in histogram boosting, LambdaMART.
built from: `trees`, `learn`, `metrics`, `exposition`

### 4.50 `kernel-methods/`: kernel regression, batch and online (scikit-learn kernel_ridge, KRLS, NORMA)

`kernel-ridge.ts` is the closed-form dual solve `(K + alpha I)^(-1) y` over any kernel of `kernels` (a name with
`gamma`, `degree`, `coef0` by `kernels/named.ts`, or a `Kernel` object), for one or several outputs. `solveKernelRidge`,
`hatDiagonal`, `looPredictions` and `looPress` work on a Gram matrix: the leave-one-out residual is
`c_i / [(K + alpha I)^(-1)]_ii`, exact, from one inverse. The estimator `kernelRidge` (kind regressor, one step) logs
`solver/looMse` and `solver/effectiveDf`; its verbs add `hatDiagonal`, `looPredictions`, `looPress` and a normal
`predictDistribution`. Sample weights are refused. `online-kernel.ts` has the learners that take one observation per step:
`krls` (Engel, Mannor and Meir: the dictionary grows by a block inversion when the approximate-linear-dependence
residual exceeds `aldThreshold`, otherwise a reduced-order recursive least squares step; `criterion: 'coherence'` for the
coherence test; `criterion: 'sliding-window'` keeps the exact ridge solution on the last `budget` observations by a
bordering update and a downdate of the inverse), `krlsStep` and `krlsPredict` on the plain-JSON `KrlsState`, and `norma`
(stochastic gradient descent in the RKHS with weight decay `1 - eta_t lambda`, squared or epsilon-insensitive loss, an
expansion truncated to `budget`). Logs `solver/dictionarySize`, `solver/novelty`, `solver/loss`, `solver/runningMse`,
`solver/alpha` and the message `solver/dictionaryAdd`; probe `model/dictionary`.

`feature-ridge.ts` is ridge regression in an explicit feature space (primal, `O(n D^2)` instead of the dual's `O(n^3)`):
`ridgeSolve` (the normal equations), `featureRidgeProblem` (the same ridge as a `SmoothProblem`, so every solver of
`smoothSolvers` reaches it), `featureRidge` (over any `FeatureMap`), the estimator `approximateKernelRidge` (draws a
Nystrom, random Fourier or orthogonal-random map itself; `solver: 'closed-form'` or a value of `smoothSolvers.space`) and
`ridgeGap`, the explicit approximation error against exact kernel ridge: the fitted-value gap, `|K - K~|_2` and the
bound `|K - K~|_2 |y| / alpha` that gap satisfies, which falls with the number of features. `kernel-selection.ts` is the
choice of the kernel and the ridge from the data: `logEvidence` (kernel ridge as a Gaussian process), `looScore` and
`gcvScore`, all differentiable in the Gram matrix; `kernelCriterionProblem` states the criterion over the log
hyperparameters and log ridge as a `SmoothProblem` (the gradient is autodiff through the Gram matrix), `tuneKernelRidge`
minimises it with any solver of the slot (L-BFGS by default), and `gridKernelRidge` scores a grid. `_slot.ts` (private) is
the slot with the model's own mode (`'closed-form'`, `'irls'`) beside the solvers of `smoothSolvers`; `_classifier.ts`
(private) the kernel options, training setup and classifier verbs; `_testdata.ts` (private) the tests' planted data.
`kernel-classifiers.ts` is classification by regularised risk in the dual: `kernelLogisticRegression` (binary; objective
`kernelLogisticProblem`; the default solver `'irls'` is Newton's method in the dual, any of `smoothSolvers` reaches the
same optimum; with a linear kernel it is L2-penalised logistic regression with `C = 1 / lambda`), the import vector machine
`importVectorMachine` (greedy sparse kernel logistic regression, one import vector per step), the least-squares SVM `lssvm`
(closed form `[[0, 1'], [1, K + I/C]]` by default, a slot mode) and `kernelRidgeClassifier` (one-versus-rest ridge on plus
and minus one). The kernel Fisher discriminant is `learn/discriminant.ts`, Gaussian-process classification `gp/`, the
hinge-loss SVMs `svm/`; multi-class logistic problems go through `learn/multiclass.ts`.
`multiple-kernel.ts` is multiple kernel learning: `alignmentWeights` (centred kernel-target alignment, ALIGNF: the
non-negative quadratic programme by projected gradient, or the clipped closed form), `easyMkl` (EasyMKL on two simplices),
`mklProblem` (the SimpleMKL-style optimal ridge loss over a softmax of the weights: a `SmoothProblem`, so any solver of
`smoothSolvers` applies) with `mklAlgorithm` and `mklContract` (typed paths: the solver's and `mkl/weights`, the weights at
every step) and `fitMultipleKernel`, which returns the combined kernel as a `Kernel` object and its Gram matrix. No
scikit-learn equivalent exists, so it has planted-truth and brute-force checks rather than a golden.
`mean-embedding.ts` is kernel mean embeddings: `meanEmbedding` and `embeddingInnerProduct`, `mmdWitness` (the witness
function and the biased MMD; the two-sample tests are `kernels/matrices.ts` and `stats/tests/kernel.ts`),
`conditionalMeanEmbedding` (the regularised operator; the expectation of the output is kernel ridge with ridge `n lambda`),
`kernelBayesRule` (basic level, prior weights on the sample's input rows) and the kernel Stein discrepancy:
`steinKernelGram` builds the Stein kernel of any differentiable kernel and score by nested autodiff vectorised with `vmap`,
`kernelSteinDiscrepancy` its V- or U-statistic, `steinGoodnessOfFit` the test by the multinomial wild bootstrap (an
`Algorithm`). Kernel herding is `learn/coresets.ts`. Checked by planted truth (a Gaussian posterior, a conditional
expectation, the closed-form Stein kernel of a Gaussian kernel) and identities (CME is kernel ridge), not goldens.
`online-learners.ts` has the online learners beside KRLS and NORMA, each an `Algorithm` with a plain-JSON state (the kernel
expansion) and an estimator on the iterative base: `klms` (kernel LMS; the linear kernel is the LMS filter) and `qklms`
(quantised: an input within the radius of a centre updates it, so centres stay a radius apart and the dictionary is
bounded on bounded inputs), `kernelPerceptron` and `kernelPassiveAggressive` (PA, PA-I, PA-II), binary, with the budget
rules `random-removal`, `oldest`, `forgetron` (a constant-factor shrink before dropping the oldest, a simplification of
the paper's schedule) and `projectron` (projection onto the span when the residual is below `eta`). Tests check the
perceptron's `(R / gamma)^2` mistake bound, the passive-aggressive `R^2 |u|^2` squared-loss bound on planted separable
data, and the Projectron's dictionary never exceeding the dimension for a linear kernel.
`kernel-multivariate.ts` is the two-block kernel methods through the Gram matrix: `kernelCca` (regularised kernel CCA: the
canonical correlations are the singular values of the product of the ridge smoothers `K (K + kappa I)^(-1)` of the two
centred Gram matrices; variates of new rows by centring the cross-kernel on the training Gram matrix) and
`kernelPlsRegression` (NIPALS on the deflated Gram matrix, one latent component per step of `kernelPlsAlgorithm`, so
`fittedAt(t)` is the model with `t` components; the linear kernel is linear PLS regression, checked against
`decomposition` and against scikit-learn `PLSRegression`).
`kernel-quantile.ts` is asymmetric kernel regression: `kernelExpectileRegression` (the expectile in the RKHS; objective
`kernelExpectileProblem` for the solver slot, default mode `'irls'` = iteratively reweighted kernel ridge, any of
`smoothSolvers` otherwise; equal to `kernelRidge` at one half) and `kernelQuantileRegression` (sparse, an L1 penalty on the
dual coefficients, an exact linear programme through `linear`'s `quantileRegression` on the Gram matrix; the RKHS-norm
quadratic-programme form is not offered). Not done: a kernel density estimator with scikit-learn's other kernels
(`stats/density.ts` has the Gaussian one).

oracle: scikit-learn `KernelRidge`: dual coefficients and predictions for the linear, RBF, polynomial, Laplacian and
sigmoid kernels, one and two outputs, and the leave-one-out predictions against n refits; `Nystroem` and `RBFSampler`
pipelines with `Ridge` (same landmarks and draws), and the leave-one-out errors of `RidgeCV`.
Kernel PLS with a linear kernel: `PLSRegression(scale=False)`. Kernel classifiers: `LogisticRegression` (linear kernel; RBF through `Nystroem` with every row a landmark, where the
penalties coincide), numpy's solve of the LS-SVM system, `RidgeClassifier` and `KernelRidge` on one-versus-rest targets.
built from: `kernels`, `learn`, `numerics`, `metrics`, `dists`, `trace`

### 4.51 `gam/`: generalised additive models (mgcv, pyGAM, statsmodels GLMGam)

`terms.ts` is the formula layer: `s` (a smooth of one or several covariates on any basis of `stats/smoothing`: `cr`, `cc`,
`ps`, `bs`, `ns`, `tp`, `ts`, `duchon`, `gp`; a `by` covariate for varying coefficients), `te` and `ti` (tensor products and
interactions, one smoothing parameter per margin), `linear` and `factor` (dummy columns or a random effect). `buildDesign`
compiles terms against training data into one block of columns per term with its penalty matrices and null-space
dimension (smooths centred by the sum-to-zero constraint so the intercept is identifiable) and `evaluateDesign` evaluates
the blocks at new rows. `penalised.ts` is the numerics: the assembled penalty, the penalised weighted least-squares
solution with the influence-matrix diagonal (effective degrees of freedom per coefficient and, by `termEdf`, per term),
`gcvScore`, `ubreScore` and `remlScore` (the Laplace-approximate marginal likelihood with the scale profiled out), and
`selectLogLambda`, their minimisation over `log lambda` by Nelder-Mead. `pirls.ts` is `gamAlgorithm`, penalised IRLS
as a stepped algorithm over any `DispersionFamily` and `Link` of `dists/families` (the performance iteration: the
smoothing parameters are chosen on each working response, with step halving on the penalised deviance); the inner solve
is the `solver` hyperparameter, a configuration of `glmSolvers` (default `irls`, one reweighted least-squares step; any
other mode minimises the penalised deviance at the chosen smoothing parameters, with the penalty matrix as its
penalty); logs
`solver/deviance`, `solver/penalisedDeviance`, `solver/edf`, `solver/criterion`, `solver/scale`, `solver/coef`,
`solver/logLambda`. `backfitting.ts` is Hastie and Tibshirani's backfitting (local scoring for an exponential family),
one cycle per step, which converges to the same penalised solution at fixed smoothing parameters. `estimators.ts` has
`gam` and `gamClassifier` on the iterative base: `predict`, `predictTerms`, `standardErrors` (Bayesian posterior
covariance), `predictDistribution`, `edfTerms`, `smoothingParameters`, `aic`. `inference.ts` is inference on a fitted
model from `fitContext(problem, state)` (the Bayesian and frequentist covariances, `edf` and `edf1` per coefficient):
`gamSummary` (mgcv's `summary.gam`: Wald tests of the parametric terms, Wood's 2013 reference test `referenceTest` of
each smooth with the fractional-rank pseudo-inverse and a `weightedChiSquareSf` null, an F reference when the scale is
estimated, deviance explained), pointwise and simultaneous credible bands (`termBand` for one partial effect,
`predictionBand` for the predictor or the mean; the simultaneous multiplier is the simulated quantile of the maximal
standardised deviation), and posterior simulation (`sampleCoefficients`, `simulateCurves`). The estimators expose them
as the verbs `summary`, `termBand`, `predictionBand`, `sampleCoefficients` and `simulateCurves`. Tested by simulation:
coverage of the bands, uniformity of the null p-values, power under a real effect.

`shape.ts` makes the shape constraints of a smooth term (`s(0, { shape: 'increasing' | 'decreasing' | 'convex' |
'concave' | [..], lower, upper })`) into linear inequalities on its coefficients (coefficient differences on a P-spline,
the fitted effect on a grid on other bases and for bounds); `solvePenalised` then solves the penalised least-squares step
as a quadratic program (the active-set method of `optim`, the interior-point method if a degenerate grid makes it
cycle) and inverts the Hessian on the null space of the binding constraints, so `edf`, the posterior covariance and the
REML determinant are those of the constrained model. The solver slot is `irls` only for such terms.

`outer.ts` is the outer iteration of mgcv (`iteration: 'outer'`; the performance iteration stays the default,
`iteration: 'performance'`): `innerFit` runs the penalised fit to convergence at a trial `log lambda` (penalised IRLS
with step halving, or any `glmSolvers` mode warm-started), `evaluateOuter` scores it, and `gamOuterAlgorithm` (one outer
iteration per step, `fitGamOuter` to the end) minimises the score. REML and ML are the Laplace-approximate marginal
likelihood with the observed information (`observedWeights`: the Fisher weight times `1 + (y - mu)(V'/V + g''/g')`),
the scale profiled out when unknown, an analytic gradient in `log lambda` (the implicit derivative of the penalised
coefficients, the weights differentiated in the linear predictor), and a Hessian from differences of gradients
(`outerHessian`), with Newton steps, an eigenvalue safeguard and step halving; GCV and UBRE are the deviance scores of the
converged fit minimised with the simplex. Its state is a `GamState` plus the gradient norm, so inference and the
estimators (`gam.with({ iteration: 'outer' })`) apply to it unchanged.

`selection.ts` is model selection: `gamInformation` (log-likelihood, AIC and BIC with the effective degrees of freedom
corrected for smoothing-parameter uncertainty by `smoothingCorrection`, the covariance widened by the inverse Hessian of
the REML score, and the conditional AIC that takes the smoothing parameters as known), `compareGams` (the approximate F
or chi-square test between nested fits), and the double penalty (`s(0, { select: true })` in `terms.ts` adds a penalty
on the null space of the smoothness penalty so the criterion can shrink a term out). `diagnostics.ts` is `gam.check`:
`gamResiduals` (deviance, Pearson, working, response), `qqData` (QQ data with a band from simulated datasets),
`residualsVsFitted`, `kIndex` (the basis-dimension check on residuals ordered by the covariate, with a shuffle
p-value), `concurvity` (worst, observed and estimate), `cooksDistance`.

`derivatives.ts` is functionals of the smooths with bands from the same posterior covariance: `predictTermsSe` (every
term's effect with its standard error, terms left out by `exclude` without being evaluated, the band of the total),
`smoothDerivative` (first or second derivative of a one-covariate smooth: exact where the basis has a closed-form
derivative, otherwise central differences or their Richardson extrapolation with the stencils shifted inward at the
range's ends) and
`smoothDifference` (the difference of two smooths, the levels of a by-variable, with pointwise or simultaneous bands from the
joint covariance). The estimators carry them as verbs.

`profile.ts` estimates a scalar parameter of the response family with the smooths (`profileFamilyParameter`: the
negative binomial's dispersion, the Tweedie power): the converged outer REML score is a marginal likelihood of the
parameter too (the families' log-likelihoods are exact), so it is minimised over the parameter by bounded
one-dimensional search, each evaluation an outer fit warm-started at the last smoothing parameters. Beta
regression is not an exponential-dispersion family in the mean, so it, and the families below, go through the
extended-family machinery of `extended.ts`.

`extended.ts` is additive modelling for extended families (Wood, Pya and Saefken 2016): the likelihood is a
`CoefficientLikelihood` (value, gradient and observed information in the stacked coefficients, and the derivative of the
information along a direction), implemented by `etaLikelihood` for the `ExtendedFamily` of `dists/families` (one design per
predictor, so a mean and a scale, or K - 1 logits, are blocks of one coefficient vector; the information along a direction
from the family's third derivatives when it has them, else from differences of its Hessian) and by `differencedLikelihood`
for anything given in the coefficients (the Cox partial likelihood). `extendedModel` and `modelFromDesigns` prepare the
designs, stacked penalties and starting point. `innerExtendedFit` is Newton's method on the observed Hessian (not Fisher
scoring) with negative eigenvalues of the penalised information flipped and floored and step halving; its `solver` is the
slot: undefined for that Newton, or any configuration of `smoothSolvers` minimising the same penalised negative
log-likelihood (an `analyticObjective` over the likelihood). `evaluateExtended` is the Laplace-approximate marginal
likelihood `-l + beta' S beta / 2 + log|H| / 2 - log|S|_+ / 2` with its analytic gradient in `log lambda` (the implicit
derivative of the coefficients and the derivative of the information along it) and a central difference of the criterion
(refitted, warm-started) for the aux parameters, which are thereby estimated jointly with the smoothing parameters;
`extendedAlgorithm` is the outer Newton iteration (Hessian by differences of gradients, eigenvalue safeguard, step halving),
an `Algorithm<ExtendedState, ExtendedPath>` with `extendedContract` whose state serialises; `fitExtended`,
`predictExtended` (linear predictors, Bayesian standard errors, parameters through the inverse links) and
`extendedTermTests` (Wood's reference test of each term). Fourth derivatives are not used: the Hessian of the criterion is
differenced from the analytic gradient. Families: beta regression, negative binomial with `theta` estimated jointly,
zero-inflated Poisson (count and inflation predictors), ordered categorical (cut points as aux parameters), multinomial,
Tweedie with power and dispersion estimated, Gaussian location-scale, and the GAMLSS distributions. `cox.ts` is Cox
proportional hazards with smooth terms: `coxGamModel` takes the design without its intercept and the partial likelihood of
`survival`'s `coxProblem` (Efron or Breslow ties, strata) as the likelihood, `coxBaseline` is Breslow's baseline cumulative
hazard at the fitted predictor and `coxSurvival` the survival curves.

`gamlss.ts` fits a GAMLSS model (every parameter of a distribution an additive predictor with its own smoothing parameters)
in one of three modes: `newton` (the generic extended iteration), `cg` (Cole and Green: one joint penalised Newton step per
iteration with the cross derivatives, the smoothing parameters re-chosen on each working model) and `rs` (Rigby and
Stasinopoulos: the parameters updated in turn, each by a penalised Newton step on its own block, cross derivatives
ignored); all three reach the same penalised maximum at fixed smoothing parameters. `fitGamlss` reports the global deviance,
effective degrees of freedom and smoothing parameters per parameter and term; `quantileResiduals` (randomised for discrete
responses) and `wormPlot` give the diagnostics.

`posterior.ts` samples the coefficient posterior `exp(l(beta) - beta' S beta / (2 scale))` in place of the Gaussian
approximation: `gamPosteriorTarget` (exponential families, by autodiff) and `extendedPosteriorTarget` are `LogDensity`
values of `sampling`, `posteriorKernel` its Hamiltonian Monte Carlo or random-walk Metropolis kernel preconditioned by the
Gaussian approximation's covariance, `samplePosterior` draws and `posteriorSampler` is the chain as an algorithm with a
serialisable state (the paths are `samplerAlgorithm`'s, untyped).

Exact basis derivatives: `SmoothBasis.derivative` of `stats/smoothing`, carried by `CompiledTerm.derivative`, is what
`smoothDerivative` uses by default (`method: 'exact'`) for `ps`, `bs`, `cr`, `cc`, `tp`, `ts`, `duchon` and `gp` smooths of
one covariate, with `richardson` and `central` differences for the other bases.

oracle: statsmodels `GLMGam` at fixed smoothing parameters (Gaussian, Poisson, binomial and gamma-log families on a
B-spline smooth: coefficients, leverages, effective degrees of freedom, deviance) and statsmodels `MixedLM` for REML
(a random-effect term is a penalty of the identity with `lambda = sigma^2 / sigma_b^2`: the REML smoothing parameter,
variance and intercept). The outer iteration is checked against an independent numpy/scipy implementation (`gam/outer`:
statsmodels' B-spline design, a brute-force profile of the Laplace REML over a grid of `log lambda` refined by scipy, for the
Gaussian, Poisson, binomial and gamma-log families: `log lambda`, criterion, effective degrees of freedom, coefficients).
pyGAM and mgcv are not in the oracle environment. The extended families are checked against an independent numpy/scipy
implementation (`gam/extended`: scipy's densities, five-point finite differences of the density for the derivatives in the
predictors, Newton for the penalised coefficients, a brute-force profile over `log lambda` then Nelder-Mead on the joint
Laplace REML of the smoothing and aux parameters), for beta, negative binomial, Gaussian location-scale, zero-inflated
Poisson, ordered categorical, multinomial, GAMLSS gamma and Student-t and the penalised Cox partial likelihood: at fixed
hyperparameters the coefficients, log-likelihood, effective degrees of freedom and criterion to 1e-5, and the selected
smoothing and family parameters, criterion and effective degrees of freedom to 1e-2. Tweedie has no scipy density and is
checked by planted truth and by the derivative tests only. Not done: estimators on the iterative base for the extended
families (they are functions: `extendedModel`, `fitExtended`, `fitGamlss`, `fitCoxGam`), smoothing-parameter-uncertainty
corrections, simultaneous bands and deviance residuals for them, analytic third derivatives beyond the Gaussian
location-scale, and the shape-constrained terms.
built from: `stats/smoothing`, `dists` (families, links), `learn`, `numerics`, `optim` (Nelder-Mead, `smoothSolvers`), `survival` (`coxProblem`), `sampling`, `metrics`, `trace`

### 4.52 `covariance/`: covariance and precision matrices (scikit-learn covariance)

`shrinkage.ts` wraps `stats/covariance.ts` (`empiricalCovariance`, `shrunkCovariance`, `ledoitWolf`, `oas`: not
duplicated here) as `covarianceEstimator` on the iterative base, with the location, covariance, precision (by
pseudo-inverse), shrinkage intensity, `mahalanobis` (squared distances) and `score` (the mean Gaussian log likelihood)
verbs; `squaredMahalanobis`, `mahalanobisDistances` and `mahalanobisRecord` (an `anomalyScores` record against the 97.5%
chi-squared quantile) are the helpers. `robust.ts` is FastMCD (Rousseeuw and Van Driessen): `cStepAlgorithm` (one
concentration step per step, the determinant falling), `fastMcdAlgorithm` (random `h`-subsets screened by two C-steps in
`init`, the best refined step by step, then scikit-learn's consistency correction of the distances and the reweighting at
the 97.5% quantile with the Croux-Haesbroeck factor), `minCovDet` and `ellipticEnvelope` (the Mahalanobis detector over
`'mcd'`, `'empirical'`, `'shrunk'`, `'ledoit-wolf'` or `'oas'`, thresholded at the contamination quantile; it uses the
outlier verbs of `anomaly`, so `covariance` imports `anomaly` and `anomaly` never imports `covariance`).
`graphical-lasso.ts` is the graphical lasso by block coordinate descent as scikit-learn does it (each column's lasso
through `optim`'s `lassoCoordinateDescent` in the Gram form, factored by Cholesky; the duality gap as the stopping rule),
`graphicalLassoAlgorithm`, `graphicalLasso` and `graphicalLassoEstimator`, whose exposition is a `regularisationPath`
of the partial correlations over a penalty grid. The block coordinate descent is intrinsic to the problem (the L1 term is
not smooth, so it is not a `smoothSolvers` slot); the inner lasso is the library's.

oracle: scikit-learn `graphical_lasso` at two penalties (tolerances tightened), `MinCovDet` (default and 0.9 support
fraction) and `EllipticEnvelope` on data with planted outliers, where the oracle picks a data seed for which
scikit-learn's random starts all find the same minimum (the determinant has several nearby local minima on the clean
rows, and a golden needs the global one). The shrinkage estimators are tested in `stats/covariance`. TODO: the nested
subsets of FastMCD for more than 500 rows, `assume_centered` in the MCD, `GraphicalLassoCV`, the LARS-solved lasso mode.
built from: `stats` (covariance), `anomaly` (outlier verbs), `learn`, `numerics` (linalg), `optim` (lasso), `dists`
(chi-squared), `random`, `trace`

### 4.53 `anomaly/`: anomaly and outlier detection (scikit-learn ensemble and neighbors, PyOD ideas)

`isolation.ts` is the isolation forest (Liu, Ting and Zhou) and the extended variant (Hariri, Kind and Brunner):
`isolationTree` (random split per node on a subsample, flat node list; or a random hyperplane through a uniform point of
the bounding box), `pathLength`, `isolationPath` (the nodes a point visits, for the explanation), `averagePathLength`
(`c(n)`), `isolationForestAlgorithm` (one tree per step, the running scores and their change), `isolationForest`,
`extendedIsolationForest` and the classes. The trees have no criterion, so they do not use `trees`; the growing is
intrinsic. `neighbours.ts` is the local outlier factor (`lofAlgorithm` in three stages: neighbourhoods and
k-distances, reachability densities, factors; `localOutlierFactors`, `localOutlierFactor`), the connectivity-based
outlier factor (`averageChainingDistance` of the set-based nearest path, `cofAlgorithm`, `connectivityOutlierFactor`)
and the k-nearest-neighbour distance (`knnAlgorithm`, `knnDetector`, aggregation `largest`, `mean` or `median`), all over
`neighbourIndex` of `learn` (any metric of `distances`, brute force or tree); a training row is left out of its own
neighbourhood as `kneighbors()` does, the verbs score new rows (novelty mode). `combine.ts` is what every detector ends
in: `outlierVerbs` (`scoreSamples` higher is more anomalous, `decisionFunction` as `threshold - score`, `predict` -1 or
1), `outlierExposition`, the thresholds (`contaminationThreshold`, `quantileThreshold`, `potThreshold` peaks over
threshold with a generalised Pareto tail, `iqrThreshold`, `madThreshold`), the normalisations (`zscoreNormalise`,
`minmaxNormalise`, `rankNormalise`, `unifyScores`) and the combinations (`averageScores`, `maximizationScores`,
`aomScores`, `moaScores`, `combineScores`). Scores here are the positive anomaly scores, so `scoreSamples` is the negative
of scikit-learn's.

oracle: scikit-learn `LocalOutlierFactor` (training factors and novelty scores at two `k` and two metrics, the decision
function, labels and threshold at a contamination and at `'auto'`). The isolation forest is tested through statistical
properties (the outlier's score, the spread of the score falling with the trees, the contamination fraction flagged,
reproducibility under a key), as its random trees do not match scikit-learn's draw for draw; the connectivity-based
factor and the neighbour-distance score have no scikit-learn reference and are tested on hand-computed values. TODO:
the statistical detectors (z-scores, Mahalanobis, PCA reconstruction, HBOS) and the one-class methods, which the chapter
README lists.
built from: `learn` (neighbours, iterative base), `distances`, `stats` (quantiles, ranks, extreme values), `random`,
`trace`, `exposition`

### 4.54 `sketches/`: probabilistic data structures for streams (datasketch, Redis Bloom, Apache DataSketches)

`hashing.ts` is the shared hash: `hash32` (FNV-1a mixing finished by Murmur3's `fmix32`) of strings and numbers (through
their IEEE-754 bytes), `hashPair` (a 64-bit pair), `doubleHashIndex` (enhanced Kirsch-Mitzenmacher probes), and `seedOf`,
which turns a number or a `Key` into a seed, so every sketch is reproducible from a key. `membership.ts`: `bloomFilter`
(with `optimalBloomParams`, `bloomFalsePositiveRate`, `estimateCount`, `merge`), `countingBloomFilter`, `cuckooFilter`
(partial-key cuckoo hashing, deletion, and a refused insertion rolls back, so no stored item is ever lost), and the
algorithms `bloomFilterAlgorithm` (fill, theoretical and measured false-positive rate per insertion) and
`cuckooFilterAlgorithm` (load factor and kicks). `cardinality.ts`: `hyperLogLog` with the classical estimator (linear
counting below `5m/2`) and Ertl's improved estimator (no bias tables), `hllMerge`, `hyperLogLogAlgorithm`.
`frequency.ts`: `countMinSketch` (optionally conservative), `countSketch` (signed, median of rows, second moment),
`misraGries`, `spaceSavingTopK` (with the guaranteed and the candidate heavy hitters), `countMinAlgorithm` and
`spaceSavingAlgorithm` (the error against exact counts and the guaranteed bound). `similarity.ts`: `minHash`,
`minHashJaccard`, `exactJaccard`, `lshIndex` (banded), `lshCollisionProbability`, `lshOptimalParams` (the false-positive
and false-negative areas of the S-curve integrated exactly). `reservoir.ts`: `reservoirSampler` (Algorithm R, Li's
Algorithm L, weighted A-Res and A-ExpJ), `reservoirSample`, `reservoirAlgorithm`. `quantiles.ts`: `gkSummary`
(Greenwald-Khanna, every rank within `epsilon n`), `tDigest` (merging, k1 scale; tails tighter than the middle),
`kllSketch` (compactors; rank error about `1/k`, mergeable). `skiplist.ts`: `skipList` (Pugh; keyed heights, a comparator,
search traces), `skipListAlgorithm`.

Each sketch states its guarantee and the property tests check it across many keys: the Bloom rate against the fill-based
and the classical formula, the cuckoo load factor and bound, HyperLogLog's error against `1.04/sqrt(m)` and its absence of
bias over seeds, the Count-Min bound `epsilon N` with probability `1 - delta`, the Space-Saving and Misra-Gries counter
bounds (also after a merge), MinHash's unbiasedness and binomial spread, the LSH S-curve, reservoir inclusion
probabilities (and the successive-sampling ones for weights), the quantile sketches' rank errors. No oracle: datasketch is
not in the `oracle-core` group, so there are no goldens. TODO: HyperLogLog++'s sparse representation, a streaming
algorithm form for the quantile sketches, sliding-window variants, serialisation of the objects (only the algorithms'
states serialise).
built from: `base`, `random`, `trace`, `numerics` (the quadrature rule of `lshOptimalParams`)

### 4.55 `survival/`: survival analysis of right-censored data (lifelines, statsmodels duration, R survival)

`data.ts` reads right-censored data (`survivalSample`: times and 0/1 event indicators, validated once) and builds the
table of distinct event times (`eventTable`: at risk, events, censored). `kaplan-meier.ts` is the product-limit
estimator with Greenwood's variance and the log-log, plain and log intervals (`kaplanMeier`, `survivalAt`,
`survivalQuantile`, `restrictedMeanSurvival`), the Nelson-Aalen cumulative hazard (`nelsonAalen`), and
`kaplanMeierAlgorithm`, which takes one subject per step (events before censorings at a tied time) so the curve forms
as the run proceeds; its state serialises, and after `n` steps it is the batch curve. `log-rank.ts` is the weighted
K-sample log-rank family (`logRankTest`: log-rank, Gehan-Breslow/Wilcoxon, Tarone-Ware, Peto-Peto, Fleming-Harrington
`(p, q)`, strata summed over; `logRankZ` for two groups). `cox.ts` is Cox regression: `coxProblem` builds the
partial likelihood (Breslow or Efron ties, strata, a ridge) as a `SmoothProblem` in differentiable tensor operations
(one reverse cumulative sum gives every risk set), `coxAlgorithm`/`coxPh` fit it with `solver` a hyperparameter from
`smoothSolvers` (default Newton to a gradient tolerance of 1e-10, which reproduces the references; every other smooth
solver reaches the same optimum), and the fit carries the covariance from the observed information, Wald,
likelihood-ratio and score tests, the concordance, Breslow's baseline cumulative hazard per stratum and
`coxPredictSurvival`. `aft.ts` is the accelerated failure time family (`aftProblem`, `aftAlgorithm`, `aft`: Weibull,
log-normal, log-logistic, exponential) on the same slot, with `aftPredictSurvival` and `aftPredictQuantile`.
`cox.ts` also has the Schoenfeld and martingale residuals. `concordance.ts` is Harrell's C (`concordanceIndex`, with pair
counts). `prediction.ts` scores predictions under censoring with inverse-probability-of-censoring weights: the
censoring distribution (`censoringSurvival`), the Brier score and its integral (`brierScore`, `integratedBrierScore`)
and Uno's cumulative/dynamic AUC (`cumulativeDynamicAuc`). The planted generator is `survivalData` of
`datasets` (Weibull or exponential proportional hazards and log-normal AFT with known coefficients), which the tests
use.

oracle: statsmodels `SurvfuncRight` (curve, Greenwood errors, risk sets), `survdiff` (log-rank, Gehan-Breslow,
Tarone-Ware, Fleming-Harrington, two and three groups, with and without tied times), `PHReg` (coefficients, errors,
partial log-likelihood, baseline hazard and Schoenfeld residuals for both tie methods, and a stratified fit); the AFT fits are compared with
a maximum-likelihood fit written from `scipy.stats` log densities (Gumbel, normal, logistic) polished by Newton steps
on numerical derivatives, and the concordance index with a direct count of the pairs (neither statsmodels nor
scikit-learn has an equivalent installed); the Brier score and AUC are tested on hand-computed values. TODO: left
truncation (entry times), case weights, time-varying covariates, interval and left censoring for the AFT family, the
Grambsch-Therneau proportional-hazards test.
built from: `optim` (`smoothSolvers`, `loggedSolver`), `autodiff`, `tensor`, `numerics` (special functions, LAPACK),
`trace`, `space`, `random`

### 4.56 `causal/`: causal inference from data (DoWhy, EconML, causalml, MatchIt, sensemakr)

Graph-side causal work (structural models, d-separation, back-door and front-door sets, the ID algorithm's cases,
instruments, counterfactuals, PC, GES, LiNGAM, NOTEARS, FCI) is `pgm/causal` and `pgm/learning`; this domain is what
runs on data. Every estimator takes `CausalData` (`x` covariates, a 0/1 `treatment`, an `outcome`; `splitTreatment`
reads the layout of the `treatmentEffects` generator) and returns an `EffectEstimate` (estimand, `Estimate` with
standard error and interval, p-value, n and, where the estimator has one, the per-unit influence values the standard
error comes from). `nuisance.ts` is the nuisance-model slot: a propensity model and per-arm outcome models are any
`Estimator` of the library (`linearModel` logistic and least squares by default; a forest, boosted trees or nearest
neighbours fit as well), `fitNuisances` fits on training rows and predicts at test rows (so cross-fitting is the same
call on another split), `crossFitFolds` gives treatment-stratified folds, and the diagnostics are `overlap`,
`effectiveSampleSize` and `balanceTable` (standardised mean differences before and after weighting).
`estimation.ts` has `regressionAdjustment` (linear with HC1 errors, Lin's interacted model, or per-arm g-computation of
any learner with a bootstrap error), `ipwWeights` and `ipw` (Horvitz-Thompson or Hajek, ATE, ATT or ATC, stabilised
weights, trimming), `stratification`, `aipw` (doubly robust, optional cross-fitting), `tmle` (targeted maximum
likelihood; its logistic fluctuation is `tmleFluctuationProblem`, a `SmoothProblem` solved by any solver of
`smoothSolvers`, Newton by default), `doubleMachineLearningAlgorithm` (one cross-fitting fold per step; the partially
linear and interactive models; the state serialises) with its batch form `doubleMachineLearning`, and
`bootstrapEffect`. `matching.ts` is nearest-neighbour matching on the propensity logit, standardised covariates or the
Mahalanobis distance, with or without replacement, `k` neighbours, a caliper, and the Abadie-Imbens variance.
`balancing.ts` is entropy balancing: the dual is `entropyBalancingProblem` (a `SmoothProblem`; `entropyBalancingAlgorithm`
is it under `loggedSolver`), and `entropyBalancing` returns weights that match the target's moments exactly.
`instrumental.ts` is the k-class family (`instrumentalVariables`: OLS, two-stage least squares, LIML, Fuller; classical or
HC errors, weights, controls) with `twoStageLeastSquares`, `liml`, the first-stage F statistic and partial R-squared per
endogenous regressor (`firstStageF`), Sargan's overidentification test and `waldEstimate`. `designs.ts` is the panel
designs: `splitPanel`, `differenceInDifferences` (two by two, cluster-robust by unit), `twoWayFixedEffects`
(Frisch-Waugh demeaning, the dummy-variable regression's small-sample factors), `eventStudy` (relative-time dummies, or
the Callaway-Sant'Anna group-time estimator against never-treated units with a unit bootstrap; the leads' Wald test of
parallel trends), and synthetic control (`syntheticControl`: simplex-constrained least squares solved exactly by
`optim`'s active-set quadratic program; `placeboInSpace` and `placeboInSpaceAlgorithm`, one donor per step).
`discontinuity.ts` is regression discontinuity: local polynomial fits on each side with triangular, uniform or
Epanechnikov kernels and HC1 errors (`regressionDiscontinuity`, sharp or fuzzy, the fuzzy one as weighted 2SLS with the
cutoff as the instrument), the Imbens-Kalyanaraman bandwidth (`imbensKalyanaraman`), `rdBandwidthSweep`, and a
McCrary-style `densityTest`. The generators added to `datasets` for these are `syntheticControlPanel` (a treated unit
that is an exact convex combination of donors under a factor model, with the effect path and counterfactual as truth) and
`upliftData` (a randomised experiment with a binary outcome, covariate-dependent uplift including harmed units, and
the uplift function as truth).
`heterogeneous.ts` is the meta-learners over the base-learner slot (`sLearner`, `tLearner`, `xLearner`, `rLearner` as a weighted
regression of the residual ratio, `drLearner` on the AIPW score; each returns a `CateModel` with `predict`; R and DR cross-fit
their nuisance models) and `pehe`. `causal-forest.ts` is the causal forest: nuisance-centred outcome and treatment, honest
trees from `trees` (`decisionTreeRegressor`) grown on the gradient pseudo-outcomes of the residual-on-residual moment, and a
forest estimate that is the residual regression weighted by shared leaves; `causalForestAlgorithm` grows one tree per step
(serialisable state), `causalForestOf` and `causalForest` give `predict`, `predictOutOfBag`, the doubly robust
`averageEffect` and split importances. `uplift-tree.ts` is the Rzepakowski-Jaroszewicz uplift tree (`upliftTree` with the
`KL`, `ED`, `Chi` divergences for 0/1 outcomes and the `tau` criterion for any outcome) and the bagged `upliftForest`.
`metrics/uplift.ts` (4.28 `metrics/`) holds the scoring: `upliftCurve` (Qini and uplift gain curves with the random
line), `qiniCoefficient`, `auuc`, `upliftAtK`, `upliftDeciles` and the inverse-propensity `policyValue` with the treat-all
and treat-none values.
`sensitivity.ts` is how strong a hidden confounder would have to be: `eValue` (VanderWeele-Ding, with the conversions from odds
and hazard ratios and standardised differences), `rosenbaumBounds` (the signed-rank bounds on matched pairs, normal
approximation) and `rosenbaumBoundsBinary` (exact McNemar tails) with the critical Gamma, `matchedPairs` (the pairs of a
matching result), the Cinelli-Hazlett omitted-variable bias (`omittedVariableBias`, `robustnessValue`, `partialR2`,
`ovbContour` for the contour grid, `sensitivityModel` and `benchmarkStrength` for benchmarks from observed covariates, which
reproduce the dropped-covariate estimate exactly when given its true partial R-squared), and `refute` (placebo treatment,
random common cause, data subset).
`graph.ts` joins `pgm/causal` to the estimators: `backdoorEffect` takes a minimal back-door set from the graph and adjusts for it
(Lin's regression adjustment or AIPW for a 0/1 treatment, robust least squares for a numeric one) and `frontdoorEffect` applies
the front-door formula through the graph's mediators for a binary treatment and discrete mediators when a hidden confounder
closes the back-door, with a bootstrap error. The graph side (d-separation, back-door and front-door sets, instruments, the
cases of the ID algorithm, the PC algorithm in `pgm/learning`, GES, LiNGAM, NOTEARS and FCI in `pgm/causal`) was already in
`pgm` and is not repeated here.
The planted generators the tests use are `treatmentEffects` (confounding, overlap, nonlinear setting, heterogeneity),
`instrumentalVariables`, `panelEffects` and `discontinuityDesign` of `datasets`.

oracle: statsmodels OLS with HC1 errors (regression adjustment), a GLM with an offset (the TMLE fluctuation), `IV2SLS` and the
k-class formulas with scipy's generalised eigensolver (LIML, Fuller), dummy-variable OLS with cluster-robust errors (difference
in differences, two-way fixed effects, event studies), WLS with HC1 errors per side (regression discontinuity) and scipy's
SLSQP (synthetic control), scipy's Wilcoxon test and binomial tails (Rosenbaum bounds; the published sensemakr and E-value
examples are unit tests);
scikit-learn's logistic and linear regressions for the nuisance models of the weighting, AIPW, stratification and TMLE
formulas, which are written out in numpy beside them, and its nearest neighbours for matching; scipy's BFGS on the
entropy-balancing dual. Planted truth (the effect of the generators, with the naive estimator shown biased) checks
recovery.
built from: `learn` (the estimator contract, neighbour search, splitters), `linear` (default nuisance models), `trees`
(causal forests), `pgm` (graph queries), `optim` (`smoothSolvers`, `quadprog`), `numerics`, `random`, `trace`, `space`, `tensor`
### 4.57 `embedding/`: embeddings learned from data (word2vec, fastText, GloVe, knowledge-graph embeddings)

oracle: numpy (the GloVe loss and gradient from the definition, `oracle/cases/embedding/glove.py`; the four knowledge-graph
scoring functions in numpy complex arithmetic and the filtered rank with scipy's `rankdata`,
`oracle/cases/embedding/knowledge_graph.py`); the PPMI and SVD steps
are `text`'s and are checked there against numpy and gensim's `KeyedVectors`. gensim and torch train stochastically and
differently, so the trainers are checked on planted truth instead: a gradient check of each analytic update against
autodiff, the loss decreasing under several solvers, and analogies and clusters recovered on `analogyCorpus`.

A model is an objective and the optimiser is a slot. Each trainer builds a `SmoothProblem` (`word2vecProblem`,
`gloveProblem`; the examples, with their negatives or Huffman paths, are drawn once from a key, so the loss is a fixed
smooth function of the matrices and the `gradient` mode can draw batches of examples), and its table of solvers
(`word2vecSolvers`, `gloveSolvers`, built by `solversWith`) holds the library's generic solvers (gradient descent,
conjugate gradient, Newton, BFGS, L-BFGS, coordinate descent, the stochastic `gradient` mode with its optimiser,
schedule and batch) beside the model's own algorithm as a named mode: word2vec's plain SGD with a linearly decaying rate
touching only the rows involved (`word2vecSgd`), GloVe's AdaGrad (`gloveAdaGrad`). The named modes are `Algorithm`s with
serialisable states; their per-example updates are the exact gradient of the problem's loss, which the tests check
against autodiff.
- `word2vec.ts`: `word2vecVocabulary`, `word2vecExamples` (skip-gram pairs or CBOW windows, a dynamic window, frequent
  word subsampling, unigram^0.75 negatives), `word2vecProblem` (negative sampling, or the hierarchical softmax over a
  Huffman tree through `losses`' `hierarchicalSoftmax`; fastText's hashed character n-gram rows as an input option),
  `word2vecSgdAlgorithm`, `word2vecSolvers`, `trainWord2vec`, `word2vecModel`, `subwordVector` (a vector for an unseen
  word from its n-grams), `solversWith` (any model's own mode beside the generic solvers).
- `glove.ts`: `gloveProblem` (weighted least squares on the log of the harmonic-weighted co-occurrences),
  `gloveAdaGradAlgorithm`, `gloveSolvers`, `trainGlove`, `gloveModel` (word vector `w + w~`).
- `count-based.ts`: `ppmiEmbedding`, the closed-form route (windowed counts, shifted PPMI, truncated SVD).
- `knowledge-graph.ts`: TransE, DistMult, ComplEx and RotatE scorers (`kgScore` in numbers, the same in tensor
  operations inside the loss, checked against each other), `kgProblem` (margin or logistic negative-sampling loss against
  corrupted heads and tails drawn once from a key, optional L2), `kgSolvers` (the generic solvers, the stochastic
  `gradient` mode the default), `trainKg`, `kgEmbedding`, `rankKg` (filtered link prediction: MRR, mean rank, Hits@k,
  tied ranks averaged). The tests train all four on `plantedRelations` and check what each can express: DistMult
  scores a fact and its reverse identically, so it cannot orient the cycle and ranks held-out symmetric facts far
  better than TransE does.
- `evaluation.ts`: `analogyAccuracy` (3CosAdd or 3CosMul, optionally over a candidate set), `similarityCorrelation`
  (Spearman against judged similarity), `neighbourPurity`.
  built from: `losses`, `optim` (`smoothSolvers`), `text` (co-occurrence, PPMI, SVD, analogies, fastText hashing), `stats`,
  `trace`, `random`, `space`, `tensor`
### 4.58 `patterns/`: pattern mining (mlxtend, SPMF, pysubgroup)

`itemsets.ts` mines frequent itemsets from baskets of item ids three ways that return the same itemsets with the same
counts on every input: Apriori (`apriori`, `aprioriAlgorithm`: one level per step, candidates joined from a shared
prefix and pruned by the downward closure, with the joined, pruned and counted candidates on the tape), FP-growth
(`fpGrowth`, `fpGrowthAlgorithm`: one transaction inserted into the FP-tree per step, then one conditional tree mined
per step, the pending conditional pattern bases in the state) and Eclat (`eclat`, `eclatAlgorithm`: one prefix class
per step, tid-lists intersected). `exhaustiveItemsets` enumerates every subset and is the reference the three are
tested against (property tests on random baskets, with and without `maxLength`); `closedItemsets` and `maximalItemsets`
filter a complete result. `association-rules.ts` has `ruleMeasures` (support, confidence, lift, leverage, conviction and
Zhang's metric from the four counts, mlxtend's definitions) and `associationRules` (every split of every frequent
itemset, filtered and ranked by any measure). `sequences.ts` mines sequences of itemsets: `prefixSpanAlgorithm` (one
prefix extension per step over projected databases that remember the earliest match) and `gspAlgorithm` (one level per
step, the join of the paper, pruned by contiguous subsequences, with minimum and maximum gap and maximum span),
`containsPattern`, `exhaustivePatterns` as the reference (PrefixSpan and GSP equal it on random sequences, and GSP equals
it under random gap constraints). `subgroups.ts` is subgroup discovery: `subgroupQuality` (the Klösgen family with
weighted relative accuracy, relative accuracy, lift, signed chi-squared), `columnSelectors` (nominal values and
equal-frequency numeric cuts), `subgroupProblem` and `discoverSubgroups` (the conjunction language of
`search/refinement.ts` searched by beam or exhaustively with the optimistic estimate of WRAcc; binary or numeric
targets), and the covering variants `cn2SdAlgorithm` (beam search under example weights that fall for covered
positives) and `aprioriSdAlgorithm` (candidate levels, then weighted selection), one rule per step. The planted
generator is `plantedTransactions` of `datasets`.

oracle: pysubgroup (`StandardQF(a)` for a = 0, 0.5, 1 and `StandardQFNumeric`, DFS with optimistic estimates, top-k
qualities); mlxtend and SPMF are not installed in the oracle group, so the itemset and sequence miners are checked by
exact agreement with exhaustive enumeration and by the worked examples of the Apriori and PrefixSpan papers. TODO:
planted sequences, SD-Map, interval selectors for numeric columns, exceptional model mining, frequent subgraphs.
built from: `search` (refinement), `trace`, `base`

### 4.59 `rules/`: rule learning on the `learn` contract (Orange, imodels, CORELS, Weka)

`rule.ts` is the shared representation: a `Literal` (a test on one feature: `==` a category, `<=` or `>` a threshold),
a `Rule` (literals, the class it predicts, the class counts of the rows it covered when learned) and a `RuleModel` (an
ordered `list` that takes the first rule that fires, or a `set` whose firing rules vote by summing their counts, with the
default counts for rows nothing covers: `predictCounts`, `firingRule`, Laplace-smoothed `smoothedProbabilities`). The
quality measures are `ruleStatistics` (support, confidence, lift, Laplace accuracy, WRAcc of a rule on data),
`ruleQuality` (entropy, Laplace, accuracy, m-estimate and WRAcc from covered class counts, the heuristics of the
searches) and CN2's `likelihoodRatio` and `ruleSignificance` (chi-squared, k - 1 degrees of freedom); `modelText`,
`ruleLine` and `modelRecord` print a model or turn it into the `rules` record. `covering.ts` is sequential covering as an
`Algorithm` with a rule per step: ordered CN2 (a beam over conjunctions, entropy or Laplace, kept only if significant,
the majority class of the covered rows, all covered rows removed), unordered CN2 (rules per class over all rows, only the
covered rows of that class removed, rules vote) and PRISM (greedy on precision to a pure rule). `ripper.ts` is RIPPER and
IREP: classes from the rarest, a rule grown by FOIL's gain (`logic/ilp`) on two thirds and pruned on a third by
(p - n) / (p + n), the 50% error stop and the description-length stop (`ruleSetDescriptionLength`, Cohen's code for rules
plus the exceptions, 64 bits of slack), then replacement and revision of each rule keeping whichever gives the set the
shortest description, a rebuild and a deletion pass; its randomness is the grow/prune split from the step's key.
`lists.ts` has the pre-mined antecedents (`mineAntecedents`: Eclat of `patterns` over the literals each row satisfies)
and two rule-list learners over them: `corelsAlgorithm`, a CORELS-style branch and bound over prefixes (objective and
look-ahead bounds, the minimum-support bound, the equivalent-points bound and the permutation bound; the queue emptying
certifies optimality, `fitCorels` reports it), and `bayesianRuleListAlgorithm`, Metropolis-Hastings over ordered lists
(truncated Poisson priors on length and rule size, Dirichlet-multinomial likelihood, add/remove/swap proposals with the
Hastings ratios; `ruleListLogPosterior` is its target density). `sets.ts` has `oneRAlgorithm` (Holte's buckets, one
feature per step), `decisionTableAlgorithm` (Kohavi's table, forward selection by leave-one-out accuracy) and
`treeRuleSet` (one rule per leaf of a `trees` classification tree, predicting exactly as the tree). `estimators.ts` puts
each learner on the iterative base: `cn2Classifier` (ordered or unordered), `prismClassifier`, `ripperClassifier` (RIPPER
or IREP), `corelsClassifier`, `bayesianRuleListClassifier`, `oneRClassifier`, `decisionTableClassifier`, each with
`method(data)`, `session(data)` and `fittedAt(t)` (the rule model after t steps), logging the README's tracks
(`rule/literal`, `rule/coverage`, `rule/precision`, `ruleset/rules`, `ruleset/accuracy`, `ruleset/covered`,
`rule/descriptionLength`, `pass/revision`, `search/bound`, `search/incumbent`, `search/nodes`, `chain/logPosterior`,
`chain/accepted`) and answering the probe `model/rules`. The planted generator is `plantedRuleList` of `datasets`
(categorical rows labelled by a planted decision list with label noise; the list, the noise and the exact accuracy are
the truth). The contracts of the estimators go through the iterative base, whose paths are not yet typed.

oracle: none for the learners (no installed reference implements these exactly: Orange and imodels are not in the oracle
group); the checks are exhaustive enumeration (CORELS equals the best list found by trying every list up to length 3),
the exact posterior (the sampler's visit frequencies converge to the enumerated posterior over all 65 lists of a small
problem), Cendrowska's contact lenses data (PRISM's pure rules, OneR's 17 of 24 as Weka reports), the likelihood ratio
and description length by hand, and recovery of the planted list with generalisation to fresh draws. TODO: RuleFit
(`explain/rules.ts` has it), Apriori-based classification (CBA, CMAR), numeric interval literals, RIPPER's exact
Weka description-length constants, the CORELS cache and symmetry-aware bounds beyond the permutation bound.
built from: `learn` (iterative base), `patterns` (Eclat), `logic/ilp` (`foilGain`), `trees`, `exposition` (`ruleSet`),
`numerics/special`, `trace`, `random`

### 4.60 `agents/`, `environments/`, `problems/` and `search/evolutionary/`: function approximation, best-arm bandits, test beds with truth, puzzles and NEAT (Sutton and Barto, Garivier and Kaufmann, Stanley and Miikkulainen)

New files in four existing domains; none changes what was there. In `environments/`, `prediction.ts` holds the test beds
of prediction and off-policy learning, each carrying its truth: `randomWalk` (the 5-, 19- and 1000-state chains, with
`randomWalkValues` in closed form or by the exact solve of its model), Baird's counterexample (`bairdCounterexample`,
with `BAIRD_FEATURES`, the behaviour and target policies, the initial weights and `BAIRD_TRUTH`: every true value is
zero and semi-gradient off-policy TD diverges) and `infiniteVariance` (Example 5.5: value 1, infinite variance of the
ordinary importance-sampling estimate). `changing-maze.ts` is the blocking and shortcut mazes of Dyna-Q+ (`blockingMaze`,
`shortcutMaze`): the clock counts steps across episodes through `resetFrom`, the change is flagged in `info`, and
`changingMazeTruth` gives the shortest path before and after (10 then 16, and 16 then 10) and the optimal actions.
`goal-reaching.ts` is `reachGoal`, the sparse-reward point-reaching task of hindsight replay (observation position then
goal, `achievedGoal`, `desiredGoal`, `goalReward`, `reachGoalOptimalSteps`, a `minDistance` knob that makes the reward
rare). `bandits.ts` gains `changePointBandit`, a Bernoulli bandit whose probabilities change at a planted pull. All are
registered in `registry.ts` with tags (59 entries now).

In `agents/`, `approximation.ts` is linear function approximation: sparse feature maps (`oneHotFeatures`,
`matrixFeatures`, `tileFeatures`, `actionTileFeatures`), `tileCoder` (Sutton's tiles with asymmetric, symmetric or no
offsets and optional hashing into a fixed memory) and the learners as agents with serialisable state:
`semiGradientTd` (TD(lambda) with accumulating, replacing and dutch traces, `offline` updating whose summed update equals
the forward view of the lambda returns, off-policy through importance ratios, and an `optimiser` slot taking any
gradient transformation of `optim` as the update rule), `semiGradientSarsa`, `gradientTd` (GTD2 and TDC) and `emphaticTd`.
`best-arm.ts` is fixed-confidence best-arm identification as algorithms with a stopping rule and typed paths
(`BestArmPath`, `bestArmContract`): `successiveEliminationAlgorithm`, `medianEliminationAlgorithm`, `lucbAlgorithm` and
`trackAndStopAlgorithm` (Gaussian arms), `runBestArm`, and the quantities the sample complexity is measured against
(`characteristicTime`, the optimal allocation and `1 / T*` of the lower bound, and `hardness`).
`non-stationary-bandits.ts` has `slidingWindowUcb` and `discountedUcb`. `off-policy-monte-carlo.ts` has
`offPolicyMonteCarloPrediction` (ordinary and weighted importance sampling) and `offPolicyMonteCarloControl`.
`goal-conditioned.ts` has `relabelEpisode` (hindsight strategies `final`, `future` and `episode`) and `goalConditionedQ`.

In `problems/`, `classic-puzzles.ts` has the sliding-tile puzzles (`slidingPuzzle`, `isSolvable` by permutation parity,
`scrambledBoard`, `optimalMoves`) and `towersOfHanoi`; `tours.ts` has travelling-salesman instances with known optimal
tours (`circleTour`, `gridTour`, `randomTour` with the exact optimum from `heldKarp`) and Hamiltonian-cycle puzzles
(`plantedHamiltonian`, `findHamiltonianCycle`); `sat-instances.ts` has `randomKSat`, `plantedKSat` (with the witness),
`pigeonhole` and `SAT_THRESHOLDS`; `registry.ts` is `problemRegistry`, tagging every family (also queens and Sudoku) by
kind and solvability, whose `make` builds an instance with its truth (`whereProblems`, `findProblem`). In
`search/evolutionary/`, `neat.ts` is `neatAlgorithm`: innovation numbers, speciation by the compatibility distance,
aligned crossover and structural mutation, with XOR (`xorFitness`, `solvesXor`) as its truth and `neatNetwork` to run a
genome. Where this differs from `LIB.md`: the continuous particle swarm was already `optim/minimize/particle-swarm.ts`
(inertia or constriction, global or ring topology) and was not duplicated; `randomWalk` and `bairdCounterexample` are in
`environments/prediction.ts` rather than `toy-text.ts`, `changingMaze` in `changing-maze.ts` rather than `gridworlds.ts`,
and the goal-reaching task is its own file instead of a variant of `pointMass`.

## 5. Methods, exposition records and views

> **Status, 2026-10-28.** The methods layer is being rebuilt on the log contract and sessions (`docs/methods.md`,
> `docs/methods_port/`): one base per way of running, particulars of a few lines each. The methods described below
> were parked under `_parked/` on 2026-10-26 and removed from the tree on 2026-10-28 (they stay in git history at
> 933f0dd8, same paths below `_parked/`); their algorithms move home to `lib/` as they are needed
> (`docs/methods_port/inventory.md`), and the constructables already have (`lib/environments`, `lib/agents`,
> `lib/problems`, `lib/graphics/rendering.ts`). Every area keeps a folder under `methods/` whose `index.ts` summary
> says what it is for, a placeholder until its methods are written, so the index of methods stays in the code and in
> the docs. The hub (the `mdml-hub` package and the site's hub section) went with the parked code: pages that explain
> a method are the encyclopedia's, composed from the render session layer. Sections 5.x below describe what was
> parked; the exposition records of `lib/exposition` and the views of render are unchanged.

Three layers, each usable without the ones after it:

| layer   | where                          | what it holds                                                                                                                                                  | draws? |
| ------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| core    | `packages/engine/src/lib/`     | the primitives of section 4 (a library, as PyTorch or Pyro are)                                                                                                | no     |
| methods | `packages/engine/src/methods/` | named algorithms and estimators (lasso, trees, k-means, UMAP, agents), headless, built from the core; each returns typed exposition records beside its results | no     |
| views   | `packages/render/`             | reusable components, including one view per kind of exposition record                                                                                          | yes    |

**Why three.** The core does not decide how it is used; the encyclopedia, a blog or a slide deck define their own uses.
Methods are worked algorithms anyone can call. Views make the exposition reusable: a lasso, an elastic net and a GAM's
smoothness sweep all return a `RegularisationPath`, which `render` draws one way everywhere. A page anywhere only
arranges views; a figure that does not exist yet becomes a new render view, available to every consumer.

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
- The records of the generic panels (wave R1 of `methods/RENDER.md`): `VectorField` (`vectorField`, `fields.ts`: a flow,
  a score or a phase portrait on a grid with nullclines and fixed points classified by stability; `sampleVectorField`),
  `Strip` (`strip`, `strips.ts`: tokens along x with a track of values per class, rows of tags, the chosen path and
  marks), `MarkedTable` (`markedTable`, `tables.ts`: a small table whose cells, rows and columns carry a status and a
  note, and the decision outlined; named apart from the coefficient and likelihood tables), `StructureSteps`
  (`structureSteps`, `structure-steps.ts`: one `Structure` with the looks of its nodes and edges at recorded steps;
  `structureStepAt`), `PointsSteps` (`pointsSteps`, `points-steps.ts`: a two- or three-dimensional point set that
  accumulates or moves, with an optional fixed `range`; `pointsAt`), `ComplexPlane` (`complexPlane`,
  `complex-plane.ts`: poles, zeros and locus branches in `s` or `z`, `isStablePole`), `Simplex` (`simplex`,
  `simplex.ts`: a triangle of three strategies or the square of two populations with paths, a field and classified
  fixed points; `simplexRecord`), `Reachability` (`reachability`, `reachability.ts`: OPTICS distances in visiting
  order with a cut; `reachabilityRecord`), `SceneSteps` (`sceneSteps`, `scene-steps.ts`: a `Scene` per recorded step;
  `sceneStepAt`) and the records of the specialised panels (wave R2): `LandscapeSlice` (`landscapeSlice`,
  `landscape-slice.ts`: the loss along one direction through the iterate, or over a plane, with the optimiser's iterates
  and the Armijo and Wolfe lines on it; `sampleLandscape`, `sliceCoordinates`, the projection of parameters onto the
  directions), `StabilityRegion` (`stabilityRegion`, `stability-region.ts`: `|R(z)|` on a grid of the complex plane
  with the eigenvalues times the step size marked; `explicitRungeKutta` from a Butcher tableau, `stabilityModulus`,
  `sampleStabilityRegion`, `stableArea`, `realAxisLimit`, `markVerdict`), `EquationFit` (`equationFit`,
  `equation-fit.ts`: terms with estimated and true coefficients or a whole expression as TeX, the fit and the truth on
  a grid, the observations; `coefficientErrors`, `fitResidual`), `AudioClip` (`audio`, `audio.ts`: mono samples, an
  optional spectrogram and the `frames` that map tape steps to times; `playheadAt`, `stepAt`, `audioDuration`),
  `Transcript` (`transcript`, `transcript.ts`: tokens by role with a weight per named track and the emitted token;
  `turnsOf`) and `SampleGrid` (`sampleGrid`, `sample-grid.ts`: images or text from fixed noise at recorded steps;
  `sampleStepAt`, `filmstripSteps`). `PointsSteps` steps may carry a `values` channel (with `valueName`, `valueRange`),
  and `pointsAt` returns each point's `born` step. `ImageSteps` steps may carry a caption under each image (`labels`);
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

**Sessions.** `render/react/session/` runs a method of the log contract (`lib/trace`'s `Method` and `Session`) in a
page, with the computation and the viewing at their own paces: `useSession` (the producer: computes the run ahead in
the spare time of each animation frame, so the tape may stand at step 1000 while the page shows step 100; versions the
tape as it grows; holds a change of method or options until play in live mode, or applies it after a short pause),
`preferences.ts` (eager against live, and holding, remembered in the browser from page to page), `SessionPlayer` (the
consumer: the cursor through the tape at a chosen rate, waiting at the head when the computation is slower; the eager
toggle computes the whole run at once and shows its end; seek to either end and step one; a reset of the page's
controls to their defaults when the page offers one), `SessionControls` (seed, steps, subscription, and the same
preferences), the `Panel` vocabulary (`layout.ts`: rows with an overlay, a probe, curves, per-element series, a
trajectory, a tensor with row and column labels and marks (`rowLabels`, `colLabels`, `marks`, `marksPath`), a sweep, a
projection, the whole tape, and the panels of probed records: `field` (a `vectorField` with tape paths over it as a
trajectory), `strip`, `table`, `images`, `graph` (a `structureSteps` record), `points` (`pointsSteps`, coloured by
class or by a `values` channel, with a fading trail and camera presets) and `scene` (`sceneSteps`); the specialised
panels (`SpecialisedPanels.tsx` dispatches them): `zoom` (a probe asked for a viewport and a raster size, panned and
zoomed by wheel, drag and pinch, with a minimap, reset and a pixel inspector; `ZoomPanel.tsx`, the maths in
`views/zoom.ts`), `landscape` (`landscapeSlice`), `stability` (`stabilityRegion`), `equation` (`equationFit`), `audio`
(`audio`: a playhead locked to the step cursor, WebAudio playback), `transcript`, `samples` (`sampleGrid` with a
filmstrip) and `inner` (a session registered by name on `SessionPanels` (`inner` prop, `InnerSessionSpec`), run for
the viewed outer step and drawn with its own layout and cursor, or the outer one; `InnerPanel.tsx`); and the layout
feature `multiples` (one panel spec repeated over a facet: the variants of a comparison, the paths of a curves panel or
the values of a probe argument, with the vertical axis of curves shared; `Multiples.tsx`, `multiples.ts`);
`record-panels` in `RecordPanels.tsx`, `FieldPanel.tsx`, `trajectory.ts`), `SessionPanels` that draws a layout under one
`StepCursor` (`cursor.tsx`; curves scrub it on click, `scrub.ts`), the rows panel, and the generic `TapeView`. A page
anywhere (the encyclopedia, the gallery's `animation/` recipes) composes these.

**Comparisons.** The same layer compares several runs of one method: `useSessions(method, options, variants, deps)`
runs one variant per `{ label, config }` with the same seed, data, budget and subscription (the shared options), so
the knobs are the only difference, under one producer that keeps the variants in step (each frame the variant furthest
behind takes the next slice, `compare.ts`'s `advanceInStep`; a step budget brings every variant up to the slowest head
plus the steps). It returns one `LiveSession` per variant and a combined handle: the head is the step every variant has
reached (`sharedHead`: the slowest unfinished variant's, or the furthest once all have ended), done when all are, and
every verb acts on all. `SessionPlayer` takes a session or a comparison (play starts every producer; pause, eager,
seeks, steps and held changes act on all; the cursor is shared; the stamp shows the slowest head) and `SessionPanels`
draws the same layout over either: curves and sweeps overlay the variants (one colour per variant, the label in the
legend, the step marker shared; `overlayCurves`), and every other panel (rows, probe, tensor, trajectory, projection,
the whole tape) is drawn once per variant, side by side on a full row, titled with its label. A single session is a
comparison of one, and the single-session API is unchanged. The **sweep panel** (`{ kind: 'sweep', path, over? }`)
draws a metric logged once per trial of a study against the knob logged at the same step (`over`; the step itself
without one), each point a trial (`sweepTrials`), those after the viewed step faint and the viewed one ringed; a click
scrubs to the nearest trial. Bayesian optimisation (`methods/tune`) logs `x`, `value` and `best` per trial, so
`{ kind: 'sweep', path: 'value', over: 'x' }` is its sweep. The gallery's `animation/05-compare` recipe runs the toy
walk as three variants under one player.

**The worker producer, page export and narratives.** `useSession`'s producer is a factory (`SessionHookOptions.producer`):
the default computes on the main thread; `workerProducer({ createWorker })` (`render/producer/`) hands a Web Worker the
method's name and its JSON options, computes the run ahead of the cursor (the player tells it where the cursor is, so a
`lookahead` holds it that many steps ahead; the eager toggle runs it to the end) and streams the tape back, probing derived
paths in the worker on request; the player and the panels read the same `Session` interface, now a `RemoteSession`. Pause,
remake and unmount terminate the worker; a failure shows in the player. The gallery's worker entry is `toys/worker.ts`
(`serveWorker(toyRegistry)`). Measured on the gallery's heavy toy (40,000 rows, 1000 steps of descent, a probe of a grid of
losses per step): the page's `requestAnimationFrame` cadence stays at 16.7 ms (p99 16.8, longest 33 ms) while the worker
computes, against a median of 133 ms on the main thread. **Page export:** `SessionPlayer`'s `pageExport` menu captures every
figure under the title and over the caption; the tape file holds the events and the probes of the layout at up to 120 steps, and
`useTape` replays it with nothing recomputed. **Narratives:** a `Narrative` is pages (`NarrativePage`: text, layout, `Moment`s
as steps or event firings) over one session; the gallery's `infrastructure/` recipes show the three.

**No hub.** The hub package and the site's hub section were removed on 2026-10-28: pages that explain a method are
the encyclopedia's purview. What such a page needs is in the engine (a `Method` run as a `Session`) and in render
(`react/session/`: the hook, the player, the settings and the panels); the gallery's `animation/` section shows the
player on a toy method.

**Where the shared pieces live.** There is no `methods/shared/`: what methods share is library code (`methods/README.md`, `LIB.md`). `lib/trace/modes.ts`: a table of modes for one slot, one `union` space over the modes' knobs and one dispatcher from a configuration to the mode it names (`make` for the library algorithm, `params` for the common fitted parameters, `extras` for the paths only that mode can say, `needs` as the validity table beside the mode, `admissible`). `lib/space/tables.ts`: `optionTable`, the same shape for options that are values (knobs, properties, `make`). `lib/optim/solvers.ts`: `smoothSolvers` (gradient descent with its step rules, conjugate gradient, Newton, BFGS, L-BFGS, coordinate descent as one slot) and `loggedSolver`; `lib/optim/schedule-table.ts`: `schedules`; `lib/losses/tables.ts`: `regressionLosses`, `classificationLosses` (with focal and a label-smoothed logistic), `multiclassLosses` (cross-entropy, label smoothing, focal and the noisy-label losses, with `bounded`, `noiseRobust`, `proper` and `needsTransition` among their properties) and `penalties` with their properties.

**Datasets live in the core.** The generators, vendored real datasets and themed dataset collections of 5.18 are the
core domain `lib/datasets/` (at the top of the core's import graph, so exact truths can come from any core domain, such as
`probprog` generating programs or `dynamics` changepoint posteriors); `lib/data` keeps the schemas, loaders, splits and
modifiers.

**Testing, by layer.** The core keeps the full standard (oracle goldens, property tests, `checkPrimitive`). Methods are
tested lightly: a smoke test per method that runs it on a small dataset and checks its exposition records' shapes,
and a golden against the familiar library only where it is cheap and matters (a lasso path, a tree's splits).

**Layering.** Methods import the core and other method areas, without cycles between areas; the core never imports methods; render
imports engine contracts (including `lib/exposition`), never algorithms.

### 5.0 First pass: scope, phases and the exposition contract (methods, with their views)

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

**Dataset registry.** Beside the hand-picked collections, `lib/datasets/catalogue.ts` builds `datasetRegistry`, every
generator, vendored real dataset and remote asset as one tagged entry (kind, modality, real or synthetic, size,
licence, source, with a title and description), from the machinery of `lib/datasets/registry.ts`: `whereDatasets`
(a query by kinds, modalities, real, form and search text), `loadDataset` and `fetchDataset`, and `registerDatasets`,
which returns a new registry with an app's own entries. `lib/environments/registry.ts` does the same for
environments (`environmentRegistry`, `whereEnvironments`, `registerEnvironments`), with the action and observation
kinds derived from each definition's spaces; mdml-render's `DatasetSelector` and `EnvironmentSelector` are the
controls over them.

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

`covariance/` (sklearn.covariance)
built from: `stats/covariance.ts`, `anomaly` (outlier verbs), `learn`, `optim` (lasso), `numerics/linalg`
- `shrinkage.ts`: the empirical, shrunk, Ledoit-Wolf and OAS estimators on the iterative base, Mahalanobis distances,
  the Gaussian log likelihood.
- `robust.ts`: the C-step, FastMCD, `minCovDet`, the elliptic envelope.
- `graphical-lasso.ts`: the graphical lasso by block coordinate descent, its regularisation path.

`anomaly/` (sklearn.ensemble.IsolationForest, sklearn.neighbors.LocalOutlierFactor, PyOD)
built from: `learn` (neighbour index, iterative base), `distances`, `stats`, `random`
- `isolation.ts`: isolation trees and forests, the extended variant, isolation paths.
- `neighbours.ts`: the local and connectivity-based outlier factors, the k-nearest-neighbour distance.
- `combine.ts`: outlier verbs and exposition, thresholds, score normalisation and combination.

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
- `search` (dataset-free methods of the session layer, in `methods/search/` for now): `gridSearch` (breadth-first,
  depth-first, uniform-cost, greedy, A* and weighted A* on the hand-drawn maps of `lib/problems`, a board of cell
  categories per step) and `annealing` (simulated annealing, hill climbing and late acceptance over the test functions
  of `datasets/objectives.ts`, the objective as the probe `objective/surface`).
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

`vision/` (the classical algorithms watched; neural detection heads parked)
oracle: torchvision.ops, torchvision.models.detection
built from: `vision` (core, 4.43), `nn`, `losses`, `metrics`
What exists: dataset-free methods of the session layer, each the library's traced algorithm on a synthetic scene from
`lib/datasets` (`_scene.ts`: a geometric scene with its clean pixels, regions and corners, or one shape with its mask)
with the picture at any step as a probe (`_picture.ts`: an image as a `decisionSurface` field with points over it):
`peronaMalik.ts` (anisotropic diffusion, PSNR to the clean scene, the conduction map), `slicSuperpixels.ts` (labels,
the share of pixels leaking across true edges, the scene in its superpixels' mean grey), `snakes.ts` (an active
contour from the frame onto a shape, its overlap with the mask), `kltTracking.ts` (corners of a drifting scene through
pyramidal Lucas-Kanade, their error against the true motion), `grabCutSegmentation.ts` (the mask as a board, its
overlap with the polygon) and `haarBoosting.ts` (AdaBoost over Haar stumps on windows of shapes, the stump and the
accumulated template as pictures).
The classical parts of this plan now live in the core `lib/vision` (4.43, with Hough transforms in `signal/image`):
boxes, non-maximum suppression, anchors, box coding and anchor matching; contours, keypoints, descriptors and matching,
line segments, Haar cascades and HOG detection; two-view geometry, calibration and stereo; segmentation, tracking and
background subtraction. Only what needs a trained network remains for methods:
- `detection.ts`: detection heads and their targets (FCOS centre-ness and regression targets, feature-pyramid level
  assignment, focal-loss classification), decoded through the core's `boxes.ts`.
- When the methods layer restarts, thin wrappers (a summary and a smoke test each) turning the core's classical
  pipelines into exposition records: panoramas from features and homographies, calibration from a chessboard, lane
  finding, tracking by detection.

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
- `odeSystems.ts` (built, on the log contract): a dataset-free `Method` that integrates a named system of
  `lib/dynamics/named-systems.ts` (Lotka-Volterra, Lorenz, Van der Pol, damped pendulum, SIR) with `RK45`, `RK23` or
  `DOP853` at a chosen `rtol`, one accepted step per step; plain `{ t, y, h, evaluations }` state, rebuilding the
  library's stepper each step with the proposed step as its first. Logs `t`, `y` (full), `h` and `evaluations`.
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
- `registry.ts`: the tagged environment registry: `environmentRegistry` (every definition of the folder and the
  control models as environments, one entry each with its title, family, `episodic`, and `actions` and
  `observations` derived from its spaces by `actionKindOf` and `observationKindOf`, plus its spec, spaces, options
  and `make`), `environmentEntry` (an entry from any definition, single- or multi-agent), `whereEnvironments`
  (by family, kinds, episodic, multi-agent, search text), `registerEnvironments` (a new registry with an app's
  definitions), `findEnvironment`.
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
records so far, which a page draws live. Private plumbing: `_train.ts` (lib/nn's `fitAlgorithm` run with per-epoch
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
- `detection.ts`: a small single-stage detector (anchor-free), built on `vision/detection.ts` and the core's
  `lib/vision/boxes.ts`.
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
  `rosenbrock`, `himmelblau`, `beale`, `rastrigin`, `ackley`, `branin`, `quadraticBowl` with a set condition number,
  and the one-dimensional `forrester`, `gramacyLee` and `dampedSine` for watching a surrogate model),
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
- `click-sessions.ts`: `dbnClickLogs`, click sessions from a dynamic Bayesian network with each item's attractiveness
  and satisfaction and the continuation probability as truth.
- `corpus.ts`: toy and topic corpora with true topics (`naiveBayesCorpus`, `topicCorpus` with Dirichlet or bars
  topics), drifting topics (`driftingTopics`), worked-example corpora (`workedExampleCorpus`: Deerwester's LSA
  titles, the IR book's Caesar passages), `nurseryRhymes`, Tiny Shakespeare (`tinyShakespeare`, remote), near-duplicate
  corpora with planted duplicates (`nearDuplicateCorpus`, for shingle matching and record linkage), tagged sentences
  from a tag HMM with its tables as truth (`taggedCorpus`), and typed words with the intended ones through an edit
  channel (`typoPairs`, for noisy-channel spelling correction).
- `planted-embeddings.ts`: `analogyCorpus` (target words that combine a concept and an attribute, so the parallelogram
  analogies and the concept clusters are planted and listed in the truth), `plantedRelations` (a knowledge graph whose
  relations are symmetric, inverse, a directed cycle and a composition, with the inferable facts held out as a test set),
  `linkageTables` (two tables of person records, the second with typos, and the matching pairs).
- `transactions.ts`: baskets with planted itemsets (`plantedTransactions`; the exact `expectedSupport` of any itemset).
- `captions.ts`: scenes of coloured shapes with template captions and questions (`captionedShapes`; a paired
  image-text schema in `data/schema/` is still to do, the text sits in the truth).
- `audio.ts`: `spokenDigits`, a stream of synthetic spoken digits (formant trajectories over a harmonic source, speakers
  with pitch, speed and vocal-tract scale) with the digit of every sample and the utterance list as truth.
- `timeseries.ts` also has `plantedSeries` (a motif planted repeatedly, a discord and a regime change with known
  positions), `garchReturns` (the conditional variance as truth) and `coupledSeries` (VAR or coupled logistic maps
  through a known directed graph).
- `planted-pgm.ts`: `irtResponses` (0/1 test responses from known abilities and 1PL, 2PL or 3PL items, the abilities as
  the regression target and the true items in the truth) and `pottsSamples` (Swendsen-Wang samples of the Potts lattice at
  a multiple of the critical coupling, with each sample's magnetisation as the target and the energy, the critical
  coupling and Onsager's magnetisation in the truth).
- `shapelet-series.ts`: `shapeletSeries`, noise series in which each class but the first carries one planted pattern
  (`shapeletPatterns`: a bump, a ramp, a pulse) at a random position; the patterns and positions are the truth.
- `hidden-markov.ts`: sequences from hidden Markov chains with the states as labels: the occasionally dishonest casino
  (`dishonestCasino`: faces 0 to 5, a fair and a loaded die swapped in secret) and a series switching between a calm
  and an active Gaussian regime (`regimeSwitching`), each with its generating model in the truth; the `sequences`
  collection.
- `sequence-tasks.ts`: copy, reverse, sort, Dyck languages, addition, associative recall, parity, modular arithmetic
  (`copyTask`, `reverseTask`, `sortTask`, `dyckTask`, `additionTask`, `associativeRecall`, `parityTask`,
  `modularArithmetic`), each with an exact solver in its truth, and in-context linear regression written as quantised
  tokens (`inContextRegression`, with the least-squares and Bayes predictions), and the tasks of the Mamba and H3
  papers with per-position supervision masks: selective copying (`selectiveCopying`, vanilla copying with `fixed`)
  and induction heads (`inductionHeads`).
- `graphs.ts`: karate club (`karateClub`), Les Miserables (`lesMiserables`), contextual stochastic block models
  (`contextualSbm`), graph families (`familyGraph`, `graphFamilies`: Erdos-Renyi, Barabasi-Albert, Watts-Strogatz,
  grids, cycles, paths, stars, complete graphs, trees, geometric graphs), small molecules (`smallMolecules`, read from
  SMILES, with formulas, weights and ring counts), knowledge-graph triples (`kinshipTriples`, family trees with
  planted rules).
- `images.ts`: `shapes` (with boxes and masks), `digitGlyphs`, `barsAndStripes`, geometric scenes with true edges and
  corners (`geometricScenes`), a small Fashion-MNIST (`fashionMnistTest`, remote, parsed by `parseIdxImages`).
- `causal.ts`: synthetic causal datasets with known structural models, ATEs and CATEs: `linearScm` (random DAGs,
  Gaussian or non-Gaussian noise, total effects, interventional means), `treatmentEffects` (confounded treatment,
  potential outcomes, CATE, propensities), `instrumentalVariables`, `panelEffects` (a unit-by-period panel with
  staggered or simultaneous adoption, a differential pre-trend and dynamic effects, for difference in differences and
  event studies), `discontinuityDesign` (sharp or fuzzy, the jump and the first stage as truth).
- `experiments.ts`: A/B test data with known effects (`abTest`: conversion, continuous or revenue metrics, several
  arms, pre-period covariates for CUPED, arrival days and novelty effects), `networkExperiment` (a small-world network
  with unit or cluster randomisation and spillover: direct, spillover and total effects), `carryoverExperiment` (two
  chained experiments with a persistent first effect and the naive bias).
- `trajectories.ts`: logged bandit data (`loggedBandit`, with exact policy values), offline RL datasets from known
  behaviour policies (`offlineMdp` on random tabular MDPs or Strens' chain, `policyEvaluation`, `TabularMdp`).
- `weak-labels.ts`: labelling functions with known accuracies (`labellingFunctions`), crowd labels (`crowdLabels`,
  Dawid-Skene), positive-unlabelled data (`positiveUnlabelled`, SCAR and SAR), label-proportion bags
  (`labelProportionBags`), multiple-instance bags with witnesses (`multipleInstanceBags`), clusters whose classes follow
  or break the cluster assumption (`clusterAssumption`, the exact posterior); crowd workers can be spammers or
  adversaries. Each carries the exact posterior or mechanism.
- `operators.ts`: operator-learning pairs (`heatOperator`, `burgersOperator` solved exactly by Cole-Hopf,
  `antiderivativeOperator` with Gaussian-process inputs).
- `catalogue.ts`: `datasetGenerators` (every generator by name, for recipes and controls), `remoteDatasets`, and
  `datasetRegistry`, the tagged registry built from them and the vendored real datasets (every entry with its kind,
  modality, real flag, size, licence and source beside its title and description).
- `registry.ts`: the registry's machinery: `DatasetEntry` (a generator with starting knobs, a vendored loader or a
  remote reference, with `DatasetTags`), `DatasetModality`, `generatorEntry`, `vendoredEntry`, `remoteEntry`,
  `whereDatasets` and `matchesDatasetQuery` (by kinds, modalities, real, form and search text), `registerDatasets`
  (a new registry with an app's entries, kinds checked), `findDataset`, `loadDataset` (synchronous: generate or
  load) and `fetchDataset` (remote assets too).
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
  So far `methods/tune/bayesianOptimisation.ts`, a dataset-free method of the session layer: one-dimensional functions
  of `datasets/objectives.ts`, a GP surrogate (`fitGpSurrogate`) and `maximizeAcquisition` per trial, the probe
  `tune/posterior` (function, surrogate band and acquisition as a `curves` record).
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
any fitted estimator, each returning exposition records that views draw and pages show; process shown
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

Where they are shown: these are model demonstrations, so their pages are the encyclopedia's (SHAP on fitted models,
counterfactuals for a credit model, audits with per-group thresholds, mitigation trading accuracy for parity, conformal
coverage). The gallery's exposition section keeps one small recipe per new view kind on simple data, showing what the
view can draw.

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

Round two built so far: the shared threshold layer (`thresholdSweep` record, `ThresholdExplorer`, section 6), group
report intervals and warnings, `subgroupReport` and `dataBias` records with their views, gallery recipes
`exposition/43-subgroup-report` and `44-data-bias`, and the `paretoFront` record (view `paretoFrontView`, React
`ParetoFrontView`). Round two's new methods have smoke tests; their pages are the encyclopedia's. Gallery: the
exposition section gains a recipe per new
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
  `panelParts` and `chartFrame` (shared with `compose.ts`); a legend too long for one line pushes the plot area down. A colour bar is calculable (draggable handles, clickable
  bands) only when its layer answers a selection (`ColourBar.select`); otherwise it is a plain legend.
- `selection.ts`: colour bar range selections for layers ECharts cannot recolour (heatmap images, matrix cells):
  `ColourSelect`, `selectionFilter` (a `datarangeselected` event as soft value weights), `dimColour` (values outside
  faded towards the background) and `colourSelection` (the option update the React `Chart` merges, once per frame).
- `echarts-core.ts`: `echartsCore`, the tree-shaken ECharts core with every chart type the builders use, both
  renderers and the registered themes (shared by the React bindings and server-side export).
- `lines.ts`: curves, steps, areas, bands, fill-between, polygons (a rule's box, a leaf, an ellipse in data coordinates),
  series from tensors.
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
  `fields.ts` (`VectorField`: arrows or streamlines, nullclines, fixed points by class; `FIXED_POINT_LOOK`), `strips.ts`
  (`Strip`), `marked-tables.ts` (`MarkedTable` as a scene), `structure-steps.ts` (`StructureSteps` as a scene at a step),
  `matrix-marks.ts` (a labelled matrix with outlined cells, rows, columns and dashed boundaries, and `cellShapes`, the
  pixel-exact outlines the strip uses too), `points-steps.ts` (`PointsSteps`: a scatter on fixed axes, a 3-D scene to
  orbit, a projected still; colour by a value channel, a fading trail, camera presets), `complex-plane.ts`, `simplex.ts`,
  `reachability.ts` (`Reachability` bars by cluster, `reachabilityClusters`), `scene-steps.ts`, `zoom.ts` (viewport
  maths and the raster of a grid), `landscape-slice.ts` (curve, contours, surface), `stability-region.ts`, `equation-fit.ts`
  (TeX of the fit and the truth, the fit against the truth, the residual strip), `audio.ts` (waveform and spectrogram
  as a scene with a playhead; `spectrogramOf`), `transcript.ts`, `sample-grid.ts` (a scene: grid and filmstrip), `images.ts` also giving `imageGridView`, `imageGridFigure` and
  `imageStepIndex` (the images at a tape step as a grid with captions),
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

`producer/` (headless: the computing half of a live session, behind one interface; a worker bundles only this and the engine)
- `producer.ts`: `Producer` (the sessions the consumer reads, `advance`, `start`/`stop`, `follow` the cursor, `probeAsync`, `subscribe`,
  `dispose`) and `ProducerFactory`; `inline.ts` is the main-thread producer (slices in the spare time of each frame),
  `make.ts` makes the sessions of a comparison (the one place, shared with the worker), `step.ts` keeps variants in step.
- `protocol.ts`: the wire. Requests: `init` (method name, JSON `WireOptions`, variants, lookahead), `run` (`eager` ignores the
  lookahead), `advance`, `pause`, `cursor`, `probe` (`latest` probes are coalesced: only the newest of a path is computed while
  the run goes on), `dispose`. Events, numbered by `seq`: `ready` (resolved configuration, seed, the kind of each path), `tape`
  (a `TapeDelta` per variant: events kept, appended, the step and status, event firings), `idle` (paused, lookahead, advanced,
  ended, failed), `probed`, `error`. `wireOptions` says what crosses (JSON; hooks by name, `On('name', options)`, resolved
  against `Method.hooks`; a hook function, a key or a start state cannot).
- `registry.ts`: `methodRegistry`, methods by name, resolved on the worker's side and when a tape file is loaded.
- `serve.ts`: `serveProducer(scope, registry)` / `serveWorker(registry)` (a worker entry is one line): makes sessions in live mode,
  computes in short slices (a pause, a cursor or a probe is heard between slices), stays within the lookahead of the cursor,
  streams deltas of the tape by identity of events (windows that forget are followed), answers probes with a duty cycle so they
  cannot starve the run.
- `client.ts`: `workerProducer({ createWorker, lookahead?, checkpoints?, available? })`: the worker starts when first needed and is
  terminated by `dispose`; Workers unavailable or options that cannot cross make the main-thread producer, with `fallback` saying why.
  `remote.ts`: `RemoteSession`, the `Session` the panels read (tape, step and status as streamed; a probe not answered yet
  shows the last answer of its path, else throws `ProbePending`; no state, fork or algorithm).
- `tape.ts`: the tape as a file, `mdml.tape/1`: method name, seed, per variant the label, configuration, head, status, the run's
  events, the kind of each path, event firings and the probes (`TapeProbe`: step, path, arguments, value as `stateToJson`) at a
  spread of steps (`probeSteps`); `tapeOf`, `parseTape`, `ReplaySession` (nearest stored step when a step was not stored),
  `replaySessions`. `_fake.ts`: an in-memory worker wire for tests.

`react/` (React 19; React and ECharts are peer dependencies)
- `Chart.tsx`: an ECharts option as a live chart (theme from mode, resizing, disposal, events).
- `SceneView.tsx`: a `Scene` as SVG elements, with hover and click hit-testing.
- `Scene3DView.tsx`: a 3-D scene with drag-to-orbit camera, re-projected on interaction.
- `StructureView.tsx`: a `Structure` in a frame (by default) with pan, zoom, hover details and overlay state (`frames`
  sizes nodes once for a whole animation).
- `TraceView.tsx`: the standard figures of a run; `useRunStream` for live runs.
- `Player.tsx`: playback of a precomputed timeline or a live algorithm, redesigned and simpler than aifn-engine's:
  a reset of the page's controls to their defaults and an eager toggle when the controller offers them, seek to the
  start, step back, play and pause with a caret menu for the target frames per second, step forward, seek to the end,
  a draggable scrubber, and an optional time stamp.
  `usePlayback.ts` (`usePlayback`, `useTimelinePlayback`, `useLiveRun`: one controller over precomputed frames and
  live algorithm steps, and `useTicker`, the animation-frame ticker every player shares) and `playhead.ts` (frame
  timing, the pure part).
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
  `Toggle`, `SeedInput`; validity greying (`SpaceControls` takes `admissible` and `why`, and an option a rule rules out is
  drawn disabled with the reason as its title; `validity.ts` reads the rule from the option tables: `tableSlot` (from the
  properties of a table of losses, penalties or schedules), `modeSlot` (from a mode table's `needs`), `tableRules`,
  `settleAdmissible`); `ModelSelector` (the options of a union as a list, each with a settings popover; the knobs set for
  an option are remembered) and `SlotVariants` (the same list for a comparison over a slot: every option that applies
  is a variant with its colour and its own knobs, held by the page as `SlotKnobs`); `DatasetPicker` (a dataset collection from `lib/datasets/collections.ts`, with the
  chosen generator's knobs as controls, `DatasetKnobs`, and `useDataset` over a collection or a registry);
  `DatasetSelector` (the tagged `datasetRegistry` filtered by its props, kinds, modalities, real, form, with a search
  box, each entry's title and tags, and the chosen entry's description, provenance and knobs) and
  `EnvironmentSelector` (the same over `environmentRegistry`: families, kinds of action and observation, episodic,
  multi-agent; the chosen entry's spaces, formatted, and its options as controls; `useEnvironment` makes it), both
  drawn by `EntryList` (`EntryList.tsx`: the search box and the list of titles with `Tags` chips, `tagLabel`); and
  `MethodControls` (controls from a method's `hyperSpace`, demo defaults
  first); `SettingsButton` (a gear that opens a modal of settings and changes nothing until it is confirmed: the dialog
  edits a draft, Apply hands the whole draft to `onApply`, Cancel, Escape and the close button drop it, a press on the
  backdrop closes it only while the draft is untouched, Reset returns the draft to given defaults, focus is kept inside and
  restored; the logic is `settings.ts`: `settingsReducer`, `sameSettings`) and `SpaceSettings` (the same over a `Space`, drawn by
  `SpaceControls`), which `FigureFrame` takes as `settings` and draws in its toolbar beside export and expand; and `ControlPanel`: collapsible, laid out in a given number of rows and columns, stacked in a
  left sidebar beside the figure or in a bar above it, or placed in named bento-grid areas (`ControlLayout` places it).
  Plain React and Tailwind over the `--wb-*` tokens; the rules are pure modules (`number.ts`, `options.ts`, `seed.ts`,
  `space-controls.ts`, `datasets.ts` (sources: a collection id, a collection or a registry), `environments.ts`
  (selections, `makeSelection`, `firstFrame` of any environment form, `formatSpace` gymnasium-style)) and menus open
  in a `Popover` portal so no container clips them.
- `views/`: a React component per exposition view, one file per headless module (`paths.tsx`: `PathView`,
  `CrossValidationView`, `LearningCurveView`, `PruningPathView`; `trees.tsx`: `TreeView`; `steps.tsx`:
  `ClusteringPlayer`, `MixturePlayer`, `EmbeddingPlayer`, `DendrogramView`, `ReachabilityView`; `surfaces.tsx`,
  `stages.tsx` (`EnsembleView`; `PipelineView`: a strip of stage cards and the selected stage's panel, with its
  data table and fitted parameters beside the figure), `components.tsx`, `tables.tsx`, `margins.tsx`, `anomaly.tsx`, `episodes.tsx`, `boards.tsx`:
  `BoardPlayer`, `PopulationPlayer`; `curves.tsx`: `CurvesView`; `densities.tsx`: `DensityView`; `sequences.tsx`:
  `StateSequenceView`; `cells.tsx`: `ValueGridView`, `ConfusionMatrixView`; `diagrams.tsx`: `DiagramPlayer`, a
  `StructureView` with the record's steps played beneath; `topics.tsx`: `TopicWordsView`; `fields.tsx`: `VectorFieldView`;
  `strips.tsx`: `StripView`; `marked-tables.tsx`: `MarkedTableView`; `structure-steps.tsx`: `StructureStepsView`;
  `points-steps.tsx`: `PointsStepsView`; `complex-plane.tsx`: `ComplexPlaneView`; `simplex.tsx`: `SimplexView`;
  `landscape-slice.tsx`: `LandscapeSliceView`; `stability-region.tsx`: `StabilityRegionView`; `equation-fit.tsx`:
  `EquationFitView`; `audio.tsx`: `AudioView`, `useClipPlayback`; `transcript.tsx`: `TranscriptView`; `sample-grid.tsx`:
  `SampleGridView`;
  `reachability.tsx`: `ReachabilityPlotView`; `scene-steps.tsx`: `SceneStepsView`; `images.tsx` also `ImageGridView`), each one figure in one
  `FigureFrame` with any player beneath it, and `ExpositionView`, which picks the standard view of any record.
- `session/`: sessions in React (the Sessions and Comparisons paragraphs of section 5): `useSession.ts` (`useSession`,
  `useSessions`: one producer over one or several sessions, `LiveSession`, `LiveSessions`, `variantsOf`), `compare.ts`
  (the pure part of a comparison: `sharedHead`, `advanceInStep`, `overlayCurves`, `sweepTrials`), `SessionPlayer.tsx`,
  `SessionControls.tsx`, `preferences.ts`, `layout.ts` (the `Panel` vocabulary), `SessionPanels.tsx`, `RowsPanel.tsx`,
  `cursor.tsx`, `scrub.ts` and `TapeView.tsx`; `useSession`'s hook options take a `producer` (see `producer/`), so the same
  hook computes in a worker; `SessionExport.tsx` (the player's `pageExport` menu: the whole page as PNG or SVG, a recording from step
  a to b as GIF or WebM, the tape as JSON in and out) over `export.ts` (`probeRequestsOf` a layout, `framesBetween`, `pageSvg`);
  `useTape.ts` (`useTape(file, registry)`: a tape file as the live handle); `narrative.ts` (moments resolved against the tape,
  the page switcher that keeps the cursor, the step-through) and `NarrativeView.tsx` (page tabs, text, panels, moments). The rows panel's channels are one mechanism in `overlays.ts` (a number
  per row as colour with a colour bar or as size, a category per row or named sets as hollow marks, regions of the plane
  as boxes or ellipses, one chosen row with its neighbours, a path, a radius and a related point; each is read at the
  viewed step from a probe or a tensor path of the tape, `readChannel`, and drawn as chart layers, `channelLayers`); the
  curve families of the `curves` and `sweep` panels are pure layer builders in `families.ts` (`curveFamily`: many runs
  faint with their mean in bold, bands and fans, a truth line, reference slopes on log-log axes, event marks;
  `intervalStrip`: many intervals as bars against a truth, the misses in red); and `slots.ts` is the comparison over a
  slot (`variantsOfSlot`: one `Variant` per option of a union of a method's configuration, each with its knobs, for
  `useSessions`). The gallery's `pages/linear-classifier` is the template of a method page (a dataset picker, the
  objective's controls with validity greying, `SlotVariants` for the solvers, one player, curves overlaid and the
  decision surface of each solver side by side).
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

One private Vite and React 19 app, served by `make dev` (http://localhost:5300) and built by `make site`, with two
sections on one shell: the gallery shows off mdml-render and the docs document the engine. It consumes the packages exactly as an outside project does. aifn-engine's
`examples/` app is the template: its shell, recipe registry, live-example pages, Markdown and docstring-driven docs
pages are recycled rather than reinvented.

`src/shell/`: the top bar with the two sections, a dark and light toggle at the top right (dark by default,
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
  parts of a whole (pies, funnels, gauges), and panels (`panels/`: one recipe per record kind of the generic panels on
  a toy method of a few dozen lines, shown with the session panel that draws it: vector field, strip, table, images,
  graph, matrix with marks, scene, accumulating points, complex plane, simplex, reachability) and `specialised/` (the
  specialised panels, each on a toy method: zoomable picture, orbiting cloud, loss landscape, stability region, equation
  against the truth, audio in sync, transcript, samples over training, small multiples, inner session). Every figure has one frame (expand, export) and a live control. Toy data
  only, no model training: model demonstrations are the encyclopedia's.

  Coverage: every exposition record kind and every view component has a gallery page, and a test keeps it so. A recipe
  lists what it really draws in an optional `shows` field of its `recipe` (`readonly string[]`: record kinds such as
  `'groupReport'`, and view components beyond a kind's standard one such as `'WaterfallView'`); the registry passes it
  through on each entry. `apps/site/src/gallery/coverage.test.ts` derives the kinds from the code (the exposition
  interfaces' `readonly kind: '...'`, checked against `viewedKinds()` of the render registry, which the compiler keeps
  exhaustive over the `Exposition` union), the views from the exported `*View` and `*Player` components of
  `packages/render/src/react/views`, and fails for any not named by some recipe's `shows` (a view is also covered when it
  is the standard component of a shown kind). Its two short exemption tables (kinds, views) start from one entry, each
  with its reason. The record recipes build their records on made-up data with the engine's constructors and library
  functions (`apps/site/src/gallery/toys/synthetic.ts` holds the shared generators) and live in four sections:
  `explanations/` (attributions in every variant, dependence, importances, concepts, rules, prototypes,
  counterfactuals, explanation quality), `trust/` (calibration, conformal, threshold sweep, group and subgroup reports,
  bias in data, Pareto fronts, anomaly scores), `selection/` (paths, tables, curves, trees, searches, surfaces,
  posteriors, margins, neighbours, components, likelihoods) and `steps/` (clustering, mixture, embedding, population,
  board, agent, ensemble, pipeline, epicycle, state, value, curve, diagram and episode records); the older records
  keep their recipes in `panels/`, `specialised/`, `evaluation/` and `animation/`.

  The gallery is for general visualisation types only; on 2026-10-25 its model demonstrations (the `exposition/`,
  `fairness/` and `trust/` sections and the RL episodes) moved to the hub, which was removed on 2026-10-28. The
  `animation/` section shows the session player on a toy method.

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
  of the gallery's recipes; the first page's section loads before the first render, so it
  appears without a Suspense fallback (which React shows for at least 300 ms).

## 9. Python oracle, `oracle/`

A uv project (root `pyproject.toml`) with pydantic and pydantic-settings, and dependency groups split by domain so a
fixture run installs only what it needs: `oracle-core` (numpy, scipy, scikit-learn, statsmodels, sympy, python-sat, clingo,
networkx, python-control with slycot), `oracle-torch` (torch, torchmetrics, torchdiffeq, torchsde), `oracle-prob` (numpyro, pyro, pgmpy, arviz),
`oracle-signal` (pywt, librosa, scikit-image, pystoi, pesq), `oracle-text` (tokenizers, tiktoken, nltk, sacrebleu,
rouge-score, jiwer, rapidfuzz, datasketch, gensim), `oracle-rl` (gymnasium, minatar, pettingzoo, open_spiel with ecos for its LPs, nashpy), `oracle-vision`
(opencv-python-headless),
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
| `tree`                                                      | `lib/trees/`                                          |
| `ensemble`                                                  | `lib/ensembles/`                                      |
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
| `metrics.pairwise`                                          | `lib/distances/`, `lib/kernels/matrices.ts`           |
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
| `torchvision.ops`                     | `lib/vision/boxes.ts`, `lib/metrics/detection-segmentation.ts`              |
| OpenCV                                | `lib/vision/`                                                               |
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
| `scipy.spatial.distance`              | `lib/distances/`, `lib/numerics/geometry/`        |
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
| DoWhy, EconML, causal-learn       | `lib/causal/`, `lib/pgm/causal/`, `methods/causal/`                                               |
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
