# Methods: the workshop draft

A working document, 2026-10-28. It says what the methods layer is for, how a method is put together so that every
method is generic (logistic regression with any solver, any loss, any penalty), how the layer is structured (the
encyclopedia's chapters, one folder each), what the chapters share on the design side and on the rendering side, and
what problems are open. Each chapter's `README.md` lists its capabilities, the examples worth making, what it needs
drawn, its shared pieces and its own problems. None of it is final: it is the thing to argue with.

## 1. What a method is

A method is a light, named composition of library pieces that runs as a session (`lib/trace`): it has a configuration
(a typed space of knobs), it builds a run from constructables (a dataset, a model, an environment, a problem), it logs
against a contract, and it answers probes at any step. Nothing in it is an algorithm of its own: if a method needs
something the library lacks, the library gets it first.

The point of a method is the explanation, not the number: the tape holds everything, a page shows what it chooses to,
and the same method can be run eagerly for its result or live for its story.

## 2. Slots: why every method is generic

Logistic regression is not one algorithm. It is a choice of objective (the logistic loss, or a hinge, or an
exponential loss), a penalty (none, ridge, lasso, elastic net), a solver (IRLS, Newton, any gradient optimiser with any
schedule, stochastic gradient over batches, coordinate descent, proximal gradient), a starting point and a stopping
rule. Each of those is a **slot**, and the method's configuration fills the slots from named options that the library
already implements. The method's own code says which options are allowed in each slot, which combinations are valid,
and what to log; it does not implement any of them.

The slots, with the library behind each:

| slot       | what it chooses                                                                     | library                                  |
| ---------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| data       | a dataset or generator with its truth, a split, a modifier (noise, imbalance)       | `data`, `datasets`                       |
| model      | the representation: linear, kernel, tree, network, mixture, graphical model, policy | `linear`, `kernels`, `nn`, `pgm`, ...    |
| objective  | a loss and a penalty, a likelihood, an energy, a return                             | `losses`, `dists`, `probprog`, `rl`      |
| solver     | how the objective is driven: closed form, IRLS, CD, EM, an `optim` optimiser, MCMC  | `linear`, `optim`, `mixture`, `sampling` |
| schedule   | steps, batches, learning-rate schedules, stopping, restarts                         | `optim`, `trace`                         |
| init       | the start: keys, warm starts, k-means++, pretrained                                 | `random`, the chapter's shared pieces    |
| evaluation | metrics on held-out data, curves, what counts as done                               | `metrics`, `learn`                       |

Rules:
- A slot is a `Space` entry (`choice`), and an option's own knobs are conditional on it (an optimiser's learning rate
  shows only when the solver is a gradient one). `lib/space` has conditional spaces; the controls must draw them.
- A method declares a **validity table**: which options go together (IRLS needs a GLM likelihood; a proximal solver
  needs a penalty with a prox). Invalid combinations are greyed, not errored.
- The default filling is the textbook one, so the method with no changes is the method everyone knows.
- The **same slots across methods** are the point: the solver slot of logistic regression and of a GAM and of a
  neural classifier offer the same gradient optimisers from `optim`, so an optimiser race is one page over any model.
- Every option on the tape: the configuration is logged, so a run says what it was.

## 3. The structure: by what a page does

Folders by what you do, sections by task or application. Not the encyclopedia's chapters: its tree is pedagogical,
and one method serves many chapters (logistic regression appears in supervised learning, loss functions,
optimisation, calibration and fairness). This tree is for building and using.

```
methods/
  README.md   this document
  shared/     what every folder uses (section 4)
  data/       registries and selectors (datasets, environments, problems); preparation, shift and selection studies
  fit/        task methods, the scikit-learn world: classify/ regress/ cluster/ reduce/ density/ detect/ rank/ forecast/ survive/
  train/      problem methods, the torch world: networks/ generative/ representation/ sequence/ graphs/
  infer/      problem methods, the Bayesian world: programs/ samplers/ variational/ graphical/ filters/
  act/        agents in environments: tabular/ deep/ bandits/ control/ planning/
  search/     procedures over problems: state-space/ local/ population/ constraint/ adversarial/
  process/    operations on signals, images, text and graphs, watched: signals/ images/ text/ graphs/
  simulate/   processes, systems and numerical procedures: stochastic/ dynamics/ numerics/ automata/ networks/ sketches/ curiosities/
  analyse/    done to data or to fitted models: statistics/ explain/ fairness/ calibrate/ evaluate/
  study/      methods over methods: sweep/ race/ learning-curve/ cross-validate/ cost/ theory/
```

Inside any folder, section or not: a `README.md` (what it should do, in the form of the pages written today), an
`index.ts` (its summary, which is its entry in the docs tree, and its exports), a `shared/` for its own slot tables,
probes, records and data slots, and `<method>.ts` with `<method>.test.ts`.

Three kinds of file in a folder:
- **Methods** (`<method>.ts`): a `Method` of the log contract, or an estimator on the `learn` contract that becomes
  one. Each is small because its slots and probes come from `shared/`.
- **Studies** (also `<method>.ts`, named for what they compare): a method whose run drives other methods: an optimiser
  race, a regularisation path, a learning curve, a restart study, a comparison of two solvers on one dataset. They
  live in `study/`, or in the folder whose question they answer when the question is that folder's alone.
- **Shared pieces** (`shared/`): slot option tables (`solvers.ts`: the solvers this folder's objectives admit),
  standard probes for the folder's kind of model, the folder's exposition records when the generic ones do not fit,
  and data slots with their truths.

Nothing under `methods/` draws. Rendering lives in `mdml-render`; a folder's README says what it needs drawn.

**The tree today.** The fifty folders named after the encyclopedia's chapters, each with a page, were written before
this layout was decided and mirror the encyclopedia too literally. They stay for now as the catalogue of
capabilities (what to build, with examples, rendering needs and problems); when building starts each capability
moves to its folder above and the chapter folders go. The twelve built areas move the same way: `cluster` and
`mixture` to `fit/cluster` (the mixture of experts to `fit/regress`), `linear` to `fit/classify` and `fit/regress`,
`tune` to `study/`, `sampling` and `sequence` (the hidden Markov model) to `infer/`, `dynamics` to
`simulate/dynamics` (ODE systems) and `infer/filters` (Kalman tracking), `linalg` to `simulate/numerics`,
`probability` and `stochastic` to `simulate/stochastic`, `search` to `search/`, `vision` to `process/images`. Nothing
moves until the core pieces of section 8 are built and a first exemplar has proved the pattern.

## 4. `methods/shared/`: what every folder needs

- `slots/`: the catalogues behind the common slots. `solvers.ts` (closed form, IRLS, coordinate descent, proximal,
  every `optim` optimiser with its schedule, EM, the samplers), `objectives.ts` (losses and penalties by name with
  their properties: convex, smooth, has a prox, is a likelihood), `schedules.ts` (steps, batches, stopping rules,
  restarts), `init.ts` (starts). Each is a `choice` space plus the conditional knobs plus the property table the
  validity rules read. The showcases are the solver race in `study/race` and the objective slot of `fit/`.
- `kinds.ts`: the kinds of model (classifier, regressor, clusterer, transformer, density, outlier, policy, sampler)
  with the verbs a fitted one offers and the **standard probes** each gets for free: a classifier its decision
  surface and probabilities over a grid, a regressor its fit and band, a clusterer its assignments and centres, a
  transformer its components and embedding, a density its log density over a grid, an outlier detector its scores, a
  policy its value and action maps, a sampler its chain diagnostics.
- `bases/`: one base per way of running (the port's vocabulary): one-shot, iterative, trainer (modules, optimiser,
  batches, evaluation at a schedule, parameter checkpoints), agent (environment, agent, schedule), procedure, sweep.
- `sweeps.ts`: the study bases: a grid or sampler over one knob of an inner method, with the inner tapes kept or
  summarised (regularisation path, CV curve, learning curve, restart study, race, repeated fits over fresh samples).
- `streams.ts`: data arriving step by step with planted drift (online learning, shift, time series, experimentation).
- `evaluation.ts`: held-out evaluation at a step with the `metrics` library, logged on the tape as `eval/<metric>`.
- `data.ts`: adapters from `datasets` (generators with truth, real datasets, modifiers) to what a method's data slot
  expects, and the truth beside the data so a probe can compare (the Bayes boundary, the true function, the planted
  clusters, the planted change point).
- `pictures.ts`: the picture helpers of the vision methods made general (an image, a grid of images, a surface with
  points over it), so any method with a 2-D field uses one path.
- `records.ts`: the generic records for anything step-like that the `exposition` kinds do not cover.

## 5. Rendering: what the chapters have in common

Already there (`render/react/session`): a session page is a layout of panels over one step cursor: rows with an
overlay, a probe, curves, per-element series, a trajectory, a tensor, a projection, a board, the whole tape; views for
every exposition record kind; the player with eager, seeks and held changes; the generic controls from a space.

Common requirements that the capability pages ask for, by how far they exist:

| requirement                                                                                                 | status                                                   |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| controls from a space with **conditional** knobs (an option's knobs appear with it)                         | space yes, controls no                                   |
| the **comparison** of two or more runs of one method with different options: overlay and side by side       | missing; asked by 20 pages                               |
| a **sweep panel**: one knob on the x axis, a curve per metric, click to open that run                       | partly (`searchTrials`)                                  |
| the **data slot** as a control: pick a generator, its knobs, a modifier, with the truth drawn               | `DatasetPicker` partly                                   |
| **fields** over a grid at the viewed step: decision surface, density, value map, image, vector field        | yes (`decisionSurface`, `fields`); no vector-field panel |
| **structures** at the viewed step: trees, graphs, graphical models, architectures, with per-step attributes | views yes, panel no                                      |
| **sequences**: tokens, state sequences, series with events, position-aligned values                         | views yes, panel partly                                  |
| **images and volumes**: one, a grid, over steps, with masks and points; two images with correspondences     | `image`, `imageSteps`; no grid or pair panel             |
| **episodes**: an environment rendered, the policy's action, a value overlay                                 | `EpisodePlayer` yes                                      |
| **3-D**: surfaces, point clouds, cameras                                                                    | yes (`Scene3DView`)                                      |
| **rows with per-point values**: scores, order, weights, explanations as colour and markers                  | partly (`rows` overlays)                                 |
| **small multiples**: one chart per layer, client, head, aligned under one cursor                            | missing                                                  |
| a **narrative**: one tape, several pages that each show part of it                                          | missing                                                  |
| **export**: a panel or a whole page as PNG, SVG, GIF, the tape as JSON                                      | panels yes, page no                                      |
| **scale**: a tape of 10,000 steps with tensors; a Web Worker producer                                       | missing (planned)                                        |

The first three are the ones that unlock the generic story: if the solver is a slot, the page must show the solver's
knobs and compare two solvers.

## 6. Problems, catalogued

1. **Conditional controls.** The controls draw flat dict spaces; slots need option-dependent knobs. Render work.
2. **Validity.** Where the validity table lives (a method's declaration) and how a page greys an option. Engine and
   render work.
3. **One contract for estimators and methods.** Today an estimator (`learn`) is made into a method by the host; the
   fitted verbs and the probes are two systems. Fold: a fitted model is a probe of the state, a verb is a probe with
   arguments. Decided on `supervised-learning`.
4. **Comparison.** Two sessions on one page with the same step cursor and overlaid curves; the same data and keys so
   the only difference is the option. The player then drives two producers.
5. **Checkpoints everywhere.** Probes seek from checkpoints, so every algorithm's state must serialise; networks need
   parameter checkpoints; samplers need their chains.
6. **Keys across solvers.** Changing the solver must not change the data or the start; the key tree has to be the same
   whatever fills the slots.
7. **Cost.** A gradient solver over a network at 60 steps a second on the main thread: the worker producer. Everything
   in `train/` waits on it for anything beyond toy size.
8. **Datasets with truth.** Every data slot should carry the truth the probes compare with; most generators have it,
   real datasets do not; what a page shows then.
9. **Evaluation on the tape.** Held-out metrics at a schedule, consistently named, so a sweep panel works over any
   method.
10. **Multi-class and multi-output.** Wrappers (one-vs-rest, chains) as slots of the base methods rather than methods.
11. **Studies that run studies.** A learning curve over a regularisation path over a solver: nesting sessions, and
    what of the inner tapes to keep. The repeated-fit studies of `statistics` and `learning-theory` are the hard case.
12. **What is in `lib` first.** Many capability pages name algorithms the library lacks (trees, ensembles, SMO, GP fitting,
    isolation forests, LOF, matrix profiles, change points, pattern mining, CKY, CRFs, label models). Each page names
    them; the order of building them is the plan's next table.
13. **The folder summary and the README.** The summary in `index.ts` is the README's first paragraph, copied by hand
    for now; a script could keep them equal.
14. **Testing depth.** A smoke test per method stays; shared pieces (slot tables, bases, sweeps, kinds) get the full
    standard.
15. **The encyclopedia's needs.** Its pages compose sessions; what they need beyond the panels (prose with live
    values, narratives, the same session on several pages) shapes render's next layer; they read the methods, they do not shape them.
16. **The simulated and the real.** Several pages (language models, engineering, multimodal learning)
    show mechanisms at toy scale or simulate what cannot run here; each page must say which it is.

## 7. The reflections of 2026-10-28, worked through

### 7.1 Modes: how a slot branches without a solver tree

A **mode** is one option of a slot with everything it needs in one module: its own knobs (a space), a factory that
builds the algorithm, the extra paths it alone logs, and what it needs of the other slots. The Python instinct (a key
to a function that solves and logs) is right, and the log contract makes the second half unnecessary: the model
declares the paths every mode logs (the objective, the coefficients), the mode declares only its extras, and the
method's contract is the union.

```ts
// fit/classify/logistic/modes/gradient.ts: one mode, one module, nothing outside it
export const gradient = mode({
  space: dict({ optimiser: optimiserSlot, schedule: scheduleSlot, batch: int(1, 4096, { nullable: true }) }),
  make: (problem, hyper, key) => gradientAlgorithm(problem, hyper, key), // library pieces only
  params: (state) => ({ coef: state.w, intercept: state.b }), // the common fitted parameters
  logs: { 'batch/loss': scalar() }, // what only this mode can say
  needs: { penalty: ['none', 'l2', 'l1', 'elasticNet'] }, // the validity table, locally
})
```

- The state differs by mode (IRLS keeps weights and halvings, SGD keeps a batch position), so each mode gives
  `params(state)`, the one fitted shape the probes and verbs read. Probes are mode-independent for free.
- Full batch against stochastic is not a second switch beside the optimiser: it is a knob of the gradient modes
  (`batch`, null for the whole set). Second-order modes have no batch knob. One interface, no `SGDClassifier`; a
  cross-validation runs over the hyperparameters including the mode.
- Branching nests: a mode's space can hold a slot (the optimiser), whose option can hold knobs (a momentum). Nesting
  is the rule, not an exception, because a mode is a space and a space can contain spaces. The rule that keeps it
  maintainable: a mode's definitions stay in its module; a mode reuses a shared slot by importing it, never by
  redefining it.
- Same model, different modes, different solutions: the lasso by coordinate descent, by LARS, by proximal gradient;
  search by DFS, BFS and A* over the same problem. The page shows the solutions side by side; the comparison panel
  is the rendering need.
- `lib/space` already has what this needs: a dict key with `when: { key, in }` is active only when another key's
  value is in a set, and `activeKeysOf` says which keys are live. `SpaceControls` does not read `when` yet: that is
  the first render job (7.4).

### 7.2 Two kinds of method: a task and a problem

- A **task method** (the scikit-learn world, `fit/`): a generic approach to a kind of problem. The data is a slot
  (any dataset of the task's kind), the model is an entry of a registry, the modes are as above, evaluation is
  standard for the kind. Pages: "a classification problem, set up five ways".
- A **problem method** (the torch and probabilistic-programming world, `train/`, `infer/`): a particular setup. The
  network's layers, widths and loss, or the program's latents and observations, are the method; the data is fixed or
  a narrow slot. What stays generic is the optimiser or sampler slot, the schedule and the logging. Pages: "this
  model on this problem, watched".
- Both run as sessions on the same contract and player; they differ in what the controls show (a registry and
  modes, or the problem's own knobs) and in how they are tested.

### 7.3 Registries and selectors: datasets, environments, models

- **Datasets.** The catalogue has generators (`name`, `kind`, `description`, `knobs`), vendored real datasets and
  remote assets pinned by hash with a licence, grouped by hand into collections (`classification2d`,
  `regressionTabular`, `timeSeries`, ...). It lacks tags. Add a **registration phase**: every dataset registers with
  `kind` (regression, classification, clustering, density, ranking, sequence, ...), `modality` (tabular, image, audio,
  time series, text, graph), `real` or synthetic, size, licence and source; collections become saved queries. A query
  API (`datasets.where({ kind: ['regression', 'classification'], real: true, modality: 'audio' })`) feeds a
  `DatasetSelector` control with those filters and a search box. New real datasets register the same way.
- **Environments.** Separate, as they should be: an `EnvSpec` carries observation and action spaces already; the
  registry adds tags (discrete or continuous actions, vector, grid or image observations, episodic or continuing,
  family) and an `EnvironmentSelector` filters on them, showing the spaces.
- **Models.** The same pattern for task methods: a registry of models per task with a `ModelSelector`: a list of
  names, each with a settings button that opens the model's space (its modes inside). A page then composes
  `DatasetSelector`, `ModelSelector`, the session player and panels, and nothing bespoke.
- The place for the registries is `data/` (datasets, environments, problems) and each task's `shared/` (models);
  the controls are render's.

### 7.4 Controls: hierarchical and compact

- Flat key-value panels stay the simple form. The responsive form renders a dict space recursively: a `choice` key
  that other keys depend on (`when`) shows its dependants only when they are active; a nested dict is a group; a
  slot's option with its own space opens as a settings popover or an inline group.
- The model list with a gear per entry is the compact form of the same thing: the list is a `choice`, the gear opens
  that option's space.
- A space is the single source: the same `hyperSpace` drives the controls, the cross-validation grid and the tuning
  samplers, so a conditional knob is handled once (`lib/space` already encodes inactive keys for samplers).

### 7.5 Pipelines and unions

`pipeline`, `featureUnion` and `columnTransformer` exist in `lib/learn`, and a pipeline's `hyperSpace` is the dict of
its steps' spaces by step name. A model's modes therefore travel with it into a pipeline unchanged: the pipeline's
space has a `logistic` key whose value is the model's space, modes and all. Nothing is reinvented. Pipelines are
particular compositions (a scaler, an encoder, a model), not arbitrary graphs, and each is written as such.

### 7.6 Cost-sensitive learning

Present in pieces: sample weights on `LearnData` and `fit`, `tableCost` and `thresholdCosts` in `metrics` (a cost
matrix over a confusion table and over thresholds), the `ThresholdExplorer`, and the fairness reductions that solve
cost-sensitive problems. Missing: a model-selection study that takes a grid, a cost matrix and a model and shows the
chosen threshold and configuration under the costs (`study/`), and a `classWeight` option on the task methods'
objective slot. Both small.

### 7.7 Logging: selective by default, adjustable per use

Already there: a subscription names paths with `Log({ every: n | 'logSpaced' | schedule, detail: 'summary' | 'full',
budget })`, each method has `defaults`, and the host offers off, defaults, all and custom. `off` keeps checkpoints
only, for a large job. Add `keep: 'all' | 'last' | n` so a path can keep only its last value or a window, and name
the two or three paths that make a method's story (the loss and the path) as its defaults, everything else on
request. A page passes its own subscription, so one method serves a page that wants every tensor and another that
wants the loss alone.

### 7.8 Predicting after training

Yes. A task method's fitted value has the verbs of its kind (`predict`, `predictProba`, `decisionFunction`,
`transform`, as the kind dictates), built from plain parameters; `revive(params)` rebuilds them anywhere. A session
exposes the parameters at any step as a probe (`model/params` in the iterative base), so "the model at step 40 on
new data" is `revive(session.probe(40, 'model/params')).predict(x)`. A `fittedAt(t)` convenience on the session
makes that one call. Train, validate, predict is then: a session on the training split, held-out metrics on the tape
(7.7), and the fitted value on anything after.

### 7.9 Testing the methods

Agreed: a method wraps library code that has the full standard (oracle goldens, properties, `checkPrimitive`), so a
method gets a smoke test (it runs, its records have the right shape, its probes answer) and no fixtures; parity with
another library's composition is not the point. Shared pieces (slot tables, bases, registries, sweeps) are library
code in all but location and get the full standard.

### 7.10 Probabilistic data structures

Missing from the library and worth a domain, `lib/sketches`: Bloom, counting Bloom and cuckoo filters, count-min and
count sketches, HyperLogLog, MinHash and locality-sensitive hashing, reservoir sampling, quantile sketches (t-digest,
KLL), top-k (space-saving), skip lists. Each is a traced `Algorithm` where it grows. Methods (`simulate/` or `data/`):
a Bloom filter filling with its false-positive rate against the formula, HyperLogLog's estimate converging on a
stream, count-min's error against its width and depth. Tests against exact counts and the theoretical bounds rather
than an oracle.

### 7.11 Extending the datasets without hand work

The mechanism exists: a remote asset pinned by SHA-256 with its licence, fetched and parsed on demand. What is missing
is the ingestion: a Python script in the oracle that takes a manifest (source URL, licence, converter), downloads,
converts to compact typed arrays, pins the hash and writes the registry entry with its tags (7.3). Candidates with
permissive terms, by modality:
- Images: MNIST (CC BY-SA 3.0), Fashion-MNIST (MIT; a test subset is vendored), KMNIST (CC BY-SA 4.0), the
  scikit-image sample images (mostly public domain), Olivetti faces (research use), CIFAR-10 (no stated licence,
  universally used).
- Audio: Free Spoken Digit Dataset (CC BY-SA 4.0), Speech Commands (CC BY 4.0), NSynth (CC BY 4.0), LibriSpeech
  subsets (CC BY 4.0); ESC-50 is non-commercial and would need a flag. Synthesised audio (tones, chirps, noise
  mixtures) as generators from `signal`.
- Time series: Mauna Loa CO2 (public domain), Jena climate, UCI HAR (CC BY 4.0), the M4 competition sets, PhysioNet
  ECG (ODC-BY), the UCR archive (free for research, licence per set).
- Tabular classics beyond the vendored ones: adult, bank marketing, bike sharing, heart disease (UCI, CC BY 4.0),
  California and Ames housing.
- Text: 20 newsgroups, SMS spam (UCI), AG News; graphs: Cora and Citeseer (research), Newman's small networks;
  recommendation: MovieLens (research terms; a sample is vendored), Book-Crossing.

### 7.12 The structure

Folded into section 3: folders by what a page does; the encyclopedia is a reader of the methods, not their shape.

## 8. Build order

Core pieces first, each proved on its own; exemplars when they hold; the catalogue after that.
1. **Core pieces** (`methods/shared/`, `lib/space`, render):
   - the controls honour conditional keys and nested spaces (groups, a settings popover per option);
   - the dataset registry with tags (kind, modality, real, size, licence, source), its query API and the
     `DatasetSelector`; the environment registry and `EnvironmentSelector` beside it;
   - the mode module (space, factory, `params`, extra logs, needs) and the slot tables (solvers, objectives,
     schedules, init) with their validity rules;
   - `kinds` with the standard probes, `fittedAt(t)` on a session, `keep` in subscriptions;
   - the comparison of runs on one cursor, and the sweep panel.
2. **Exemplars**, one per kind of method, each a page that uses only the pieces above:
   - a task method: logistic regression with every mode, five solvers on one dataset, boundaries side by side;
   - a dataset page: the selector over real and synthetic sets of one kind, one model following the choice;
   - a problem method: a small network on a planted problem, the trainer base;
   - a Bayesian problem: a program and two samplers, the chains compared;
   - an agent: one environment from the selector, one agent, value map and episode;
   - a study: a grid, a cost matrix and the chosen configuration.
3. **The catalogue**, folder by folder from the capability pages, each capability moving into its folder as it is
   built, the library getting first whatever a method needs.
