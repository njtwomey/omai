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

**The tree today.** The fifty-odd folders named after the encyclopedia's chapters, each with a page, were written
before this layout was decided and mirror the encyclopedia too literally. They stay for now as the catalogue of
capabilities (what to build, with examples, rendering needs and problems); when building starts each capability
moves to its folder above and the chapter folders go. One chapter was re-cut on 2026-10-28: `transfer-and-meta-learning`
conflated three things and is now `transfer-learning` (fine-tuning, domain adaptation, domain generalisation,
continual, federated), `meta-learning` (MAML and its kin, metric and model-based learners, hypernetworks, learned
optimisers, hypergradients) and `multi-task-learning` (parameter sharing, task weighting, gradient surgery, auxiliary
tasks, task affinity), which the old chapter did not mention at all.

**The README format.** Each chapter's README is the to-do list for its area, in the form prototyped on `mathematics`
(linear algebra in full), `time-series`, `probabilistic-inference` (the sequence models laid out) and the three
chapters above: the opening paragraph; a **Structure** block giving the folder tree the chapter's methods will form
under the layout above, with the full proposed path of every group and the chapter's `shared/` helpers named, so the
move is a lookup; one **Methods** table per group with the columns `method` | `why it is interesting` (what to look
into, which outputs matter, the knob worth turning, the comparison worth making) | `tracks` (the paths on the tape,
each tagged scalar, tensor, tensor kept in full, probe, projection or event) | `needs from lib` (the module and the
function that exists, or what is missing marked TODO) | `status` (`built <path>`, `lib ready`, `lib partly`, `lib
missing`); a **shared/** table (`helper` | `what it does` | `used by`); then the Examples, Rendering and Problems
sections as before. A family of modes is one row that names the slot carrying the modes (`arima` with the fit as a
mode), not a row per mode. The tracks column is the contract a method will declare and the paths a page can ask for,
named as the built methods name theirs (`solver/objective`, `model/params`, `image/pixels`, `chain/position`). The twelve built areas move the same way: `cluster` and
`mixture` to `fit/cluster` (the mixture of experts to `fit/regress`), `linear` to `fit/classify` and `fit/regress`,
`tune` to `study/`, `sampling` and `sequence` (the hidden Markov model) to `infer/`, `dynamics` to
`simulate/dynamics` (ODE systems) and `infer/filters` (Kalman tracking), `linalg` to `simulate/numerics`,
`probability` and `stochastic` to `simulate/stochastic`, `search` to `search/`, `vision` to `process/images`. Nothing
moves until the core pieces of section 8 are built and a first exemplar has proved the pattern.

## 4. `methods/shared/`: the option sets and the bases

What a page exposes is a small number of **externally facing option sets**: which dataset, which environment, which
model, which solver, which loss. Each is one thing in one place: a space (what the controls draw and the tuners
sample), a table (the options and what the library does for each), and what a selector shows for an option (a name,
a line, its tags). The shared directory is organised around them; the name `shared` is provisional.

```
shared/
  options/
    datasets.ts       the dataset registry: every entry tagged (kind, modality, real, size, licence, source), queries,
                      and what DatasetSelector shows; the library pre-registers its catalogue, an app adds its own
    environments.ts   the environment registry over EnvSpec, tagged by its spaces (discrete or continuous actions;
                      vector, grid or image observations), episodic or continuing; EnvironmentSelector
    problems.ts       search problems and worlds, the same way
    models.ts         per-task model registries (the classifiers, the regressors, ...): each model with its space and
                      its modes; ModelSelector draws the list with a settings button per model
    solvers.ts        the solver union the task methods share (closed form, IRLS, coordinate descent, proximal,
                      the optimisers of `optim` with their schedules and batches, EM, the samplers)
    objectives.ts     losses and penalties with their properties (convex, smooth, has a prox, is a likelihood)
    schedules.ts      steps, batches, stopping rules, restarts
    init.ts           starts (keys, warm starts, k-means++, pretrained)
  modes.ts            the Mode type and `modes({ ... })`: a table keyed by the mode hyperparameter's values
  kinds.ts            the kinds of model with their verbs and standard probes
  bases/              one base per way of running: one-shot, iterative, trainer, agent, procedure, sweep
  sweeps.ts           studies over an inner method (paths, curves, races, repeated fits)
  streams.ts          data arriving step by step with planted drift
  evaluation.ts       held-out metrics at a schedule on the tape (`eval/<metric>`)
  data.ts             adapters from the registry to a method's data slot, with the truth beside the data
  pictures.ts         images, grids of images, surfaces with points, for any method with a 2-D field
  records.ts          generic step-like records the `exposition` kinds do not cover
```

Registration is explicit data, never an import-time side effect (the library's entry points have none): the library
ships its registries built from its catalogues with tags at the point of definition, and a page or an app extends
one with `register`, getting a registry back. A folder's own `shared/` holds what only it needs: its model registry
entries, its probes, its records, its planted data.

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

### 7.1 Modes and the hyperparameter tree

**The tree.** A configuration has two shapes. The simple one is a flat dict of knobs. The responsive one is a
**union**: a key whose value names an option, and each option with a space of its own, nested as deep as it goes.
Choosing a mode specifies its child pieces; the batch size exists only inside the gradient modes; the optimiser's
momentum exists only inside the momentum optimiser inside a gradient mode. Nesting is the rule, not an exception,
because an option's space is a space. This is a new kind in `lib/space` beside `dict` (`union`, with `encode`,
`sample`, `grid`, `defaults` and `contains` extended to it: an inactive branch encodes as the midpoint, as `when` keys
do today); `when` stays for the simple cross-condition between two sibling knobs. One tree drives the controls, the
cross-validation grid and the tuning samplers, so a conditional knob is handled once.

```ts
const linear = dict({
  loss: union('loss', {
    squared: dict({}),
    huber: dict({ delta: real(0.1, 10) }),
    quantile: dict({ q: real(0.01, 0.99) }),
  }),
  penalty: union('penalty', {
    none: dict({}),
    l2: dict({ alpha }),
    l1: dict({ alpha }),
    elasticNet: dict({ alpha, l1Ratio }),
  }),
  solver: union('solver', {
    closedForm: dict({}),
    irls: dict({ tol, maxIter }),
    coordinateDescent: dict({ tol, maxIter }),
    proximalGradient: dict({ step, maxIter }),
    lars: dict({}),
    gradient: dict({ optimiser: optimiserUnion, batch: int(1, 4096, { nullable: true }), schedule: scheduleUnion }),
  }),
})
```

**The table.** The mode hyperparameter is the union's tag (`solver`), and the method holds a table keyed by its
values, each value a module: the Python instinct, a key to the function that solves.

```ts
export const solvers = modes({
  closedForm: {
    make: (problem, h, key) => leastSquaresAlgorithm(problem, h),
    params: paramsOfLstsq,
    needs: { loss: ['squared'], penalty: ['none', 'l2'] },
  },
  irls: { make: irls, params: paramsOfIrls, needs: { loss: ['huber', 'quantile'], penalty: ['none', 'l2'] } },
  lars: {
    make: lars,
    params: paramsOfLars,
    needs: { loss: ['squared'], penalty: ['l1'] },
    extras: { 'path/coefficients': tensor({ role: 'state' }), 'path/alphas': tensor({ role: 'state' }) },
  },
  gradient: { make: gradient, params: paramsOfGradient },
  // ...
})
```

- `make` returns a library algorithm; the state differs by mode, so each gives `params(state)`, the one fitted shape.
- **Solving and logging are separate objects.** A mode never logs. The method's contract is derived from the state:
  the model's paths (`model/coef`, `model/objective`, the decision surface) are computed from `params(state)` and
  are the same whatever the mode; a mode with something only it can say (LARS has the whole path) declares
  `extras`, also derived from its state. The log contract is intrinsic to the method, as hoped, and a mode is the
  solver alone.
- `needs` is the validity table, kept beside the mode it belongs to and nowhere else; the union's options are
  filtered by it, so an impossible solver is greyed, not errored. A mode's definitions never reach outside its
  module; it imports a shared union (the optimiser's) rather than redefining it.
- Full batch against stochastic is a knob inside the gradient modes (`batch`, null for the whole set), not a second
  switch beside the optimiser; second-order modes have none. One interface, no `SGDClassifier`.

**Linear regression, worked.** Loss and penalty are the model (the objective); the solver is the mode. The
textbook default follows from the objective: squared with no penalty or l2 is closed form, squared with l1 or
elastic net is coordinate descent, Huber and quantile are IRLS, anything smooth may also take a gradient mode. The
overlap you noted is exactly `needs`: LARS admits squared loss with l1 and nothing else, and it alone produces the
regularisation path as a native output (its `extras`); coordinate descent produces a path only as a study over
alpha with warm starts (`study/sweep`). A page that compares the two shows LARS's path against the swept one. "Some
have modes others don't" is then the rule: a mode's `extras` are shown when present, and the layout asks for them
by path.

**The same shape elsewhere.** A GAM's modes are backfitting and penalised IRLS over one model; graph search's modes
are DFS, BFS and A* over one problem, the same output (a path) by different means; a GP's modes are exact, sparse
and Laplace. In every case the model (or problem) fixes what is logged and the mode fixes how it is reached.

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
  `regressionTabular`, `timeSeries`, ...). It lacks tags. The **registration phase** adds them: every entry
  registers with `kind` (regression, classification, clustering, density, ranking, sequence, ...), `modality`
  (tabular, image, audio, time series, text, graph), `real` or synthetic, size, licence and source; collections
  become saved queries. The library pre-registers its own catalogue at the point of definition (tags beside the
  entry, no import-time side effects); an app registers more and gets a registry back. A query API
  (`datasets.where({ kind: ['regression', 'classification'], real: true, modality: 'audio' })`) feeds the
  `DatasetSelector` control, whose props are those filters plus a search box:
  `<DatasetSelector registry={datasets} kind={['regression', 'classification']} real modality="audio" />`.
- **Environments.** Separate, as they should be: an `EnvSpec` carries observation and action spaces already; the
  registry adds tags derived from them (discrete or continuous actions; vector, grid or image observations) and
  by hand (episodic or continuing, family), and an `EnvironmentSelector` filters on them and shows the spaces.
- **Models.** The same pattern for task methods: a registry of models per task, each entry a model with its space
  and its modes table (7.1), drawn by a `ModelSelector` as a list of names with a settings button that opens the
  model's tree. A page then composes `DatasetSelector`, `ModelSelector`, the session player and panels, and nothing
  bespoke.
- All three live in `shared/options/` (section 4); the controls are render's.

### 7.4 Controls: hierarchical and compact

- The flat dict stays the simple form, drawn as today. The responsive form draws the tree of 7.1: a union is its
  choice plus the chosen option's space, recursively; a nested dict is a group; `when` hides a sibling. The compact
  form is the list with a gear: the list is the union's choice, the gear opens that option's space in a popover.
- `SpaceControls` reads neither `when` nor a union today; both are the first render piece of section 8.
- Because the tree is the single source, the controls, the cross-validation grid and the tuners agree on which knobs
  exist for a given choice, and a knob is never described twice.

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
