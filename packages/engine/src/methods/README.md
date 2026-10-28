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

## 3. The structure: the encyclopedia's chapters

The methods are organised as the encyclopedia (`ml-concepts`, `content/taxonomy.yaml`) is: its chapters, one folder
each under `methods/`, named with the chapter's slug; a chapter with many sections has them as subfolders named as
the encyclopedia names them (`supervised-learning/trees-and-ensembles/`). The parts are a grouping, not folders. Where
a method has no chapter in the encyclopedia the methods tree adds one in the same style (`active-learning`,
`search-and-planning`, `causal-inference`, `complex-systems`), and the encyclopedia can follow. Names are the
encyclopedia's, which settles the old inconsistency (`active-learning` against `transfer`): `transfer-and-meta-learning`,
`semi-supervised-learning`, `anomaly-detection`, `reinforcement-learning`.

```
methods/
  README.md               this document
  shared/                 what every chapter uses (section 4)
  <chapter>/
    README.md             the chapter's workshop page: capabilities, examples, rendering, shared pieces, problems
    index.ts              the chapter's summary (its entry in the docs tree) and its exports
    shared/               the chapter's own pieces: its slots' option tables, its probes, its records, its data slots
    <section>/            when the chapter is big: the encyclopedia's sections as subfolders
    <method>.ts           one method: a few dozen lines naming the slots, the contract and the probes
    <method>.test.ts      the smoke test (runs on a small dataset; the records have the right shape)
```

The chapters, by part (a `README.md` in each; a dagger marks a chapter the encyclopedia does not have):

| part                               | chapters                                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 Mathematical foundations         | `mathematics`, `probability`, `statistics`, `optimisation`                                                                                                                                                                                                                                                            |
| 2 Signals, systems and information | `information-theory`, `signal-processing`, `control-theory`                                                                                                                                                                                                                                                           |
| 3 Data                             | `data-preparation`, `features`, `exploration-and-discovery`, `shift-and-drift`, `data-centric-learning`                                                                                                                                                                                                               |
| 4 Principles of learning           | `learning-theory`, `model-selection`, `evaluation-metrics`, `decision-theory-and-calibration`, `loss-functions`, `probabilistic-inference`                                                                                                                                                                            |
| 5 Learning paradigms               | `supervised-learning`, `semi-supervised-learning`, `weak-supervision`, `unsupervised-learning`, `representation-learning`, `anomaly-detection`, `online-learning`, `active-learning`†, `transfer-and-meta-learning`, `reinforcement-learning`, `logic-and-rule-learning`, `search-and-planning`†, `causal-inference`† |
| 6 Neural architectures             | `feedforward-networks`, `convolutional-networks`, `graph-neural-networks`, `sequence-models`, `transformers`, `large-language-models`, `generative-models`, `scientific-machine-learning`                                                                                                                             |
| 7 Application domains              | `time-series`, `online-experimentation`, `recommendation-and-retrieval`, `natural-language-processing`, `computer-vision`, `multimodal-learning`, `games`, `complex-systems`†                                                                                                                                         |
| 8 Practice                         | `trustworthy-machine-learning`, `engineering`                                                                                                                                                                                                                                                                         |
| 9 Curiosities                      | `curiosities`                                                                                                                                                                                                                                                                                                         |

Three kinds of file in a chapter:
- **Methods** (`<method>.ts`): a `Method` of the log contract, or an estimator on the `learn` contract that becomes
  one. Each is small because its slots and probes come from `shared/`.
- **Studies** (also `<method>.ts`, named for what they compare): a method whose run drives other methods: an optimiser
  race, a regularisation path, a learning curve, a restart study, a comparison of two solvers on one dataset. They sit
  in the chapter whose question they answer (an optimiser race in `optimisation/`, a lasso path in
  `supervised-learning/`).
- **Shared pieces** (`shared/`): slot option tables (`solvers.ts`: the solvers this chapter's objectives admit),
  standard probes for the chapter's kind of model, the chapter's exposition records when the generic ones do not fit,
  and data slots with their truths.

Nothing under `methods/` draws. Rendering lives in `mdml-render`; the chapter README says what it needs drawn.

### Where the old areas go

The 56 areas of the old index and the 12 built this week, mapped onto the chapters. The built ones stay where they
are until the structure is agreed; moving them is a `git mv` and an import path.

| old area                                                                                                                                                      | chapter (section)                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `foundations`, `algorithms`, `linalg` (built)                                                                                                                 | `mathematics`; curiosities to `curiosities`                                                                                                                                                                                                        |
| `monte-carlo`, `probability` (built), `stochastic` (built)                                                                                                    | `probability`                                                                                                                                                                                                                                      |
| `survival`, the statistical half of `experimentation`                                                                                                         | `statistics`                                                                                                                                                                                                                                       |
| `optimisation`, evolution strategies of `evolutionary`                                                                                                        | `optimisation`                                                                                                                                                                                                                                     |
| `information`                                                                                                                                                 | `information-theory`                                                                                                                                                                                                                               |
| `signal`, `audio`                                                                                                                                             | `signal-processing` (audio a section)                                                                                                                                                                                                              |
| `dynamics` (built)                                                                                                                                            | `mathematics` (ODE systems), `control-theory` (Kalman tracking)                                                                                                                                                                                    |
| `patterns`, causal discovery of `causal`                                                                                                                      | `exploration-and-discovery`                                                                                                                                                                                                                        |
| `theory`                                                                                                                                                      | `learning-theory`                                                                                                                                                                                                                                  |
| `tuning`, `tune` (built)                                                                                                                                      | `model-selection`                                                                                                                                                                                                                                  |
| calibration, conformal and uncertainty of `trust`                                                                                                             | `decision-theory-and-calibration`; shift to `shift-and-drift`; robustness below                                                                                                                                                                    |
| `models`, `sampling` (built), `sequence` (built)                                                                                                              | `probabilistic-inference` (topic models to `unsupervised-learning/topic-models`)                                                                                                                                                                   |
| `linear` (built), `gam`, `svm`, `kernel-methods`, `gp`, `naive-bayes`, `discriminant`, `neighbours`, `trees`, `ensemble`, `multiclass`, `mixture/moe` (built) | `supervised-learning` (one section each)                                                                                                                                                                                                           |
| `semi-supervised`                                                                                                                                             | `semi-supervised-learning`                                                                                                                                                                                                                         |
| `cluster` (built), `mixture` (built), `decomposition`, `manifold`, `covariance`                                                                               | `unsupervised-learning` (one section each)                                                                                                                                                                                                         |
| `embedding`                                                                                                                                                   | `representation-learning`                                                                                                                                                                                                                          |
| `anomaly`                                                                                                                                                     | `anomaly-detection`                                                                                                                                                                                                                                |
| `transfer`, `federated`                                                                                                                                       | `transfer-and-meta-learning` (federated a section)                                                                                                                                                                                                 |
| `rl`                                                                                                                                                          | `reinforcement-learning`                                                                                                                                                                                                                           |
| `rules`, the logic of `gofai`                                                                                                                                 | `logic-and-rule-learning`                                                                                                                                                                                                                          |
| `search` (built), populations of `evolutionary`, planning and CSP of `gofai`                                                                                  | `search-and-planning`                                                                                                                                                                                                                              |
| treatment effects of `causal`                                                                                                                                 | `causal-inference`                                                                                                                                                                                                                                 |
| `neural`                                                                                                                                                      | `feedforward-networks`, `convolutional-networks`, `sequence-models`, `transformers`, `graph-neural-networks`                                                                                                                                       |
| `llm`                                                                                                                                                         | `large-language-models`                                                                                                                                                                                                                            |
| `generative`                                                                                                                                                  | `generative-models`                                                                                                                                                                                                                                |
| `scientific`                                                                                                                                                  | `scientific-machine-learning`                                                                                                                                                                                                                      |
| `timeseries`                                                                                                                                                  | `time-series`                                                                                                                                                                                                                                      |
| the online half of `experimentation`                                                                                                                          | `online-experimentation`                                                                                                                                                                                                                           |
| `retrieval`                                                                                                                                                   | `recommendation-and-retrieval`                                                                                                                                                                                                                     |
| `text`                                                                                                                                                        | `natural-language-processing`                                                                                                                                                                                                                      |
| `vision` (built)                                                                                                                                              | `computer-vision`                                                                                                                                                                                                                                  |
| `explain`, `fairness`, `privacy`, `adversarial`, robustness of `trust`                                                                                        | `trustworthy-machine-learning` (one section each)                                                                                                                                                                                                  |
| `complex-systems`, `active-learning`, `weak-supervision`                                                                                                      | the same                                                                                                                                                                                                                                           |
| (new)                                                                                                                                                         | `data-preparation`, `features`, `shift-and-drift`, `data-centric-learning`, `evaluation-metrics`, `loss-functions`, `online-learning`, `control-theory`, `games`, `multimodal-learning`, `engineering`, `curiosities`, `statistics`, `mathematics` |

## 4. `methods/shared/`: what every chapter needs

- `slots/`: the catalogues behind the common slots. `solvers.ts` (closed form, IRLS, coordinate descent, proximal,
  every `optim` optimiser with its schedule, EM, the samplers), `objectives.ts` (losses and penalties by name with
  their properties: convex, smooth, has a prox, is a likelihood), `schedules.ts` (steps, batches, stopping rules,
  restarts), `init.ts` (starts). Each is a `choice` space plus the conditional knobs plus the property table the
  validity rules read. The showcases are `optimisation/` (solvers) and `loss-functions/` (objectives).
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

Common requirements that the chapter READMEs ask for, by how far they exist:

| requirement                                                                                                 | status                                                   |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| controls from a space with **conditional** knobs (an option's knobs appear with it)                         | space yes, controls no                                   |
| the **comparison** of two or more runs of one method with different options: overlay and side by side       | missing; asked by 20 chapters                            |
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
7. **Cost.** A gradient solver over a network at 60 steps a second on the main thread: the worker producer. Every
   part-6 chapter waits on it for anything beyond toy size.
8. **Datasets with truth.** Every data slot should carry the truth the probes compare with; most generators have it,
   real datasets do not; what a page shows then.
9. **Evaluation on the tape.** Held-out metrics at a schedule, consistently named, so a sweep panel works over any
   method.
10. **Multi-class and multi-output.** Wrappers (one-vs-rest, chains) as slots of the base methods rather than methods.
11. **Studies that run studies.** A learning curve over a regularisation path over a solver: nesting sessions, and
    what of the inner tapes to keep. The repeated-fit studies of `statistics` and `learning-theory` are the hard case.
12. **What is in `lib` first.** Many chapters name algorithms the library lacks (trees, ensembles, SMO, GP fitting,
    isolation forests, LOF, matrix profiles, change points, pattern mining, CKY, CRFs, label models). Each chapter's
    README names them; the order of building them is the plan's next table.
13. **The chapter summary and the README.** The summary in `index.ts` is the README's first paragraph, copied by hand
    for now; a script could keep them equal.
14. **Testing depth.** A smoke test per method stays; shared pieces (slot tables, bases, sweeps, kinds) get the full
    standard.
15. **The encyclopedia's needs.** Its pages compose sessions; what they need beyond the panels (prose with live
    values, narratives, the same session on several pages) shapes render's next layer.
16. **The simulated and the real.** Several chapters (`large-language-models`, `engineering`, `multimodal-learning`)
    show mechanisms at toy scale or simulate what cannot run here; each page must say which it is.
