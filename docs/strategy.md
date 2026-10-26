# Library, methods, setups: the strategy

What each layer is for, the one mechanism that ties them together, the standard every method meets, and what is
actually needed to get there. `docs/structure.md` stays the table of contents; this document is the reasoning behind
its section 5 and will replace the parts of it that disagree.

## 0. In one paragraph

The library (`lib/`) does everything and decides nothing about use. Methods (`methods/`) package the library into
things a person builds systems from and explains with: **constructables** (problems, models, agents you can make with
a few knobs), **procedures** (the one way each kind of thing runs, always on a tape), and **expositions** (what a
figure or a sentence needs, projected from the tape). A **session** (engine, not hub) is a procedure on its tape with
the ability to advance it live, seek any past step, fork from it, probe it, store it and load it, so "train up front and
interrogate" and "train live in the page" are the same object in two modes. A **setup** is data: which constructables,
which procedure, which mode, which probes, which views. The hub is a catalogue of setups and a host for sessions, and
nothing else; it can live anywhere.

The one principle: **everything iterative runs on the tape, the tape is the source of truth, and anything you want to
see at step `t` is read from the tape or recomputed from its nearest checkpoint, never chosen in advance.**

## 1. What is unstructured today

The pieces exist; they are wired four different ways.
- **Four snapshot mechanisms.** The tape has checkpoints (`checkpointEvery`, `seek`, `resume` in `lib/trace`).
  `lib/rl`'s `TrainState` keeps its own doubling-spaced `checkpoints` inside the state. `methods/neural/_train.ts`
  keeps its own `Snapshot`s at `snapshotAt` epochs. `methods/rl/playground.ts` snapshots tabular values at a
  pre-chosen `snapshots` set of episodes. Only the first can be sought; the others fix in advance what a page may
  later ask for.
- **Two hook systems.** `lib/nn/training` has Lightning's callbacks (`Callback`, `HookContext.log`, `stop`,
  `setLearningRateScale`, `tapsActivations`). `methods/neural/_train.ts` re-implements stopping, the adaptive learning
  rate and an `onEpoch` report on top of it; `playground` has `onEpisode`; hub entries have `progress`.
- **Records are built by hand inside loops.** `collecting`, `_records.ts`, `representationSteps`, the playground's
  episode recorder: each method decides during the run what a view will get. The tape, which already records metrics,
  tensors, frames and checkpoints under a policy, is not what the records come from, so a record cannot be re-derived at
  another stride, and a view cannot ask for anything the method did not foresee.
- **The hub sees records, never the run.** `HubEntry.run` returns `Explained<unknown>`; the worker posts JSON records
  back. There is no object on the page that can answer "what was the policy at episode 40, and how does it play". The
  thing the user asked for (click an iteration, see that iteration's policy act) is impossible by construction, not by
  omission.
- **Live or stored is undecided.** Every entry trains live under `maxRows`; nothing is precomputed, though
  `exportRun`, `serialiseFitted` and `saveState`/`loadState` all exist.
- **Entries mix four concerns.** A `HubEntry` chooses constructables, sets procedure defaults, builds records and
  declares layout in one object, so no part is reusable outside the hub.

None of this needs new machinery. It needs the existing machinery used once, in one place.

## 2. The layers, defined once

### 2.1 `lib/`: capabilities

Unchanged in role: the thirteen contracts of `structure.md` section 3, primitives, algorithms, no opinions about demos.
One addition, the session (section 4), which belongs here because "pre-train and interrogate" is an execution concern,
not a hub concern.

### 2.2 `methods/`: three kinds of export, kept apart

Every method area exports some of each, and the file names say which.

**Constructables.** Named builders of things people want elsewhere, each with a `Space` of knobs and defaults that work
in a demo budget, JSON configuration, and no closures over data:

| kind        | examples already in the tree                                                                            | contract                                            |
| ----------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| problems    | dataset generators with truth, `EnvDefinition`s, `SearchProblem`s, worlds, `cornellBox`, outline shapes | `DatasetGenerator`, `Env`, `SearchProblem`, `Scene` |
| models      | `Module`s (`_nets.ts`), estimators with `hyperSpace`                                                    | `Module`, `Estimator`                               |
| agents      | tabular, DQN, PPO, planners                                                                             | `Agent`                                             |
| simulations | `boidsAlgorithm`, `epidemicAlgorithm`, cellular automata, epicycles                                     | `Algorithm`                                         |

A constructable is reusable on its own: in a notebook, another repo, a slide. This is the "useful things people may want
to use elsewhere".

**Procedures.** How a kind of thing runs. Five families, one wrapper each (section 3); nothing in a method writes its
own loop. Every procedure is an `Algorithm` with `name`, JSON `config`, a `TraceSchema`, and `serialise`/`deserialise`,
or it is a one-shot computation. That is the whole requirement for being on the tape.

**Expositions.** Pure projections from a tape and a final state to records: `(run, state) → Exposition[]`. Step-like
records are derived from the tape at any stride the caller wants; structural records (a tree, a coefficient table, a
margin) are constructors from the final state, as now. Methods return `{ ...result, run, exposition }`.

### 2.3 Setups and the hub

A setup is data (section 5): constructables and knobs, a procedure and its parameters, a mode (live, stored, hybrid),
the probes it exposes, and view slots. The hub is the catalogue of setups plus one React host that runs sessions. It
draws nothing, computes nothing, and holds no logic that a notebook could not reproduce by reading the setup. Whether
it stays in `packages/hub` or moves to its own repo changes nothing above it, so that decision can wait.

## 3. The five families and their one wrapper

| family                | what is in it                                                                            | wrapper                                                                                                       | state on the tape                                | status                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| one-shot              | OLS, PCA, k-NN, tests, metrics, closed-form Bayes                                        | plain function or `Estimator.fit`; `traceCall` when it has steps worth showing (elimination, a factorisation) | none needed                                      | fine as is                                                                |
| sklearn-iterative     | coordinate descent, IRLS, k-means, EM, SMO, t-SNE, UMAP, boosting stages                 | `defineExplainedEstimator` over an inner `Algorithm`                                                          | the solver's state; `serialise` added            | fine; expose the inner algorithm so a session can step and seek it        |
| torch-like            | MLPs, CNNs, RNNs, transformers, VAEs, GANs, flows, diffusion, contrastive, GNNs          | one `Trainer`: a Lightning-style spec run as `Algorithm<TrainState>`                                          | params, buffers, optimiser state, epoch and step | `fit.ts` is most of it; `_train.ts` and the generative loops fold into it |
| sequential-decision   | bandits, tabular control, DQN, policy gradients, planning a sweep at a time, model-based | `trainingAlgorithm`                                                                                           | agent state `G`, environment state, counters     | exists; its in-state checkpoints move to the tape; `G` serialises         |
| search and simulation | A\*, CSP, MCTS, planning, boids, SEIR, automata, path tracing, epicycles                 | already `Algorithm`s                                                                                          | the search or world state                        | fine; `serialise` and a `frame` channel where missing                     |

**The sklearn family stays as it is.** `fit` returning a fitted value with verbs is the right shape; the only addition
is that an iterative estimator can hand out its algorithm (`estimator.algorithm(data)`) for a session to drive.

**The torch family gets a Lightning-style spec**, because `fit(model, data, loss)` cannot express a GAN's two
optimisers, a VAE's two-term loss with a schedule, diffusion's noise sampling or contrastive pairs, so each of those
methods currently writes a loop. The spec is what Lightning's `LightningModule` is, minus the class:

```ts
interface TrainingSpec<B> {
  readonly name: string
  readonly config: Json
  readonly modules: Readonly<Record<string, Module>> // 'net', or 'generator' and 'discriminator'
  readonly optimizers: Readonly<Record<string, GradientTransformation>>
  readonly batches: (key: Key, epoch: number) => Iterable<B> // from lib/data batching
  /** One optimisation step: compute losses, take the optimiser steps you choose, log what you like. */
  trainingStep(batch: B, ctx: StepContext & TrainHooks): void
  validationStep?(batch: B, ctx: TrainHooks): Readonly<Record<string, number>>
  /** Named probes a session may run on any stored step (section 4). */
  readonly probes?: Readonly<Record<string, Probe<TrainState>>>
}
```

`TrainHooks` is the existing `HookContext` (`log`, `logTensor`, `stop`, `setLearningRateScale`, `run`) plus
`step(optimizerName, loss)` for manual optimisation. The callbacks of `lib/nn/training/callbacks.ts` stay the hook
system; `_train.ts`'s stopping rules and adaptive learning rate become callbacks; its `Snapshot`s become tape
checkpoints. An MLP classifier method is then thirty lines: a module, a loss, an optimiser, a probe for the decision
surface.

**The sequential-decision family keeps `trainingAlgorithm`** and loses its private checkpoint list: the tape's
`checkpoint` channel holds `TrainState` (so `seek` and `fork` work on it like on anything else), `Agent` gains
`serialise`/`deserialise` of `G` (tabular agents are arrays already; neural agents are `saveState`), and
`policyAt(session, t)` is `greedyPolicyOf(agent, session.stateAt(t).agent)`.

**Search and simulation need nothing** beyond `serialise` where it is missing. The user is right that these are easy:
they were written on the tape from the start.

## 4. The session

The missing engine piece: a procedure on its tape, with six verbs. It lives in `lib/trace` (or a small `lib/session`)
and knows nothing of React, workers or the hub.

```ts
interface Session<S> {
  readonly algorithm: Algorithm<S>
  readonly seed: number
  readonly policy: TracePolicy
  /** The tape so far. */
  run(): Run
  /** Live mode: advance by a budget of steps or milliseconds (runFor), recording on the policy's schedule. */
  advance(budget: SliceBudget): Slice<S>
  /** The state at any recorded step: the nearest earlier checkpoint re-run forward with the original keys (seek). */
  stateAt(t: number): S
  /** A new session starting from stateAt(t): same seed to continue exactly, another to branch. */
  fork(t: number, options?: { seed?: number; config?: Json }): Session<S>
  /** Run a named probe on a past state: a decision surface on a grid, a greedy episode, the parameters. */
  probe<T>(t: number, name: string, args: Json): T
  /** Stored mode: the tape (with checkpoints) as JSON, and a session rebuilt from one. */
  export(): Run
}
function session<S>(algorithm: Algorithm<S>, options: { seed; policy; probes }): Session<S>
function sessionFrom<S>(algorithm: Algorithm<S>, run: Run, options): Session<S> // stored or hybrid
```

Why this is cheap: `runFor` advances in slices that equal an uninterrupted run; `foldIn(stepsKey, t)` makes a fork at
`t` the run's own continuation unless the seed changes; `seek` re-runs at most one checkpoint interval; `exportRun` and
`parseRun` already validate the JSON. The session is a thin object over four existing functions.

**Probes are named, not arbitrary.** The page cannot send a closure across a worker, so a method declares what may be
asked of a state (`decisionSurface(grid)`, `greedyEpisodes(n, seed)`, `tabularSnapshot()`, `params()`,
`reconstruct(images)`, `sample(n, seed)`), each returning JSON or an exposition record. This is Lightning's hooks turned
round: hooks pull values out _while_ training; probes pull them out of _any stored step_ afterwards. It is also the
answer to "the various hooks so we can get the things we need externally".

**The host.** `packages/hub/src/react/useHubRun.ts` becomes a `SessionHost`: one session per page, in a worker when the
setup is in the catalogue, answering `{ advance | stateAt | probe | fork | export }` messages with JSON. Progress is no
longer a callback: the host reads `metric` events as they arrive on the tape.

**Three modes, one object.**

| mode   | starts from                                                                    | can do                                              | when                                                                                             |
| ------ | ------------------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| live   | `init`                                                                         | advance, seek, fork, probe, export                  | the default; anything that learns visibly inside the page's budget                               |
| stored | an exported run with checkpoints and final state (`make artefacts`, section 7) | seek, fork, probe; `advance` continues from the end | anything too slow or too large to train live: a language model, a diffusion model, a long RL run |
| hybrid | a stored run                                                                   | stored, then live                                   | "here is a trained policy; keep training it with your settings"                                  |

**The RL story the user asked for, end to end.** A setup pairs cliff walking with Q-learning, mode live. The page shows
the return per episode from the tape. Clicking episode 40 sends `probe(stepOfEpisode(40), 'greedyEpisodes', { n: 3,
seed })`; the host seeks the state, runs three greedy episodes with that policy, and returns their frames; the
`EpisodePlayer` (which already takes frames) plays them beside the value grid from `probe(t, 'tabularSnapshot')`. No
`snapshots` option, no `recordEpisodes` list: any episode, any time. `fork(t, { seed: 7 })` answers "what if it had
explored differently from here"; `fork(t, { config: { epsilon: 0 } })` answers "stop exploring now".

## 5. Setups

What was a `HubEntry`, with the session's concerns separated from the page's, and a mode:

```ts
interface Setup {
  readonly id: string
  readonly title: string
  readonly summary: string
  readonly tags: readonly string[]
  readonly problem:
    | { readonly collection: string; readonly dataset?: string; readonly knobs?: Knobs }
    | { readonly env: string; readonly knobs?: Knobs }
    | undefined
  readonly procedure: {
    readonly ref: MethodRef
    readonly params: DictSpace
    readonly budget: { steps: number; ms?: number }
  }
  readonly mode: 'live' | 'stored' | 'hybrid'
  readonly probes: readonly string[] // which of the procedure's probes the page offers
  readonly views: readonly ViewSlot[] // record kinds and the probe-driven views (a curve with a click handler)
}
```

`fitEstimator` and `fitEither` stay as the way to make a sklearn setup in one line. Setups are plain data, so the
same list drives the hub, a notebook or an external site. Where they live is a small decision: next to their methods
(`methods/<area>/setups.ts`, so an external hub repo imports them with the methods) or in `packages/hub/collections`
as now. Beside the methods is recommended; it is a file move either way.

The site's standalone hub pages (`apps/site/src/hub/pages/`) are one-off code and stay as they are; they are not
setups and need not become any.

## 6. Exposition from the tape

Two kinds of record, two sources:
- **Step-like records** (`curves`, `learningCurve`, `imageSteps`, `agentSteps`, `valueGrids`, `embeddingSteps`,
  `clusteringSteps`, `mixtureSteps`, `episodes`, `boardSteps`, `populationSteps`, `decisionSurface.steps`) are
  projections of the tape: `curvesFrom(run, names)`, `imageStepsFrom(run, 'samples')`, `agentStepsFrom(frames)`,
  built from `metric`, `tensor` and `frame` events by name. A view can ask for every fourth step or the last ten; the
  method did not decide. `collecting`, `_records.ts`'s curve builders and the playground's recorder retire into these.
- **Structural records** (`tree`, `coefficientTable`, `margin`, `rules`, `components`, `confusionMatrix`, the
  attribution and fairness records) are constructors from a final state, as now.

A method with a `TraceSchema` and `serialise` is exposition-ready before anyone writes a record: `render/trace`'s
metric, tensor, message and episode views draw any tape. Bespoke records are for when the generic view is not the
explanation.

## 7. Stored sessions (`make artefacts`)

For setups in `stored` or `hybrid` mode, a build step runs the session in Node and writes
`apps/site/.generated/artefacts/<setup-id>.<hash>.json` (an exported run with checkpoints and the final state), as
`make docs-catalogue` writes the docs. The hash is the engine version, the setup id, its configuration and seed, so a
changed method invalidates its artefact. A test re-runs a sample of artefacts and compares `end` summaries, which
determinism makes exact. Sizes are bounded by the policy's `maxBytes`, and the checkpoint interval is chosen from the
byte budget rather than fixed. Final states also serialise on their own where a page only predicts
(`serialiseFitted`, `saveState`), which is the small case of the same thing.

## 8. The standard

What every method meets; each line is checkable, most by a test over the method exports.
1. **On the tape.** Every iterative procedure is an `Algorithm` with `name`, JSON `config`, `schema`, `serialise` and
   `deserialise`; a test round-trips a state of each. One-shots are plain functions.
2. **No private snapshots.** Nothing stores past states in its own state or picks snapshot steps up front. Past states
   are tape checkpoints; anything wanted at step `t` is `stateAt(t)` or a probe.
3. **No private hooks.** Training hooks are `Callback`s; everything else is `observe` and probes. No `onEpoch`,
   `onEpisode` or `progress` in method signatures; progress is the tape.
4. **Records are projections.** Step-like records come from the tape; structural ones from the final state. Methods
   return `{ ...result, run, exposition }`.
5. **Constructables stand alone.** Exported with a `Space` of knobs and working defaults, JSON config, no closures
   over data, usable without the rest of the method.
6. **Setups are data.** No computation and no drawing in a setup; the host runs sessions and arranges views.
7. **Budgets are declared.** A live setup names its step and millisecond budget and finishes its defaults inside it in
   a worker; otherwise its mode is `stored`.
8. **Tests by layer, unchanged.** Full in `lib/`; a smoke test and the round-trip of rule 1 per method; the catalogue
   builds and every live setup finishes in budget.

## 9. What is actually needed

In order, each a batch built back to back and tested once at the end. Sizes are relative.

| #   | need                                                                                                                                                                                                                                                 | where                                                             | size                   | why first                                         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------- | ------------------------------------------------- |
| 1   | `Session` and `sessionFrom` over `runFor`, `seek`, `resume`, `exportRun`; named probes                                                                                                                                                               | `lib/trace/session.ts`                                            | small                  | everything else is expressed through it           |
| 2   | `serialise`/`deserialise` on every method-exported `Algorithm` and on `Agent` state; the round-trip test                                                                                                                                             | across methods, `lib/rl/agent.ts`                                 | mechanical, many files | without it nothing can be sought                  |
| 3   | RL on the tape: `TrainState.checkpoints` removed in favour of tape checkpoints; `policyAt`; `greedyEpisodes` and `tabularSnapshot` probes; the clickable curve beside `EpisodePlayer`; playground's `snapshots` and `recordEpisodes` options retired | `lib/rl/training.ts`, `methods/rl/playground.ts`, one render view | medium                 | the proving case the user named                   |
| 4   | The `Trainer`: `TrainingSpec` with manual optimisation over `fitAlgorithm`; `_train.ts`'s stopping and adaptive rate as callbacks; neural and generative methods as specs                                                                            | `lib/nn/training`, `methods/neural`, `methods/generative`         | medium to large        | removes the second hook system and the hand loops |
| 5   | Tape projections for the step-like records; retire `collecting`, `_records.ts` curves, the playground recorder                                                                                                                                       | `lib/exposition`, `lib/learn/explained.ts`                        | medium                 | makes records re-derivable                        |
| 6   | `Setup` with `mode` and `probes`; `HubEntry` renamed; `useHubRun` becomes the `SessionHost`                                                                                                                                                          | `packages/hub`                                                    | medium                 | the hub becomes the thin thing it should be       |
| 7   | `make artefacts` and the verification test                                                                                                                                                                                                           | `scripts/`, `apps/site/plugins`                                   | small to medium        | stored mode                                       |

**Not needed:** more hub entries or standalone pages; a new record system; a plugin or hook registry; changes to render
beyond one probe-driven view; moving the hub out of the repo now (a data catalogue moves in an afternoon whenever that
is wanted); consolidating the `-demo` pages (they are not setups).

## 10. Decisions to take

1. Does `Explained<T>` carry the `run` always, or only when `fit` is asked to trace? Always is simpler and the tape is
   small under `policies.minimal`; recommended.
2. Setups beside their methods or in the hub package. Beside the methods is recommended (section 5).
3. The default checkpoint byte budget for live sessions (a few megabytes keeps neural parameters at toy sizes).
4. Whether rule 3 retires `HubRunInput.progress` at once or after the host reads progress from the tape. At once is
   cleaner; the host change is small.
