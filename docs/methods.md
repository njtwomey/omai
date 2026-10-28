# Methods: what they are for, the log contract, and eager against live

A design for the methods layer on its own terms. Nothing depends on methods except the pages that show them (the
encyclopedia's, since the hub was removed on 2026-10-28), so methods are designed as the complete, self-sufficient unit: what a person reaches for to _do_ something with the
engine and to _see_ how it was done. The current state of the code is in `strategy.md` section 1; this document does
not start from it.

## 0. The two jobs

A method does two things, and the second is what makes it different from a function in a library:
1. **Wrap a core idea of the engine for an application.** Fit a classifier to rows, train a generative model on
   images, learn a policy in an environment, solve a puzzle, simulate a flock. The person names the thing and a few
   knobs; the method assembles the primitives.
2. **Capture everything needed to elucidate the model and its training.** Not the result alone, but the intermediate
   things an explanation, a figure or a question later will need: parameters as they changed, the decisions taken and
   the candidates not taken, the signals of learning, the internal activity, the data that went in, the randomness that
   made it reproducible. A method that returns only a fitted model has done half its job.

And one rule over both: **with nothing asked for, a method is as fast as the raw algorithm and returns answers.**
Elucidation is paid for only by whoever subscribes to it.

## 1. The unit: a method, its tape, a session

- **The tape** is the engine's `Run` (`lib/trace`): metrics, tensors, messages, frames and checkpoints, recorded as an
  `Algorithm` runs. Deterministic: step `t` draws the same numbers however the run was paused, resumed or replayed, so
  any step can be rebuilt from the nearest checkpoint and any step can be branched.
- **A method** is a declaration, not a loop:

  ```ts
  interface Method<S, C extends Json> {
    readonly name: string
    /** Knobs with defaults: the application's parameters, never functions. */
    readonly config: DictSpace
    /** Build the run from constructables (a dataset, a model, an environment, a problem). */
    make(config: C, key: Key): Algorithm<S>
    /** What can be logged and hooked, by path (section 3). */
    readonly contract: Contract<S>
    /** The subscription used when the caller gives none: a few cheap paths, or nothing. */
    readonly defaults?: Subscription
    /** Bespoke records from the tape, for views the generic ones cannot give. */
    readonly projections?: Readonly<Record<string, (run: Run, options?: Json) => Exposition>>
  }
  ```

  The algorithm it makes is the engine's; the method adds what can be known about it. It writes no loop and no hook
  of its own: loops are `runAlgorithm`, and everything it has to say goes through `ctx.log` against its contract.
- **A session** is a method running or run: the tape plus six verbs.

  ```ts
  interface Session<S> {
    advance(budget: SliceBudget): Slice<S>          // compute more
    stateAt(t: number): S                            // seek: nearest checkpoint, re-run forward with the same keys
    fork(t: number, options?: { seed?: number; config?: Json }): Session<S>   // branch from a past step
    probe(t: number, path: string, args?: Json): Json | Exposition            // a derived path, at any step, on demand
    project(name: string, options?: Json): Exposition                        // a bespoke record from the tape so far
    run(): Run                                       // the tape, exportable
  }
  session(method, { config, seed, mode, budget, logs }): Session<S>
  ```

  A session is the thing the old hub was trying to be, and belongs in the engine so a test, a notebook or any page can
  hold one.

## 2. The elucidation contract

What "everything needed" means: nine kinds of intermediate thing, each with the channel it travels on. A method's
contract (section 3) names which of them it offers; render draws each kind generically from the tape before any
bespoke record exists.

| intermediate thing               | captured as                                             | examples                                                                                                                 |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| state over time                  | checkpoints (seekable), `param`/`state` tensors by path | weights at epoch 7, centres at iteration 3, the Q-table at episode 40, the frontier                                      |
| decisions and their alternatives | messages: the candidates considered and the one chosen  | every split evaluated and the split taken; SMO's working set; the node expanded and why; a proposal accepted or rejected |
| signals of learning              | scalars                                                 | loss, score, gradient norm, learning rate, return, acceptance rate, duality gap                                          |
| internal activity                | taps → `activation` tensors                             | hidden activations, attention, responsibilities, residuals, messages in belief propagation                               |
| data flow                        | messages                                                | which rows were in the batch, sample weights per round, replay-buffer draws                                              |
| outputs over time                | derived paths on a fixed probe set, or frames           | predictions on a grid, samples from fixed noise, the decoded path, the rendered world                                    |
| structure                        | `describe()` → a `diagram` record                       | the network, the search tree, the factor graph, the pipeline                                                             |
| randomness                       | the keys themselves (`foldIn(stepsKey, t)`)             | exact replay of any episode; the same run with another seed                                                              |
| counterfactual handles           | `fork(t, { seed, config })`                             | "stop exploring from here", "what if the learning rate dropped now"                                                      |

Two consequences. A method never stores its own past: the past is the tape, and anything wanted at step `t` is
`stateAt(t)` or a derived path read there, so nothing has to be chosen in advance. And a method never decides what a
figure shows: it offers, and the subscription chooses.

## 3. Saying what to keep: paths, `Log` and `On`

The contract is the method's side; the subscription is the caller's; the session reconciles them. Paths give the
hierarchy (`training/loss`, `model/params/layers/0/weight`, `agent/q`, `env/frame`) and patterns select over it
(`model/params/**`, `**/loss`).

**The contract** names every path and what it is:

```ts
type Contract<S> = Readonly<Record<string, Source<S> | Event>>

type Source<S> =
  | { kind: 'scalar'; description: string; unit?: string; direction?: 'min' | 'max' }
  | { kind: 'tensor'; role: TensorRole; shape: readonly (number | string)[]; description: string }
  | { kind: 'message'; payload: ZodType; description: string }
  | { kind: 'frame'; description: string }
  /** Computed from the state when due or when probed; never emitted inline. `dear` ones default to rare schedules. */
  | {
      kind: 'derived'
      description: string
      cost?: 'cheap' | 'dear'
      compute: (state: S, args?: Json) => Json | Exposition
    }

/** A lifecycle point hooks may attach to, with what a hook receives. */
type Event = { kind: 'event'; description: string; context: ZodType }
```

Inline sources are emitted where they happen, by `ctx.log(path, value)`, which is a no-op unless the path is
subscribed; `ctx.wants(path)` lets code skip a computation nobody asked for. Derived sources are the probes of
section 1: the same function serves a schedule (`Log`) and a click (`session.probe`). Events are emitted by the
runners (`training/epoch/end`, `rl/episode/end`, `smo/converged`) and by method code where it has its own.

**The subscription** picks paths and says how:

```ts
type Subscription = 'off' | 'defaults' | 'all' | Readonly<Record<PathPattern, LogSpec | HookSpec>>

Log(options?: {
  every?: number | EventPath | [EventPath, number]   // steps; whenever an event fires; every nth firing
  detail?: 'summary' | 'full'                        // tensors: norms and histograms, or the values
  args?: Json                                        // for derived paths (a grid, a sample count)
  budget?: number                                    // bytes held, after which this path stops
  keep?: 'all' | 'last' | number                     // how much of the path the tape holds: everything, the last, the last n
}): LogSpec
On(hook: (ctx: HookContext) => void): HookSpec     // on an event path; HookContext has log, stop, run, the event's context
On(name: string, options?: Json): HookSpec         // a hook the method exports by name, so it can cross to a worker
```

`'off'` is the trivial mode: no metrics, tensors, messages or frames, no derived computations, no hooks; the method is
the raw algorithm, with only checkpoints kept so `stateAt`, `fork` and `probe` still work (`past: false` drops them too
and uses `nullTracer`). `'defaults'` is the method's own small choice (usually a loss). `'all'` is everything at summary detail,
for debugging. Scheduling by event is what makes `every: 'training/epoch/end'` and `every: ['rl/episode/end', 10]`
one mechanism rather than an epoch concept and an episode concept. `keep` bounds what the tape holds of a path rather
than when it is logged: `Log({ keep: 'last' })` on the parameters logs them every step and keeps the newest alone, so a
page that shows the current model pays for one; older events are forgotten as new ones arrive, checkpoints are never
among them (seeking is unaffected), and the byte budget counts what is held.

**Reconciliation**, once, when the session is made: `resolve(contract, subscription)` matches patterns to paths
(an exact path that does not exist is an error; a pattern matching nothing is a warning), compiles the matched inline
sources into a `TracePolicy` with per-path schedules and detail, schedules the derived sources, installs the hooks on
their events, and turns checkpoints on at an interval chosen from the byte budget whenever anything may be sought.
After that the session does nothing clever: the runner calls `due(path, t)`, `ctx.log` consults `wants`, events run
their hooks.

**What the page gets for free.** A path's kind and shape fix its default view: scalars become curves, grouped by
their prefix onto shared axes (`training/*` on one, `training/val/*` beside it); a tensor of shape `[rows, cols]`
becomes a grid player; a `frame` a scene player; messages a table; a derived path returning an exposition record its
record view. A page lists what it subscribed to and has a layout; hand-arranged views stay possible and become rare.

**Where hooks run.** `Log` specs are JSON and cross to a worker. A hook written as a closure runs only in a
main-thread session; to run in a worker it must be a hook the method exports by name (`On('earlyStopping',
{ patience: 5 })`). Early stopping, learning-rate schedules and best-checkpoint are exported by the trainer, so the
common cases cross.

**What this costs the engine.** `TracePolicy` gains per-path schedules and detail (today one schedule serves every
channel); `StepContext` gains `log` and `wants`; the runners emit their lifecycle events; `resolve` is new and small.
The channel schemas, policies, taps and callbacks that exist are the parts this is assembled from.

## 4. Four sketches

Not complete; enough to see whether the shape holds across the families.

### 4.1 An MLP classifier (torch-like)

```ts
export const mlpClassifier = defineMethod({
  name: 'neural/mlp-classifier',
  config: dict({
    hidden: int(1, 256, { default: 32 }),
    lr: real(1e-4, 1, { log: true, default: 1e-2 }),
    epochs: int(1, 500, { default: 50 }),
  }),
  make: (c, key, data) =>
    trainer({
      modules: { net: mlpNetwork([data.d, c.hidden, data.k], 'relu', key) },
      optimizers: { adam: adam(c.lr) },
      batches: minibatches(data, 32),
      trainingStep: ({ net }, batch, ctx) => {
        const logits = net.apply(batch.x, { taps: ctx.taps('model/activations') }) // taps log only if subscribed
        const loss = crossEntropy(logits, batch.y)
        ctx.log('training/loss', loss)
        ctx.step('adam', loss) // manual optimisation
      },
      validationStep: ({ net }, batch, ctx) => ctx.log('training/val/accuracy', accuracy(net.apply(batch.x), batch.y)),
    }),
  contract: {
    ...trainerContract, // training/loss, training/lr, training/epoch/end, training/batch/after-backward, model/params/**, model/grads/**, model/activations/**
    'model/decision-surface': derived({
      cost: 'dear',
      compute: (s, { grid }) => decisionSurface(forward(s.params, grid)),
    }),
    'model/confusion': derived({ compute: (s) => confusionCounts(predictWith(s.params, data.x), data.y) }),
  },
  defaults: { 'training/loss': Log(), 'training/val/**': Log({ every: 'training/epoch/end' }) },
})
```

Three uses of the one method:

```ts
// Answers only: no tape, no derived work, no hooks; the trainer at full speed.
const fitted = session(mlpClassifier, { config, seed: 0, logs: 'off' }).advance({ steps: Infinity })

// A teaching page: play training, scrub it, see the surface move, stop early.
session(mlpClassifier, {
  config,
  seed: 0,
  mode: 'live',
  logs: {
    'training/**': Log(),
    'model/params/**': Log({ every: 'training/epoch/end', detail: 'summary' }),
    'model/activations/h1': Log({ every: 10, detail: 'full' }),
    'model/decision-surface': Log({ every: 'training/epoch/end', args: { grid } }),
    'training/epoch/end': On('earlyStopping', { monitor: 'training/val/accuracy', patience: 5 }),
  },
})

// Later, from the same tape: the surface at epoch 7, never scheduled.
s.probe(stepOfEpoch(7), 'model/decision-surface', { grid })
```

### 4.2 SMO (sklearn-iterative)

The innards of SMO are which pair it chooses and why, and what that does to the dual. They are emitted where they
happen; no hook is needed.

```ts
const smoAlgorithm = (data, c): Algorithm<SmoState> => ({
  name: 'svm/smo', config: { C: c.C, kernel: c.kernel },
  init: (key) => initialDual(data, key),
  step: (s, ctx) => {
    const { i, j, violation } = selectWorkingSet(s)             // the decision
    ctx.log('smo/working-set', { i, j, violation })             // and its reason
    const next = updatePair(s, i, j)
    ctx.log('smo/dual-objective', next.dual)
    ctx.log('smo/alpha', next.alpha)                            // a no-op unless someone asked
    return next
  },
  done: (s) => (s.maxViolation < tol ? 'converged' : undefined),
})

contract: {
  'smo/dual-objective': scalar({ direction: 'max' }),
  'smo/working-set': message({ payload: z.object({ i: z.number(), j: z.number(), violation: z.number() }) }),
  'smo/alpha': tensor({ role: 'state', shape: ['n'] }),
  'smo/support-vectors': derived({ compute: (s) => indicesWhere(s.alpha, (a) => a > 0) }),
  'smo/decision-surface': derived({ cost: 'dear', compute: (s, { grid }) => decisionSurface(decisionFunction(s, grid)) }),
  'smo/converged': event({ context: z.object({ steps: z.number(), dual: z.number() }) }),
}

// The margin view: the dual climbing, the pairs chosen, the surface at log-spaced steps.
logs: { 'smo/dual-objective': Log(), 'smo/working-set': Log(), 'smo/decision-surface': Log({ every: 'log-spaced', args: { grid } }) }
```

`Estimator.fit` stays the surface for the scikit-learn-shaped call: `svc(c).fit(data)` is `session(svc, { logs:
'defaults' })` run to completion, and `fit(data, { logs })` passes a subscription through.

### 4.3 Q-learning on cliff walking (sequential decision)

```ts
contract: {
  ...rlContract,     // rl/step/reward, rl/episode/return, rl/episode/length, rl/episode/end (event), rl/eval/return, env/frame
  'agent/q': tensor({ role: 'state', shape: ['states', 'actions'] }),
  'agent/epsilon': scalar(),
  'agent/td-error': scalar(),
  'agent/policy': derived({ compute: (s) => greedyPolicyTable(s.agent) }),
  'agent/rollout': derived({ cost: 'dear', compute: (s, { n, seed }) => rolloutFrames(env, greedyPolicyOf(agent, s.agent), n, seed) }),
}

// Train 300 episodes on load, then let the person continue; keep what the page draws.
const s = session(qLearning, { config, seed: 0, mode: { eager: { until: ['rl/episode/end', 300] } }, logs: {
  'rl/episode/**': Log(),
  'agent/q': Log({ every: ['rl/episode/end', 10], detail: 'full' }),
  'env/frame': Log({ every: ['rl/episode/end', 25] }),
}})

// Click episode 40 on the return curve: that policy, acting now.
s.probe(stepOfEpisode(s, 40), 'agent/rollout', { n: 3, seed: 1 })
// "What if it stopped exploring here?"
s.fork(stepOfEpisode(s, 40), { config: { epsilon: 0 } }).advance({ steps: 5000 })
```

The agent logs its own paths from `learn` through the same `ctx.log`, so a DQN offers `agent/loss`,
`agent/target-updates` and `model/params/**` under the same prefixes with nothing else changed.

### 4.4 A\* on a grid (search; the trivial end)

```ts
contract: {
  'search/expand': message({ payload: z.object({ node: z.string(), g: z.number(), h: z.number() }) }),
  'search/frontier': scalar(), 'search/explored': scalar(),
  'world/frame': frame(),
  'search/solution': event({ context: z.object({ cost: z.number(), length: z.number() }) }),
}

aStar(problem).solve()                                                        // logs: 'off' by default: the path, nothing else
session(aStar, { config, logs: { 'search/expand': Log(), 'world/frame': Log({ every: 5 }) } })   // the board player
```

Across the four, the method author wrote `ctx.log` where things happen and a contract line per path; the differences
between families stayed in `make`.

## 5. Eager against live

The other knob a person must be able to set trivially: whether the work happens up front or when they press play.

|                          | **eager**                                                                       | **live**                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| when the procedure runs  | on creation, to its budget, in a worker while the page loads                    | when the person presses play, in slices per frame or per click                             |
| what the page gets first | the finished tape                                                               | the tape at step 0                                                                         |
| interaction              | seek, probe, fork: instant, because everything is recorded                      | play, pause, scrub (`stateAt`), probe the recorded prefix; results appear as they are made |
| suits                    | fits that finish in a second or two; anything you interrogate rather than watch | anything you want to watch learn; long runs; "press play and see"                          |

The same session and the same tape serve both: `mode` only says who calls `advance` and when, and determinism means
switching changes no number. Two refinements fall out:
- **Eager to a prefix, then live** (`mode: { eager: { until: ... } }`, as in 4.3) is "pre-train and interrogate" and
  "train live in the demo" as one setting, not two mechanisms.
- **Derived paths are always live.** Even in an eager session a probe is computed when asked; the click in 4.3 rolls
  out _then_, as a small session of its own. Nothing was decided when the session was made.

A stored tape (computed at build time, loaded instead of run) is a third source for an eager session and nothing more;
a later addition that changes no interface.

### 5.1 What is logged is not what is shown

A page declares a **layout**: a few panels, each naming what it draws (the training rows with a probe's overlay at the
scrubbed step; a probe; a few scalar paths as curves; a projection of the tape so far). The host draws those and
nothing else; the generic everything-on-the-tape view is a debugging toggle. The entry's subscription stays as rich as
it likes, so the tape holds more than the page shows: a later panel, or a later stage of a narrative, draws from the
same session (a Bayesian mixture trained once; its components dying off in one place, its component priors over
training in another). Live is the default mode. Layouts are data, like entries, so the same panel can be placed on
any page that holds a session.

The panel vocabulary and everything that runs a session in a page live in `mdml-render` (`react/session/`): the hook
that runs a method live, the player, the settings, the layout type and the panels. A page anywhere (the encyclopedia, a notebook,
someone else's app) composes them. The panels share one
**step cursor**: the viewed step is a single value, panels of a step (rows, a probe, a tensor, a trajectory) read it,
panels of the tape (curves, per-element series) move it when clicked, and a custom panel reads it through a context
with no wiring. Scrubbing is opt-in: only a chart whose x axis is the step (or a per-step position such as time)
asks to move the cursor.

A method need not run on data. A process, an equation, a sampler or a search is a `Method` of nothing but its
configuration, and a page can hold one directly in place of an estimator with its datasets, showing its knobs and
the session controls and nothing about data. Two things such a method must do for the
panels to work: declare `serialise` and `deserialise` on its algorithm (probes at a step seek from checkpoints, and
only serialisable states are checkpointed), and keep in full the tensor paths it wants drawn element by element
(`Log({ detail: 'full' })` in its `defaults`; `all` keeps summaries).

## 6. The standard

1. **A method is a declaration**: `config` space, `make`, `contract`, `defaults`, `projections`. No loop, hook or
   callback of its own; everything it says goes through `ctx.log` against its contract; no function in `config`.
2. **Everything iterative is an `Algorithm`** with `name`, JSON `config`, `serialise` and `deserialise`. A test
   round-trips a state of each.
3. **No private past.** Nothing keeps earlier states in its state or picks snapshot steps up front.
4. **`'off'` is free.** With no subscription a method costs what the raw algorithm costs; a test runs each both ways
   and compares.
5. **Every contract path is honest**: a test subscribes to `'all'` and checks each inline path is emitted at least
   once and each derived path computes on the final state.
6. **Eager and live are a session's business.** A method is correct in both and does not know which it is in.
7. **Constructables stand alone**: datasets, models, environments, problems and simulations exported separately, each
   with a `Space` of knobs and working defaults.
8. **Budgets are declared**: steps, and milliseconds in a worker, for the defaults.

## 7. What we need, in order

1. The contract and subscription types, `Log`, `On`, `resolve`; `ctx.log` and `ctx.wants`; per-path schedule and
   detail in `TracePolicy`; lifecycle events from the runners. Small, in `lib/trace`.
2. `Session` with `mode`, over the existing `runFor`, `seek`, `resume` and `exportRun`; `defineMethod`.
3. `serialise`/`deserialise` on the algorithms methods make and on agent state. Mechanical.
4. The sequential-decision family on it: `rlContract`, agent paths, `agent/rollout`, a curve a person can click. The
   proving case.
5. One trainer for the torch-like family with `trainerContract`, manual optimisation and named hooks; neural and
   generative methods become specs.
6. Projections for the step-like records that the generic views cannot cover, replacing side histories.
7. Pages read sessions instead of records (the hub did, until it was removed on 2026-10-28).

Not needed: more entries or pages, a new record system, render changes beyond the clickable curve and the
path-to-view defaults, filling the empty method areas before the standard exists.

## 8. Open choices

1. Where `Method`, `Session` and the contract types live: inside `lib/trace`, or a small `lib/session` beside it.
2. Typing of subscriptions: exact paths typed against the contract (`keyof Contract`), patterns as strings; or all
   strings checked at `resolve`. The first catches typos at compile time and costs a generic on every method.
3. Whether `Estimator.fit` returns its tape always or only when a subscription is passed.
4. The default checkpoint byte budget for sessions that may be sought.
