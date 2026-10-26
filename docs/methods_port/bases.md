# The bases: one per way of running, and what a particular method is left to write

The leverage in the port is that almost everything a method does today by hand (run the loop, keep a history, pick
snapshot steps, build records, add the decision surface, carry the training rows, handle restarts, respect
`explain: false`, nest under cross-validation) is the same across every method of one execution shape. A base does it
once. Below: the five bases, what each hand-holds, the kind add-ons, and sketches of particular methods before and
after.

## 1. The shape every base produces

Every base returns a `Method` (`lib/trace/session.ts`): `config`, `make`, `contract`, `defaults`, `hooks`,
`projections`. So every method, whatever its family, is driven the same way (`session`, `fit`, a test, a page) and
reads the same way. The base is a function from a _particular_ (a small declaration) to that method.

```ts
// What every base owns, and what a particular never writes:
//   - the loop (the engine's `Algorithm`, driven by sessions),
//   - the standard events of its family (`fit/start`, `iteration/end`, `epoch/end`, `episode/end`, `restart/end`),
//   - `serialise`/`deserialise` (the generic `stateToJson`, overridable),
//   - the standard probes of the family and the kind (a decision surface for any classifier, ...),
//   - the record plumbing (`withInputRows`, feature and class names, `explain: false`, child runs under CV),
//   - a `fit`/`solve`/`train` surface that is "a session run to completion with `defaults`".
```

## 2. One base per way of running, kinds as add-ons

| base        | for                                                                                 | the particular declares                                                                                   | the base adds                                                                                                                                                   |
| ----------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `oneShot`   | closed-form fits, tests, metrics                                                    | `fit(data, hyper) → params`, verbs, `hyperSpace`                                                          | nothing iterative; a `traceCall` wrapper when the steps of a direct method are worth showing; the kind's probes                                                 |
| `iterative` | coordinate descent, IRLS, k-means, EM, SMO, t-SNE, isolation-forest growth          | `algorithm(data, hyper, key) → Algorithm<S>`, `params(state) → P`, verbs, a contract of the state's paths | the loop, `iteration/end`, restarts as child runs, convergence as status, serialise, the kind's probes                                                          |
| `trainer`   | networks, VAEs, GANs, flows, diffusion, contrastive                                 | modules, optimisers, batches, `trainingStep`, optional `validationStep`, probes                           | the minibatch loop, epochs, `epoch/end` and batch events, parameter and gradient paths, taps, callbacks as named hooks, serialise of params and optimiser state |
| `agent`     | bandits, tabular, deep RL, planning by sweeps                                       | an environment, an agent, probes                                                                          | `trainingAlgorithm`, `episode/end`, reward and agent-statistic paths, frames, `policyAt`, rollout and snapshot probes, serialise                                |
| `procedure` | A\*, CSP, MCTS, boids, SEIR, automata, path tracing, epicycles                      | the `Algorithm`, a contract of aliases and a frame                                                        | the loop, serialise where the state allows, the frame on a schedule                                                                                             |
| `sweep`     | regularisation paths, CV curves, learning curves, restarts, hyperparameter searches | a grid or sampler, the inner method                                                                       | one child session per point, `point/end`, the record as a projection over the children                                                                          |

**Why not one base per estimator kind.** The kind (classifier, regressor, transformer, clusterer, density, outlier,
other) decides three things only: the verbs a fitted value offers, the standard probes (a classifier gets
`model/decision-surface` and `model/probabilities` over a grid; a clusterer `model/assignments` and `model/centres`;
a transformer `model/components` and `model/embedding`; a density `model/density` over a grid; an outlier detector
`model/scores`), and how its records carry the training rows. None of that is about _how it runs_. So the kind is an
add-on the base applies:

```ts
const kinds = {
  classifier: { verbs: classifierVerbs, probes: { 'model/decision-surface': surfaceProbe, 'model/probabilities': probaProbe }, rows: 'labelled' },
  clusterer:  { verbs: clustererVerbs, probes: { 'model/assignments': ..., 'model/centres': ... }, rows: 'indexed' },
  // ...
}
```

A particular says `kind: 'classifier'` and gets all of it. Adding a standard probe to every classifier is one line in
`kinds.classifier.probes`.

## 3. The iterative base, in full

The one that absorbs the most. A particular:

```ts
interface IterativeParticular<H, S, P> {
  readonly name: string
  readonly kind: EstimatorKind
  readonly hyperSpace: (h: H) => Space
  readonly defaults: H
  /** The engine's algorithm over the data: untouched solvers from `lib/`. */
  algorithm(data: LearnData, hyper: H, key: Key): Algorithm<S>
  /** The fitted parameters from a state (what `revive` rebuilds verbs from). */
  params(state: S, data: LearnData, hyper: H): P
  readonly verbs: (params: P, hyper: H) => Verbs & FittedFacts
  /** What the state can tell: aliases of the algorithm's native channels, and derived paths. */
  contract?(data: LearnData, hyper: H): Contract<S>
  /** Restarts: how many, and which final state wins (default none). */
  readonly restarts?: { count: (h: H) => number; better: (a: S, b: S) => boolean }
  /** Structural records from the final state (a tree, a coefficient table); step-like ones come from the tape. */
  records?(state: S, data: LearnData, hyper: H): Exposition[]
}
```

The base produces the `Method`, and from it the scikit-learn surface:

```ts
export function iterative<H, S, P>(p: IterativeParticular<H, S, P>): ExplainedEstimator<H, ...> & { method: Method<...> }
// fit(data, options) =
//   session(method, { config: hyper, seed, logs: options.logs ?? (options.explain === false ? 'off' : 'defaults') })
//     .advance()   // to completion (restarts are child sessions; the winner's state is the result)
//   then params(state), verbs(params), records = [...p.records(state), ...projections(run)] with the rows attached.
```

What the base's contract contains before the particular adds anything: `fit/start`, `iteration/end` (context: the
iteration, the metrics the algorithm observed), `fit/converged`, `restart/end`; the algorithm's own `schema` channels
exposed under `solver/<name>` automatically (so an untouched solver is loggable by path with no aliases written);
`model/params` as a derived path (`params(state)` as JSON); the kind's probes.

## 4. Sketches: before and after

### 4.1 k-means (`methods/cluster/kmeans.ts`, about 60 lines of fit today)

Today the fit runs `lloydAlgorithm` for each restart through `runAlgorithm`, keeps the best, and builds
`lloydRecord` from a `history` the state carries, plus `centroidSurface`. After:

```ts
export const kmeans = iterative<KMeansHyper, LloydState, CentroidParams>({
  name: 'kmeans',
  kind: 'clusterer',
  hyperSpace: kmeansSpace,
  defaults: { nClusters: 3, init: 'k-means++', nInit: 1, maxIter: 300, tol: 1e-4 },
  algorithm: (data, h, key) => lloydAlgorithm(data.x, initialCentres(data.x, h.nClusters, h.init, key), { tol: h.tol }),
  restarts: { count: (h) => (typeof h.init === 'string' ? h.nInit : 1), better: (a, b) => a.inertia < b.inertia },
  params: (s) => ({ centres: s.centres, labels: s.labels, inertia: s.inertia }),
  verbs: centroidVerbs,
  contract: () => ({
    'lloyd/inertia': scalar({ from: 'inertia', direction: 'lower' }),
    'lloyd/moved': scalar({ from: 'moved', description: 'how far the centres moved' }),
    'lloyd/centres': tensor({ from: 'centres', role: 'state', shape: ['k', 'd'] }),
    'lloyd/assignments': tensor({ from: 'labels', role: 'state', shape: ['n'] }),
  }),
})
```

Gone: the restart loop, `history` inside `LloydState`, `lloydRecord`, the surface call (the clusterer kind adds
`model/assignments` and `model/centres`; `clusteringSteps` becomes `clusteringStepsFrom(run)` over the
`lloyd/centres` and `lloyd/assignments` tensors at whatever stride the view asks). About 15 lines of particular.

### 4.2 The lasso and its path (`methods/linear/regression.ts`)

Two things are tangled today: the coordinate-descent _solve_ (one run) and the _path_ over thirty penalties (thirty
solves). After, they are two methods, one a sweep over the other:

```ts
export const lasso = iterative<LassoHyper, CoordinateDescentState, LinearModelParams>({
  name: 'lasso',
  kind: 'regressor',
  hyperSpace: enetSpace,
  defaults: { alpha: 1, fitIntercept: true, maxIter: 1000, tol: 1e-4 },
  algorithm: (data, h) => coordinateDescentAlgorithm(design(data, h), { alpha: h.alpha, l1Ratio: 1, tol: h.tol }),
  params: (s, data, h) => linearParams(s, data, h),
  verbs: linearVerbs,
  contract: () => ({
    'cd/objective': scalar({ from: 'objective', direction: 'lower' }),
    'cd/duality-gap': scalar({ from: 'gap' }),
    'cd/coefficients': tensor({ from: 'coef', role: 'param', shape: ['d'] }),
    'cd/active-set': derived({ compute: (s) => Array.from(s.coef, (c, j) => (c !== 0 ? j : -1)).filter((j) => j >= 0) }),
  }),
  records: (s, data, h) => [coefficientTable(...)],
})

export const lassoPath = sweep({
  name: 'lasso-path',
  over: (h) => alphaGrid(h),                       // the points
  inner: (alpha) => lasso.with({ alpha }),          // one child session per point
  warmStart: (previous) => previous.state,          // optional: start the next point from the last state
  projections: { regularisationPath: (children) => regularisationPathFrom(children) },
})
```

The `regularisationPath` record is a projection over the children's final `cd/coefficients`; the duality gap per
penalty is a projection over their last `cd/duality-gap`. `lassoCV` is `sweep` over `lassoPath` with folds as
children of children, which `runTree` already reads.

### 4.3 A GAN on the trainer base

```ts
export const gan = trainer<GanHyper, Batch>({
  name: 'gan',
  modules: (h, key, data) => ({
    generator: mlp([h.latent, 64, data.d], 'relu', key),
    discriminator: mlp([data.d, 64, 1], 'relu', foldIn(key, 1)),
  }),
  optimizers: (h) => ({ generator: adam({ learningRate: h.lr }), discriminator: adam({ learningRate: h.lr }) }),
  batches: (data, h) => minibatches(data, h.batchSize),
  trainingStep: ({ generator, discriminator }, batch, ctx) => {
    const z = normal(ctx.key, [batch.n, h.latent])
    const dLoss = discriminatorLoss(discriminator, generator, batch.x, z)
    ctx.log('training/discriminator/loss', dLoss)
    ctx.step('discriminator', dLoss)
    const gLoss = generatorLoss(discriminator, generator, z)
    ctx.log('training/generator/loss', gLoss)
    ctx.step('generator', gLoss)
  },
  probes: {
    'model/samples': derived({ cost: 'dear', compute: (s, { n, seed }) => sample(s.modules.generator, n, seed) }),
  },
})
```

The trainer base gives `training/epoch/end`, `training/batch/after-backward`, `model/params/**`, `model/grads/**`,
`model/activations/**` by tap, the learning-rate path, early stopping and best-checkpoint as named hooks, and serialise
of every module's parameters and optimiser state. The GAN writes its two-optimiser step and one probe. Today's
`trainLoop` in `methods/generative/_shared.ts` (about 90 lines plus frames and progress) goes.

### 4.4 DQN on the agent base

```ts
export const dqnCartPole = agent<DqnHyper>({
  name: 'dqn/cartpole',
  env: () => make(cartPole),
  agent: (env, h, key) => dqn(env, { ...h, key }),
  probes: {
    'agent/q-on-grid': derived({ cost: 'dear', compute: (s, { grid }) => qValues(s.agent, grid) }),
  },
})
```

The agent base gives `rl/episode/end`, `rl/step/reward`, `rl/eval/return`, every `agent.metrics` channel under
`agent/`, `env/frame`, `agent/policy`, `agent/rollout` (greedy episodes with frames at any step), and serialise via
`stateToJson` (a DQN's state is parameters and a replay buffer of typed arrays; it serialises as is). Today's
`playground.ts` (850 lines) is these pairings as data plus the generic base.

### 4.5 Boids on the procedure base

```ts
export const boids = procedure<BoidsOptions, BoidsState>({
  name: 'boids',
  algorithm: (c, key) => boidsAlgorithm(c),
  contract: {
    'flock/polarisation': scalar({ from: 'polarisation' }),
    'world/frame': derived({ compute: (s) => flockScene(s) }),
  },
})
```

Nothing else: the base's loop, serialise (typed arrays) and the frame on a schedule do the rest.

## 5. What a particular is, by family, in lines

| family              | today (typical)                                                   | after | what remains in the particular                                    |
| ------------------- | ----------------------------------------------------------------- | ----- | ----------------------------------------------------------------- |
| iterative estimator | 40–120 lines of fit, a private history, a record builder          | 10–25 | algorithm, params, verbs, a contract of aliases and derived paths |
| trainer-based       | a loop per family (`_train.ts`, `trainLoop`), snapshots, progress | 15–40 | modules, optimisers, the step, probes                             |
| agent               | `playground.ts` pairings with snapshots and recorders             | 5–15  | env, agent, probes                                                |
| procedure           | already an algorithm, plus a frame builder                        | 5–10  | aliases and a frame                                               |
| sweep               | a loop over fits building one record                              | 5–15  | the grid, the inner method, the projection                        |
