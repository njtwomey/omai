# Leverage and complexity: what is there to build on, and where the port is hard

An honest inventory, so the estimate in the conversation is checkable.

## 1. What exists and is kept

| piece                                                                                                 | where                                            | role in the port                                                                    |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `Algorithm`, `runAlgorithm`, `runFor`, `iterate`                                                      | `lib/trace/algorithm.ts`                         | the loop every base drives; untouched                                               |
| `TracePolicy`, channels, `RunHandle.wants`                                                            | `lib/trace/policy.ts`, `tracer.ts`               | what `resolve` compiles a subscription into; the per-path schedule sits above it    |
| checkpoints, `seek`, `resume`                                                                         | `lib/trace/checkpoint.ts`                        | `stateAt`, `fork`, `probe`                                                          |
| contract, `Log`, `On`, `resolve`, log context, gating tracer                                          | `lib/trace/contract.ts`, `logging.ts`            | the mechanism (proof of concept, to harden)                                         |
| `Method`, `session`, eager/live                                                                       | `lib/trace/session.ts`                           | the unit every base produces                                                        |
| `stateToJson`                                                                                         | `lib/trace/serialise.ts`                         | default `serialise` for states of tensors, typed arrays and plain trees             |
| `defineEstimator`, `defineExplainedEstimator`, `withInputRows`, `decisionSurfaceOf`, `featureNamesOf` | `lib/learn/estimator.ts`, `explained.ts`         | the estimator surface and the record plumbing the iterative and one-shot bases wrap |
| `fitAlgorithm`, `Callback`, `HookContext`, `lossAndGrads`, `applyGradients`                           | `lib/nn/training`                                | the trainer base's loop and its named hooks                                         |
| `trainingAlgorithm`, `greedyPolicyOf`, `runEpisode`, `evaluatePolicy`                                 | `lib/rl`                                         | the agent base's loop (already emits `rl/episode/end` and serialises)               |
| `observer`, `PATH_SCHEMA`, `SearchState`                                                              | `lib/search`                                     | the procedure base's aliases for searches                                           |
| `subRun`, `runTree`, `childPolicy`                                                                    | `lib/trace/nesting.ts`, `query.ts`               | the sweep base's child sessions and the projections over them                       |
| the 36 `*Algorithm` factories outside RL, 7 inside                                                    | across `methods/` and `lib/`                     | the `algorithm(...)` of each iterative particular; untouched                        |
| `datasetGenerators`, `EnvDefinition`, `World`, `Module`                                               | `lib/datasets`, `lib/rl`, `lib/worlds`, `lib/nn` | the constructables; untouched                                                       |
| render's record views, `metricCurves`, `episodeCurves`, `Player`, `SceneView`                         | `packages/render`                                | the generic viewer composes them; no change needed for the port                     |

## 2. What goes

| piece                                                                                            | lines (about)      | replaced by                                                     |
| ------------------------------------------------------------------------------------------------ | ------------------ | --------------------------------------------------------------- |
| `collecting` and the `history` fields inside states (26 files use the first, several the second) | 300 across methods | the tape and `stateAt`                                          |
| `methods/neural/_train.ts` (stopping, adaptive rate, snapshots, `onEpoch`)                       | 260                | the trainer base, with stopping and rate changes as named hooks |
| `methods/generative/_shared.ts` `trainLoop`, frames and `GenerativeProgress`                     | 150                | the trainer base                                                |
| `methods/rl/playground.ts` session, recorder and snapshots                                       | 500 of 850         | the agent base; the pairings stay as data                       |
| `methods/svm/_dual.ts` snapshots and history                                                     | 60                 | aliases on the untouched `smoAlgorithm`                         |
| `geometricSteps`, `onGeometricSchedule`, `stagedSteps`                                           | 60                 | `Log({ every: 'logSpaced' })` and projections at any stride     |
| `MlpHyper.onProgress`, `onEpisode`, `onTrial`, `HubRunInput.progress`                            | scattered          | the tape, read by whoever hosts the session                     |
| `TrainState.checkpoints` in `lib/rl/training.ts` and `replayEpisode`'s use of it                 | 60                 | tape checkpoints and `fork`                                     |

## 3. Where it is complicated

1. **Sweeps.** Regularisation paths, CV curves, learning curves and k-means restarts are many fits, not one. Today
   each is a hand loop building one record. The sweep base runs them as child sessions and projects the record over
   the children. Warm starts (the lasso path starts each penalty from the last solution) are a `start` state handed
   from child to child, which sessions support. Cost: one base, medium; the particulars become small. Risk: deep
   nesting (CV over a path over a solver) makes tapes with thousands of child runs; the `inner` policy of child runs
   exists to keep them quiet.
2. **States that keep their past.** `LloydState.history`, SVM dual snapshots, mini-batch histories: removing them
   changes the state types and the record builders at once. Mechanical, but the two must move together per method.
3. **States that do not serialise.** Search nodes with parent links; a few RL buffers that are classes. Either a
   custom `serialise` or no seeking. Decide per case; default to none for search.
4. **Native channels with ambiguous names.** Several algorithms emit `loss`, `objective`, `step`; aliases (`from`)
   rename them per method, so two methods sharing a solver can name the same channel differently. Fine, but the
   base should alias the solver's whole `schema` under `solver/<name>` automatically so a particular only aliases
   what it wants prettier.
5. **Records that index training rows.** `withInputRows` attaches the rows to clusterings, margins, neighbours and
   surfaces; a projection from the tape must do the same. The base passes the data to projections; no new idea.
6. **`explain: false` and cross-validation folds.** Folds fit with records off and nested under the outer run. In the
   port: `logs: 'off'` and `parent`. The iterative base maps `FitOptions` onto session options once.
7. **The worker boundary.** Closures in `derived` and in `make` are fine (they live where the session lives). Only
   subscriptions cross, and `On` hooks must be named to cross. The proof of concept runs sessions on the main thread;
   a worker host is a later, separate piece.
8. **Pattern paths and taps.** `model/activations/*` is only known as the model runs; the scheduler matches patterns on
   every `log` call. Cheap at toy sizes, worth a cache before a transformer logs a hundred taps a step.
9. **Per-path detail.** The proof of concept takes one `detail` for the whole policy; `Log({ detail: 'full' })` on one
   path turns it on for all. Needs the gating tracer to summarise per path. Small.
10. **Typing of contracts.** Plain strings now; `keyof` typing later if typos bite.

## 4. The order that keeps the codebase coherent at every stop

1. Harden the mechanism (per-path detail, caching, tests) and add the iterative base over `defineExplainedEstimator`.
2. Port one area end to end (`cluster`): proves the base, removes `history` and `collecting` there, gives the first
   `clusteringStepsFrom` projection.
3. The sweep base, then `linear` (paths, CV).
4. The trainer base, then `neural` and `generative`.
5. The agent base, then `rl/playground` and the pairings.
6. The procedure base, then `gofai`, `complex-systems`, `evolutionary`, `graphics`, `signal`.
7. One-shot methods: nothing, unless a kind add-on wants a probe they lack.

After any step the old and new coexist: sessions wrap algorithms, estimators keep their `fit`, records keep their
views. Stop at any point and nothing is half-broken.
