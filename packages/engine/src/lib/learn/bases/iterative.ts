/**
 * The iterative base: an estimator whose fit is an `Algorithm` of the library run as a session. A particular declares
 * the algorithm over the data, how its state becomes parameters, the verbs those parameters give, and what the state
 * can tell beyond the solver's own channels; the base owns the loop, the `fit/iteration/end` event, restarts, the
 * solver's channels as `solver/...` paths, `model/params` and the kind's standard probes, serialisation, the
 * scikit-learn `fit` surface (a session run to completion with the method's defaults), the records' rows, and a
 * `session` to drive the same fit live, with `fittedAt(t)` for the fitted value at any reached step.
 */

import type { Json } from '../../base/index.ts'
import { resolveKey } from '../../context/index.ts'
import type { Exposition } from '../../exposition/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { type DictSpace, dict, type SpaceValue } from '../../space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  type EventSource,
  event,
  frame,
  type Method,
  type Mode,
  message,
  type Run,
  type Session,
  type Source,
  type StepContext,
  type Subscription,
  scalar,
  session,
  tensor,
} from '../../trace/index.ts'
import type { Estimator, EstimatorKind, FitOptions, Fitted, FittedFacts, LearnData, Verbs } from '../estimator.ts'
import { withInputRows } from '../explained.ts'
import { extendSession } from './_session.ts'
import { type KindPath, kindProbes } from './kinds.ts'

/**
 * The paths of a session of the iterative base: the particular's own (`K`), the base's (`fit/iteration/end`,
 * `model/params`, the kind's probes) and the solver's channels (`solver/<name>`, named by the algorithm's schema at run
 * time, hence a template literal). With `K` left at `string` every path is allowed, as before.
 *
 * @example
 * ```ts
 * import type { IterativePath } from 'mdml-engine/learn'
 * type P = IterativePath<'tree/split'>   // 'tree/split' | 'fit/iteration/end' | 'model/params' | ... | `solver/${string}`
 * ```
 */
export type IterativePath<K extends string = string> =
  | K
  | 'fit/iteration/end'
  | 'model/params'
  | KindPath
  | `solver/${string}`

/**
 * What a particular method declares over the iterative base.
 *
 * The last parameter is the type of the paths the particular adds to the base's (`string` by default: unchecked). To
 * type them, in the file of the particular (the same mechanical steps as {@link defineContract}):
 *
 * 1. `export type FooPath = 'foo/a' | 'foo/b'` (template literals for computed parts).
 * 2. `contract(data, h): Contract<S, FooPath>` returning `defineContract({ ... })`.
 * 3. `algorithm(...): Algorithm<S, FooPath>`, so its `ctx.log`, `ctx.emit` and `ctx.wants` are checked; a library
 *    algorithm that is itself typed (`Algorithm<S, SolverPath>`) is returned as it is when its paths are in `FooPath`.
 * 4. Type `logs` and give the estimator its type: `IterativeEstimator<H, S, P, V, FooPath>`. The estimator's `method`
 *    and `session` then carry `IterativePath<FooPath>`: `logs:` keys and `probe` paths of a session are checked.
 *
 * The solver's channels (`solver/<name>` renames of the algorithm's `observe`) and the base's paths need no entry in
 * `FooPath`; only what the particular's algorithm logs by path and what its `contract` adds.
 */
export interface IterativeParticular<
  H extends object,
  S,
  P extends Json,
  V extends Verbs & FittedFacts,
  K extends string = string,
> {
  readonly name: string
  readonly kind: EstimatorKind
  /** The hyperparameters, with their defaults. */
  readonly defaults: H
  /** The knobs a demo or a search varies (the rest of `H` keeps its values). */
  readonly hyperSpace?: (hyper: H) => DictSpace
  /** The library's algorithm over the data: untouched solvers from `lib/`. */
  algorithm(data: LearnData, hyper: H, key: Key): Algorithm<S, K>
  /** The fitted parameters of a state (plain JSON, what `revive` rebuilds verbs from). */
  params(state: S, data: LearnData, hyper: H): P
  /** The verbs and facts of fitted parameters. */
  verbs(params: P, hyper: H): V
  /** What the state can tell beyond the solver's channels: aliases, derived paths, events. */
  contract?(data: LearnData, hyper: H): Contract<S, K>
  /** The step budget of a fit, which may depend on the data (default 10,000; the algorithm's `done` usually ends it sooner). */
  readonly maxSteps?: (hyper: H, data: LearnData) => number
  /** Restarts: how many, and which final state wins. */
  readonly restarts?: { readonly count: (hyper: H) => number; readonly better: (a: S, b: S) => boolean }
  /** Structural records from the final state (a tree, a coefficient table). */
  records?(state: S, data: LearnData, hyper: H): readonly Exposition[]
  /** Step-like records from the tape; `undefined` when the tape has nothing to project (a fit leaves it out). */
  readonly projections?: Readonly<Record<string, (run: Run, data: LearnData, hyper: H) => Exposition | undefined>>
  /** What `fit` keeps by default (default nothing: the raw algorithm). */
  readonly logs?: Subscription<NoInfer<IterativePath<K>>>
}

/** A fitted value of the iterative base: the fitted estimator, its records and the tape of its fit. */
export type IterativeFitted<P extends Json, V> = Fitted<P> &
  V & { readonly exposition: readonly Exposition[]; readonly run: Run }

/**
 * Options of {@link IterativeEstimator.session}: the fit options, and when to compute. The parameter is the type of the
 * session's paths (`logs` keys are checked against it; `string`: any).
 */
export interface IterativeSessionOptions<K extends string = string> extends Omit<FitOptions, 'logs'> {
  /** What to keep (default the estimator's `logs`). */
  readonly logs?: Subscription<NoInfer<K>>
  readonly mode?: Mode
  readonly steps?: number
  /** Keep checkpoints under `logs: 'off'` so the session can seek and probe (default `true`; a fit sets `false`). */
  readonly past?: boolean
}

/**
 * A session of a fit on the iterative base: the session's verbs, plus the fitted value at any reached step. Its forks
 * are iterative sessions too. The last parameter is the type of the session's paths (see {@link IterativePath}).
 */
export interface IterativeSession<S, P extends Json, V extends Verbs & FittedFacts, K extends string = string>
  extends Session<S, Config, K> {
  /**
   * The fitted value at a step (the verbs of the kind, built from the parameters of `stateAt(t)`), so "the model at
   * step 40 on new data" is `fittedAt(40).predict(x)`. Any reached step, through the checkpoints; at the head, the
   * value `fit` would return.
   */
  fittedAt(t: number): Fitted<P> & V
  fork(t: number, options?: { readonly seed?: number; readonly config?: Partial<Config> }): IterativeSession<S, P, V, K>
}

/**
 * An estimator on the iterative base: scikit-learn's surface, plus its method and a session of its fit. The last
 * parameter is the type of the paths the particular adds (see {@link IterativeParticular}); the method and the sessions
 * carry it together with the base's, `IterativePath<K>`.
 */
export interface IterativeEstimator<
  H extends object,
  S,
  P extends Json,
  V extends Verbs & FittedFacts,
  K extends string = string,
> extends Estimator<H, IterativeFitted<P, V>> {
  /** The method of this estimator on a dataset (to drive as a session, or to read its contract). */
  method(data: LearnData): Method<S, Config, IterativePath<K>>
  /** A session of the fit on a dataset: the tape to advance, seek, probe and fork, and the fitted value at any step. */
  session(
    data: LearnData,
    options?: IterativeSessionOptions<IterativePath<K>>,
  ): IterativeSession<S, P, V, IterativePath<K>>
  with(changes: Partial<H>): IterativeEstimator<H, S, P, V, K>
}

/**
 * A session with the fitted value at any reached step: the session's verbs delegated one by one (`t`, `state` and
 * `status` are getters, so no spread), and `fittedAt(t)` the kind's verbs built from the state there. The bases that
 * run a library algorithm as a fit (iterative, trainer) share it; forks are fitted sessions too.
 *
 * @param s - The session.
 * @param fitted - The fitted value of a state under a configuration.
 * @returns The session with `fittedAt`.
 * @example
 * ```ts
 * import { fittedSession } from 'mdml-engine/learn'
 * import { defineMethod, session } from 'mdml-engine/trace'
 * import { dict } from 'mdml-engine/space'
 * const m = defineMethod({
 *   name: 'count', config: dict({}),
 *   make: () => ({ name: 'count', config: {}, init: () => 0, step: (s) => s + 1, serialise: (s) => s, deserialise: (j) => j }),
 *   contract: {},
 * })
 * const s = fittedSession(session(m, { steps: 5, mode: 'eager' }), (n) => ({ estimator: 'count', kind: 'other', params: { n } }))
 * s.fittedAt(3).params // => { n: 3 }
 * ```
 */
export function fittedSession<S, P extends Json, V extends Verbs & FittedFacts, K extends string = string>(
  s: Session<S, Config, K>,
  fitted: (state: S, config: Config) => Fitted<P> & V,
): IterativeSession<S, P, V, K> {
  return extendSession(s, (inner) => ({
    fittedAt: (t: number): Fitted<P> & V => fitted(inner.stateAt(t), inner.config),
  })) as IterativeSession<S, P, V, K>
}

const EMPTY_RUN: Run = { schema: 'mdml.run/1', events: [] }

/**
 * An estimator over the iterative base.
 *
 * @param p - The particular.
 * @param hyper - Its hyperparameters (default the particular's defaults).
 * @returns The estimator, with `method` and `session` beside `fit`.
 * @concept log-contract
 * @example
 * ```ts
 * import { iterative } from 'mdml-engine/learn'
 * import { asTensor, full } from 'mdml-engine/tensor'
 * // The mean of the targets, found by averaging one more row per step.
 * const mean = iterative({
 *   name: 'running-mean', kind: 'regressor', defaults: {},
 *   algorithm: (data) => {
 *     const y = Array.from(data.y)
 *     return { name: 'mean', config: {}, init: () => ({ i: 0, m: 0 }),
 *       step: (s) => ({ i: s.i + 1, m: s.m + (y[s.i] - s.m) / (s.i + 1) }),
 *       done: (s) => (s.i >= y.length ? 'converged' : undefined),
 *       observe: (s) => ({ metrics: { mean: s.m } }) }
 *   },
 *   params: (s) => ({ mean: s.m }),
 *   verbs: (p) => ({ predict: (x) => full([asTensor(x).shape[0]], p.mean) }),
 * })
 * mean.fit({ x: [[0], [1], [2]], y: [1, 2, 6] }).params // => { mean: 3 }
 * ```
 */
export function iterative<
  H extends object,
  S,
  P extends Json,
  V extends Verbs & FittedFacts,
  K extends string = string,
>(p: IterativeParticular<H, S, P, V, K>, hyper: H = p.defaults): IterativeEstimator<H, S, P, V, K> {
  const space: DictSpace = p.hyperSpace?.(hyper) ?? dict({})
  const knobs = Object.keys(space.spaces)
  const hyperOf = (c: Config): H => ({ ...hyper, ...c }) as H
  const configOf = (h: H): Config =>
    Object.fromEntries(knobs.map((k) => [k, (h as Record<string, unknown>)[k] as SpaceValue])) as Config
  const build = (params: P, h: H = hyper): Fitted<P> & V => ({
    ...p.verbs(params, h),
    estimator: p.name,
    kind: p.kind,
    params,
  })
  // The algorithm is built once per (data, hyper, key): `make` and the contract both need it.
  const cache = new WeakMap<LearnData, Map<string, Algorithm<S, K>>>()
  const algorithmOf = (data: LearnData, h: H, key: Key): Algorithm<S, K> => {
    const id = `${JSON.stringify(h)}|${key.words.join(',')}`
    let forData = cache.get(data)
    if (forData === undefined) {
      forData = new Map()
      cache.set(data, forData)
    }
    let alg = forData.get(id)
    if (alg === undefined) {
      alg = p.algorithm(data, h, key)
      forData.set(id, alg)
    }
    return alg
  }
  const contractOf = (data: LearnData, h: H, key: Key): Contract<S, IterativePath<K>> => {
    const alg = algorithmOf(data, h, key)
    const out: Record<string, Source<S> | EventSource> = {}
    const schema = alg.schema
    if (schema !== undefined) {
      for (const [name, m] of Object.entries(schema.metrics)) {
        out[`solver/${name}`] = scalar({
          from: name,
          description: m.description,
          ...(m.unit === undefined ? {} : { unit: m.unit }),
          ...(m.direction === undefined ? {} : { direction: m.direction }),
        })
      }
      for (const [name, t] of Object.entries(schema.tensors)) {
        out[`solver/${name}`] = tensor({
          from: name,
          role: t.role,
          description: t.description,
          ...(t.shape === undefined ? {} : { shape: t.shape }),
        })
      }
      for (const [name, m] of Object.entries(schema.messages)) {
        out[`solver/${name}`] = message({
          from: name,
          description: m.description,
          ...(m.payload === undefined ? {} : { payload: m.payload }),
        })
      }
      if (schema.frames) out['solver/frame'] = frame({ description: 'the solver’s own picture of its state' })
    }
    out['fit/iteration/end'] = event({ context: { t: 'the iteration just finished' } })
    out['model/params'] = derived<S>({
      description: 'the fitted parameters at this step',
      compute: (s) => p.params(s, data, h) as Json,
    })
    Object.assign(out, kindProbes<S>(p.kind, { data, fitted: (s) => build(p.params(s, data, h), h) }))
    Object.assign(out, p.contract?.(data, h) ?? {})
    // The base's paths are computed from the algorithm's schema, so `out` is a plain record; its keys are `IterativePath<K>`.
    return out as Contract<S, IterativePath<K>>
  }
  const method = (data: LearnData): Method<S, Config, IterativePath<K>> =>
    defineMethod<S, Config, IterativePath<K>>({
      name: p.name,
      config: space,
      make: (c, key) => {
        const alg = algorithmOf(data, hyperOf(c), key)
        return {
          ...alg,
          step: (s, ctx) => {
            const next = alg.step(s, ctx)
            ;(ctx as StepContext<'fit/iteration/end'>).emit('fit/iteration/end', { t: ctx.t + 1 })
            return next
          },
        }
      },
      contract: (c, key) => contractOf(data, hyperOf(c), key),
      ...(p.logs === undefined ? {} : { defaults: p.logs }),
      // The same projections a fit returns, available on a live session's tape so far.
      ...(p.projections === undefined
        ? {}
        : {
            projections: Object.fromEntries(
              Object.entries(p.projections).map(([name, f]) => [name, (run: Run) => f(run, data, hyper)]),
            ),
          }),
    })
  // The fitted value at a step is the kind's verbs over the parameters there, under the session's own configuration.
  const wrap = (
    s: Session<S, Config, IterativePath<K>>,
    data: LearnData,
  ): IterativeSession<S, P, V, IterativePath<K>> =>
    fittedSession(s, (state, config) => {
      const h = hyperOf(config)
      return build(p.params(state, data, h), h)
    })
  const sessionOf = (
    data: LearnData,
    options: IterativeSessionOptions<IterativePath<K>>,
    key: Key,
  ): IterativeSession<S, P, V, IterativePath<K>> => {
    const steps = p.maxSteps?.(hyper, data) ?? 10_000
    return wrap(
      session(method(data), {
        key,
        config: configOf(hyper),
        logs: options.logs ?? (options.explain === false ? 'off' : 'defaults'),
        mode: options.mode ?? 'live',
        steps: options.steps ?? steps,
        ...(options.past === undefined ? {} : { past: options.past }),
        checkpointEvery: Math.max(1, Math.floor(steps / 100)),
      }),
      data,
    )
  }
  const fit = (data: LearnData, options: FitOptions = {}): IterativeFitted<P, V> => {
    const key = resolveKey(options, p.name)
    const n = Math.max(1, p.restarts?.count(hyper) ?? 1)
    let best: IterativeSession<S, P, V, IterativePath<K>> | undefined
    for (let r = 0; r < n; r++) {
      const s = sessionOf(
        data,
        { ...options, mode: 'eager', past: false } as IterativeSessionOptions<IterativePath<K>>,
        n === 1 ? key : foldIn(key, r),
      )
      if (best === undefined || p.restarts?.better(s.state, best.state) === true) best = s
    }
    const chosen = best as IterativeSession<S, P, V, IterativePath<K>>
    const state = chosen.state
    const run = chosen.run()
    const fitted = chosen.fittedAt(chosen.t)
    const records: Exposition[] = [
      ...(p.records?.(state, data, hyper) ?? []),
      ...Object.values(p.projections ?? {}).flatMap((f) => f(run, data, hyper) ?? []),
    ]
    const exposition = options.explain === false ? records : withInputRows(records, data, fitted.classes, fitted.kind)
    return { ...fitted, exposition, run }
  }
  return {
    name: p.name,
    kind: p.kind,
    hyper,
    ...(knobs.length === 0 ? {} : { hyperSpace: space }),
    fit,
    revive: (params) => ({ ...build(params as P), exposition: [], run: EMPTY_RUN }),
    with: (changes) => iterative(p, { ...hyper, ...changes }),
    method,
    session: (data, options = {}) => sessionOf(data, options, resolveKey(options, p.name)),
  }
}
