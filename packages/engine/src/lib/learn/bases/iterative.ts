/**
 * The iterative base: an estimator whose fit is an `Algorithm` of the library run as a session. A particular declares
 * the algorithm over the data, how its state becomes parameters, the verbs those parameters give, and what the state
 * can tell beyond the solver's own channels; the base owns the loop, the `fit/iteration/end` event, restarts, the
 * solver's channels as `solver/...` paths, `model/params` and the kind's standard probes, serialisation, the
 * scikit-learn `fit` surface (a session run to completion with the method's defaults), the records' rows, and a
 * `session` to drive the same fit live.
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
  type Subscription,
  scalar,
  session,
  tensor,
} from '../../trace/index.ts'
import type { Estimator, EstimatorKind, FitOptions, Fitted, FittedFacts, LearnData, Verbs } from '../estimator.ts'
import { withInputRows } from '../explained.ts'
import { kindProbes } from './kinds.ts'

/** What a particular method declares over the iterative base. */
export interface IterativeParticular<H extends object, S, P extends Json, V extends Verbs & FittedFacts> {
  readonly name: string
  readonly kind: EstimatorKind
  /** The hyperparameters, with their defaults. */
  readonly defaults: H
  /** The knobs a demo or a search varies (the rest of `H` keeps its values). */
  readonly hyperSpace?: (hyper: H) => DictSpace
  /** The library's algorithm over the data: untouched solvers from `lib/`. */
  algorithm(data: LearnData, hyper: H, key: Key): Algorithm<S>
  /** The fitted parameters of a state (plain JSON, what `revive` rebuilds verbs from). */
  params(state: S, data: LearnData, hyper: H): P
  /** The verbs and facts of fitted parameters. */
  verbs(params: P, hyper: H): V
  /** What the state can tell beyond the solver's channels: aliases, derived paths, events. */
  contract?(data: LearnData, hyper: H): Contract<S>
  /** The step budget of a fit (default 10,000; the algorithm's `done` usually ends it sooner). */
  readonly maxSteps?: (hyper: H) => number
  /** Restarts: how many, and which final state wins. */
  readonly restarts?: { readonly count: (hyper: H) => number; readonly better: (a: S, b: S) => boolean }
  /** Structural records from the final state (a tree, a coefficient table). */
  records?(state: S, data: LearnData, hyper: H): readonly Exposition[]
  /** Step-like records from the tape; `undefined` when the tape has nothing to project (a fit leaves it out). */
  readonly projections?: Readonly<Record<string, (run: Run, data: LearnData, hyper: H) => Exposition | undefined>>
  /** What `fit` keeps by default (default nothing: the raw algorithm). */
  readonly logs?: Subscription
}

/** A fitted value of the iterative base: the fitted estimator, its records and the tape of its fit. */
export type IterativeFitted<P extends Json, V> = Fitted<P> &
  V & { readonly exposition: readonly Exposition[]; readonly run: Run }

/** Options of {@link IterativeEstimator.session}: the fit options, and when to compute. */
export interface IterativeSessionOptions extends FitOptions {
  readonly mode?: Mode
  readonly steps?: number
  /** Keep checkpoints under `logs: 'off'` so the session can seek and probe (default `true`; a fit sets `false`). */
  readonly past?: boolean
}

/** An estimator on the iterative base: scikit-learn's surface, plus its method and a session of its fit. */
export interface IterativeEstimator<H extends object, S, P extends Json, V extends Verbs & FittedFacts>
  extends Estimator<H, IterativeFitted<P, V>> {
  /** The method of this estimator on a dataset (to drive as a session, or to read its contract). */
  method(data: LearnData): Method<S, Config>
  /** A session of the fit on a dataset: the tape to advance, seek, probe and fork. */
  session(data: LearnData, options?: IterativeSessionOptions): Session<S, Config>
  with(changes: Partial<H>): IterativeEstimator<H, S, P, V>
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
export function iterative<H extends object, S, P extends Json, V extends Verbs & FittedFacts>(
  p: IterativeParticular<H, S, P, V>,
  hyper: H = p.defaults,
): IterativeEstimator<H, S, P, V> {
  const space: DictSpace = p.hyperSpace?.(hyper) ?? dict({})
  const knobs = Object.keys(space.spaces)
  const hyperOf = (c: Config): H => ({ ...hyper, ...c }) as H
  const configOf = (h: H): Config =>
    Object.fromEntries(knobs.map((k) => [k, (h as Record<string, unknown>)[k] as SpaceValue])) as Config
  const build = (params: P): Fitted<P> & V => ({ ...p.verbs(params, hyper), estimator: p.name, kind: p.kind, params })
  // The algorithm is built once per (data, hyper, key): `make` and the contract both need it.
  const cache = new WeakMap<LearnData, Map<string, Algorithm<S>>>()
  const algorithmOf = (data: LearnData, h: H, key: Key): Algorithm<S> => {
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
  const contractOf = (data: LearnData, h: H, key: Key): Contract<S> => {
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
    Object.assign(out, kindProbes<S>(p.kind, { data, fitted: (s) => build(p.params(s, data, h)) }))
    Object.assign(out, p.contract?.(data, h) ?? {})
    return out
  }
  const method = (data: LearnData): Method<S, Config> =>
    defineMethod<S, Config>({
      name: p.name,
      config: space,
      make: (c, key) => {
        const alg = algorithmOf(data, hyperOf(c), key)
        return {
          ...alg,
          step: (s, ctx) => {
            const next = alg.step(s, ctx)
            ctx.emit('fit/iteration/end', { t: ctx.t + 1 })
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
  const steps = p.maxSteps?.(hyper) ?? 10_000
  const checkpointEvery = Math.max(1, Math.floor(steps / 100))
  const sessionOf = (data: LearnData, options: IterativeSessionOptions, key: Key): Session<S, Config> =>
    session(method(data), {
      key,
      config: configOf(hyper),
      logs: options.logs ?? (options.explain === false ? 'off' : 'defaults'),
      mode: options.mode ?? 'live',
      steps: options.steps ?? steps,
      ...(options.past === undefined ? {} : { past: options.past }),
      checkpointEvery,
    })
  const fit = (data: LearnData, options: FitOptions = {}): IterativeFitted<P, V> => {
    const key = resolveKey(options, p.name)
    const n = Math.max(1, p.restarts?.count(hyper) ?? 1)
    let best: Session<S, Config> | undefined
    for (let r = 0; r < n; r++) {
      const s = sessionOf(data, { ...options, mode: 'eager', past: false }, n === 1 ? key : foldIn(key, r))
      if (best === undefined || p.restarts?.better(s.state, best.state) === true) best = s
    }
    const chosen = best as Session<S, Config>
    const state = chosen.state
    const run = chosen.run()
    const fitted = build(p.params(state, data, hyper))
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
