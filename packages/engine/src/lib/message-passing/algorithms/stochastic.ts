/**
 * Stochastic and averaged expectation propagation (Li, Hernandez-Lobato and Turner 2015; Dehaene and Barthelme 2018)
 * for a model of one shared latent variable and many exchangeable data factors (a regression over rows, a mixture
 * over observations): one averaged factor approximation stands for all N data factors, so the state is a single message
 * and its size does not depend on N, where EP keeps a message per factor. An update removes one copy of the average
 * from the belief to make the cavity, matches the moments of one data factor against it, and moves the average a fraction
 * 1 / N toward the factor's own message (a stochastic-approximation step); the belief is the fixed part (the priors) times
 * the average to the power N. It is the data-parallel, minibatchable, anytime cousin of EP: the fixed point is where the
 * average of the factors' messages equals the average, and it reproduces EP when the factors are alike. The graph is
 * the same factor graph the engine runs (one shared variable, observed data in the factors' parameters: the regression rows of
 * `operators/rows.ts` are made for it), so one graph runs under `epAlgorithm` and under this.
 *
 * @index module Stochastic and averaged EP: one averaged factor for N exchangeable data factors, memory independent of N.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { foldIn, key, uniform } from '../../random/index.ts'
import { choice, dict, int, real } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  channels,
  defineContract,
  defineMethod,
  frame,
  type Method,
  message,
  type RunStatus,
  scalar,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import type { CompiledFactor, CompiledGraph } from '../core/graph.ts'
import type { Arg, Eta, Family } from '../core/types.ts'
import { type ImproperCounts, type ImproperPolicy, type MpSnapshot, operatorMessage } from '../engine/engine.ts'
import { momentProjection } from '../engine/projections.ts'

/** The knobs of stochastic EP. */
export interface StochasticEpOptions {
  /** The shared latent variable (default: the one latent variable of the graph, when there is exactly one). */
  readonly shared?: string
  /** The ids of the data factors (default: every factor on the shared variable whose operator is not `prior`). */
  readonly data?: readonly string[]
  /** Updates per step: the granularity of a step, so of cancellation and of anytime reports (default 1). */
  readonly batchSize?: number
  /** `sequential` visits the data factors in order, `random` draws each one uniformly with replacement from the seed (default). */
  readonly order?: 'sequential' | 'random'
  /** The seed of the random order. */
  readonly seed?: number
  /** Damping in [0, 1): the average moves `(1 - damping) / N` toward the factor's message. */
  readonly damping?: number
  /**
   * The step of the average: `constant` (1 / N, as published: the average keeps a fluctuation of the size of the spread of
   * the factors' messages over the square root of N) or `decay` (1 / N shrunk by `1 + sweeps / decaySweeps`, a
   * Robbins-Monro schedule under which the average converges to the fixed point: the average of the factors' messages).
   */
  readonly step?: 'constant' | 'decay'
  /** The sweeps over which the `decay` step halves (default 10). */
  readonly decaySweeps?: number
  /** What to do with an update that would make the belief improper. */
  readonly improper?: ImproperPolicy
  /** Stop when the belief moves by no more than this over a sweep (N updates). */
  readonly tol?: number
  /** Stop, uncompleted, after this many sweeps. */
  readonly maxSweeps?: number
}

/** The defaults of stochastic EP. */
export const stochasticEpDefaults: Required<Omit<StochasticEpOptions, 'shared' | 'data'>> = {
  batchSize: 1,
  order: 'random',
  seed: 0,
  damping: 0,
  step: 'constant',
  decaySweeps: 10,
  improper: 'damp',
  tol: 1e-6,
  maxSweeps: 500,
}

/** The state of a stochastic EP run: the averaged message, and where the run is in its sweep. Its size is that of one message. */
export interface SepState {
  /** The averaged factor approximation: one message for the shared variable. */
  readonly site: readonly number[]
  /** The belief at the start of the sweep, for the change over it. */
  readonly start: readonly number[]
  /** Completed sweeps (N updates each). */
  readonly sweeps: number
  /** Updates done in the current sweep. */
  readonly cursor: number
  /** Updates done in all. */
  readonly updates: number
  /** The change of the belief over the last completed sweep (Infinity before the first). */
  readonly change: number
  readonly improper: ImproperCounts
}

/** The paths of a stochastic EP run. */
export type SepPath =
  | 'sep/change'
  | 'sep/sweeps'
  | 'sep/updates'
  | 'sep/improper'
  | 'sep/skipped'
  | 'sep/damped'
  | 'sep/reset'
  | 'sep/marginals'
  | 'sep/frame'
  | 'sep/snapshot'

/**
 * What a stochastic EP run logs, by path.
 *
 * @concept stochastic-ep
 */
export const sepContract: Contract<SepState, SepPath> = defineContract({
  'sep/change': scalar({ from: 'change', description: 'the change of the belief over the last completed sweep' }),
  'sep/sweeps': scalar({ from: 'sweeps', description: 'completed sweeps (N updates each)' }),
  'sep/updates': scalar({ from: 'updates', description: 'updates so far, one data factor each' }),
  'sep/improper': scalar({
    from: 'improper',
    description: 'updates that would have made the belief improper, or failed',
  }),
  'sep/skipped': scalar({ from: 'skipped', description: 'updates dropped by the improper policy' }),
  'sep/damped': scalar({ from: 'damped', description: 'updates shortened until the belief stayed proper' }),
  'sep/reset': scalar({ from: 'reset', description: 'updates that reset the average to flat' }),
  'sep/marginals': message({
    from: 'marginals',
    description: 'the current marginal of every variable, by variable id',
  }),
  'sep/frame': frame({ description: 'the marginals by variable id' }),
  'sep/snapshot': message({
    description: 'an anytime snapshot (see runAnytimeWith): where the run is and every marginal',
  }),
})

const NO_IMPROPER: ImproperCounts = { events: 0, skipped: 0, damped: 0, reset: 0 }

/** What the run is built from: the graph cut into the shared variable, its fixed part and the data factors. */
interface Layout {
  readonly shared: number
  readonly family: Family
  readonly fixed: Eta
  readonly data: readonly CompiledFactor[]
}

function layoutOf(g: CompiledGraph, options: StochasticEpOptions): Layout {
  const latent = g.variables.map((v, i) => ({ v, i })).filter(({ v }) => v.edges.length > 0)
  let shared: number
  if (options.shared === undefined) {
    if (latent.length !== 1)
      throw new OmaiError(
        'value',
        `stochastic EP: name the shared variable (the graph has ${latent.length} latent ones)`,
      )
    shared = (latent[0] as { i: number }).i
  } else {
    const i = g.index.get(options.shared)
    if (i === undefined) throw new OmaiError('value', `stochastic EP: there is no variable ${options.shared}`)
    shared = i
  }
  if (latent.some(({ i }) => i !== shared))
    throw new OmaiError(
      'value',
      'stochastic EP: every factor must touch the shared variable only (the others are observed)',
    )
  const sharedVar = g.variables[shared] as CompiledGraph['variables'][number]
  const family = sharedVar.family
  const named = options.data === undefined ? undefined : new Set(options.data)
  if (named !== undefined) {
    const ids = new Set(g.factors.map((f) => f.id))
    for (const id of named) if (!ids.has(id)) throw new OmaiError('value', `stochastic EP: there is no factor ${id}`)
  }
  const isData = (f: CompiledFactor): boolean => (named === undefined ? f.operator.name !== 'prior' : named.has(f.id))
  const data = g.factors.filter(isData)
  if (data.length === 0) throw new OmaiError('value', 'stochastic EP: the graph has no data factor')
  let fixed: Eta = family.flat
  for (const f of g.factors) {
    if (isData(f)) continue
    const port = f.ports.findIndex((p) => p.edge >= 0)
    const args = f.ports.map(
      (p, q): Arg =>
        q === port
          ? { family, eta: family.flat, value: undefined }
          : { family: p.family, eta: undefined, value: p.value },
    )
    const p = operatorMessage(f, momentProjection, port, args)
    if (p.kind !== 'message')
      throw new OmaiError(
        'value',
        `stochastic EP: the fixed factor ${f.id} must send a message that does not depend on the cavity`,
      )
    fixed = family.product(fixed, p.eta)
  }
  return { shared, family, fixed, data }
}

/** The belief: the fixed part times the average to the power N. */
const beliefOf = (l: Layout, site: Eta): Eta => l.family.product(l.fixed, l.family.power(site, l.data.length))

/**
 * The marginals of a state, by variable id: the shared variable's belief, and the observed variables as such.
 */
function marginalsOf(g: CompiledGraph, l: Layout, state: SepState): Record<string, Record<string, Json>> {
  const out: Record<string, Record<string, Json>> = {}
  for (const [i, v] of g.variables.entries()) {
    if (v.observed !== undefined) out[v.name] = { family: v.family.kind, observed: v.observed as Json }
    else if (i === l.shared) out[v.name] = { family: v.family.kind, ...v.family.summary(beliefOf(l, state.site)) }
  }
  return out
}

/**
 * An anytime snapshot of a stochastic EP state: where it is and the marginals, in the shape `MpSnapshot` has for the
 * engine's runs, so a page reads every message-passing run the same way. Pass it to `runAnytimeWith`.
 *
 * @param g - The compiled graph.
 * @param options - The options the algorithm was built with (they choose the shared variable and the data).
 * @returns A function of a state alone.
 * @concept stochastic-ep
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, sepSnapshot, stochasticEpAlgorithm } from 'omai-engine/message-passing'
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'w', family: { kind: 'gaussian-mv', size: 1 } }, { name: 'y', family: { kind: 'bernoulli' }, observed: 1 }],
 *     factors: [{ operator: 'prior', args: ['w'], params: { eta: [0, 1] } }, { operator: 'probit-row', args: ['y', 'w'], params: { a: [1] } }],
 *   },
 *   defaultOperators(),
 * )
 * const snapshot = sepSnapshot(g)
 * const alg = stochasticEpAlgorithm(g)
 * snapshot(alg.init({ kind: 'key', words: [0, 0] })).sweeps // => 0
 * ```
 */
export function sepSnapshot(g: CompiledGraph, options: StochasticEpOptions = {}): (state: SepState) => MpSnapshot {
  const l = layoutOf(g, options)
  return (state) => ({
    sweeps: state.sweeps,
    updates: state.updates,
    change: state.change,
    improper: state.improper,
    marginals: marginalsOf(g, l, state),
  })
}

/** A pseudo-random index in [0, n): a pure function of the seed and the update's number. */
function drawIndex(seed: number, t: number, n: number): number {
  const u = Number(toFlat(uniform(foldIn(key(seed), t), [1]))[0])
  return Math.min(n - 1, Math.floor(u * n))
}

/**
 * Stochastic expectation propagation (SEP, also called averaged EP) on a factor graph of one shared latent variable and
 * N exchangeable data factors. The state is the single averaged message of the data factors, so memory does not grow
 * with N; an update draws one data factor, removes one copy of the average from the belief for the cavity, moment-matches
 * the factor against it by the operator's own rule (`probit-row` and `logistic-row` for a regression, any operator of one
 * Gaussian vector argument), and moves the average a fraction `1 / N` (times `1 - damping`) toward the message that
 * results. A step is `batchSize` updates, a sweep is N of them, and the run stops when the belief moves less than `tol` over
 * a sweep. The belief is the fixed factors (the priors) times the average to the power N. It is close to EP when the
 * factors are alike, uses `N` times less memory, and is the form of EP that can be stopped and read at any step; it has
 * no evidence estimate on its tape. The improper policy acts on an update that would make the belief improper, and is
 * counted. The state is plain numbers: a run serialises and resumes to the same state (the random order is a function of the
 * seed and the update's number), and `sepSnapshot` with `runAnytimeWith` gives anytime snapshots.
 *
 * @param g - The compiled graph.
 * @param options - The shared variable, the data factors, the batch size, the order and seed, damping, improper policy, tolerance and sweep cap.
 * @returns The algorithm; its tape is described by {@link sepContract}.
 * @throws OmaiError with code `value` for a graph with more than one latent variable (and none named), a factor on a variable other than the shared one, no data factors, or a fixed factor whose message depends on the cavity.
 * @concept stochastic-ep
 * @reference Li, Y., Hernandez-Lobato, J. M. and Turner, R. E. (2015). Stochastic expectation propagation. Advances in Neural Information Processing Systems 28.
 * @reference Dehaene, G. and Barthelme, S. (2018). Expectation propagation in the large data limit. Journal of the Royal Statistical Society B 80.
 * @index algorithm Stochastic (averaged) EP: one averaged site for N data factors, memory independent of N.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, marginalsOf, stochasticEpAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const rows = [[1, 0.5], [1, -1.2], [1, 0.8], [1, 1.5]]
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'w', family: { kind: 'gaussian-mv', size: 2 } }, ...rows.map((_, i) => ({ name: `y${i}`, family: { kind: 'bernoulli' as const }, observed: i % 2 }))],
 *     factors: [{ operator: 'prior', args: ['w'], params: { eta: [0, 0, 0.25, 0, 0, 0.25] } }, ...rows.map((a, i) => ({ operator: 'probit-row', args: [`y${i}`, 'w'], params: { a } }))],
 *   },
 *   defaultOperators(),
 * )
 * const r = runAlgorithm(stochasticEpAlgorithm(g, { order: 'sequential' }), { maxSteps: 100000 })
 * r.state.site.length // => 6
 * ```
 */
export function stochasticEpAlgorithm(
  g: CompiledGraph,
  options: StochasticEpOptions = {},
): Algorithm<SepState, SepPath> {
  const o = { ...stochasticEpDefaults, ...options }
  if (!(o.damping >= 0 && o.damping < 1))
    throw new OmaiError('value', `stochastic EP: damping must be in [0, 1), got ${o.damping}`)
  if (!(Number.isInteger(o.batchSize) && o.batchSize >= 1))
    throw new OmaiError('value', 'stochastic EP: the batch size must be a positive integer')
  const l = layoutOf(g, options)
  const n = l.data.length
  const family = l.family

  const update = (
    site: Eta,
    t: number,
    sweeps: number,
    counts: { events: number; skipped: number; damped: number; reset: number },
  ): Eta => {
    const rate = (1 - o.damping) / n / (o.step === 'decay' ? 1 + sweeps / o.decaySweeps : 1)
    const index = o.order === 'sequential' ? t % n : drawIndex(o.seed, t, n)
    const f = l.data[index] as CompiledFactor
    const port = f.ports.findIndex((p) => p.edge >= 0)
    // The cavity: the belief with one copy of the average removed.
    const belief = beliefOf(l, site)
    const cavity = family.quotient(belief, site)
    if (cavity.improper) return site
    const args = f.ports.map(
      (p, q): Arg =>
        q === port
          ? { family, eta: cavity.eta, value: undefined }
          : { family: p.family, eta: undefined, value: p.value },
    )
    const p = operatorMessage(f, momentProjection, port, args)
    if (p.kind === 'blocked') return site
    if (p.kind !== 'message') {
      counts.events++
      counts.skipped++
      return site
    }
    const next = site.map((v, i) => v + rate * ((p.eta[i] ?? 0) - v))
    if (family.isProper(beliefOf(l, next))) return next
    counts.events++
    if (o.improper === 'error')
      throw new OmaiError('convergence', `stochastic EP: an update of ${f.id} makes the belief improper`)
    if (o.improper === 'reset') {
      counts.reset++
      return family.flat
    }
    if (o.improper === 'damp') {
      let step = 1
      for (let k = 0; k < 10; k++) {
        step /= 2
        const trial = site.map((v, i) => v + step * ((next[i] ?? 0) - v))
        if (family.isProper(beliefOf(l, trial))) {
          counts.damped++
          return trial
        }
      }
    }
    counts.skipped++
    return site
  }

  return {
    name: 'sep',
    config: {
      shared: g.variables[l.shared]?.name ?? '',
      data: n,
      batchSize: o.batchSize,
      order: o.order,
      seed: o.seed,
      damping: o.damping,
      step: o.step,
      decaySweeps: o.decaySweeps,
      improper: o.improper,
      tol: o.tol,
      maxSweeps: o.maxSweeps,
    } as Json,
    schema: channels({
      metrics: {
        change: { description: 'the change of the belief over the last completed sweep' },
        sweeps: { description: 'completed sweeps (N updates each)' },
        updates: { description: 'updates so far, one data factor each' },
        improper: { description: 'updates that would have made the belief improper, or failed' },
        skipped: { description: 'updates dropped by the improper policy' },
        damped: { description: 'updates shortened until the belief stayed proper' },
        reset: { description: 'updates that reset the average to flat' },
      },
      messages: { marginals: { description: 'the current marginal of every variable, by variable id' } },
      frames: true,
    }),
    init: (): SepState => ({
      site: [...family.flat],
      start: [...beliefOf(l, family.flat)],
      sweeps: 0,
      cursor: 0,
      updates: 0,
      change: Number.POSITIVE_INFINITY,
      improper: NO_IMPROPER,
    }),
    step: (state): SepState => {
      const counts = { ...state.improper }
      let site: Eta = state.site
      let cursor = state.cursor
      let updates = state.updates
      let sweeps = state.sweeps
      let start = state.start
      let change = state.change
      for (let k = 0; k < o.batchSize; k++) {
        site = update(site, updates, sweeps, counts)
        updates++
        cursor++
        if (cursor >= n) {
          const now = beliefOf(l, site)
          change = family.isProper(now) && family.isProper(start) ? family.change(start, now) : Number.POSITIVE_INFINITY
          start = [...now]
          cursor = 0
          sweeps++
          break
        }
      }
      return { site: [...site], start, sweeps, cursor, updates, change, improper: counts }
    },
    done: (state): RunStatus | undefined => {
      if (state.site.some((v) => Number.isNaN(v) || !Number.isFinite(v))) return 'diverged'
      if (state.sweeps >= 1 && state.cursor === 0 && state.change < o.tol) return 'converged'
      return state.sweeps >= o.maxSweeps ? 'completed' : undefined
    },
    observe: (state) => {
      const marginals = marginalsOf(g, l, state)
      return {
        metrics: {
          sweeps: state.sweeps,
          updates: state.updates,
          improper: state.improper.events,
          skipped: state.improper.skipped,
          damped: state.improper.damped,
          reset: state.improper.reset,
          ...(Number.isFinite(state.change) ? { change: state.change } : {}),
        },
        messages: [{ kind: 'marginals', payload: marginals as Json }],
        frame: { marginals: marginals as Json },
      }
    },
    serialise: (state) => stateToJson(state),
    deserialise: (json) => stateFromJson(json) as SepState,
  }
}

/**
 * Stochastic EP on a graph as a method: the knobs as a space (batch size, order, seed, damping, improper policy,
 * tolerance, sweep cap), the tape as the typed contract {@link sepContract}.
 *
 * @param g - The compiled graph.
 * @param shared - The shared variable and data factors, when the graph does not say them itself.
 * @returns The method, named `sep`.
 * @concept stochastic-ep
 * @index function The stochastic EP method of a graph: knobs as a space, typed tape.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, stochasticEpMethod } from 'omai-engine/message-passing'
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'w', family: { kind: 'gaussian-mv', size: 1 } }, { name: 'y', family: { kind: 'bernoulli' }, observed: 1 }],
 *     factors: [{ operator: 'prior', args: ['w'], params: { eta: [0, 1] } }, { operator: 'probit-row', args: ['y', 'w'], params: { a: [1] } }],
 *   },
 *   defaultOperators(),
 * )
 * stochasticEpMethod(g).name // => 'sep'
 * ```
 */
export function stochasticEpMethod(
  g: CompiledGraph,
  shared: Pick<StochasticEpOptions, 'shared' | 'data'> = {},
): Method<SepState, Config, SepPath> {
  return defineMethod<SepState, Config, SepPath>({
    name: 'sep',
    config: dict({
      batchSize: int(1, 100_000, { default: 1, log: true, description: 'updates per step' }),
      order: choice(['random', 'sequential'] as const, {
        default: 'random',
        description: 'the order the data factors are visited in',
      }),
      seed: int(0, 1_000_000, { default: 0, description: 'the seed of the random order' }),
      damping: real(0, 0.95, { default: 0, description: 'weight of the old average in the new' }),
      step: choice(['constant', 'decay'] as const, {
        default: 'constant',
        description: 'a fixed 1 / N step, or one that decays over the sweeps',
      }),
      decaySweeps: real(0.5, 1000, {
        log: true,
        default: 10,
        description: 'the sweeps over which the decaying step halves',
      }),
      improper: choice(['skip', 'damp', 'reset', 'error'] as const, {
        default: 'damp',
        description: 'the repair of an update that makes the belief improper',
      }),
      tol: real(1e-12, 1e-2, {
        log: true,
        default: 1e-6,
        description: 'stop when the belief moves less than this over a sweep',
      }),
      maxSweeps: int(1, 100_000, { default: 500, description: 'the sweep cap' }),
    }),
    make: (c) => {
      const num = (name: string): number | undefined => (typeof c[name] === 'number' ? (c[name] as number) : undefined)
      const str = (name: string): string | undefined => (typeof c[name] === 'string' ? (c[name] as string) : undefined)
      const order = str('order')
      const improper = str('improper')
      const step = str('step')
      return stochasticEpAlgorithm(g, {
        ...shared,
        ...(num('batchSize') === undefined ? {} : { batchSize: num('batchSize') as number }),
        ...(order === 'random' || order === 'sequential' ? { order } : {}),
        ...(num('seed') === undefined ? {} : { seed: num('seed') as number }),
        ...(num('damping') === undefined ? {} : { damping: num('damping') as number }),
        ...(step === 'constant' || step === 'decay' ? { step } : {}),
        ...(num('decaySweeps') === undefined ? {} : { decaySweeps: num('decaySweeps') as number }),
        ...(improper === 'skip' || improper === 'damp' || improper === 'reset' || improper === 'error'
          ? { improper }
          : {}),
        ...(num('tol') === undefined ? {} : { tol: num('tol') as number }),
        ...(num('maxSweeps') === undefined ? {} : { maxSweeps: num('maxSweeps') as number }),
      })
    },
    contract: sepContract,
  })
}
