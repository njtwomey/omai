/**
 * Variational inference on discrete graphical models: naive mean field and variational message passing (Winn and Bishop
 * 2005), which return marginals with an ELBO lower bound on log Z, and tree-reweighted belief propagation (Wainwright
 * et al. 2005), which returns pseudo-marginals with an upper bound, using spanning-tree edge probabilities. Each is a
 * traced algorithm.
 */

import { type Json, logSumExpOf, OmaiError } from '../../base/index.ts'
import { type Key, uniform } from '../../random/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { inverse } from '../_dense.ts'
import type { Factor } from '../factor.ts'
import type { DiscreteModel } from '../models/factor-graph.ts'
import { type Compiled, compile, marginalFactor, normaliseLog, observedMarginals } from './_compiled.ts'

/** Options of {@link meanField} and {@link variationalMessagePassing}. */
export interface MeanFieldOptions {
  /** Observed states (names or indices). */
  readonly evidence?: Readonly<Record<string, string | number>>
  /** Damping in [0, 1) of each update (in probability space). */
  readonly damping?: number
  /** Stop when no marginal changes by more than this (default 1e-10). */
  readonly tol?: number
  /** Iteration cap (default 500). */
  readonly maxIter?: number
  /** Start from random marginals drawn with this key (default: uniform marginals). */
  readonly key?: Key
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of a mean-field run: one log marginal per unobserved variable. */
export interface MeanFieldState {
  readonly q: readonly Float64Array[]
  readonly change: number
  readonly elbo: number
}

/** Result of {@link meanField}, {@link variationalMessagePassing} and {@link treeReweighted}. */
export interface VariationalResult {
  /** The approximate marginal of every variable (observed ones are point masses). */
  readonly marginals: Record<string, Factor>
  /** The bound on log Z (a lower bound for mean field, an upper bound for tree-reweighted BP). */
  readonly logPartitionBound: number
  readonly converged: boolean
  readonly status: RunStatus
  readonly iterations: number
}

/**
 * E_{q}[log f_a] as a function of variable position k's state, with the other variables of the factor averaged under
 * their marginals (configurations of zero probability are skipped, so hard zeros elsewhere do not poison the result).
 */
function expectedLogFactor(c: Compiled, a: number, k: number, q: readonly Float64Array[]): Float64Array {
  const f = c.factors[a]
  if (f === undefined) return new Float64Array(0)
  const d = f.vars.length
  const i = f.vars[k] ?? 0
  const out = new Float64Array(c.card[i] ?? 1)
  for (let e = 0; e < f.logValues.length; e++) {
    let w = 1
    for (let j = 0; j < d && w > 0; j++) {
      if (j === k) continue
      w *= Math.exp(q[f.vars[j] ?? 0]?.[f.digits[e * d + j] ?? 0] ?? Number.NEGATIVE_INFINITY)
    }
    if (w === 0) continue
    const x = f.digits[e * d + k] ?? 0
    out[x] = (out[x] ?? 0) + w * (f.logValues[e] ?? 0)
  }
  return out
}

/** The evidence lower bound sum_a E_q[log f_a] + sum_i H(q_i) + the constant of the evidence. */
function elboOf(c: Compiled, q: readonly Float64Array[]): number {
  let s = c.constant
  for (const f of c.factors) {
    const d = f.vars.length
    for (let e = 0; e < f.logValues.length; e++) {
      let w = 1
      for (let j = 0; j < d && w > 0; j++) w *= Math.exp(q[f.vars[j] ?? 0]?.[f.digits[e * d + j] ?? 0] ?? 0)
      if (w > 0) s += w * (f.logValues[e] ?? 0)
    }
  }
  for (const qi of q) for (const l of qi) if (l > Number.NEGATIVE_INFINITY) s -= Math.exp(l) * l
  return s
}

/** The new log marginal of variable i from the expected log factors. */
function update(c: Compiled, i: number, q: readonly Float64Array[]): Float64Array {
  const out = new Float64Array(c.card[i] ?? 1)
  for (const e of c.variableEdges[i] ?? []) {
    const { factor, position } = c.edges[e] ?? { factor: 0, position: 0 }
    const m = expectedLogFactor(c, factor, position, q)
    for (let x = 0; x < out.length; x++) out[x] = (out[x] ?? 0) + (m[x] ?? 0)
  }
  normaliseLog(out)
  return out
}

function dampLog(next: Float64Array, old: Float64Array, damping: number): Float64Array {
  if (damping <= 0) return next
  const out = next.map((v, x) => Math.log((1 - damping) * Math.exp(v) + damping * Math.exp(old[x] ?? 0)))
  normaliseLog(out)
  return out
}

function maxChange(a: readonly Float64Array[], b: readonly Float64Array[]): number {
  let d = 0
  for (const [i, ai] of a.entries())
    for (let x = 0; x < ai.length; x++) d = Math.max(d, Math.abs(Math.exp(ai[x] ?? 0) - Math.exp(b[i]?.[x] ?? 0)))
  return d
}

function initial(c: Compiled, key: Key | undefined): Float64Array[] {
  return c.names.map((_, i) => {
    const k = c.card[i] ?? 1
    if (key === undefined) return new Float64Array(k).fill(-Math.log(k))
    const u = uniform(key, [c.names.length, k]).data
    const out = Float64Array.from({ length: k }, (_, x) => Math.log(0.5 + (u[i * k + x] ?? 0)))
    normaliseLog(out)
    return out
  })
}

function meanFieldLike(model: DiscreteModel, options: MeanFieldOptions, parallel: boolean): Algorithm<MeanFieldState> {
  const c = compile(model, options.evidence ?? {})
  const damping = options.damping ?? 0
  const tol = options.tol ?? 1e-10
  if (!(damping >= 0 && damping < 1)) throw new OmaiError('value', 'variational: damping must be in [0, 1)')
  const name = parallel ? 'variationalMessagePassing' : 'meanField'
  return {
    name,
    config: { damping, tol, schedule: parallel ? 'parallel' : 'sequential' } as Json,
    schema: channels({
      metrics: {
        elbo: { description: 'evidence lower bound on log Z', direction: 'higher' },
        change: { description: 'largest change of a marginal', direction: 'lower' },
      },
      messages: parallel ? { vmp: { description: 'a factor-to-variable message E[log f]' } } : {},
    }),
    init: () => {
      const q = initial(c, options.key)
      return { q, change: Number.POSITIVE_INFINITY, elbo: elboOf(c, q) }
    },
    step: (s) => {
      const q = [...s.q]
      if (parallel) {
        const next = c.names.map((_, i) => dampLog(update(c, i, s.q), s.q[i] as Float64Array, damping))
        for (const [i, v] of next.entries()) q[i] = v
      } else {
        for (let i = 0; i < c.names.length; i++) q[i] = dampLog(update(c, i, q), s.q[i] as Float64Array, damping)
      }
      return { q, change: maxChange(q, s.q), elbo: elboOf(c, q) }
    },
    done: (s, t) => (t > 0 && s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { elbo: s.elbo, change: Number.isFinite(s.change) ? s.change : -1 },
      ...(parallel
        ? {
            messages: c.edges.map((e) => ({
              kind: 'vmp',
              payload: {
                factor: e.factor,
                variable: c.names[e.variable] ?? '',
                values: Array.from(expectedLogFactor(c, e.factor, e.position, s.q)),
              },
              from: `f${e.factor}`,
              to: c.names[e.variable] ?? '',
            })),
          }
        : {}),
    }),
  }
}

function finish(c: Compiled, q: readonly Float64Array[]): Record<string, Factor> {
  const marginals: Record<string, Factor> = observedMarginals(c)
  for (const [i, name] of c.names.entries()) marginals[name] = marginalFactor(name, q[i] as Float64Array)
  return Object.fromEntries(c.all.map((v) => [v.name, marginals[v.name] as Factor]))
}

/**
 * Naive mean field as an {@link Algorithm}: a fully factorised q(x) = prod_i q_i(x_i), updated by coordinate ascent,
 * q_i proportional to exp(sum_{a containing i} E_{q}[log f_a]), each step one sweep over the variables. The evidence
 * lower bound (ELBO) on log Z never decreases; it is recorded with the largest change of a marginal.
 *
 * @param model - The model.
 * @param options - Evidence, damping, stopping rule and a key for a random start.
 * @returns The algorithm.
 * @concept mean-field
 * @example
 * ```ts
 * import { isingModel, meanFieldAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(meanFieldAlgorithm(isingModel({ rows: 2, cols: 2, coupling: 0.3, field: 0.1 })), { maxSteps: 100 }).status // => 'converged'
 * ```
 */
export function meanFieldAlgorithm(model: DiscreteModel, options: MeanFieldOptions = {}): Algorithm<MeanFieldState> {
  return meanFieldLike(model, options, false)
}

/**
 * Variational message passing (Winn and Bishop 2005) on the factor graph of a discrete model, as an
 * {@link Algorithm}: each factor sends each of its variables the message E_{q}[log f_a] under the current marginals of
 * its other variables, and every variable then sets q_i proportional to exp(sum of its messages), all in parallel
 * (damping helps it settle). It has the fixed points of mean field; its `vmp` messages are recorded.
 *
 * @param model - The model.
 * @param options - Evidence, damping, stopping rule and a key for a random start.
 * @returns The algorithm.
 * @concept variational-message-passing
 * @example
 * ```ts
 * import { isingModel, variationalMessagePassingAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(variationalMessagePassingAlgorithm(isingModel({ rows: 2, cols: 2, coupling: 0.3, field: 0.1 }), { damping: 0.5 }), { maxSteps: 200 }).status // => 'converged'
 * ```
 */
export function variationalMessagePassingAlgorithm(
  model: DiscreteModel,
  options: MeanFieldOptions = {},
): Algorithm<MeanFieldState> {
  return meanFieldLike(model, options, true)
}

function runVariational(model: DiscreteModel, options: MeanFieldOptions, parallel: boolean): VariationalResult {
  const c = compile(model, options.evidence ?? {})
  const result = runAlgorithm(meanFieldLike(model, options, parallel), {
    maxSteps: options.maxIter ?? 500,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  return {
    marginals: finish(c, result.state.q),
    logPartitionBound: result.state.elbo,
    converged: result.status === 'converged',
    status: result.status,
    iterations: result.steps,
  }
}

/**
 * Run naive mean field ({@link meanFieldAlgorithm}) and return the factorised marginals and the ELBO, a lower bound on
 * log Z (log P(evidence) for a Bayes network).
 *
 * @param model - The model.
 * @param options - Evidence, damping, stopping rule, start and tracer.
 * @returns Marginals, the lower bound and convergence information.
 * @concept mean-field
 * @example
 * ```ts
 * import { isingModel, meanField, logPartition } from 'omai-engine/pgm'
 * const m = isingModel({ rows: 2, cols: 2, coupling: 0.3, field: 0.1 })
 * meanField(m).logPartitionBound <= logPartition(m) // => true
 * ```
 */
export function meanField(model: DiscreteModel, options: MeanFieldOptions = {}): VariationalResult {
  return runVariational(model, options, false)
}

/**
 * Run variational message passing ({@link variationalMessagePassingAlgorithm}) and return the marginals and the ELBO.
 *
 * @param model - The model.
 * @param options - Evidence, damping (default 0.5), stopping rule, start and tracer.
 * @returns Marginals, the lower bound and convergence information.
 * @concept variational-message-passing
 * @example
 * ```ts
 * import { isingModel, variationalMessagePassing, meanField } from 'omai-engine/pgm'
 * const m = isingModel({ rows: 2, cols: 2, coupling: 0.3, field: 0.1 })
 * Math.abs(variationalMessagePassing(m).logPartitionBound - meanField(m).logPartitionBound) < 1e-8 // => true
 * ```
 */
export function variationalMessagePassing(model: DiscreteModel, options: MeanFieldOptions = {}): VariationalResult {
  return runVariational(model, { damping: 0.5, ...options }, true)
}

/** Options of {@link treeReweighted}. */
export interface TreeReweightedOptions {
  readonly evidence?: Readonly<Record<string, string | number>>
  /**
   * Edge appearance probabilities, one for every edge or per edge keyed `"a|b"` (default: those of the uniform
   * distribution over spanning trees, the effective resistances of the graph).
   */
  readonly rho?: number | Readonly<Record<string, number>>
  /** Damping in [0, 1) of the messages (default 0.5). */
  readonly damping?: number
  readonly tol?: number
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** A pairwise model: unary and pairwise log potentials over the unobserved variables. */
interface Pairwise {
  readonly c: Compiled
  readonly theta: Float64Array[]
  /** Edges (s < t) with their log potential theta_st[x_s * K_t + x_t]. */
  readonly edges: { s: number; t: number; theta: Float64Array; rho: number }[]
  /** Edge ids touching each variable. */
  readonly touching: number[][]
}

/** Edge appearance probabilities of the uniform spanning-tree distribution: effective resistances, component-wise. */
function spanningTreeRho(n: number, pairs: readonly [number, number][]): number[] {
  const adj = Array.from({ length: n }, () => new Set<number>())
  for (const [s, t] of pairs) {
    adj[s]?.add(t)
    adj[t]?.add(s)
  }
  const comp = new Int32Array(n).fill(-1)
  let count = 0
  for (let i = 0; i < n; i++) {
    if ((comp[i] ?? 0) >= 0) continue
    const stack = [i]
    comp[i] = count
    while (stack.length > 0) {
      const v = stack.pop() as number
      for (const w of adj[v] ?? [])
        if ((comp[w] ?? 0) < 0) {
          comp[w] = count
          stack.push(w)
        }
    }
    count++
  }
  const inverses = new Map<number, { members: number[]; inv: number[][] }>()
  for (let k = 0; k < count; k++) {
    const members = Array.from({ length: n }, (_, i) => i).filter((i) => comp[i] === k)
    const m = members.length
    const L = members.map((i) => members.map((j) => (i === j ? (adj[i]?.size ?? 0) : adj[i]?.has(j) ? -1 : 0) + 1 / m))
    inverses.set(k, { members, inv: inverse(L) })
  }
  return pairs.map(([s, t]) => {
    const block = inverses.get(comp[s] ?? 0)
    if (block === undefined) return 1
    const a = block.members.indexOf(s)
    const b = block.members.indexOf(t)
    const g = block.inv
    return (g[a]?.[a] ?? 0) + (g[b]?.[b] ?? 0) - 2 * (g[a]?.[b] ?? 0)
  })
}

function pairwise(model: DiscreteModel, options: TreeReweightedOptions): Pairwise {
  const c = compile(model, options.evidence ?? {})
  const theta = c.names.map((_, i) => new Float64Array(c.card[i] ?? 1))
  const byPair = new Map<string, { s: number; t: number; theta: Float64Array }>()
  for (const f of c.factors) {
    if (f.vars.length === 1) {
      const i = f.vars[0] ?? 0
      for (let x = 0; x < f.logValues.length; x++)
        (theta[i] as Float64Array)[x] = (theta[i]?.[x] ?? 0) + (f.logValues[x] ?? 0)
      continue
    }
    if (f.vars.length !== 2)
      throw new OmaiError('value', 'treeReweighted: needs a pairwise model (factors of one or two variables)')
    const [a, b] = [f.vars[0] ?? 0, f.vars[1] ?? 0]
    const [s, t] = a < b ? [a, b] : [b, a]
    const key = `${s},${t}`
    const kt = c.card[t] ?? 1
    let entry = byPair.get(key)
    if (entry === undefined) {
      entry = { s, t, theta: new Float64Array((c.card[s] ?? 1) * kt) }
      byPair.set(key, entry)
    }
    for (let e = 0; e < f.logValues.length; e++) {
      const xa = f.digits[2 * e] ?? 0
      const xb = f.digits[2 * e + 1] ?? 0
      const [xs, xt] = a < b ? [xa, xb] : [xb, xa]
      entry.theta[xs * kt + xt] = (entry.theta[xs * kt + xt] ?? 0) + (f.logValues[e] ?? 0)
    }
  }
  const list = [...byPair.values()]
  const rhoOption = options.rho
  const defaults =
    rhoOption === undefined
      ? spanningTreeRho(
          c.names.length,
          list.map((e) => [e.s, e.t]),
        )
      : []
  const edges = list.map((e, k) => {
    const named = `${c.names[e.s]}|${c.names[e.t]}`
    const reversed = `${c.names[e.t]}|${c.names[e.s]}`
    const rho =
      rhoOption === undefined
        ? Math.min(1, defaults[k] ?? 1) // effective resistances are at most one, up to rounding
        : typeof rhoOption === 'number'
          ? rhoOption
          : (rhoOption[named] ?? rhoOption[reversed] ?? 1)
    if (!(rho > 0 && rho <= 1)) throw new OmaiError('value', `treeReweighted: edge appearance ${rho} must be in (0, 1]`)
    return { ...e, rho }
  })
  const touching = c.names.map(() => [] as number[])
  for (const [k, e] of edges.entries()) {
    touching[e.s]?.push(k)
    touching[e.t]?.push(k)
  }
  return { c, theta, edges, touching }
}

/** The state of a tree-reweighted run: log messages toS[e] (t to s) and toT[e] (s to t). */
export interface TreeReweightedState {
  readonly toS: readonly Float64Array[]
  readonly toT: readonly Float64Array[]
  readonly change: number
}

/** The weighted sum theta_v + sum_e rho_e log M_e->v over the edges at v, excluding (if given) edge `skip`. */
function weightedIn(p: Pairwise, st: TreeReweightedState, v: number, skip: number): Float64Array {
  const out = Float64Array.from(p.theta[v] ?? [])
  for (const k of p.touching[v] ?? []) {
    if (k === skip) continue
    const e = p.edges[k]
    if (e === undefined) continue
    const m = e.s === v ? st.toS[k] : st.toT[k]
    for (let x = 0; x < out.length; x++) out[x] = (out[x] ?? 0) + e.rho * (m?.[x] ?? 0)
  }
  return out
}

/** Divide a log potential by rho (a hard zero stays a hard zero). */
const scaled = (v: number, rho: number): number => (v === Number.NEGATIVE_INFINITY ? v : v / rho)

/** New message from `from` to `to` along edge k. */
function trwMessage(p: Pairwise, st: TreeReweightedState, k: number, towardsS: boolean): Float64Array {
  const e = p.edges[k] as Pairwise['edges'][number]
  const [from, to] = towardsS ? [e.t, e.s] : [e.s, e.t]
  const kFrom = p.c.card[from] ?? 1
  const kTo = p.c.card[to] ?? 1
  const kt = p.c.card[e.t] ?? 1
  const base = weightedIn(p, st, from, k)
  // The reverse message along this edge, raised to (1 - rho).
  const reverse = towardsS ? st.toT[k] : st.toS[k]
  const out = new Float64Array(kTo)
  for (let y = 0; y < kTo; y++) {
    const terms = new Float64Array(kFrom)
    for (let x = 0; x < kFrom; x++) {
      const th = towardsS ? (e.theta[y * kt + x] ?? 0) : (e.theta[x * kt + y] ?? 0)
      terms[x] = scaled(th, e.rho) + (base[x] ?? 0) - (1 - e.rho) * (reverse?.[x] ?? 0)
    }
    out[y] = logSumExpOf(terms)
  }
  normaliseLog(out)
  return out
}

/** Beliefs and the TRW bound at a message state. */
function trwBound(p: Pairwise, st: TreeReweightedState): { unary: Float64Array[]; bound: number } {
  const unary = p.c.names.map((_, v) => {
    const b = weightedIn(p, st, v, -1)
    normaliseLog(b)
    return b
  })
  let bound = p.c.constant
  for (const [v, b] of unary.entries()) {
    for (let x = 0; x < b.length; x++) {
      const pb = Math.exp(b[x] ?? 0)
      if (pb > 0) bound += pb * ((p.theta[v]?.[x] ?? 0) - (b[x] ?? 0))
    }
  }
  for (const [k, e] of p.edges.entries()) {
    const ks = p.c.card[e.s] ?? 1
    const kt = p.c.card[e.t] ?? 1
    const as = weightedIn(p, st, e.s, k)
    const at = weightedIn(p, st, e.t, k)
    const logs = new Float64Array(ks * kt)
    for (let x = 0; x < ks; x++) {
      for (let y = 0; y < kt; y++) {
        logs[x * kt + y] =
          scaled(e.theta[x * kt + y] ?? 0, e.rho) +
          (as[x] ?? 0) -
          (1 - e.rho) * (st.toS[k]?.[x] ?? 0) +
          (at[y] ?? 0) -
          (1 - e.rho) * (st.toT[k]?.[y] ?? 0)
      }
    }
    normaliseLog(logs)
    for (let x = 0; x < ks; x++) {
      for (let y = 0; y < kt; y++) {
        const l = logs[x * kt + y] ?? Number.NEGATIVE_INFINITY
        if (l === Number.NEGATIVE_INFINITY) continue
        const pb = Math.exp(l)
        // theta_st and minus rho times the mutual information term.
        bound += pb * ((e.theta[x * kt + y] ?? 0) - e.rho * (l - (unary[e.s]?.[x] ?? 0) - (unary[e.t]?.[y] ?? 0)))
      }
    }
  }
  return { unary, bound }
}

/**
 * Tree-reweighted belief propagation (Wainwright, Jaakkola and Willsky 2005) as an {@link Algorithm}, on a pairwise
 * model: messages M_ts(x_s) proportional to sum_{x_t} exp(theta_st / rho_st + theta_t) prod_{v in N(t)\s} M_vt^rho_vt /
 * M_st^(1 - rho_st), with edge appearance probabilities rho of a distribution over spanning trees. Its fixed point
 * gives pseudo-marginals and an upper bound on log Z (the TRW free energy is convex). With every rho = 1 it is
 * ordinary belief propagation.
 *
 * @param model - A pairwise model (factors of one or two variables).
 * @param options - Evidence, edge appearance probabilities, damping and stopping rule.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for factors of three or more variables or rho outside (0, 1].
 * @concept tree-reweighted
 * @example
 * ```ts
 * import { isingModel, treeReweightedAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(treeReweightedAlgorithm(isingModel({ rows: 3, cols: 3, coupling: 0.4 })), { maxSteps: 500 }).status // => 'converged'
 * ```
 */
export function treeReweightedAlgorithm(
  model: DiscreteModel,
  options: TreeReweightedOptions = {},
): Algorithm<TreeReweightedState> {
  const p = pairwise(model, options)
  const damping = options.damping ?? 0.5
  const tol = options.tol ?? 1e-10
  const dampMsg = (next: Float64Array, old: Float64Array): Float64Array => dampLog(next, old, damping)
  return {
    name: 'treeReweighted',
    config: { damping, tol, rho: p.edges.map((e) => e.rho) } as Json,
    schema: channels({
      metrics: {
        bound: { description: 'TRW upper bound on log Z at the current messages', direction: 'lower' },
        change: { description: 'largest message change', direction: 'lower' },
      },
    }),
    init: () => ({
      toS: p.edges.map((e) => new Float64Array(p.c.card[e.s] ?? 1).fill(-Math.log(p.c.card[e.s] ?? 1))),
      toT: p.edges.map((e) => new Float64Array(p.c.card[e.t] ?? 1).fill(-Math.log(p.c.card[e.t] ?? 1))),
      change: Number.POSITIVE_INFINITY,
    }),
    step: (st) => {
      const toS = p.edges.map((_, k) => dampMsg(trwMessage(p, st, k, true), st.toS[k] as Float64Array))
      const toT = p.edges.map((_, k) => dampMsg(trwMessage(p, st, k, false), st.toT[k] as Float64Array))
      return { toS, toT, change: Math.max(maxChange(toS, st.toS), maxChange(toT, st.toT)) }
    },
    done: (st, t) => (t > 0 && st.change <= tol ? 'converged' : undefined),
    observe: (st) => ({
      metrics: { bound: trwBound(p, st).bound, change: Number.isFinite(st.change) ? st.change : -1 },
    }),
  }
}

/**
 * Run tree-reweighted belief propagation ({@link treeReweightedAlgorithm}) and return the pseudo-marginals and the
 * upper bound on log Z; with the mean-field lower bound it brackets the partition function.
 *
 * @param model - A pairwise model.
 * @param options - Evidence, edge appearance probabilities, damping, stopping rule and tracer.
 * @returns Pseudo-marginals, the upper bound and convergence information.
 * @concept tree-reweighted
 * @example
 * ```ts
 * import { isingModel, treeReweighted, logPartition } from 'omai-engine/pgm'
 * const m = isingModel({ rows: 3, cols: 3, coupling: 0.4 })
 * treeReweighted(m).logPartitionBound >= logPartition(m) // => true
 * ```
 */
export function treeReweighted(model: DiscreteModel, options: TreeReweightedOptions = {}): VariationalResult {
  const p = pairwise(model, options)
  const result = runAlgorithm(treeReweightedAlgorithm(model, options), {
    maxSteps: options.maxIter ?? 1000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const { unary, bound } = trwBound(p, result.state)
  return {
    marginals: finish(p.c, unary),
    logPartitionBound: bound,
    converged: result.status === 'converged',
    status: result.status,
    iterations: result.steps,
  }
}

/**
 * The edge appearance probabilities of the uniform distribution over spanning trees of a pairwise model's graph (the
 * effective resistances between the ends of each edge, which sum to n - 1 per connected component).
 *
 * @param model - A pairwise model.
 * @param evidence - Observed states, removed before building the graph.
 * @returns rho by edge, keyed `"a|b"`.
 * @concept tree-reweighted
 * @example
 * ```ts
 * import { isingModel, spanningTreeEdgeProbabilities } from 'omai-engine/pgm'
 * Object.values(spanningTreeEdgeProbabilities(isingModel({ rows: 1, cols: 3 }))) // => [1, 1]
 * ```
 */
export function spanningTreeEdgeProbabilities(
  model: DiscreteModel,
  evidence: Readonly<Record<string, string | number>> = {},
): Record<string, number> {
  const p = pairwise(model, { evidence })
  return Object.fromEntries(
    p.edges.map((e) => [`${p.c.names[e.s]}|${p.c.names[e.t]}`, Math.round(e.rho * 1e12) / 1e12]),
  )
}
