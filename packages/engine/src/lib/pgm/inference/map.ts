/**
 * Finding the most probable joint assignment (MAP) of a discrete model: max-product belief propagation, MPLP (a dual
 * linear-programming relaxation, Globerson and Jaakkola 2007) with its bound and integrality gap, and local search by
 * iterated conditional modes with random restarts.
 */

import { argmaxOf, type Json, maxOf } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import { type DiscreteModel, stateNames } from '../models/factor-graph.ts'
import { type Compiled, compile } from './_compiled.ts'
import { type BeliefPropagationOptions, beliefPropagation } from './belief-propagation.ts'

/** A MAP assignment with its score. */
export interface MapResult {
  /** The state names of the unobserved variables (in model order). */
  readonly assignment: Record<string, string>
  readonly indices: Record<string, number>
  /** The unnormalised log probability of the assignment (with the evidence). */
  readonly logValue: number
}

function logValueOf(c: Compiled, x: Int32Array): number {
  let s = c.constant
  for (const f of c.factors) {
    let idx = 0
    for (const [k, v] of f.vars.entries()) idx += (x[v] ?? 0) * (f.strides[k] ?? 0)
    s += f.logValues[idx] ?? Number.NEGATIVE_INFINITY
  }
  return s
}

function resultOf(model: DiscreteModel, c: Compiled, x: Int32Array): MapResult {
  const indices = Object.fromEntries(c.names.map((v, i) => [v, x[i] ?? 0]))
  return { assignment: stateNames(model, indices), indices, logValue: logValueOf(c, x) }
}

/**
 * MAP by max-product belief propagation: run max-product BP (exact on trees, a heuristic on loopy graphs) and decode
 * each variable as the argmax of its max-marginal.
 *
 * @param model - The model.
 * @param options - Belief-propagation options (evidence, schedule, damping, stopping rule; the mode is max).
 * @returns The decoded assignment, its log value, and whether BP converged.
 * @concept map-inference
 * @example
 * ```ts
 * import { isingModel, maxProductMap } from 'mdml-engine/pgm'
 * maxProductMap(isingModel({ rows: 1, cols: 3, coupling: 1, field: { s_0_0: 0.5 } })).assignment // => { s_0_0: '+1', s_0_1: '+1', s_0_2: '+1' }
 * ```
 */
export function maxProductMap(
  model: DiscreteModel,
  options: Omit<BeliefPropagationOptions, 'mode'> = {},
): MapResult & { readonly converged: boolean } {
  const c = compile(model, options.evidence ?? {})
  const bp = beliefPropagation(model, { ...options, mode: 'max' })
  const x = Int32Array.from(c.names, (v) => argmaxOf(bp.marginals[v]?.logValues ?? []))
  return { ...resultOf(model, c, x), converged: bp.converged }
}

/** Options of {@link mplp}. */
export interface MplpOptions {
  readonly evidence?: Readonly<Record<string, string | number>>
  /** Stop when the dual objective decreases by less than this in a sweep (default 1e-10). */
  readonly tol?: number
  /** Stop when the integrality gap falls below this (default 1e-10). */
  readonly gapTol?: number
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of an MPLP run. */
export interface MplpState {
  /** Messages lambda_{c -> i}, by edge (cluster, variable). */
  readonly lambda: readonly Float64Array[]
  /** The dual objective: an upper bound on the MAP log value. */
  readonly dual: number
  /** The best decoded assignment so far and its log value. */
  readonly best: Int32Array
  readonly bestValue: number
  readonly decrease: number
}

/**
 * MAP by linear-programming relaxation, solved through its dual by MPLP (max-product linear programming; Globerson and
 * Jaakkola 2007, generalised to clusters as in Sontag, Globerson and Jaakkola 2011), as an {@link Algorithm}. Factors
 * over one variable are node potentials theta_i; larger factors are clusters c with messages lambda_{c->i}. Each sweep
 * performs the block-coordinate descent update of every cluster,
 * lambda_{c->i} = -lambda_i^{-c} + (1/|c|) max_{x_{c\i}} [theta_c + sum_{j in c} lambda_j^{-c}], which never increases
 * the dual objective, an upper bound on the MAP value; local decoding gives a primal assignment, and a zero
 * integrality gap certifies it optimal.
 *
 * @param model - The model.
 * @param options - Evidence and stopping rules.
 * @returns The algorithm.
 * @concept map-inference
 * @example
 * ```ts
 * import { isingModel, mplpAlgorithm } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(mplpAlgorithm(isingModel({ rows: 3, cols: 3, coupling: 0.5, field: 0.2 })), { maxSteps: 200 }).status // => 'converged'
 * ```
 */
export function mplpAlgorithm(model: DiscreteModel, options: MplpOptions = {}): Algorithm<MplpState> {
  const c = compile(model, options.evidence ?? {})
  const tol = options.tol ?? 1e-10
  const gapTol = options.gapTol ?? 1e-10
  // Node potentials (sum of single-variable factors) and clusters (the rest).
  const theta = c.names.map((_, i) => new Float64Array(c.card[i] ?? 1))
  const clusters: number[] = []
  for (const [a, f] of c.factors.entries()) {
    if (f.vars.length === 1) {
      const i = f.vars[0] ?? 0
      for (let x = 0; x < f.logValues.length; x++)
        (theta[i] as Float64Array)[x] = (theta[i]?.[x] ?? 0) + (f.logValues[x] ?? 0)
    } else clusters.push(a)
  }
  const nodeBelief = (lambda: readonly Float64Array[], i: number, skipCluster: number): Float64Array => {
    const out = Float64Array.from(theta[i] ?? [])
    for (const e of c.variableEdges[i] ?? []) {
      const a = c.edges[e]?.factor ?? -1
      if (a === skipCluster || (c.factors[a]?.vars.length ?? 0) < 2) continue
      const m = lambda[e]
      for (let x = 0; x < out.length; x++) out[x] = (out[x] ?? 0) + (m?.[x] ?? 0)
    }
    return out
  }
  const dualOf = (lambda: readonly Float64Array[]): number => {
    let d = c.constant
    for (let i = 0; i < c.names.length; i++) d += maxOf(nodeBelief(lambda, i, -1))
    for (const a of clusters) {
      const f = c.factors[a]
      if (f === undefined) continue
      const es = c.factorEdges[a] ?? []
      const k = f.vars.length
      let best = Number.NEGATIVE_INFINITY
      for (let entry = 0; entry < f.logValues.length; entry++) {
        let v = f.logValues[entry] ?? Number.NEGATIVE_INFINITY
        for (let j = 0; j < k; j++) v -= lambda[es[j] ?? 0]?.[f.digits[entry * k + j] ?? 0] ?? 0
        if (v > best) best = v
      }
      d += best
    }
    return d
  }
  const decode = (lambda: readonly Float64Array[]): Int32Array =>
    Int32Array.from(c.names, (_, i) => argmaxOf(nodeBelief(lambda, i, -1)))
  return {
    name: 'mplp',
    config: { tol, gapTol, clusters: clusters.length } as Json,
    schema: channels({
      metrics: {
        dual: { description: 'dual objective (upper bound on the MAP log value)', direction: 'lower' },
        primal: { description: 'log value of the best decoded assignment', direction: 'higher' },
        gap: { description: 'integrality gap', direction: 'lower' },
      },
    }),
    init: () => {
      const lambda = c.edges.map((e) => new Float64Array(c.card[e.variable] ?? 1))
      const best = decode(lambda)
      return { lambda, dual: dualOf(lambda), best, bestValue: logValueOf(c, best), decrease: Number.POSITIVE_INFINITY }
    },
    step: (s) => {
      const lambda = [...s.lambda]
      for (const a of clusters) {
        const f = c.factors[a]
        if (f === undefined) continue
        const es = c.factorEdges[a] ?? []
        const k = f.vars.length
        const minus = f.vars.map((i) => nodeBelief(lambda, i, a))
        for (const [p, e] of es.entries()) {
          const i = f.vars[p] ?? 0
          const out = new Float64Array(c.card[i] ?? 1).fill(Number.NEGATIVE_INFINITY)
          for (let entry = 0; entry < f.logValues.length; entry++) {
            let v = f.logValues[entry] ?? Number.NEGATIVE_INFINITY
            for (let j = 0; j < k; j++) v += minus[j]?.[f.digits[entry * k + j] ?? 0] ?? 0
            const x = f.digits[entry * k + p] ?? 0
            if (v > (out[x] ?? Number.NEGATIVE_INFINITY)) out[x] = v
          }
          lambda[e] = out.map((v, x) => v / k - (minus[p]?.[x] ?? 0))
        }
      }
      const dual = dualOf(lambda)
      const candidate = decode(lambda)
      const value = logValueOf(c, candidate)
      const better = value > s.bestValue
      return {
        lambda,
        dual,
        best: better ? candidate : s.best,
        bestValue: better ? value : s.bestValue,
        decrease: s.dual - dual,
      }
    },
    done: (s, t) => (t > 0 && (s.dual - s.bestValue <= gapTol || s.decrease <= tol) ? 'converged' : undefined),
    observe: (s) => ({ metrics: { dual: s.dual, primal: s.bestValue, gap: s.dual - s.bestValue } }),
  }
}

/**
 * Run MPLP ({@link mplpAlgorithm}) and return the best decoded assignment, the dual bound and the integrality gap
 * (zero when the relaxation is tight and the assignment is a certified MAP).
 *
 * @param model - The model.
 * @param options - Evidence, stopping rules and tracer.
 * @returns The assignment, its log value, the dual bound and the gap.
 * @concept map-inference
 * @example
 * ```ts
 * import { isingModel, mplp } from 'mdml-engine/pgm'
 * const r = mplp(isingModel({ rows: 1, cols: 3, coupling: 1, field: { s_0_0: -0.5 } }))
 * [r.assignment.s_0_2, Math.abs(r.integralityGap) < 1e-9] // => ['-1', true]
 * ```
 */
export function mplp(
  model: DiscreteModel,
  options: MplpOptions = {},
): MapResult & { readonly dualBound: number; readonly integralityGap: number; readonly status: RunStatus } {
  const c = compile(model, options.evidence ?? {})
  const result = runAlgorithm(mplpAlgorithm(model, options), {
    maxSteps: options.maxIter ?? 1000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const s = result.state
  return {
    ...resultOf(model, c, s.best),
    dualBound: s.dual,
    integralityGap: s.dual - s.bestValue,
    status: result.status,
  }
}

/** Options of {@link localSearchMap}. */
export interface LocalSearchOptions {
  readonly evidence?: Readonly<Record<string, string | number>>
  /** A starting assignment (state indices) for the first run. */
  readonly init?: Readonly<Record<string, number>>
  /** Random restarts after the first run (default 10), drawn with `key`. */
  readonly restarts?: number
  readonly key?: Key
  /** Sweeps per run (default 100). */
  readonly maxIter?: number
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of an ICM run. */
export interface IcmState {
  readonly x: Int32Array
  readonly logValue: number
  readonly changed: number
}

/**
 * Iterated conditional modes (Besag 1986) as an {@link Algorithm}: each sweep sets every variable, in turn, to its most
 * probable state given all the others (coordinate ascent on the log probability), stopping at a local maximum where a
 * sweep changes nothing. Records the log value and the number of changes per sweep.
 *
 * @param model - The model.
 * @param init - The starting state (indices) of the unobserved variables (default: all first states).
 * @param options - Evidence.
 * @returns The algorithm.
 * @concept map-inference
 * @example
 * ```ts
 * import { isingModel, iteratedConditionalModes } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const r = runAlgorithm(iteratedConditionalModes(isingModel({ rows: 1, cols: 3, coupling: 1 }), { s_0_0: 1, s_0_1: 0, s_0_2: 1 }), { maxSteps: 10 })
 * Array.from(r.state.x) // => [0, 0, 0]
 * ```
 */
export function iteratedConditionalModes(
  model: DiscreteModel,
  init: Readonly<Record<string, number>> = {},
  options: Pick<LocalSearchOptions, 'evidence'> = {},
): Algorithm<IcmState> {
  const c = compile(model, options.evidence ?? {})
  const start = Int32Array.from(c.names, (v) => init[v] ?? 0)
  return {
    name: 'iteratedConditionalModes',
    config: { variables: [...c.names] } as Json,
    schema: channels({
      metrics: {
        logValue: { description: 'unnormalised log probability', direction: 'higher' },
        changed: { description: 'variables changed in the sweep' },
      },
    }),
    init: () => ({ x: start, logValue: logValueOf(c, start), changed: -1 }),
    step: (s) => {
      const x = Int32Array.from(s.x)
      let changed = 0
      for (let i = 0; i < c.names.length; i++) {
        const k = c.card[i] ?? 1
        const scores = new Float64Array(k)
        for (const e of c.variableEdges[i] ?? []) {
          const { factor, position } = c.edges[e] ?? { factor: 0, position: 0 }
          const f = c.factors[factor]
          if (f === undefined) continue
          let base = 0
          for (const [j, v] of f.vars.entries()) if (j !== position) base += (x[v] ?? 0) * (f.strides[j] ?? 0)
          const stride = f.strides[position] ?? 0
          for (let st = 0; st < k; st++)
            scores[st] = (scores[st] ?? 0) + (f.logValues[base + st * stride] ?? Number.NEGATIVE_INFINITY)
        }
        const best = argmaxOf(scores)
        if ((scores[best] ?? 0) > (scores[x[i] ?? 0] ?? Number.NEGATIVE_INFINITY) && best !== x[i]) {
          x[i] = best
          changed++
        }
      }
      return { x, logValue: logValueOf(c, x), changed }
    },
    done: (s, t) => (t > 0 && s.changed === 0 ? 'converged' : undefined),
    observe: (s) => ({ metrics: { logValue: s.logValue, changed: s.changed } }),
  }
}

/**
 * MAP by local search: iterated conditional modes from the given start, then from random restarts, keeping the best
 * local maximum found (a fast heuristic with no optimality guarantee).
 *
 * @param model - The model.
 * @param options - Evidence, start, restarts, key, sweeps and tracer.
 * @returns The best assignment found and its log value.
 * @concept map-inference
 * @example
 * ```ts
 * import { isingModel, localSearchMap } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * localSearchMap(isingModel({ rows: 2, cols: 2, coupling: 1, field: 0.1 }), { key: key(0) }).assignment // => { s_0_0: '+1', s_0_1: '+1', s_1_0: '+1', s_1_1: '+1' }
 * ```
 */
export function localSearchMap(model: DiscreteModel, options: LocalSearchOptions = {}): MapResult {
  const c = compile(model, options.evidence ?? {})
  const restarts = options.key === undefined ? 0 : (options.restarts ?? 10)
  let best: IcmState | null = null
  for (let r = 0; r <= restarts; r++) {
    let init: Record<string, number> = { ...(options.init ?? {}) }
    if (r > 0 && options.key !== undefined) {
      const u = uniform(foldIn(options.key, r), [Math.max(c.names.length, 1)]).data
      init = Object.fromEntries(c.names.map((v, i) => [v, Math.floor((u[i] ?? 0) * (c.card[i] ?? 1))]))
    }
    const run = runAlgorithm(
      iteratedConditionalModes(model, init, options.evidence === undefined ? {} : { evidence: options.evidence }),
      {
        maxSteps: options.maxIter ?? 100,
        runId: `localSearch:${r}`,
        ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
        ...(options.policy === undefined ? {} : { policy: options.policy }),
      },
    )
    if (best === null || run.state.logValue > best.logValue) best = run.state
  }
  return resultOf(model, c, (best as IcmState).x)
}
