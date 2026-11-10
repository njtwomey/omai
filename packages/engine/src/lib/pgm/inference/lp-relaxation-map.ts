/**
 * MAP by the linear-programming relaxation over the local polytope, for models with factors of any size: solved
 * directly as a linear program, and by AD3 (alternating directions dual decomposition, Martins et al. 2011), a traced
 * algorithm that splits the model into its factors and reconciles them with an augmented-Lagrangian consensus. Both
 * report the LP value (an upper bound on the MAP log value), whether the relaxed solution is integral and the gap.
 */

import { argmaxOf, type Json, maxOf, OmaiError } from '../../base/index.ts'
import { linprog, projectSimplex, qpActiveSet, qpArrays } from '../../optim/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type RunStatus,
  runAlgorithm,
  type TracePolicy,
  type Tracer,
} from '../../trace/index.ts'
import type { DiscreteModel } from '../models/factor-graph.ts'
import { stateNames } from '../models/factor-graph.ts'
import { type Compiled, compile } from './_compiled.ts'
import { iteratedConditionalModes, type MapResult } from './map.ts'

function logValueOf(c: Compiled, x: ArrayLike<number>): number {
  let s = c.constant
  for (const f of c.factors) {
    let idx = 0
    for (const [k, v] of f.vars.entries()) idx += (x[v] ?? 0) * (f.strides[k] ?? 0)
    s += f.logValues[idx] ?? Number.NEGATIVE_INFINITY
  }
  return s
}

function resultOf(model: DiscreteModel, c: Compiled, x: ArrayLike<number>): MapResult {
  const indices = Object.fromEntries(c.names.map((v, i) => [v, x[i] ?? 0]))
  return { assignment: stateNames(model, indices), indices, logValue: logValueOf(c, x) }
}

/** The compiled model split into node potentials (single-variable factors summed) and clusters (the other factors). */
function splitFactors(c: Compiled, caller: string): { theta: Float64Array[]; clusters: number[] } {
  const theta = c.names.map((_, i) => new Float64Array(c.card[i] ?? 1))
  const clusters: number[] = []
  for (const [a, f] of c.factors.entries()) {
    for (const v of f.logValues)
      if (!Number.isFinite(v))
        throw new OmaiError('value', `${caller}: a factor value is ${v}; log values must be finite`)
    if (f.vars.length === 1) {
      const t = theta[f.vars[0] ?? 0] as Float64Array
      for (let x = 0; x < t.length; x++) t[x] = (t[x] ?? 0) + (f.logValues[x] ?? 0)
    } else clusters.push(a)
  }
  return { theta, clusters }
}

/** Options of {@link lpRelaxationMap}. */
export interface LpRelaxationOptions {
  readonly evidence?: Readonly<Record<string, string | number>>
  /** An integrality tolerance for deciding whether the solution is integral (default 1e-6). */
  readonly tol?: number
}

/** The result of {@link lpRelaxationMap} and {@link ad3Map}. */
export interface LpRelaxationResult extends MapResult {
  /** The optimum of the relaxation (AD3: the best dual bound found): an upper bound on the MAP log value. */
  readonly dualBound: number
  /** The bound minus the log value of the decoded assignment; zero certifies the assignment is a MAP. */
  readonly integralityGap: number
  /**
   * Whether the relaxation is tight: the relaxed solution is integral (every node marginal is one-hot) or, for AD3, the
   * bound met the value of the rounded assignment. Either way the assignment is a MAP.
   */
  readonly integral: boolean
  /** The node marginals of the relaxed solution, by variable. */
  readonly marginals: Readonly<Record<string, readonly number[]>>
}

/**
 * MAP by the local-polytope relaxation solved as a linear program with the `optim` solvers: maximise the expected log
 * value over node marginals mu_i and factor marginals mu_c that sum to one and agree on the variables they share. Its
 * optimum bounds the MAP value from above and is attained at an integral point exactly when the relaxation is tight
 * (always on trees); an integral solution is the MAP assignment, a fractional one is rounded by node argmax. The size
 * grows with the tables of the factors, so it suits small models and checks of the dual methods ({@link mplp},
 * {@link ad3Algorithm}).
 *
 * @param model - A model with factors of any size (log values must be finite).
 * @param options - Evidence and the integrality tolerance.
 * @returns The rounded assignment, the LP optimum as the bound, the gap, whether it is integral and the marginals.
 * @throws OmaiError with code `value` for a non-finite log value, `convergence` when the linear program does not solve.
 * @concept map-inference
 * @concept linear-programming
 * @index function MAP by the local-polytope LP relaxation, solved with the linear-programming solvers.
 * @example
 * ```ts
 * import { isingModel, lpRelaxationMap } from 'omai-engine/pgm'
 * const r = lpRelaxationMap(isingModel({ rows: 1, cols: 3, coupling: 1, field: { s_0_0: 0.5 } }))
 * [r.integral, r.assignment.s_0_2, Math.abs(r.integralityGap) < 1e-9] // => [true, '+1', true]
 * ```
 */
export function lpRelaxationMap(model: DiscreteModel, options: LpRelaxationOptions = {}): LpRelaxationResult {
  const c = compile(model, options.evidence ?? {})
  const { theta, clusters } = splitFactors(c, 'lpRelaxationMap')
  const n = c.names.length
  const nodeAt: number[] = []
  let size = 0
  for (let i = 0; i < n; i++) {
    nodeAt.push(size)
    size += c.card[i] ?? 1
  }
  const clusterAt: number[] = []
  for (const a of clusters) {
    clusterAt.push(size)
    size += c.factors[a]?.logValues.length ?? 0
  }
  const cost = new Float64Array(size)
  for (let i = 0; i < n; i++)
    for (let x = 0; x < (c.card[i] ?? 1); x++) cost[(nodeAt[i] ?? 0) + x] = -(theta[i]?.[x] ?? 0)
  for (const [m, a] of clusters.entries()) {
    const f = c.factors[a]
    for (let e = 0; e < (f?.logValues.length ?? 0); e++) cost[(clusterAt[m] ?? 0) + e] = -(f?.logValues[e] ?? 0)
  }
  const rows: Float64Array[] = []
  const rhs: number[] = []
  const row = (): Float64Array => new Float64Array(size)
  for (let i = 0; i < n; i++) {
    const r = row()
    for (let x = 0; x < (c.card[i] ?? 1); x++) r[(nodeAt[i] ?? 0) + x] = 1
    rows.push(r)
    rhs.push(1)
  }
  for (const [m, a] of clusters.entries()) {
    const f = c.factors[a]
    if (f === undefined) continue
    const k = f.vars.length
    for (const [p, v] of f.vars.entries()) {
      for (let x = 0; x < (c.card[v] ?? 1); x++) {
        const r = row()
        for (let e = 0; e < f.logValues.length; e++)
          if ((f.digits[e * k + p] ?? 0) === x) r[(clusterAt[m] ?? 0) + e] = 1
        r[(nodeAt[v] ?? 0) + x] = -1
        rows.push(r)
        rhs.push(0)
      }
    }
  }
  const sol = linprog({ c: cost, aEq: rows.map((r) => Array.from(r)), bEq: rhs })
  if (sol.status !== 'optimal')
    throw new OmaiError('convergence', `lpRelaxationMap: the linear program ended ${sol.status}`)
  const mu = toFlat(sol.x)
  const marginals: Record<string, number[]> = {}
  const x = new Int32Array(n)
  let integral = true
  const tol = options.tol ?? 1e-6
  for (let i = 0; i < n; i++) {
    const m = Array.from({ length: c.card[i] ?? 1 }, (_, s) => mu[(nodeAt[i] ?? 0) + s] ?? 0)
    marginals[c.names[i] ?? ''] = m
    x[i] = argmaxOf(m)
    if (!m.every((v) => v < tol || v > 1 - tol)) integral = false
  }
  const dualBound = -sol.fun + c.constant
  const result = resultOf(model, c, x)
  return { ...result, dualBound, integralityGap: dualBound - result.logValue, integral, marginals }
}

/** Options of {@link ad3Algorithm} and {@link ad3Map}. */
export interface Ad3Options {
  readonly evidence?: Readonly<Record<string, string | number>>
  /** The initial penalty of the augmented Lagrangian (default 1). */
  readonly rho?: number
  /** Whether to rescale the penalty from the primal and dual residuals (Boyd et al. 2011; default true). */
  readonly adaptive?: boolean
  /** Stop when both residuals fall below this (default 1e-6). */
  readonly tol?: number
  /** Stop when the integrality gap falls below this (default 1e-9). */
  readonly gapTol?: number
  readonly maxIter?: number
  /** In {@link ad3Map}: polish the rounded assignment by iterated conditional modes (default true). */
  readonly polish?: boolean
  readonly tracer?: Tracer
  readonly policy?: TracePolicy
}

/** The state of an AD3 run. */
export interface Ad3State {
  /** Node marginals u_i (the consensus variables). */
  readonly u: readonly Float64Array[]
  /** Factor marginals q_c, by cluster. */
  readonly q: readonly Float64Array[]
  /** Multipliers y_{c,i}, by edge (cluster, variable). */
  readonly y: readonly Float64Array[]
  readonly rho: number
  /** The norm of the disagreement between factor and node marginals. */
  readonly primalResidual: number
  /** rho times the norm of the change of the node marginals. */
  readonly dualResidual: number
  /** The dual bound for the current multipliers: an upper bound on the MAP log value. */
  readonly bound: number
  /** The best dual bound so far. */
  readonly bestBound: number
  /** The best rounded assignment so far and its log value. */
  readonly best: Int32Array
  readonly bestValue: number
  /** Whether every node marginal is one-hot (to 1e-6). */
  readonly integral: boolean
}

/**
 * AD3 (Martins, Figueiredo, Aguiar, Smith and Xing 2011) as an {@link Algorithm}: MAP by the local-polytope relaxation,
 * solved by alternating directions on a consensus form. Each factor c keeps its own marginal q_c over the table of its
 * assignments, tied to the node marginals u_i by multipliers y_{c,i}; a step solves, for every factor, the small
 * quadratic program min over the simplex of s . q + rho/2 sum_i |marginal_i(q) - u_i|^2 exactly (active set), averages
 * the factor marginals into each u_i (projected onto the simplex) and moves the multipliers by rho times the
 * disagreement. Unlike subgradient dual decomposition it converges at rate O(1/t) with a stopping rule from the
 * residuals, and unlike {@link mplp} it reaches the optimum of the relaxation rather than a fixed point of block
 * coordinate descent. The dual bound of the current multipliers (an upper bound on the MAP value) and the rounded
 * primal assignment are recorded, so the integrality gap is on the tape.
 *
 * @param model - A model with factors of any size (log values must be finite).
 * @param options - Evidence, the penalty and its adaptation, tolerances and tracing.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a non-finite log value or a penalty that is not positive.
 * @concept map-inference
 * @concept dual-decomposition
 * @example
 * ```ts
 * import { ad3Algorithm, isingModel } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(ad3Algorithm(isingModel({ rows: 2, cols: 2, coupling: 0.5, field: 0.2 })), { maxSteps: 500 }).status // => 'converged'
 * ```
 */
export function ad3Algorithm(model: DiscreteModel, options: Ad3Options = {}): Algorithm<Ad3State> {
  const c = compile(model, options.evidence ?? {})
  const { theta, clusters } = splitFactors(c, 'ad3')
  const rho0 = options.rho ?? 1
  if (!(rho0 > 0)) throw new OmaiError('value', `ad3: rho must be positive, got ${rho0}`)
  const tol = options.tol ?? 1e-6
  const gapTol = options.gapTol ?? 1e-9
  const adaptive = options.adaptive ?? true
  const n = c.names.length
  /** The factor of each cluster, its edge ids (in scope order) and its number of entries. */
  const parts = clusters.map((a) => ({ f: c.factors[a] as Compiled['factors'][number], edges: c.factorEdges[a] ?? [] }))
  const degree = c.names.map(
    (_, i) =>
      (c.variableEdges[i] ?? []).filter((e) => (c.factors[c.edges[e]?.factor ?? 0]?.vars.length ?? 0) > 1).length,
  )

  const marginalOf = (m: number, k: number, q: Float64Array): Float64Array => {
    const { f } = parts[m] as (typeof parts)[number]
    const out = new Float64Array(c.card[f.vars[k] ?? 0] ?? 1)
    const d = f.vars.length
    for (let e = 0; e < q.length; e++) {
      const slot = f.digits[e * d + k] ?? 0
      out[slot] = (out[slot] ?? 0) + (q[e] ?? 0)
    }
    return out
  }
  const nodeScore = (y: readonly Float64Array[], i: number): Float64Array => {
    const out = Float64Array.from(theta[i] ?? [])
    for (const part of parts) {
      const k = part.f.vars.indexOf(i)
      if (k < 0) continue
      const ye = y[part.edges[k] ?? 0]
      for (let x = 0; x < out.length; x++) out[x] = (out[x] ?? 0) + (ye?.[x] ?? 0)
    }
    return out
  }
  const boundOf = (y: readonly Float64Array[]): number => {
    let bound = c.constant
    for (let i = 0; i < n; i++) bound += maxOf(nodeScore(y, i))
    for (const part of parts) {
      const { f } = part
      const d = f.vars.length
      let best = Number.NEGATIVE_INFINITY
      for (let e = 0; e < f.logValues.length; e++) {
        let v = f.logValues[e] ?? 0
        for (let k = 0; k < d; k++) v -= y[part.edges[k] ?? 0]?.[f.digits[e * d + k] ?? 0] ?? 0
        if (v > best) best = v
      }
      bound += best
    }
    return bound
  }
  const decode = (u: readonly Float64Array[]): Int32Array => Int32Array.from(u, (v) => argmaxOf(v))
  const isIntegral = (u: readonly Float64Array[]): boolean => u.every((v) => v.every((p) => p < 1e-6 || p > 1 - 1e-6))

  return {
    name: 'ad3',
    config: { rho: rho0, adaptive, tol, gapTol, factors: clusters.length } as Json,
    schema: channels({
      metrics: {
        bound: { description: 'dual bound (upper bound on the MAP log value)', direction: 'lower' },
        primal: { description: 'log value of the best rounded assignment', direction: 'higher' },
        gap: { description: 'integrality gap', direction: 'lower' },
        primalResidual: { description: 'disagreement of factor and node marginals', direction: 'lower' },
        dualResidual: { description: 'rho times the change of the node marginals', direction: 'lower' },
        rho: { description: 'penalty parameter' },
      },
    }),
    init: () => {
      const u = c.names.map((_, i) => {
        const k = c.card[i] ?? 1
        return degree[i] === 0
          ? Float64Array.from({ length: k }, (_, x) => (x === argmaxOf(theta[i] ?? [0]) ? 1 : 0))
          : new Float64Array(k).fill(1 / k)
      })
      const q = parts.map((p) => new Float64Array(p.f.logValues.length).fill(1 / Math.max(p.f.logValues.length, 1)))
      const y = c.edges.map((e) => new Float64Array(c.card[e.variable] ?? 1))
      const best = decode(u)
      const bound = boundOf(y)
      return {
        u,
        q,
        y,
        rho: rho0,
        primalResidual: Number.POSITIVE_INFINITY,
        dualResidual: Number.POSITIVE_INFINITY,
        bound,
        bestBound: bound,
        best,
        bestValue: logValueOf(c, best),
        integral: isIntegral(u),
      }
    },
    step: (s) => {
      const rho = s.rho
      // 1. Each factor's quadratic program.
      const q = parts.map((part, m) => {
        const { f, edges } = part
        const d = f.vars.length
        const entries = f.logValues.length
        const linear = new Float64Array(entries)
        for (let e = 0; e < entries; e++) {
          let v = -(f.logValues[e] ?? 0)
          for (let k = 0; k < d; k++) {
            const slot = f.digits[e * d + k] ?? 0
            v += (s.y[edges[k] ?? 0]?.[slot] ?? 0) - rho * (s.u[f.vars[k] ?? 0]?.[slot] ?? 0)
          }
          linear[e] = v
        }
        const P: number[][] = Array.from({ length: entries }, (_, a) =>
          Array.from({ length: entries }, (_, b) => {
            let same = 0
            for (let k = 0; k < d; k++) if (f.digits[a * d + k] === f.digits[b * d + k]) same++
            return rho * same
          }),
        )
        const sol = qpActiveSet(
          qpArrays({
            P,
            q: linear,
            A: [new Array<number>(entries).fill(1)],
            b: [1],
            lb: new Array<number>(entries).fill(0),
          }),
          { x0: s.q[m] ?? new Float64Array(entries).fill(1 / entries) },
        )
        return sol.x
      })
      // 2. The node marginals: the projected average of the factor marginals, shifted by the multipliers.
      const u = s.u.map((old, i) => {
        if (degree[i] === 0) return old
        const avg = Float64Array.from(theta[i] ?? [])
        for (let x = 0; x < avg.length; x++) avg[x] = (avg[x] ?? 0) / (rho * (degree[i] ?? 1))
        for (const [m, part] of parts.entries()) {
          const k = part.f.vars.indexOf(i)
          if (k < 0) continue
          const marg = marginalOf(m, k, q[m] as Float64Array)
          const ye = s.y[part.edges[k] ?? 0]
          for (let x = 0; x < avg.length; x++)
            avg[x] = (avg[x] ?? 0) + ((marg[x] ?? 0) + (ye?.[x] ?? 0) / rho) / (degree[i] ?? 1)
        }
        return Float64Array.from(toFlat(projectSimplex(avg)))
      })
      // 3. The multipliers and the residuals.
      const y = [...s.y]
      let primal = 0
      let dual = 0
      for (const [m, part] of parts.entries()) {
        for (const [k, v] of part.f.vars.entries()) {
          const e = part.edges[k] ?? 0
          const marg = marginalOf(m, k, q[m] as Float64Array)
          const next = new Float64Array(marg.length)
          for (let x = 0; x < marg.length; x++) {
            const r = (marg[x] ?? 0) - (u[v]?.[x] ?? 0)
            primal += r * r
            const du = (u[v]?.[x] ?? 0) - (s.u[v]?.[x] ?? 0)
            dual += du * du
            next[x] = (s.y[e]?.[x] ?? 0) + rho * r
          }
          y[e] = next
        }
      }
      const primalResidual = Math.sqrt(primal)
      const dualResidual = rho * Math.sqrt(dual)
      let nextRho = rho
      if (adaptive) {
        if (primalResidual > 10 * dualResidual) nextRho = Math.min(rho * 2, 1e6)
        else if (dualResidual > 10 * primalResidual) nextRho = Math.max(rho / 2, 1e-6)
      }
      const bound = boundOf(y)
      const candidate = decode(u)
      const value = logValueOf(c, candidate)
      const better = value > s.bestValue
      return {
        u,
        q,
        y,
        rho: nextRho,
        primalResidual,
        dualResidual,
        bound,
        bestBound: Math.min(s.bestBound, bound),
        best: better ? candidate : s.best,
        bestValue: better ? value : s.bestValue,
        integral: isIntegral(u),
      }
    },
    done: (s, t) =>
      t > 0 && (s.bestBound - s.bestValue <= gapTol || (s.primalResidual <= tol && s.dualResidual <= tol))
        ? 'converged'
        : undefined,
    observe: (s) => ({
      metrics: {
        bound: s.bound,
        primal: s.bestValue,
        gap: s.bestBound - s.bestValue,
        primalResidual: s.primalResidual,
        dualResidual: s.dualResidual,
        rho: s.rho,
      },
    }),
  }
}

/**
 * MAP by AD3 ({@link ad3Algorithm} run to convergence): the best rounded assignment, the dual bound and the
 * integrality gap; a zero gap certifies the assignment is a MAP, and `integral` says the relaxation was tight.
 *
 * @param model - A model with factors of any size.
 * @param options - Evidence, penalty, tolerances, iteration limit and tracer.
 * @returns The assignment, its log value, the bound, the gap, whether the relaxation was tight, the marginals
 *   and how the run ended.
 * @concept map-inference
 * @concept dual-decomposition
 * @index function MAP by AD3, alternating-directions dual decomposition of the LP relaxation.
 * @example
 * ```ts
 * import { ad3Map, isingModel } from 'omai-engine/pgm'
 * const r = ad3Map(isingModel({ rows: 1, cols: 3, coupling: 1, field: { s_0_0: -0.5 } }))
 * [r.assignment.s_0_2, r.integral] // => ['-1', true]
 * ```
 */
export function ad3Map(
  model: DiscreteModel,
  options: Ad3Options = {},
): LpRelaxationResult & { readonly status: RunStatus; readonly iterations: number } {
  const c = compile(model, options.evidence ?? {})
  const run = runAlgorithm(ad3Algorithm(model, options), {
    maxSteps: options.maxIter ?? 1000,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const s = run.state
  const marginals = Object.fromEntries(c.names.map((v, i) => [v, Array.from(s.u[i] ?? [])]))
  let result = resultOf(model, c, s.best)
  if ((options.polish ?? true) && !s.integral) {
    // Rounding fractional marginals can tie or disagree with the factors; a few conditional-mode sweeps repair that.
    const init = Object.fromEntries(c.names.map((v, i) => [v, s.best[i] ?? 0]))
    const icm = runAlgorithm(
      iteratedConditionalModes(model, init, options.evidence === undefined ? {} : { evidence: options.evidence }),
      { maxSteps: 100 },
    ).state
    if (icm.logValue > result.logValue) result = resultOf(model, c, icm.x)
  }
  return {
    ...result,
    dualBound: s.bestBound,
    integralityGap: s.bestBound - result.logValue,
    integral: s.integral || s.bestBound - result.logValue <= (options.gapTol ?? 1e-9),
    marginals,
    status: run.status,
    iterations: run.steps,
  }
}
