/**
 * Entropy balancing (Hainmueller 2012): reweight one arm so its covariate moments equal a target's exactly, with the
 * weights closest to uniform in relative entropy. The weights are the softmax of a linear function of the covariates,
 * whose coefficients minimise a smooth convex dual, exposed as a `SmoothProblem` for the solver slot (Newton by
 * default; every solver of `smoothSolvers` reaches the same weights) and as a logged algorithm.
 */

import { MdmlError } from '../base/index.ts'
import {
  analyticObjective,
  type GradientPath,
  loggedSolver,
  type SmoothProblem,
  type SolverPath,
} from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { mean, type Prepared, prepare, variance } from './_dense.ts'
import { newtonSolver, solveSmooth } from './_solve.ts'
import { type CausalData, type EffectEstimate, effectFromSe } from './data.ts'
import type { AverageEstimand } from './estimation.ts'
import { type BalanceRow, balanceTable } from './nuisance.ts'

/**
 * The dual of entropy balancing as a smooth problem: `log sum_i exp(-lambda . c_i)` for the centred moments
 * `c_i = z_i - target` of the units to reweight (rows of `moments`, width `q`), with gradient `-sum w_i c_i` and Hessian
 * the weighted covariance of the `c_i`, where `w` is the softmax of `-lambda . c_i`. At its minimum the weighted moments
 * equal the target.
 *
 * @param moments - The units' moment features, row-major `[m, q]`.
 * @param q - The number of features.
 * @param target - The moments to reach, length `q`.
 * @returns The objective and the start `lambda = 0`.
 * @example
 * ```ts
 * import { entropyBalancingProblem } from 'mdml-engine/causal'
 * entropyBalancingProblem([-1, 0, 1, 2], 1, [0.5]).x0 // => [0]
 * ```
 */
export function entropyBalancingProblem(
  moments: ArrayLike<number>,
  q: number,
  target: ArrayLike<number>,
): SmoothProblem {
  const m = moments.length / q
  const centred = Float64Array.from({ length: moments.length }, (_, k) => (moments[k] ?? 0) - (target[k % q] ?? 0))
  const softmax = (lambda: ArrayLike<number>): Float64Array => {
    const a = new Float64Array(m)
    let top = Number.NEGATIVE_INFINITY
    for (let i = 0; i < m; i++) {
      let s = 0
      for (let j = 0; j < q; j++) s -= (lambda[j] ?? 0) * (centred[i * q + j] ?? 0)
      a[i] = s
      top = Math.max(top, s)
    }
    let z = 0
    for (let i = 0; i < m; i++) z += Math.exp((a[i] ?? 0) - top)
    for (let i = 0; i < m; i++) a[i] = Math.exp((a[i] ?? 0) - top) / z
    return a
  }
  return {
    objective: analyticObjective({
      value: (lambda) => {
        let top = Number.NEGATIVE_INFINITY
        const a = new Float64Array(m)
        for (let i = 0; i < m; i++) {
          let s = 0
          for (let j = 0; j < q; j++) s -= (lambda[j] ?? 0) * (centred[i * q + j] ?? 0)
          a[i] = s
          top = Math.max(top, s)
        }
        let z = 0
        for (let i = 0; i < m; i++) z += Math.exp((a[i] ?? 0) - top)
        return top + Math.log(z)
      },
      gradient: (lambda) => {
        const w = softmax(lambda)
        const g = new Float64Array(q)
        for (let i = 0; i < m; i++)
          for (let j = 0; j < q; j++) g[j] = (g[j] ?? 0) - (w[i] ?? 0) * (centred[i * q + j] ?? 0)
        return g
      },
      hessian: (lambda) => {
        const w = softmax(lambda)
        const bar = new Float64Array(q)
        for (let i = 0; i < m; i++)
          for (let j = 0; j < q; j++) bar[j] = (bar[j] ?? 0) + (w[i] ?? 0) * (centred[i * q + j] ?? 0)
        const h = new Float64Array(q * q)
        for (let i = 0; i < m; i++)
          for (let a = 0; a < q; a++)
            for (let b = 0; b < q; b++)
              h[a * q + b] = (h[a * q + b] ?? 0) + (w[i] ?? 0) * (centred[i * q + a] ?? 0) * (centred[i * q + b] ?? 0)
        for (let a = 0; a < q; a++)
          for (let b = 0; b < q; b++) h[a * q + b] = (h[a * q + b] ?? 0) - (bar[a] ?? 0) * (bar[b] ?? 0)
        return h
      },
    }),
    x0: new Array<number>(q).fill(0),
  }
}

/**
 * Entropy balancing as a logged algorithm over the solver slot: each step of the chosen solver (Newton, L-BFGS,
 * gradient descent, ...) logs the dual objective, its gradient norm and the step length.
 *
 * @param moments - The units' moment features, row-major `[m, q]`.
 * @param q - The number of features.
 * @param target - The moments to reach.
 * @param solver - A value of `smoothSolvers.space` (default Newton).
 * @param key - The run's key.
 * @returns The algorithm; its state holds the dual coefficients `x`.
 * @example
 * ```ts
 * import { entropyBalancingAlgorithm } from 'mdml-engine/causal'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = entropyBalancingAlgorithm([-1, 0, 1, 2], 1, [0.5])
 * runAlgorithm(alg, { maxSteps: 50 }).status // => 'converged'
 * ```
 */
export function entropyBalancingAlgorithm(
  moments: ArrayLike<number>,
  q: number,
  target: ArrayLike<number>,
  solver: SpaceValue = newtonSolver(1e-10),
  key: Key = makeKey(0),
): Algorithm<unknown, SolverPath | GradientPath> {
  return loggedSolver(entropyBalancingProblem(moments, q, target), solver, key)
}

/** Options of {@link entropyBalancing}. */
export interface EntropyBalancingOptions {
  /** The target: the treated (default), the controls or everyone. */
  readonly estimand?: AverageEstimand
  /** Match the means only (1, default) or the means and variances (2). */
  readonly moments?: 1 | 2
  /** A solver of the slot (default Newton). */
  readonly solver?: SpaceValue
  readonly key?: Key
  readonly level?: number
}

/** An entropy-balancing estimate. */
export interface EntropyBalancingResult extends EffectEstimate {
  /** The weights, summing to one within each arm that was reweighted (the target arm has uniform weights). */
  readonly weights: Float64Array
  /** Standardised mean differences before and after weighting. */
  readonly balance: readonly BalanceRow[]
  /** The dual coefficients of each reweighted arm (`treated` and/or `control`). */
  readonly coefficients: Readonly<Record<string, readonly number[]>>
}

function momentFeatures(p: Prepared, order: 1 | 2): { z: Float64Array; q: number } {
  const q = p.d * order
  const z = new Float64Array(p.n * q)
  for (let j = 0; j < p.d; j++) {
    const col = Array.from({ length: p.n }, (_, i) => p.x[i * p.d + j] ?? 0)
    const mu = mean(col)
    const sd = Math.sqrt(variance(col, 1)) || 1
    for (let i = 0; i < p.n; i++) {
      const u = ((col[i] ?? 0) - mu) / sd
      z[i * q + j] = u
      if (order === 2) z[i * q + p.d + j] = u * u
    }
  }
  return { z, q }
}

/**
 * Entropy balancing estimate of a treatment effect: weight the controls (ATT), the treated (ATC) or both arms (ATE)
 * so their covariate means (and optionally variances) equal the target population's exactly, then difference the
 * weighted outcome means. The standard error treats the weights as fixed, adding the arms' weighted variances.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The estimand, moments and solver.
 * @returns The effect with the weights, the balance table and the dual coefficients.
 * @example
 * ```ts
 * import { entropyBalancing } from 'mdml-engine/causal'
 * const x = [[0], [1], [2], [3], [1], [2], [3], [4]]
 * const r = entropyBalancing({ x, treatment: [0, 0, 0, 0, 1, 1, 1, 1], outcome: [0, 1, 2, 3, 3, 4, 5, 6] })
 * r.estimate.value // => 2
 * ```
 */
export function entropyBalancing(data: CausalData, options: EntropyBalancingOptions = {}): EntropyBalancingResult {
  const p = prepare(data.x, data.treatment, data.outcome, 'entropyBalancing')
  const estimand = options.estimand ?? 'att'
  const { z, q } = momentFeatures(p, options.moments ?? 1)
  const solver = options.solver ?? newtonSolver(1e-10)
  const key = options.key ?? makeKey(0)
  const treated = Array.from({ length: p.n }, (_, i) => i).filter((i) => p.t[i] === 1)
  const control = Array.from({ length: p.n }, (_, i) => i).filter((i) => p.t[i] === 0)
  const meanOf = (rows: readonly number[]): Float64Array =>
    Float64Array.from({ length: q }, (_, j) => mean(rows.map((r) => z[r * q + j] ?? 0)))
  const targetRows =
    estimand === 'att' ? treated : estimand === 'atc' ? control : Array.from({ length: p.n }, (_, i) => i)
  const target = meanOf(targetRows)
  const weights = new Float64Array(p.n)
  const coefficients: Record<string, readonly number[]> = {}
  const reweigh = (rows: readonly number[], name: string): void => {
    const moments = Float64Array.from(
      { length: rows.length * q },
      (_, k) => z[(rows[Math.floor(k / q)] ?? 0) * q + (k % q)] ?? 0,
    )
    const solved = solveSmooth(entropyBalancingProblem(moments, q, target), solver, key, 500, 'entropyBalancing')
    if (!(solved.gradNorm < 1e-6))
      throw new MdmlError(
        'value',
        `entropyBalancing: the ${name} arm cannot be balanced to the target (it lies outside their convex hull)`,
      )
    coefficients[name] = Array.from(solved.x)
    const a = rows.map((r) => -solved.x.reduce((s, l, j) => s + l * ((z[r * q + j] ?? 0) - (target[j] ?? 0)), 0))
    const top = Math.max(...a)
    const e = a.map((v) => Math.exp(v - top))
    const s = e.reduce((u, v) => u + v, 0)
    rows.forEach((r, i) => {
      weights[r] = (e[i] ?? 0) / s
    })
  }
  const uniform = (rows: readonly number[]): void => {
    for (const r of rows) weights[r] = 1 / rows.length
  }
  if (estimand === 'att') {
    uniform(treated)
    reweigh(control, 'control')
  } else if (estimand === 'atc') {
    uniform(control)
    reweigh(treated, 'treated')
  } else {
    reweigh(treated, 'treated')
    reweigh(control, 'control')
  }
  const armMean = (rows: readonly number[]): number => rows.reduce((s, r) => s + (weights[r] ?? 0) * (p.y[r] ?? 0), 0)
  const armVar = (rows: readonly number[]): number => {
    const m = armMean(rows)
    return rows.reduce((s, r) => s + (weights[r] ?? 0) ** 2 * ((p.y[r] ?? 0) - m) ** 2, 0)
  }
  const value = armMean(treated) - armMean(control)
  const se = Math.sqrt(armVar(treated) + armVar(control))
  const effect = effectFromSe(estimand, value, se, p.n, {
    ...(options.level === undefined ? {} : { level: options.level }),
    method: 'entropy balancing',
  })
  return { ...effect, weights, balance: balanceTable(data.x, data.treatment, weights), coefficients }
}
