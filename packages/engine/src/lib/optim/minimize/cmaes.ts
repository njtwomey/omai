/**
 * CMA-ES, the covariance matrix adaptation evolution strategy (Hansen): a derivative-free global optimiser that samples
 * candidates from a Gaussian and adapts its mean, step size and covariance towards the best of them, as a traced
 * algorithm with a `cmaes` front end.
 */

import { MdmlError } from '../../base/index.ts'
import { eigh } from '../../numerics/linalg/index.ts'
import { Generator } from '../../random/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import type { Algorithm, StepContext } from '../../trace/index.ts'
import { boundArrays, clipInto } from './_bounds.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import type { Bounds } from './objective.ts'
import {
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

const EMPTY = new Float64Array(0)

// ---------------------------------------------------------------------------------------------------------------
// CMA-ES

/** Options of {@link cmaes}. */
export interface CmaesOptions extends VectorMethodOptions {
  /** The initial step size (default 0.3 of the mean bound range, else 0.5). */
  readonly sigma0?: number
  /** Offspring per generation lambda (default 4 + floor(3 ln n)). */
  readonly popSize?: number
  /** Box bounds: offspring are repaired by clipping into the box. */
  readonly bounds?: Bounds
  /** Stop when the best values of recent generations and the current spread lie within this (default 1e-11). */
  readonly tolFun?: number
  /** Stop when sigma times every coordinate standard deviation falls below this (default 1e-11). */
  readonly tolX?: number
  /** The evaluation budget (default unlimited). */
  readonly maxEvaluations?: number
}

/** State of a CMA-ES run. */
export interface CmaesState extends VectorState {
  readonly mean: Float64Array
  readonly sigma: number
  /** The covariance matrix (n x n, row-major), its eigenvectors (columns) and the square roots of its eigenvalues. */
  readonly cov: Float64Array
  readonly basis: Float64Array
  readonly scales: Float64Array
  readonly pathSigma: Float64Array
  readonly pathC: Float64Array
  readonly generation: number
  /** Best value of each recent generation, for the tolFun test. */
  readonly recent: readonly number[]
  /** Spread (max - min) of the last generation's values. */
  readonly spread: number
}

/**
 * The covariance matrix adaptation evolution strategy, (mu/mu_w, lambda)-CMA-ES (Hansen and Ostermeier 2001; Hansen's
 * tutorial 2016), as a traced algorithm: offspring sampled from N(m, sigma^2 C), weighted recombination of the best mu,
 * cumulative step-size adaptation, rank-one and rank-mu covariance updates, eigendecomposition of C each generation.
 * Each step is one generation; the trace records sigma, the condition number of C and the mean.
 *
 * @param f - The objective.
 * @param x0 - The initial mean (any parameter tree).
 * @param options - Step size, population, bounds, tolerances and budget.
 * @returns The algorithm.
 * @concept evolution-strategies
 * @example
 * ```ts
 * import { cmaesAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * cmaesAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'cmaes'
 * ```
 */
export function cmaesAlgorithm(f: Objective, x0: Parameters, options: CmaesOptions = {}): Algorithm<CmaesState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new MdmlError('value', 'cmaes: no parameters')
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'cmaes')
  const lambda = options.popSize ?? 4 + Math.floor(3 * Math.log(n))
  if (lambda < 2) throw new MdmlError('value', 'cmaes: popSize must be at least 2')
  const mu = Math.floor(lambda / 2)
  const raw = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1))
  const total = raw.reduce((a, b) => a + b, 0)
  const weights = raw.map((w) => w / total)
  const mueff = 1 / weights.reduce((a, w) => a + w * w, 0)
  const cc = (4 + mueff / n) / (n + 4 + (2 * mueff) / n)
  const cs = (mueff + 2) / (n + mueff + 5)
  const c1 = 2 / ((n + 1.3) ** 2 + mueff)
  const cmu = Math.min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2) ** 2 + mueff))
  const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
  const tolFun = options.tolFun ?? 1e-11
  const tolX = options.tolX ?? 1e-11
  const maxfun = options.maxEvaluations ?? Number.POSITIVE_INFINITY
  const history = 10 + Math.ceil((30 * n) / lambda)
  let range = 0
  if (box !== undefined) {
    let count = 0
    for (let i = 0; i < n; i++) {
      const r = (box.upper[i] ?? 0) - (box.lower[i] ?? 0)
      if (Number.isFinite(r)) {
        range += r
        count += 1
      }
    }
    range = count > 0 ? range / count : 0
  }
  const sigma0 = options.sigma0 ?? (range > 0 ? 0.3 * range : 0.5)
  const identity = (): Float64Array => Float64Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? 1 : 0))

  return {
    name: 'cmaes',
    config: configOf({ lambda, mu, sigma0, tolFun, tolX, cc, cs, c1, cmu, damps }),
    schema: vectorSchema({
      metrics: {
        sigma: { description: 'global step size' },
        condition: { description: 'condition number of the covariance matrix' },
        spread: { description: 'range of the objective over the generation' },
      },
      tensors: { mean: { role: 'state', description: 'mean of the search distribution' } },
    }),
    init: () => {
      const m = box === undefined ? Float64Array.from(obj.x0) : clipInto(obj.x0, box)
      const fm = obj.value(m)
      return {
        x: m,
        value: fm,
        grad: EMPTY,
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: sigma0,
        mean: m,
        sigma: sigma0,
        cov: identity(),
        basis: identity(),
        scales: new Float64Array(n).fill(1),
        pathSigma: new Float64Array(n),
        pathC: new Float64Array(n),
        generation: 0,
        recent: [],
        spread: Number.POSITIVE_INFINITY,
      }
    },
    done: vectorDone,
    observe: (s) => {
      let lo = Number.POSITIVE_INFINITY
      let hi = 0
      for (const d of s.scales) {
        lo = Math.min(lo, d)
        hi = Math.max(hi, d)
      }
      return observeVector(s, {
        metrics: { sigma: s.sigma, condition: (hi / lo) ** 2, spread: s.spread },
        tensors: { mean: { role: 'state', value: s.mean } },
      })
    },
    step: (s: CmaesState, ctx: StepContext): CmaesState => {
      const rng = new Generator(ctx.key)
      const z = rng.normal([lambda, n]).data
      const offspring: { x: Float64Array; y: Float64Array; f: number }[] = []
      let nfev = s.nfev
      for (let k = 0; k < lambda; k++) {
        // y = B D z, x = m + sigma y (repaired into the box)
        const y = new Float64Array(n)
        for (let i = 0; i < n; i++) {
          let v = 0
          for (let j = 0; j < n; j++) v += (s.basis[i * n + j] ?? 0) * (s.scales[j] ?? 0) * (z[k * n + j] ?? 0)
          y[i] = v
        }
        let x: Float64Array = Float64Array.from(s.mean, (mi, i) => mi + s.sigma * (y[i] ?? 0))
        if (box !== undefined) {
          x = clipInto(x, box)
          for (let i = 0; i < n; i++) y[i] = ((x[i] ?? 0) - (s.mean[i] ?? 0)) / s.sigma
        }
        offspring.push({ x, y, f: obj.value(x) })
        nfev += 1
      }
      offspring.sort((a, b) => a.f - b.f)
      const bestChild = offspring[0] as { x: Float64Array; f: number }
      const mean = new Float64Array(n)
      const yw = new Float64Array(n)
      for (let k = 0; k < mu; k++) {
        const o = offspring[k] as { x: Float64Array; y: Float64Array }
        for (let i = 0; i < n; i++) {
          mean[i] = (mean[i] ?? 0) + (weights[k] ?? 0) * (o.x[i] ?? 0)
          yw[i] = (yw[i] ?? 0) + (weights[k] ?? 0) * (o.y[i] ?? 0)
        }
      }
      // C^{-1/2} yw = B D^{-1} B^T yw
      const bty = new Float64Array(n)
      for (let j = 0; j < n; j++) {
        let v = 0
        for (let i = 0; i < n; i++) v += (s.basis[i * n + j] ?? 0) * (yw[i] ?? 0)
        bty[j] = v / (s.scales[j] ?? 1)
      }
      const pathSigma = new Float64Array(n)
      const csn = Math.sqrt(cs * (2 - cs) * mueff)
      for (let i = 0; i < n; i++) {
        let v = 0
        for (let j = 0; j < n; j++) v += (s.basis[i * n + j] ?? 0) * (bty[j] ?? 0)
        pathSigma[i] = (1 - cs) * (s.pathSigma[i] ?? 0) + csn * v
      }
      const psNorm = Math.sqrt(pathSigma.reduce((a, v) => a + v * v, 0))
      const generation = s.generation + 1
      const hsig = psNorm / Math.sqrt(1 - (1 - cs) ** (2 * generation)) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0
      const ccn = Math.sqrt(cc * (2 - cc) * mueff)
      const pathC = Float64Array.from(s.pathC, (p, i) => (1 - cc) * p + hsig * ccn * (yw[i] ?? 0))
      const cov = new Float64Array(n * n)
      const keep = 1 - c1 - cmu + (1 - hsig) * c1 * cc * (2 - cc)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
          let rankMu = 0
          for (let k = 0; k < mu; k++) {
            const o = offspring[k] as { y: Float64Array }
            rankMu += (weights[k] ?? 0) * (o.y[i] ?? 0) * (o.y[j] ?? 0)
          }
          const v = keep * (s.cov[i * n + j] ?? 0) + c1 * (pathC[i] ?? 0) * (pathC[j] ?? 0) + cmu * rankMu
          cov[i * n + j] = v
          cov[j * n + i] = v
        }
      }
      const sigma = s.sigma * Math.exp((cs / damps) * (psNorm / chiN - 1))
      const { eigenvalues, eigenvectors } = eigh(fromStorage(cov, [n, n], 'float64'))
      const values = toFlat(eigenvalues)
      const basis = Float64Array.from(toFlat(eigenvectors))
      const scales = Float64Array.from(values, (v) => Math.sqrt(Math.max(v, 1e-300)))
      const better = bestChild.f < s.value
      const recent = [...s.recent, bestChild.f].slice(-history)
      const spread = (offspring[lambda - 1] as { f: number }).f - bestChild.f
      let stop: CmaesState['stop']
      let recentRange = 0
      if (recent.length >= history) {
        let lo = Number.POSITIVE_INFINITY
        let hi = Number.NEGATIVE_INFINITY
        for (const v of recent) {
          lo = Math.min(lo, v)
          hi = Math.max(hi, v)
        }
        recentRange = hi - lo
      } else recentRange = Number.POSITIVE_INFINITY
      let sdMax = 0
      for (let i = 0; i < n; i++) sdMax = Math.max(sdMax, Math.sqrt(Math.max(cov[i * n + i] ?? 0, 0)))
      const pcMax = pathC.reduce((a, v) => Math.max(a, Math.abs(v)), 0)
      let dmin = Number.POSITIVE_INFINITY
      let dmax = 0
      for (const d of scales) {
        dmin = Math.min(dmin, d)
        dmax = Math.max(dmax, d)
      }
      if (Math.max(recentRange, spread) < tolFun)
        stop = { status: 'converged', message: 'tolFun: objective range below tolerance' }
      else if (sigma * Math.max(sdMax, pcMax) < tolX)
        stop = { status: 'converged', message: 'tolX: step sizes below tolerance' }
      else if ((dmax / dmin) ** 2 > 1e14) stop = { status: 'failed', message: 'condition number of C exceeds 1e14' }
      else if (nfev >= maxfun) stop = { status: 'failed', message: 'maximum number of function evaluations reached' }
      return {
        x: better ? bestChild.x : s.x,
        value: better ? bestChild.f : s.value,
        grad: EMPTY,
        nfev,
        ngev: 0,
        nhev: 0,
        step: sigma,
        mean,
        sigma,
        cov,
        basis,
        scales,
        pathSigma,
        pathC,
        generation,
        recent,
        spread,
        ...(stop === undefined ? {} : { stop }),
      }
    },
  }
}

/**
 * Minimise by CMA-ES. `maxIterations` counts generations (default 100 + 150 (n + 3)^2 / sqrt(lambda)).
 *
 * @param f - The objective.
 * @param x0 - The initial mean (any parameter tree).
 * @param options - Step size, population, bounds, tolerances, budget, key or seed and run options.
 * @returns The best point found.
 * @concept evolution-strategies
 * @index function Covariance matrix adaptation evolution strategy.
 * @example
 * ```ts
 * import { cmaes } from 'mdml-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'mdml-engine/tensor'
 * const r = cmaes((x) => sum(square(sub(x as Value, [1, -2, 3]))), [0, 0, 0], { seed: 0 })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e4) / 1e4) // => [1, -2, 3]
 * ```
 */
export function cmaes(f: Objective, x0: Parameters, options: CmaesOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  const lambda = options.popSize ?? 4 + Math.floor(3 * Math.log(obj.n))
  return runVector(
    cmaesAlgorithm(f, x0, options),
    obj,
    options,
    Math.ceil(100 + (150 * (obj.n + 3) ** 2) / Math.sqrt(lambda)),
  )
}
