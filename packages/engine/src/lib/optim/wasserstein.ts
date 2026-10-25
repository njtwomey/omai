/**
 * Wasserstein distances and barycentres: exact 1-D Wasserstein distances by matching quantile functions (as
 * `scipy.stats.wasserstein_distance`), sliced Wasserstein distances over random projections, and Wasserstein
 * barycentres, the average of several distributions in transport geometry, with fixed support (iterative Bregman
 * projections) or free support.
 */

import { type Json, logSumExpOf, WhiteboxError } from '../base/index.ts'
import { type Key, key as makeKey, normal } from '../random/index.ts'
import {
  abs,
  asTensor,
  div,
  fromStorage,
  matmul,
  mean,
  mul,
  norm,
  pow,
  primalOf,
  promoteFloat,
  reshape,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../trace/index.ts'
import { emd } from './transport.ts'

type Vec = Float64Array

/** Options of {@link wasserstein1d}. */
export interface Wasserstein1dOptions {
  /** Weights of the samples (default uniform; normalised to sum 1). */
  readonly uWeights?: TensorLike
  readonly vWeights?: TensorLike
  /** The order p >= 1 (default 1). */
  readonly p?: number
}

function normalisedWeights(w: TensorLike | undefined, n: number, name: string): Vec {
  const out = w === undefined ? new Float64Array(n).fill(1) : Float64Array.from(toFlat(primalOf(asTensor(w))))
  if (out.length !== n) throw new WhiteboxError('shape', `${name}: one weight per value`)
  let s = 0
  for (const v of out) s += v
  if (!(s > 0)) throw new WhiteboxError('value', `${name}: the weights must have positive mass`)
  return Float64Array.from(out, (v) => v / s)
}

/** W_p^p between two weighted samples on the line, as a value differentiable in the sample positions. */
function wassersteinPower(u: Value, v: Value, uw: Vec, vw: Vec, p: number): Value {
  const uu = toFlat(primalOf(u))
  const vv = toFlat(primalOf(v))
  const su = Array.from(uu.keys()).sort((i, j) => (uu[i] ?? 0) - (uu[j] ?? 0))
  const sv = Array.from(vv.keys()).sort((i, j) => (vv[i] ?? 0) - (vv[j] ?? 0))
  // Merge the two quantile functions: on each interval of t both are constant.
  const iu: number[] = []
  const iv: number[] = []
  const dt: number[] = []
  let a = 0
  let b = 0
  let cu = uw[su[0] ?? 0] ?? 0
  let cv = vw[sv[0] ?? 0] ?? 0
  let t = 0
  while (a < su.length && b < sv.length) {
    const next = Math.min(cu, cv)
    if (next > t) {
      iu.push(su[a] ?? 0)
      iv.push(sv[b] ?? 0)
      dt.push(next - t)
      t = next
    }
    if (cu <= next + 1e-15 && a < su.length) {
      a += 1
      cu += uw[su[a] ?? 0] ?? 0
    }
    if (cv <= next + 1e-15 && b < sv.length) {
      b += 1
      cv += vw[sv[b] ?? 0] ?? 0
    }
  }
  const idx = (xs: number[]): Tensor => fromStorage(Int32Array.from(xs), [xs.length], 'int32')
  const diff = abs(sub(take(u, idx(iu)), take(v, idx(iv))))
  return sum(mul(fromStorage(Float64Array.from(dt), [dt.length], 'float64'), p === 1 ? diff : pow(diff, p)))
}

/**
 * The p-Wasserstein distance between two weighted samples on the real line (scipy.stats.wasserstein_distance for
 * p = 1; POT's ot.wasserstein_1d to the power 1/p), exactly, by merging the quantile functions. Differentiable in the
 * sample positions.
 *
 * @param u - Samples of the first distribution `[n]`.
 * @param v - Samples of the second `[k]`.
 * @param options - Weights and the order p.
 * @returns W_p (0-d).
 * @concept optimal-transport
 * @example
 * ```ts
 * import { wasserstein1d } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(wasserstein1d([0, 1, 3], [5, 6, 8])) // => 5
 * ```
 */
export function wasserstein1d(u: TensorLike, v: TensorLike, options: Wasserstein1dOptions = {}): Value {
  const [uv, vv] = promoteFloat(u, v) as [Value, Value]
  const p = options.p ?? 1
  if (!(p >= 1)) throw new WhiteboxError('value', `wasserstein1d: p must be at least 1, got ${p}`)
  const n = uv.shape[0] ?? 0
  const k = vv.shape[0] ?? 0
  const power = wassersteinPower(
    uv,
    vv,
    normalisedWeights(options.uWeights, n, 'wasserstein1d'),
    normalisedWeights(options.vWeights, k, 'wasserstein1d'),
    p,
  )
  return p === 1 ? power : pow(power, 1 / p)
}

/** Options of {@link slicedWasserstein}. */
export interface SlicedOptions {
  /** Weights of the two point clouds (default uniform). */
  readonly a?: TensorLike
  readonly b?: TensorLike
  /** The projection directions as columns `[d, L]`; otherwise `nProjections` random unit directions from `key`. */
  readonly projections?: TensorLike
  readonly nProjections?: number
  readonly key?: Key
  /** The order p (default 2). */
  readonly p?: number
}

/**
 * The sliced Wasserstein distance (Rabin et al. 2011; POT's ot.sliced_wasserstein_distance): the p-th root of the
 * mean over directions of W_p^p between the projected clouds, with explicit directions or random ones from a key.
 * Differentiable in the point positions.
 *
 * @param x - The first cloud `[n, d]`.
 * @param y - The second cloud `[k, d]`.
 * @param options - Weights, projections (or their number and key) and the order.
 * @returns The distance (0-d).
 * @concept optimal-transport
 * @example
 * ```ts
 * import { slicedWasserstein } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(slicedWasserstein([[0, 0], [1, 0]], [[0, 2], [1, 2]], { projections: [[0], [1]] })) // => 2
 * ```
 */
export function slicedWasserstein(x: TensorLike, y: TensorLike, options: SlicedOptions = {}): Value {
  const [xv, yv] = promoteFloat(x, y) as [Value, Value]
  if (xv.shape.length !== 2 || yv.shape.length !== 2 || xv.shape[1] !== yv.shape[1])
    throw new WhiteboxError('shape', 'slicedWasserstein: the clouds must be [n, d] and [k, d]')
  const d = xv.shape[1] ?? 0
  const p = options.p ?? 2
  let theta: Value
  if (options.projections !== undefined) theta = promoteFloat(options.projections)[0] as Value
  else {
    const raw = normal(options.key ?? makeKey(0), [d, options.nProjections ?? 50])
    theta = div(raw, norm(raw, 2, 0, true))
  }
  const px = matmul(xv, theta)
  const py = matmul(yv, theta)
  const uw = normalisedWeights(options.a, xv.shape[0] ?? 0, 'slicedWasserstein')
  const vw = normalisedWeights(options.b, yv.shape[0] ?? 0, 'slicedWasserstein')
  const L = theta.shape[1] ?? 0
  const columnOf = (m: Value, j: number): Value => take(m, fromStorage(new Int32Array([j]), [1], 'int32'), 1)
  const powers = Array.from({ length: L }, (_, j) =>
    wassersteinPower(reshapeColumn(columnOf(px, j)), reshapeColumn(columnOf(py, j)), uw, vw, p),
  )
  return pow(mean(stack(powers)), 1 / p)
}

const reshapeColumn = (c: Value): Value => reshape(c, [-1])

/** Options of {@link sinkhornBarycenter}. */
export interface BarycenterOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Barycentric weights (default uniform). */
  readonly weights?: ArrayLike<number>
  /** Stop once sum_j std_k(the k-th plan's barycentre marginal at j) falls to this (default 1e-9). */
  readonly tol?: number
  readonly maxIterations?: number
}

/** State of {@link sinkhornBarycenterAlgorithm}: the log-scalings, one column per input histogram. */
export interface BarycenterState {
  readonly logU: Vec
  readonly logMarginals: Vec
  readonly spread: number
}

/**
 * Fixed-support entropic Wasserstein barycentre by iterative Bregman projections (Benamou et al. 2015; POT's
 * ot.bregman.barycenter), in the log domain, as a traced algorithm recording the spread of the per-input marginals.
 *
 * @param histograms - The inputs as columns `[n, N]`.
 * @param cost - The ground cost `[n, n]` on the common support.
 * @param reg - The entropic regularisation.
 * @param options - Weights and tolerance.
 * @returns The algorithm.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { sinkhornBarycenterAlgorithm } from 'whitebox-engine/optim'
 * sinkhornBarycenterAlgorithm([[0.5, 0.1], [0.5, 0.9]], [[0, 1], [1, 0]], 0.1).name // => 'sinkhornBarycenter'
 * ```
 */
export function sinkhornBarycenterAlgorithm(
  histograms: TensorLike,
  cost: TensorLike,
  reg: number,
  options: BarycenterOptions = {},
): Algorithm<BarycenterState> {
  const A = primalOf(promoteFloat(histograms)[0] as Value) as Tensor
  const M = primalOf(promoteFloat(cost)[0] as Value) as Tensor
  const n = A.shape[0] ?? 0
  const N = A.shape[1] ?? 0
  if (M.shape[0] !== n || M.shape[1] !== n) throw new WhiteboxError('shape', 'sinkhornBarycenter: cost must be [n, n]')
  const w = Float64Array.from(options.weights ?? new Float64Array(N).fill(1 / N))
  const tol = options.tol ?? 1e-9
  const logA = Float64Array.from(A.data, (v) => Math.log(v))
  const logK = Float64Array.from(M.data, (v) => -v / reg)
  const at = (x: Vec, i: number, k: number): number => x[i * N + k] ?? 0
  const marginals = (logU: Vec): Vec => {
    // log(u_k * K^T (a_k / (K u_k))) for each k, stored n x N.
    const out = new Float64Array(n * N)
    const row = new Float64Array(n)
    for (let k = 0; k < N; k++) {
      const logV = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) row[j] = (logK[i * n + j] ?? 0) + at(logU, j, k)
        logV[i] = at(logA, i, k) - logSumExpOf(row)
      }
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) row[i] = (logK[i * n + j] ?? 0) + (logV[i] ?? 0)
        out[j * N + k] = at(logU, j, k) + logSumExpOf(row)
      }
    }
    return out
  }
  const spreadOf = (logM: Vec): number => {
    let s = 0
    for (let j = 0; j < n; j++) {
      let m = 0
      for (let k = 0; k < N; k++) m += Math.exp(at(logM, j, k)) / N
      let v = 0
      for (let k = 0; k < N; k++) v += (Math.exp(at(logM, j, k)) - m) ** 2 / N
      s += Math.sqrt(v)
    }
    return s
  }
  return {
    name: 'sinkhornBarycenter',
    config: { reg, tol, weights: Array.from(w) } as Json,
    schema: channels({
      metrics: { spread: { description: 'sum over the support of the std of the inputs’ marginals' } },
      tensors: { barycenter: { role: 'output', description: 'current barycentre' } },
    }),
    init: () => {
      const logU = new Float64Array(n * N)
      const logM = marginals(logU)
      return { logU, logMarginals: logM, spread: spreadOf(logM) }
    },
    step: (s) => {
      const logU = Float64Array.from(s.logU)
      for (let j = 0; j < n; j++) {
        let bar = 0
        for (let k = 0; k < N; k++) bar += (w[k] ?? 0) * at(s.logMarginals, j, k)
        for (let k = 0; k < N; k++) logU[j * N + k] = at(s.logU, j, k) + bar - at(s.logMarginals, j, k)
      }
      const logM = marginals(logU)
      return { logU, logMarginals: logM, spread: spreadOf(logM) }
    },
    done: (s) => (s.spread <= tol ? 'converged' : Number.isNaN(s.spread) ? 'diverged' : undefined),
    observe: (s) => ({
      metrics: { spread: s.spread },
      tensors: { barycenter: { role: 'output', value: barycentreOf(s, w, n, N) } },
    }),
  }
}

function barycentreOf(s: BarycenterState, w: Vec, n: number, N: number): Vec {
  return Float64Array.from({ length: n }, (_, j) => {
    let bar = 0
    for (let k = 0; k < N; k++) bar += (w[k] ?? 0) * (s.logMarginals[j * N + k] ?? 0)
    return Math.exp(bar)
  })
}

/** Result of the barycentre solvers. */
export interface BarycenterResult {
  readonly barycenter: Tensor
  readonly iterations: number
  readonly status: RunStatus
  readonly converged: boolean
}

/**
 * The entropic Wasserstein barycentre of histograms on a common support.
 *
 * @param histograms - The inputs as columns `[n, N]`.
 * @param cost - The ground cost `[n, n]`.
 * @param reg - The entropic regularisation.
 * @param options - Weights, tolerance, iteration cap (default 10000) and run options.
 * @returns The barycentre `[n]`.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { sinkhornBarycenter } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * // Two identical inputs: the barycentre is (close to) the input.
 * toArray(sinkhornBarycenter([[0.3, 0.3], [0.7, 0.7]], [[0, 1], [1, 0]], 0.05).barycenter).map((v) => Math.round(v * 1e6) / 1e6) // => [0.3, 0.7]
 * ```
 */
export function sinkhornBarycenter(
  histograms: TensorLike,
  cost: TensorLike,
  reg: number,
  options: BarycenterOptions = {},
): BarycenterResult {
  const { weights, tol: _t, maxIterations, ...run } = options
  const alg = sinkhornBarycenterAlgorithm(histograms, cost, reg, options)
  const result = runAlgorithm(alg, { ...run, maxSteps: maxIterations ?? 10000 })
  const A = primalOf(promoteFloat(histograms)[0] as Value)
  const n = A.shape[0] ?? 0
  const N = A.shape[1] ?? 0
  const w = Float64Array.from(weights ?? new Float64Array(N).fill(1 / N))
  return {
    barycenter: fromStorage(barycentreOf(result.state, w, n, N), [n], 'float64'),
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
  }
}

/** Options of {@link freeSupportBarycenter}. */
export interface FreeSupportOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The barycentre's weights (default uniform). */
  readonly b?: ArrayLike<number>
  /** Barycentric weights of the inputs (default uniform). */
  readonly weights?: ArrayLike<number>
  /** Stop once the squared displacement of the support falls to this (default 1e-7, POT's). */
  readonly tol?: number
  readonly maxIterations?: number
}

/** State of {@link freeSupportBarycenterAlgorithm}. */
export interface FreeSupportState {
  readonly x: Vec
  readonly displacement: number
}

/**
 * The free-support Wasserstein barycentre (Cuturi and Doucet 2014; POT's ot.lp.free_support_barycenter) as a traced
 * algorithm: with the barycentre's weights fixed, alternately solve exact transport from the current support to each
 * input (squared Euclidean costs) and move each support point to its plans' weighted mean.
 *
 * @param locations - The inputs' supports, each `[k_i, d]`.
 * @param masses - The inputs' weights, each `[k_i]`.
 * @param init - The initial support `[k, d]`.
 * @param options - Barycentre weights, input weights and tolerance.
 * @returns The algorithm.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { freeSupportBarycenterAlgorithm } from 'whitebox-engine/optim'
 * freeSupportBarycenterAlgorithm([[[0]], [[2]]], [[1], [1]], [[5]]).name // => 'freeSupportBarycenter'
 * ```
 */
export function freeSupportBarycenterAlgorithm(
  locations: readonly TensorLike[],
  masses: readonly TensorLike[],
  init: TensorLike,
  options: FreeSupportOptions = {},
): Algorithm<FreeSupportState> {
  const X0 = primalOf(promoteFloat(init)[0] as Value) as Tensor
  const k = X0.shape[0] ?? 0
  const d = X0.shape[1] ?? 0
  const inputs = locations.map((l) => primalOf(promoteFloat(l)[0] as Value) as Tensor)
  const weights = masses.map((m) => Float64Array.from(toFlat(primalOf(asTensor(m)))))
  const N = inputs.length
  const b = Float64Array.from(options.b ?? new Float64Array(k).fill(1 / k))
  const lambdas = Float64Array.from(options.weights ?? new Float64Array(N).fill(1 / N))
  const tol = options.tol ?? 1e-7
  return {
    name: 'freeSupportBarycenter',
    config: { k, d, inputs: N, tol } as Json,
    schema: channels({
      metrics: { displacement: { description: 'squared displacement of the support', direction: 'lower' } },
      tensors: { x: { role: 'param', description: 'barycentre support' } },
    }),
    init: () => ({ x: Float64Array.from(X0.data), displacement: Infinity }),
    step: (s) => {
      const next = new Float64Array(k * d)
      for (const [t, Xi] of inputs.entries()) {
        const ki = Xi.shape[0] ?? 0
        const cost = new Float64Array(k * ki)
        for (let r = 0; r < k; r++)
          for (let c = 0; c < ki; c++) {
            let q = 0
            for (let e = 0; e < d; e++) q += ((s.x[r * d + e] ?? 0) - ((Xi.data[c * d + e] as number) ?? 0)) ** 2
            cost[r * ki + c] = q
          }
        const plan = toFlat(emd(b, weights[t] as Vec, fromStorage(cost, [k, ki], 'float64')).plan)
        for (let r = 0; r < k; r++)
          for (let e = 0; e < d; e++) {
            let q = 0
            for (let c = 0; c < ki; c++) q += (plan[r * ki + c] ?? 0) * ((Xi.data[c * d + e] as number) ?? 0)
            next[r * d + e] = (next[r * d + e] ?? 0) + ((lambdas[t] ?? 0) * q) / (b[r] ?? 1)
          }
      }
      let displacement = 0
      for (let q = 0; q < next.length; q++) displacement += ((next[q] ?? 0) - (s.x[q] ?? 0)) ** 2
      return { x: next, displacement }
    },
    done: (s) => (s.displacement <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { displacement: s.displacement },
      tensors: { x: { role: 'param', value: s.x, shape: [k, d] } },
    }),
  }
}

/**
 * The free-support Wasserstein barycentre of point clouds.
 *
 * @param locations - The inputs' supports, each `[k_i, d]`.
 * @param masses - The inputs' weights, each `[k_i]` summing to one.
 * @param init - The initial support `[k, d]`.
 * @param options - Barycentre weights, input weights, tolerance, iteration cap (default 100, POT's) and run options.
 * @returns The barycentre's support `[k, d]` with the run's status.
 * @concept optimal-transport
 * @example
 * ```ts
 * import { freeSupportBarycenter } from 'whitebox-engine/optim'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(freeSupportBarycenter([[[0]], [[2]]], [[1], [1]], [[5]]).support) // => [[1]]
 * ```
 */
export function freeSupportBarycenter(
  locations: readonly TensorLike[],
  masses: readonly TensorLike[],
  init: TensorLike,
  options: FreeSupportOptions = {},
): { readonly support: Tensor; readonly iterations: number; readonly converged: boolean } {
  if (locations.length !== masses.length || locations.length === 0)
    throw new WhiteboxError('value', 'freeSupportBarycenter: give matching locations and masses')
  const { b: _b, weights: _w, tol: _t, maxIterations, ...run } = options
  const X0 = primalOf(promoteFloat(init)[0] as Value)
  const result = runAlgorithm(freeSupportBarycenterAlgorithm(locations, masses, init, options), {
    ...run,
    maxSteps: maxIterations ?? 100,
  })
  return {
    support: fromStorage(result.state.x, X0.shape, 'float64'),
    iterations: result.steps,
    converged: result.status === 'converged',
  }
}
