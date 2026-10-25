/**
 * Rate-distortion theory, the least number of bits per symbol needed to describe a source within a given average
 * distortion: Blahut's algorithm for points on the rate-distortion curve of a discrete source (also as a traced
 * algorithm), and the closed forms for Bernoulli and Gaussian sources, including reverse water-filling over independent
 * Gaussian components.
 */

import { type Json, logSumExpOf, MdmlError, pairwiseSum } from '../base/index.ts'
import { type Tensor, type TensorLike, tensor } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { checkProbability, distribution, h2, matrixOf } from './_matrices.ts'
import { logOfBase } from './_shared.ts'
import type { BlahutOptions, WaterFillingResult } from './channels.ts'

/** The state of {@link rateDistortionAlgorithm}. */
export interface RateDistortionState {
  /** The reproduction distribution q(xhat). */
  readonly output: Float64Array
  /** The test channel Q(xhat | x), row-major `[x, xhat]`. */
  readonly conditional: Float64Array
  /** I(X; Xhat) in nats and the expected distortion. */
  readonly rate: number
  readonly distortion: number
  /** The largest change of q in the last step. */
  readonly change: number
}

/**
 * Blahut's algorithm (1972) for a point on the rate-distortion curve of a discrete source, as a traced algorithm. For a
 * slope parameter beta > 0 it alternates Q(xhat | x) proportional to q(xhat) exp(-beta d(x, xhat)) with
 * q(xhat) = sum p(x) Q(xhat | x); the fixed point minimises I(X; Xhat) + beta E d(X, Xhat), giving the point of the
 * curve R(D) with slope -beta.
 *
 * @param source - The source distribution p(x).
 * @param distortion - The distortion matrix d `[x, xhat]`.
 * @param beta - The slope parameter (larger: less distortion, more rate).
 * @param options - Tolerance on the change of q (default 1e-12).
 * @returns The algorithm.
 * @concept rate-distortion
 * @example
 * ```ts
 * import { rateDistortionAlgorithm } from 'mdml-engine/information'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const hamming = [[0, 1], [1, 0]]
 * runAlgorithm(rateDistortionAlgorithm([0.5, 0.5], hamming, 2), { maxSteps: 500 }).status // => 'converged'
 * ```
 */
export function rateDistortionAlgorithm(
  source: TensorLike,
  distortion: TensorLike,
  beta: number,
  options: { readonly tol?: number } = {},
): Algorithm<RateDistortionState> {
  const d = matrixOf(distortion, 'rateDistortionAlgorithm')
  const p = distribution(source, d.nx, 'rateDistortionAlgorithm')
  if (!(beta > 0)) throw new MdmlError('value', 'rateDistortionAlgorithm: beta must be positive')
  const tol = options.tol ?? 1e-12
  const { nx, ny } = d
  const evaluate = (q: Float64Array, change: number): RateDistortionState => {
    const conditional = new Float64Array(nx * ny)
    for (let x = 0; x < nx; x++) {
      const logs = Array.from({ length: ny }, (_, j) =>
        (q[j] ?? 0) > 0 ? Math.log(q[j] ?? 0) - beta * (d.w[x * ny + j] ?? 0) : Number.NEGATIVE_INFINITY,
      )
      const z = logSumExpOf(logs)
      for (let j = 0; j < ny; j++) conditional[x * ny + j] = Math.exp((logs[j] ?? 0) - z)
    }
    const output = new Float64Array(ny)
    for (let x = 0; x < nx; x++)
      for (let j = 0; j < ny; j++) output[j] = (output[j] ?? 0) + (p[x] ?? 0) * (conditional[x * ny + j] ?? 0)
    const rateTerms: number[] = []
    const distortionTerms: number[] = []
    for (let x = 0; x < nx; x++)
      for (let j = 0; j < ny; j++) {
        const pq = (p[x] ?? 0) * (conditional[x * ny + j] ?? 0)
        if (pq > 0) rateTerms.push(pq * Math.log((conditional[x * ny + j] ?? 0) / (output[j] ?? 0)))
        distortionTerms.push(pq * (d.w[x * ny + j] ?? 0))
      }
    return { output, conditional, rate: pairwiseSum(rateTerms), distortion: pairwiseSum(distortionTerms), change }
  }
  return {
    name: 'blahutRateDistortion',
    config: { symbols: nx, reproductions: ny, beta, tol } as Json,
    schema: channels({
      metrics: {
        rate: { description: 'I(X; Xhat) of the current test channel', unit: 'nats' },
        distortion: { description: 'expected distortion of the current test channel' },
        change: { description: 'largest change of the reproduction distribution', direction: 'lower' },
      },
      tensors: { output: { role: 'state', description: 'the reproduction distribution q(xhat)' } },
    }),
    init: () => evaluate(new Float64Array(ny).fill(1 / ny), Number.POSITIVE_INFINITY),
    step: (s) => {
      const next = evaluate(s.output, 0)
      let change = 0
      for (let j = 0; j < ny; j++) change = Math.max(change, Math.abs((next.output[j] ?? 0) - (s.output[j] ?? 0)))
      return { ...next, change }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { rate: s.rate, distortion: s.distortion, change: s.change },
      tensors: { output: { role: 'state', value: s.output } },
    }),
  }
}

/** A point of the rate-distortion curve and its test channel. */
export interface RateDistortionPoint {
  readonly beta: number
  readonly rate: number
  readonly distortion: number
  /** The optimal test channel Q(xhat | x) `[x, xhat]`. */
  readonly conditional: Tensor
  /** The reproduction distribution q(xhat). */
  readonly output: Tensor
  readonly iterations: number
  /** Whether the reproduction distribution settled within the tolerance before the iteration cap. */
  readonly converged: boolean
}

/**
 * The point of the rate-distortion function R(D) = min I(X; Xhat) subject to E d <= D with slope -beta, by the traced
 * Blahut algorithm.
 *
 * @param source - The source distribution p(x).
 * @param distortion - The distortion matrix `[x, xhat]`.
 * @param beta - The slope parameter.
 * @param options - Tolerance, iteration cap (default 10000), base for the rate, tracer and policy.
 * @returns The rate, distortion, test channel, reproduction distribution and whether the iteration settled (the point
 *   is a valid upper bound on R at its distortion either way).
 * @concept rate-distortion
 * @example
 * ```ts
 * import { binaryEntropy, rateDistortion } from 'mdml-engine/information'
 * import { item } from 'mdml-engine/tensor'
 * // A fair bit under Hamming distortion: R(D) = 1 - h(D).
 * const point = rateDistortion([0.5, 0.5], [[0, 1], [1, 0]], 2, { base: 2 })
 * Math.abs(point.rate - (1 - item(binaryEntropy(point.distortion, { base: 2 })))) < 1e-9 // => true
 * ```
 */
export function rateDistortion(
  source: TensorLike,
  distortion: TensorLike,
  beta: number,
  options: BlahutOptions = {},
): RateDistortionPoint {
  const { tol, maxIter, base, ...run } = options
  const algorithm = rateDistortionAlgorithm(source, distortion, beta, tol === undefined ? {} : { tol })
  const result = runAlgorithm(algorithm, { ...run, maxSteps: maxIter ?? 10000 })
  const { ny } = matrixOf(distortion, 'rateDistortion')
  const s = result.state
  return {
    beta,
    rate: s.rate / logOfBase(base),
    distortion: s.distortion,
    conditional: tensor(
      Array.from({ length: s.conditional.length / ny }, (_, x) =>
        Array.from(s.conditional.subarray(x * ny, (x + 1) * ny)),
      ),
    ),
    output: tensor(Array.from(s.output)),
    iterations: result.steps,
    converged: result.status === 'converged',
  }
}

/**
 * Points of the rate-distortion curve for several slopes (each by {@link rateDistortion}).
 *
 * @param source - The source distribution p(x).
 * @param distortion - The distortion matrix `[x, xhat]`.
 * @param betas - Slope parameters.
 * @param options - As for {@link rateDistortion} (one tracer records every run).
 * @returns The points, in the order of `betas`.
 * @concept rate-distortion
 * @example
 * ```ts
 * import { rateDistortionCurve } from 'mdml-engine/information'
 * const curve = rateDistortionCurve([0.5, 0.5], [[0, 1], [1, 0]], [1, 2, 4])
 * curve.every((p, i) => i === 0 || p.rate > (curve[i - 1]?.rate ?? 0)) // => true
 * ```
 */
export function rateDistortionCurve(
  source: TensorLike,
  distortion: TensorLike,
  betas: readonly number[],
  options: BlahutOptions = {},
): RateDistortionPoint[] {
  return betas.map((beta) => rateDistortion(source, distortion, beta, options))
}

/**
 * Reverse water-filling: the rate-distortion function of independent Gaussian components with variances s_i under
 * total squared-error distortion D. Each component gets distortion min(lambda, s_i), with lambda set so that they
 * sum to D, and the rate is sum (1/2) log(s_i / D_i).
 *
 * @param variances - Component variances s_i > 0.
 * @param distortion - The total distortion D, between 0 and sum s_i.
 * @param options - The base of the rate (default e).
 * @returns The distortions, the level lambda and the rate.
 * @concept rate-distortion
 * @example
 * ```ts
 * import { reverseWaterFilling } from 'mdml-engine/information'
 * reverseWaterFilling([4, 1, 0.25], 1.25).allocation // => [0.5, 0.5, 0.25]
 * ```
 */
export function reverseWaterFilling(
  variances: readonly number[],
  distortion: number,
  options: { readonly base?: number } = {},
): WaterFillingResult {
  const total = pairwiseSum(variances)
  if (variances.length === 0 || variances.some((s) => !(s > 0)) || !(distortion > 0 && distortion <= total))
    throw new MdmlError('value', 'reverseWaterFilling: variances must be positive and 0 < D <= sum of variances')
  const sorted = [...variances].sort((a, b) => a - b)
  let level = sorted[sorted.length - 1] ?? 0
  let below = 0
  for (let k = 0; k < sorted.length; k++) {
    // The k smallest variances are fully distorted; the rest share the level.
    const candidate = (distortion - below) / (sorted.length - k)
    if (candidate <= (sorted[k] ?? 0)) {
      level = candidate
      break
    }
    below += sorted[k] ?? 0
  }
  const allocation = variances.map((s) => Math.min(level, s))
  const rate = pairwiseSum(variances.map((s, i) => 0.5 * Math.log(s / (allocation[i] ?? s)))) / logOfBase(options.base)
  return { allocation, level, total: rate }
}

/**
 * The rate-distortion function of a Bernoulli(p) source under Hamming distortion, R(D) = h(p) - h(D) for
 * D < min(p, 1 - p) and 0 beyond.
 *
 * @param p - The source probability.
 * @param distortion - The allowed bit-error rate D.
 * @param options - The base (default e).
 * @returns R(D).
 * @concept rate-distortion
 * @example
 * ```ts
 * import { binaryRateDistortion } from 'mdml-engine/information'
 * binaryRateDistortion(0.5, 0, { base: 2 }) // => 1
 * ```
 */
export function binaryRateDistortion(p: number, distortion: number, options: { readonly base?: number } = {}): number {
  checkProbability(p, 'binaryRateDistortion')
  if (!(distortion >= 0)) throw new MdmlError('value', 'binaryRateDistortion: distortion must be non-negative')
  return distortion >= Math.min(p, 1 - p) ? 0 : (h2(p) - h2(distortion)) / logOfBase(options.base)
}

/**
 * The rate-distortion function of a Gaussian source under squared error, R(D) = max(0, (1/2) log(variance / D)).
 *
 * @param variance - The source variance.
 * @param distortion - The allowed mean squared error D > 0.
 * @param options - The base (default e).
 * @returns R(D).
 * @concept rate-distortion
 * @example
 * ```ts
 * import { gaussianRateDistortion } from 'mdml-engine/information'
 * gaussianRateDistortion(4, 1, { base: 2 }) // => 1
 * ```
 */
export function gaussianRateDistortion(
  variance: number,
  distortion: number,
  options: { readonly base?: number } = {},
): number {
  if (!(variance > 0 && distortion > 0))
    throw new MdmlError('value', 'gaussianRateDistortion: arguments must be positive')
  return Math.max(0, 0.5 * Math.log(variance / distortion)) / logOfBase(options.base)
}
