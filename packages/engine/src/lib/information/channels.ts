import { type Json, logSumExpOf, maxOf, pairwiseSum, WhiteboxError } from '../base/index.ts'
import { expandDims, mul, type Tensor, type TensorLike, tensor, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { float, logOfBase } from './_shared.ts'
import { mutualInformation } from './discrete.ts'

/** A discrete memoryless channel given by its transition matrix W[x][y] = P(y | x) (rows sum to one). */
interface Channel {
  readonly w: Float64Array
  readonly nx: number
  readonly ny: number
}

function matrixOf(m: TensorLike, name: string): Channel {
  const v = float(m)
  if (v.shape.length !== 2) throw new WhiteboxError('shape', `${name}: expected a matrix [rows, columns]`)
  const [nx = 0, ny = 0] = v.shape
  return { w: Float64Array.from(toFlat(v)), nx, ny }
}

function channelOf(transition: TensorLike, name: string): Channel {
  const c = matrixOf(transition, name)
  for (let x = 0; x < c.nx; x++) {
    const row = c.w.subarray(x * c.ny, (x + 1) * c.ny)
    if (row.some((p) => !(p >= 0)) || Math.abs(pairwiseSum(row) - 1) > 1e-9)
      throw new WhiteboxError('value', `${name}: row ${x} of the transition matrix is not a distribution`)
  }
  return c
}

function distribution(p: TensorLike, n: number, name: string): Float64Array {
  const xs = Float64Array.from(toFlat(float(p)))
  if (xs.length !== n || xs.some((v) => !(v >= 0)) || Math.abs(pairwiseSum(xs) - 1) > 1e-9)
    throw new WhiteboxError('value', `${name}: expected a distribution over ${n} values`)
  return xs
}

/**
 * The mutual information I(X; Y) between the input and output of a channel, for an input distribution p(x) and
 * transition matrix W[x][y] = P(y | x). Differentiable in both, so it can also be maximised by gradient methods.
 *
 * @param input - The input distribution p(x).
 * @param transition - The transition matrix `[inputs, outputs]`.
 * @param options - The logarithm base (default e).
 * @returns I(X; Y).
 * @concept channel-capacity
 * @example
 * ```ts
 * import { channelMutualInformation } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * // A noiseless binary channel with uniform input carries one bit.
 * item(channelMutualInformation([0.5, 0.5], [[1, 0], [0, 1]], { base: 2 })) // => 1
 * ```
 */
export function channelMutualInformation(
  input: TensorLike,
  transition: TensorLike,
  options: { readonly base?: number } = {},
): Value {
  const p = float(input)
  const w = float(transition)
  if (p.shape.length !== 1 || w.shape.length !== 2 || w.shape[0] !== p.shape[0])
    throw new WhiteboxError('shape', 'channelMutualInformation: expected p [n] and W [n, m]')
  return mutualInformation(mul(expandDims(p, -1), w), options)
}

/** The output distribution q = p W and D(W_x || q) for each input x, in nats. */
function divergences(c: Channel, input: Float64Array): { readonly d: Float64Array; readonly output: Float64Array } {
  const output = new Float64Array(c.ny)
  for (let x = 0; x < c.nx; x++)
    for (let y = 0; y < c.ny; y++) output[y] = (output[y] ?? 0) + (input[x] ?? 0) * (c.w[x * c.ny + y] ?? 0)
  const d = new Float64Array(c.nx)
  for (let x = 0; x < c.nx; x++) {
    const terms: number[] = []
    for (let y = 0; y < c.ny; y++) {
      const wxy = c.w[x * c.ny + y] ?? 0
      if (wxy > 0) terms.push(wxy * Math.log(wxy / (output[y] ?? 0)))
    }
    d[x] = pairwiseSum(terms)
  }
  return { d, output }
}

/** The state of {@link blahutArimotoAlgorithm}. */
export interface BlahutArimotoState {
  /** The current input distribution. */
  readonly input: Float64Array
  /** Lower and upper bounds on the capacity, in nats. */
  readonly lower: number
  readonly upper: number
}

/**
 * The Blahut-Arimoto algorithm (1972) for the capacity of a discrete memoryless channel, as a traced algorithm: each
 * step reweights the input distribution by exp(D(W_x || q)), the divergence of each input's output distribution from
 * the current output distribution q. The bounds log sum p(x) exp(D_x) <= C <= max D_x close in on the capacity; the
 * run converges when they are within `tol`.
 *
 * @param transition - The transition matrix `[inputs, outputs]`.
 * @param options - Tolerance on the gap between the bounds, in nats (default 1e-10), and a starting input distribution
 *   (default uniform).
 * @returns The algorithm.
 * @concept channel-capacity
 * @example
 * ```ts
 * import { blahutArimotoAlgorithm } from 'whitebox-engine/information'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * runAlgorithm(blahutArimotoAlgorithm([[1, 0], [0, 1]]), { maxSteps: 100 }).status // => 'converged'
 * ```
 */
export function blahutArimotoAlgorithm(
  transition: TensorLike,
  options: { readonly tol?: number; readonly init?: TensorLike } = {},
): Algorithm<BlahutArimotoState> {
  const c = channelOf(transition, 'blahutArimotoAlgorithm')
  const tol = options.tol ?? 1e-10
  const start =
    options.init === undefined
      ? new Float64Array(c.nx).fill(1 / c.nx)
      : distribution(options.init, c.nx, 'blahutArimotoAlgorithm')
  const logWeights = (input: Float64Array, d: Float64Array): number[] =>
    Array.from(d, (v, x) => ((input[x] ?? 0) > 0 ? Math.log(input[x] ?? 0) + v : Number.NEGATIVE_INFINITY))
  const bounds = (input: Float64Array): BlahutArimotoState => {
    const { d } = divergences(c, input)
    return { input, lower: logSumExpOf(logWeights(input, d)), upper: maxOf(d) }
  }
  return {
    name: 'blahutArimoto',
    config: { inputs: c.nx, outputs: c.ny, tol } as Json,
    schema: channels({
      metrics: {
        lower: { description: 'lower bound on the capacity', unit: 'nats' },
        upper: { description: 'upper bound on the capacity', unit: 'nats' },
        gap: { description: 'upper minus lower bound', unit: 'nats', direction: 'lower' },
      },
      tensors: { input: { role: 'state', description: 'the input distribution' } },
    }),
    init: () => bounds(start),
    step: (s) => {
      const w = logWeights(s.input, divergences(c, s.input).d)
      const z = logSumExpOf(w)
      return bounds(Float64Array.from(w, (v) => Math.exp(v - z)))
    },
    done: (s) => (s.upper - s.lower < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { lower: s.lower, upper: s.upper, gap: s.upper - s.lower },
      tensors: { input: { role: 'state', value: s.input } },
    }),
  }
}

/** The capacity of a channel and the input distribution that achieves it. */
export interface CapacityResult {
  readonly capacity: number
  /** A capacity-achieving input distribution. */
  readonly input: Tensor
  /** The output distribution it induces (unique). */
  readonly output: Tensor
  readonly iterations: number
  /** Final lower and upper bounds, in the chosen base. */
  readonly lower: number
  readonly upper: number
}

/** Options of the iterative channel and source computations. */
export type BlahutOptions = { readonly tol?: number; readonly maxIter?: number; readonly base?: number } & Partial<
  Omit<RunOptions, 'maxSteps' | 'start'>
>

/**
 * The capacity C = max over p(x) of I(X; Y) of a discrete memoryless channel, by the traced Blahut-Arimoto algorithm.
 *
 * @param transition - The transition matrix `[inputs, outputs]`.
 * @param options - Tolerance on the gap (default 1e-10 nats), iteration cap (default 100000; convergence is slow when
 *   the optimal input leaves some symbols unused), base, and a tracer and policy.
 * @returns The capacity, the optimal input and output distributions and the bounds.
 * @throws WhiteboxError with code `convergence` when the bounds do not meet within the iteration cap.
 * @concept channel-capacity
 * @example
 * ```ts
 * import { channelCapacity } from 'whitebox-engine/information'
 * // A binary symmetric channel with crossover 0.11 carries 1 - h(0.11) = 0.5 bits per use.
 * Math.round(channelCapacity([[0.89, 0.11], [0.11, 0.89]], { base: 2 }).capacity * 1e3) / 1e3 // => 0.5
 * ```
 */
export function channelCapacity(transition: TensorLike, options: BlahutOptions = {}): CapacityResult {
  const { tol, maxIter, base, ...run } = options
  const c = channelOf(transition, 'channelCapacity')
  const result = runAlgorithm(blahutArimotoAlgorithm(transition, tol === undefined ? {} : { tol }), {
    ...run,
    maxSteps: maxIter ?? 100000,
  })
  if (result.status !== 'converged') throw new WhiteboxError('convergence', 'channelCapacity: the bounds did not meet')
  const scale = logOfBase(base)
  return {
    capacity: (result.state.lower + result.state.upper) / 2 / scale,
    input: tensor(Array.from(result.state.input)),
    output: tensor(Array.from(divergences(c, result.state.input).output)),
    iterations: result.steps,
    lower: result.state.lower / scale,
    upper: result.state.upper / scale,
  }
}

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
 * import { rateDistortionAlgorithm } from 'whitebox-engine/information'
 * import { runAlgorithm } from 'whitebox-engine/trace'
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
  if (!(beta > 0)) throw new WhiteboxError('value', 'rateDistortionAlgorithm: beta must be positive')
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
}

/**
 * The point of the rate-distortion function R(D) = min I(X; Xhat) subject to E d <= D with slope -beta, by the traced
 * Blahut algorithm.
 *
 * @param source - The source distribution p(x).
 * @param distortion - The distortion matrix `[x, xhat]`.
 * @param beta - The slope parameter.
 * @param options - Tolerance, iteration cap (default 10000), base for the rate, tracer and policy.
 * @returns The rate, distortion, test channel and reproduction distribution.
 * @throws WhiteboxError with code `convergence` when the iteration cap is reached.
 * @concept rate-distortion
 * @example
 * ```ts
 * import { binaryEntropy, rateDistortion } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
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
  if (result.status !== 'converged') throw new WhiteboxError('convergence', 'rateDistortion: no fixed point reached')
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
 * import { rateDistortionCurve } from 'whitebox-engine/information'
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

/** A named channel: its transition matrix and its capacity. */
export interface NamedChannel {
  readonly transition: Tensor
  readonly capacity: number
}

const h2 = (p: number): number => (p <= 0 || p >= 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log(1 - p))

function checkProbability(p: number, name: string): void {
  if (!(p >= 0 && p <= 1)) throw new WhiteboxError('value', `${name}: the probability must lie in [0, 1]`)
}

/**
 * The binary symmetric channel: each bit is flipped with probability p. Capacity 1 - h(p) bits, achieved by a uniform
 * input.
 *
 * @param p - The crossover probability.
 * @param options - The base of the capacity (default e).
 * @returns The transition matrix and capacity.
 * @concept channel-capacity
 * @example
 * ```ts
 * import { binarySymmetricChannel } from 'whitebox-engine/information'
 * binarySymmetricChannel(0.5, { base: 2 }).capacity // => 0
 * ```
 */
export function binarySymmetricChannel(p: number, options: { readonly base?: number } = {}): NamedChannel {
  checkProbability(p, 'binarySymmetricChannel')
  return {
    transition: tensor([
      [1 - p, p],
      [p, 1 - p],
    ]),
    capacity: (Math.LN2 - h2(p)) / logOfBase(options.base),
  }
}

/**
 * The binary erasure channel: each bit is erased with probability e (outputs 0, 1 and the erasure, in that order).
 * Capacity 1 - e bits.
 *
 * @param e - The erasure probability.
 * @param options - The base of the capacity (default e).
 * @returns The transition matrix `[2, 3]` and capacity.
 * @concept channel-capacity
 * @example
 * ```ts
 * import { binaryErasureChannel } from 'whitebox-engine/information'
 * binaryErasureChannel(0.25, { base: 2 }).capacity // => 0.75
 * ```
 */
export function binaryErasureChannel(e: number, options: { readonly base?: number } = {}): NamedChannel {
  checkProbability(e, 'binaryErasureChannel')
  return {
    transition: tensor([
      [1 - e, 0, e],
      [0, 1 - e, e],
    ]),
    capacity: ((1 - e) * Math.LN2) / logOfBase(options.base),
  }
}

/**
 * The Z channel: a 0 always arrives intact, a 1 turns into a 0 with probability p. Capacity
 * log2(1 + (1 - p) p^(p / (1 - p))) bits, achieved by a slightly biased input.
 *
 * @param p - The probability that a 1 is received as 0.
 * @param options - The base of the capacity (default e).
 * @returns The transition matrix and capacity.
 * @concept channel-capacity
 * @example
 * ```ts
 * import { zChannel } from 'whitebox-engine/information'
 * Math.round(zChannel(0.5, { base: 2 }).capacity * 1e6) / 1e6 // => 0.321928
 * ```
 */
export function zChannel(p: number, options: { readonly base?: number } = {}): NamedChannel {
  checkProbability(p, 'zChannel')
  const capacity = p >= 1 ? 0 : Math.log1p((1 - p) * p ** (p / (1 - p)))
  return {
    transition: tensor([
      [1, 0],
      [p, 1 - p],
    ]),
    capacity: capacity / logOfBase(options.base),
  }
}

/**
 * The capacity of the additive white Gaussian noise channel, (1/2) log(1 + SNR) per real channel use, or, with a
 * bandwidth W, the Shannon-Hartley rate W log(1 + SNR) per second.
 *
 * @param snr - The signal-to-noise power ratio P / N (not in decibels).
 * @param options - Bandwidth in hertz and the base (default e).
 * @returns The capacity.
 * @concept channel-capacity
 * @example
 * ```ts
 * import { gaussianCapacity } from 'whitebox-engine/information'
 * gaussianCapacity(3, { base: 2 }) // => 1
 * gaussianCapacity(3, { base: 2, bandwidth: 3000 }) // => 6000
 * ```
 */
export function gaussianCapacity(
  snr: number,
  options: { readonly bandwidth?: number; readonly base?: number } = {},
): number {
  if (!(snr >= 0)) throw new WhiteboxError('value', 'gaussianCapacity: snr must be non-negative')
  const perUse = Math.log1p(snr) / logOfBase(options.base)
  return options.bandwidth === undefined ? perUse / 2 : options.bandwidth * perUse
}

/** The allocation of water-filling. */
export interface WaterFillingResult {
  /** The power (or distortion) given to each channel (or component). */
  readonly allocation: readonly number[]
  /** The water level. */
  readonly level: number
  /** The total capacity (or rate). */
  readonly total: number
}

/**
 * Water-filling: the power allocation that maximises sum (1/2) log(1 + P_i / N_i) over parallel Gaussian channels
 * with noise powers N_i and total power P. Each channel gets max(0, nu - N_i), the water level nu set so the powers
 * sum to P (found exactly by sorting the noise levels).
 *
 * @param noise - Noise powers N_i > 0.
 * @param power - The total power P >= 0.
 * @param options - The base of the capacity (default e).
 * @returns The powers, the water level and the capacity.
 * @concept water-filling
 * @example
 * ```ts
 * import { waterFilling } from 'whitebox-engine/information'
 * waterFilling([1, 2, 4], 3).allocation // => [2, 1, 0]
 * ```
 */
export function waterFilling(
  noise: readonly number[],
  power: number,
  options: { readonly base?: number } = {},
): WaterFillingResult {
  if (noise.length === 0 || noise.some((n) => !(n > 0)) || !(power >= 0))
    throw new WhiteboxError('value', 'waterFilling: noise powers must be positive and the power non-negative')
  const sorted = [...noise].sort((a, b) => a - b)
  let level = 0
  for (let k = sorted.length; k >= 1; k--) {
    const candidate = (power + pairwiseSum(sorted.slice(0, k))) / k
    if (candidate > (sorted[k - 1] ?? 0) || k === 1) {
      level = candidate
      break
    }
  }
  const allocation = noise.map((n) => Math.max(0, level - n))
  const total = pairwiseSum(allocation.map((a, i) => 0.5 * Math.log1p(a / (noise[i] ?? 1)))) / logOfBase(options.base)
  return { allocation, level, total }
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
 * import { reverseWaterFilling } from 'whitebox-engine/information'
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
    throw new WhiteboxError('value', 'reverseWaterFilling: variances must be positive and 0 < D <= sum of variances')
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
 * import { binaryRateDistortion } from 'whitebox-engine/information'
 * binaryRateDistortion(0.5, 0, { base: 2 }) // => 1
 * ```
 */
export function binaryRateDistortion(p: number, distortion: number, options: { readonly base?: number } = {}): number {
  checkProbability(p, 'binaryRateDistortion')
  if (!(distortion >= 0)) throw new WhiteboxError('value', 'binaryRateDistortion: distortion must be non-negative')
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
 * import { gaussianRateDistortion } from 'whitebox-engine/information'
 * gaussianRateDistortion(4, 1, { base: 2 }) // => 1
 * ```
 */
export function gaussianRateDistortion(
  variance: number,
  distortion: number,
  options: { readonly base?: number } = {},
): number {
  if (!(variance > 0 && distortion > 0))
    throw new WhiteboxError('value', 'gaussianRateDistortion: arguments must be positive')
  return Math.max(0, 0.5 * Math.log(variance / distortion)) / logOfBase(options.base)
}
