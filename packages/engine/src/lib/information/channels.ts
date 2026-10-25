/**
 * Communication channels and their capacity, the highest rate at which information can be sent through them reliably:
 * the mutual information between a channel's input and output, the Blahut-Arimoto algorithm for the capacity of any
 * discrete memoryless channel (also as a traced algorithm), the binary symmetric, erasure and Z channels, the Gaussian
 * channel's Shannon-Hartley capacity, and water-filling power allocation over parallel Gaussian channels.
 */

import { type Json, logSumExpOf, maxOf, pairwiseSum, WhiteboxError } from '../base/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { expandDims, mul, type Tensor, type TensorLike, tensor, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { type Channel, channelOf, checkProbability, distribution, h2 } from './_matrices.ts'
import { float, logOfBase } from './_shared.ts'
import { mutualInformation } from './discrete.ts'

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

/** I(p) = sum p(x) D(W_x || q) in nats. */
function information(c: Channel, input: Float64Array): number {
  const { d } = divergences(c, input)
  return pairwiseSum(Array.from(d, (v, x) => (input[x] ?? 0) * v))
}

/** Divide by the sum, in place. */
function normalise(p: Float64Array): void {
  const total = pairwiseSum(p)
  for (let x = 0; x < p.length; x++) p[x] = (p[x] ?? 0) / total
}

/**
 * Newton steps for max I(p) on the face of the simplex where p is positive: the Karush-Kuhn-Tucker system
 * [H 1; 1' 0] [step; lambda] = [-D; 0] with the Hessian H(x, x') = -sum_y W(y | x) W(y | x') / q(y), a step that stops
 * at the boundary when a coordinate would turn negative (that symbol leaves the support), and backtracking until I
 * improves. Returns false when the system cannot be solved.
 */
function newtonOnSupport(c: Channel, p: Float64Array, support: number[]): boolean {
  for (let iteration = 0; iteration < 50 && support.length > 1; iteration++) {
    const { d, output } = divergences(c, p)
    const value = pairwiseSum(Array.from(d, (v, x) => (p[x] ?? 0) * v))
    const s = support.length
    const kkt = Array.from({ length: s + 1 }, (_, i) =>
      Array.from({ length: s + 1 }, (_, j) => {
        if (i === s || j === s) return i === s && j === s ? 0 : 1
        const a = support[i] ?? 0
        const b = support[j] ?? 0
        // A slight ridge keeps the system regular when rows repeat (the optimal input is then not unique).
        let h = i === j ? -1e-12 : 0
        for (let y = 0; y < c.ny; y++) {
          const qy = output[y] ?? 0
          if (qy > 0) h -= ((c.w[a * c.ny + y] ?? 0) * (c.w[b * c.ny + y] ?? 0)) / qy
        }
        return h
      }),
    )
    let step: number[]
    try {
      step = toFlat(solve(kkt, [...support.map((x) => -(d[x] ?? 0)), 0])).slice(0, s)
    } catch {
      return false
    }
    if (!step.every(Number.isFinite)) return false
    if (maxOf(step.map(Math.abs)) < 1e-15) return true
    let limit = 1
    let blocking = -1
    support.forEach((x, i) => {
      const di = step[i] ?? 0
      if (di < 0 && -(p[x] ?? 0) / di < limit) {
        limit = -(p[x] ?? 0) / di
        blocking = x
      }
    })
    let accepted: Float64Array | undefined
    for (let k = 0, t = limit; k < 40; k++, t /= 2) {
      const candidate = Float64Array.from(p)
      support.forEach((x, i) => {
        candidate[x] = Math.max(0, (p[x] ?? 0) + t * (step[i] ?? 0))
      })
      if (k === 0 && blocking >= 0) candidate[blocking] = 0
      normalise(candidate)
      if (information(c, candidate) >= value) {
        accepted = candidate
        break
      }
    }
    if (accepted === undefined) return true
    p.set(accepted)
    for (let i = support.length - 1; i >= 0; i--) if ((p[support[i] ?? 0] ?? 0) <= 0) support.splice(i, 1)
  }
  return true
}

/**
 * An active-set Newton refinement of an input distribution for the capacity problem: Newton steps on the current
 * support, then the optimality condition D(W_x || q) <= I(p) for the symbols outside it, bringing back the most
 * violating symbol until none violates. Blahut-Arimoto alone converges only sublinearly when the optimal input leaves
 * symbols unused (or nearly so); this converges quadratically once the support is right. Undefined when a Newton
 * system cannot be solved.
 */
function newtonRefine(c: Channel, start: Float64Array): Float64Array | undefined {
  const p = Float64Array.from(start)
  const largest = maxOf(p)
  for (let x = 0; x < c.nx; x++) if ((p[x] ?? 0) <= 1e-9 * largest) p[x] = 0
  normalise(p)
  const support = Array.from({ length: c.nx }, (_, x) => x).filter((x) => (p[x] ?? 0) > 0)
  for (let round = 0; round <= 2 * c.nx; round++) {
    if (!newtonOnSupport(c, p, support)) return undefined
    const { d } = divergences(c, p)
    const value = pairwiseSum(Array.from(d, (v, x) => (p[x] ?? 0) * v))
    let violator = -1
    let worst = value + 1e-13
    for (let x = 0; x < c.nx; x++) {
      if (!support.includes(x) && (d[x] ?? 0) > worst) {
        worst = d[x] ?? 0
        violator = x
      }
    }
    if (violator < 0) return p
    p[violator] = 1e-3
    normalise(p)
    support.push(violator)
    support.sort((a, b) => a - b)
  }
  return p
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
 * run converges when they are within `tol`. Every tenth step also tries a Newton refinement on the current support
 * (kept when it narrows the bounds), which fixes the sublinear convergence of the plain iteration when the optimal input
 * leaves some symbols unused.
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
    step: (s, ctx) => {
      const w = logWeights(s.input, divergences(c, s.input).d)
      const z = logSumExpOf(w)
      const next = bounds(Float64Array.from(w, (v) => Math.exp(v - z)))
      if (ctx.t % 10 !== 9 || next.upper - next.lower < tol) return next
      const refined = newtonRefine(c, next.input)
      if (refined === undefined) return next
      // Keep a trace of every symbol so that a wrongly dropped one can still grow back.
      const floored = Float64Array.from(refined, (v) => Math.max(v, 1e-300))
      const candidate = bounds(floored)
      return candidate.upper - candidate.lower < next.upper - next.lower ? candidate : next
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
  /** Whether the bounds met within the tolerance (when false, `lower` and `upper` still bracket the capacity). */
  readonly converged: boolean
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
 * @param options - Tolerance on the gap (default 1e-10 nats), iteration cap (default 10000), base, and a tracer and
 *   policy.
 * @returns The capacity (the midpoint of the bounds), the optimal input and output distributions, the bounds and
 *   whether they met within the tolerance.
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
    maxSteps: maxIter ?? 10000,
  })
  const scale = logOfBase(base)
  return {
    capacity: (result.state.lower + result.state.upper) / 2 / scale,
    input: tensor(Array.from(result.state.input)),
    output: tensor(Array.from(divergences(c, result.state.input).output)),
    iterations: result.steps,
    converged: result.status === 'converged',
    lower: result.state.lower / scale,
    upper: result.state.upper / scale,
  }
}

/** A named channel: its transition matrix and its capacity. */
export interface NamedChannel {
  readonly transition: Tensor
  readonly capacity: number
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
