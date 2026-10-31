/**
 * SWAG, stochastic weight averaging Gaussian (Maddox et al. 2019): collect the iterates of constant-step-size SGD late in
 * training into their running mean (the SWA solution), their running second moment (a diagonal covariance) and the
 * last few deviations from the mean (a low-rank covariance), and sample networks from the Gaussian with that mean and
 * the average of the diagonal and low-rank covariances. The moments are plain functions of flat parameter vectors, the
 * collection over a training run is an `Algorithm`, and the predictive is the average of the network over samples.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../random/index.ts'
import { add, mul, ravel, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, type StepContext } from '../trace/index.ts'
import type { Params } from './module/module.ts'

/** The moments SWAG keeps: how many iterates, their mean and mean square, and the last deviations from the mean. */
export interface SwagState {
  /** The number of iterates collected. */
  readonly count: number
  /** The maximum number of deviation columns kept (the rank of the low-rank covariance). */
  readonly rank: number
  /** The running mean of the iterates, the SWA solution. */
  readonly mean: readonly number[]
  /** The running mean of the squared iterates. */
  readonly squares: readonly number[]
  /** The deviations `theta_i - mean_i` (taken against the running mean at that time) of the last `rank` iterates, oldest first. */
  readonly deviations: readonly (readonly number[])[]
}

/**
 * An empty SWAG state for `dim` parameters.
 *
 * @param dim - The number of parameters.
 * @param rank - The number of deviation columns to keep (default 20).
 * @returns The state.
 * @example
 * ```ts
 * import { swagState } from 'mdml-engine/nn'
 * swagState(3, 5).count // => 0
 * ```
 */
export function swagState(dim: number, rank = 20): SwagState {
  if (!Number.isInteger(rank) || rank < 1)
    throw new MdmlError('value', `swagState: rank must be a positive integer, got ${rank}`)
  return {
    count: 0,
    rank,
    mean: new Array<number>(dim).fill(0),
    squares: new Array<number>(dim).fill(0),
    deviations: [],
  }
}

/**
 * Collect one iterate: update the running mean and mean square and push the deviation from the new mean, dropping
 * the oldest column beyond the rank.
 *
 * @param state - The state.
 * @param theta - The flat parameter vector.
 * @returns The new state.
 * @example
 * ```ts
 * import { swagCollect, swagState } from 'mdml-engine/nn'
 * const s = swagCollect(swagCollect(swagState(2), [1, 2]), [3, 2])
 * s.mean // => [2, 2]
 * ```
 */
export function swagCollect(state: SwagState, theta: ArrayLike<number>): SwagState {
  const d = state.mean.length
  if (theta.length !== d) throw new MdmlError('shape', `swagCollect: expected ${d} parameters, got ${theta.length}`)
  const n = state.count
  const mean = state.mean.map((m, i) => (n * m + (theta[i] as number)) / (n + 1))
  const squares = state.squares.map((m, i) => (n * m + (theta[i] as number) ** 2) / (n + 1))
  const deviation = mean.map((m, i) => (theta[i] as number) - m)
  const deviations = [...state.deviations, deviation].slice(-state.rank)
  return { ...state, count: n + 1, mean, squares, deviations }
}

/**
 * The diagonal of the covariance, `mean(theta^2) - mean(theta)^2`, floored at a tiny positive number.
 *
 * @param state - The state.
 * @returns One variance per parameter.
 * @example
 * ```ts
 * import { swagCollect, swagState, swagVariance } from 'mdml-engine/nn'
 * swagVariance(swagCollect(swagCollect(swagState(1), [1]), [3])) // => [1]
 * ```
 */
export function swagVariance(state: SwagState): number[] {
  return state.squares.map((q, i) => Math.max(q - (state.mean[i] as number) ** 2, 1e-30))
}

/** Which covariance to sample from. */
export type SwagMode = 'swag' | 'diagonal'

/** Options of the SWAG samplers. */
export interface SwagSampleOptions {
  /** `swag`: the average of the diagonal and low-rank covariances (default); `diagonal`: the diagonal only (SWAG-Diag). */
  readonly mode?: SwagMode
  /** A multiplier of the covariance (default 1: the covariance of the paper; 0 is the SWA point). */
  readonly scale?: number
}

/**
 * The dense covariance of the SWAG Gaussian, for small models: `scale (diag + D D^T / (K - 1)) / 2` (the paper's
 * average of the diagonal and low-rank estimates) or `scale diag` in the diagonal mode.
 *
 * @param state - The state.
 * @param options - The mode and scale.
 * @returns `[d, d]`.
 * @example
 * ```ts
 * import { swagCollect, swagCovariance, swagState } from 'mdml-engine/nn'
 * import { toArray } from 'mdml-engine/tensor'
 * const s = [[0], [2]].reduce(swagCollect, swagState(1, 5))
 * toArray(swagCovariance(s)) // => [[1]]
 * ```
 */
export function swagCovariance(state: SwagState, options: SwagSampleOptions = {}): Tensor {
  const d = state.mean.length
  const scale = options.scale ?? 1
  const variance = swagVariance(state)
  const k = state.deviations.length
  const out = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => {
      const diag = i === j ? (variance[i] as number) : 0
      if ((options.mode ?? 'swag') === 'diagonal') return scale * diag
      let low = 0
      for (const col of state.deviations) low += (col[i] as number) * (col[j] as number)
      return (scale * (diag + (k > 1 ? low / (k - 1) : 0))) / 2
    }),
  )
  return tensor(out)
}

/**
 * Sample a flat parameter vector from the SWAG Gaussian (Maddox et al. 2019, algorithm 1):
 * `mean + sqrt(scale / 2) (diag^{1/2} z1 + D z2 / sqrt(K - 1))` with standard normal `z1` (one per parameter) and `z2` (one
 * per deviation column), or `mean + sqrt(scale) diag^{1/2} z1` in the diagonal mode.
 *
 * @param state - The state (at least two columns for the low-rank part; with fewer the sample is diagonal).
 * @param key - The key.
 * @param options - The mode and scale.
 * @returns The sample.
 * @example
 * ```ts
 * import { swagSample, swagState } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * swagSample(swagState(2), key(0), { scale: 0 }) // => [0, 0]
 * ```
 */
export function swagSample(state: SwagState, key: Key, options: SwagSampleOptions = {}): number[] {
  const d = state.mean.length
  const scale = options.scale ?? 1
  const mode = options.mode ?? 'swag'
  const sd = swagVariance(state).map(Math.sqrt)
  const z1 = toFlat(normal(foldIn(key, 0), [d]))
  const k = state.deviations.length
  if (mode === 'diagonal' || k < 2) {
    const s = Math.sqrt(scale)
    return state.mean.map((m, i) => m + s * (sd[i] as number) * (z1[i] as number))
  }
  const z2 = toFlat(normal(foldIn(key, 1), [k]))
  const half = Math.sqrt(scale / 2)
  const low = Math.sqrt(1 / (k - 1))
  return state.mean.map((m, i) => {
    let cov = 0
    for (const [c, col] of state.deviations.entries()) cov += (col[i] as number) * (z2[c] as number)
    return m + half * ((sd[i] as number) * (z1[i] as number) + low * cov)
  })
}

/** The paths of {@link swagAlgorithm}. */
export type SwagPath = 'swag/collected' | 'swag/rank' | 'swag/mean' | 'swag/deviations'

/** A training trajectory SWAG collects from: any optimiser that turns parameters into the next parameters. */
export interface SwagTrajectory {
  /** The initial parameters. */
  init(key: Key): Params
  /** One optimiser step (constant step size, so the iterates explore the loss basin). */
  step(params: Params, ctx: StepContext): Params
}

/** Options of {@link swagAlgorithm}. */
export interface SwagAlgorithmOptions {
  /** Steps before collection starts. */
  readonly burnIn: number
  /** Collect every this many steps after the burn-in (default 1). */
  readonly collectEvery?: number
  /** The number of deviation columns kept (default 20). */
  readonly rank?: number
}

/** The state of {@link swagAlgorithm}. */
export interface SwagRunState {
  readonly params: Params
  readonly swag: SwagState
}

/**
 * Collect SWAG moments along a training trajectory: each step advances the optimiser, and after the burn-in every
 * `collectEvery` steps the iterate goes into the moments. The mean, the deviation matrix and the number collected are
 * on the tape; `unravel` of the result (see {@link swagUnravel}) turns a sample back into a parameter tree.
 *
 * @param trajectory - The optimiser: initial parameters and one step.
 * @param options - Burn-in, collection interval and rank.
 * @returns The algorithm.
 * @reference Maddox, Izmailov, Garipov, Vetrov and Wilson (2019), A simple baseline for Bayesian uncertainty in deep learning.
 * @example
 * ```ts
 * import { swagAlgorithm } from 'mdml-engine/nn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { add, tensor, type Value } from 'mdml-engine/tensor'
 * const walk = { init: () => ({ w: tensor([0]) }), step: (p: { w: Value }) => ({ w: add(p.w, 1) }) }
 * runAlgorithm(swagAlgorithm(walk, { burnIn: 2 }), { maxSteps: 6 }).state.swag.count // => 4
 * ```
 */
export function swagAlgorithm(trajectory: SwagTrajectory, options: SwagAlgorithmOptions): Algorithm<SwagRunState> {
  const every = options.collectEvery ?? 1
  const rank = options.rank ?? 20
  const dimOf = (p: Params): number => ravel(p).flat.length
  return {
    name: 'swag',
    config: { burnIn: options.burnIn, collectEvery: every, rank },
    schema: channels({
      metrics: {
        'swag/collected': { description: 'iterates collected so far' },
        'swag/rank': { description: 'deviation columns held' },
      },
      tensors: {
        'swag/mean': { role: 'state', description: 'the SWA mean of the iterates' },
        'swag/deviations': { role: 'state', description: 'the deviation matrix, one column per kept iterate' },
      },
    }),
    init: (key) => {
      const params = trajectory.init(key)
      return { params, swag: swagState(dimOf(params), rank) }
    },
    step: (s, ctx) => {
      const params = trajectory.step(s.params, ctx)
      const collect = ctx.t >= options.burnIn && (ctx.t - options.burnIn) % every === 0
      return { params, swag: collect ? swagCollect(s.swag, ravel(params).flat) : s.swag }
    },
    observe: (s) => {
      const k = s.swag.deviations.length
      const d = s.swag.mean.length
      const matrix = new Float64Array(d * k)
      for (const [c, col] of s.swag.deviations.entries())
        for (let i = 0; i < d; i++) matrix[i * k + c] = col[i] as number
      return {
        metrics: { 'swag/collected': s.swag.count, 'swag/rank': k },
        tensors: {
          'swag/mean': { role: 'state', value: s.swag.mean },
          ...(k === 0 ? {} : { 'swag/deviations': { role: 'state' as const, value: matrix, shape: [d, k] } }),
        },
      }
    },
    serialise: (s) => ({
      params: Array.from(ravel(s.params).flat),
      count: s.swag.count,
      rank: s.swag.rank,
      mean: [...s.swag.mean],
      squares: [...s.swag.squares],
      deviations: s.swag.deviations.map((c) => [...c]),
    }),
    deserialise: (j) => {
      const o = j as {
        params: number[]
        count: number
        rank: number
        mean: number[]
        squares: number[]
        deviations: number[][]
      }
      // Initial parameters only supply the structure of the tree; any fixed key does.
      const template = ravel(trajectory.init(makeKey(0)))
      return {
        params: template.unravel(o.params),
        swag: { count: o.count, rank: o.rank, mean: o.mean, squares: o.squares, deviations: o.deviations },
      }
    },
  }
}

/**
 * The function that rebuilds a parameter tree from a flat vector (a SWAG sample or the mean), given a tree of the same
 * structure.
 *
 * @param template - A parameter tree.
 * @returns `unravel(flat)`.
 * @example
 * ```ts
 * import { swagUnravel } from 'mdml-engine/nn'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * toArray(swagUnravel({ w: tensor([0, 0]) })([3, 4]).w) // => [3, 4]
 * ```
 */
export function swagUnravel<T>(template: T): (flat: ArrayLike<number>) => T {
  return ravel(template).unravel
}

/** The Bayesian model average of a network under the SWAG posterior. */
export interface SwagPredictive {
  /** The average of the predictions over the sampled networks. */
  readonly mean: Value
  /** The prediction of each sampled network. */
  readonly samples: readonly Value[]
}

/**
 * The SWAG predictive: sample `count` parameter vectors, run the network with each, and average (for classifiers pass
 * a function that returns probabilities, so the average is of probabilities and not of logits).
 *
 * @param state - The SWAG state.
 * @param unravel - Rebuilds a parameter tree from a flat vector (see {@link swagUnravel}).
 * @param predict - The network as a function of the parameters.
 * @param key - The key.
 * @param count - The number of samples (default 30).
 * @param options - The mode and scale.
 * @returns The mean prediction and every sample's.
 * @example
 * ```ts
 * import { swagCollect, swagPredictive, swagState, swagUnravel } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { tensor, toArray } from 'mdml-engine/tensor'
 * const state = [[1], [3]].reduce(swagCollect, swagState(1, 5))
 * const template = { w: tensor([0]) }
 * toArray(swagPredictive(state, swagUnravel(template), (p) => p.w, key(0), 4, { scale: 0 }).mean) // => [2]
 * ```
 */
export function swagPredictive<P>(
  state: SwagState,
  unravel: (flat: ArrayLike<number>) => P,
  predict: (params: P) => Value,
  key: Key,
  count = 30,
  options: SwagSampleOptions = {},
): SwagPredictive {
  const samples = Array.from({ length: count }, (_, i) => predict(unravel(swagSample(state, foldIn(key, i), options))))
  const total = samples.reduce((a, v) => add(a, v))
  return { mean: mul(total, 1 / count), samples }
}
