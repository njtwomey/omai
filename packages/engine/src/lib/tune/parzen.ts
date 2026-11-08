/**
 * The Parzen estimators behind the tree-structured Parzen estimator (TPE): kernel density mixtures over a search space
 * with truncated normal kernels for numbers and categorical kernels for options, as Optuna builds them. Construction
 * with Optuna's and hyperopt's group sizes and weights, log densities, sampling, the leaves of a space, and TPE's
 * acquisition, the log density ratio of good to bad trials.
 */

import { logSumExpOf, OmaiError } from '../base/index.ts'
import { truncatedNormal } from '../dists/index.ts'
import { logIntervalProbability } from '../numerics/special/index.ts'
import { categorical, foldIn, type Key, uniform } from '../random/index.ts'
import type { Space } from '../space/index.ts'
import { asTensor, div, reshape, sub, toFlat, type Value } from '../tensor/index.ts'
import { type Leaf, leavesOf } from './_leaves.ts'

export type { Leaf } from './_leaves.ts'

const EPS = 1e-12

/**
 * Optuna's default gamma: the number of "good" trials, min(ceil(0.1 n), 25).
 *
 * @param n - Finished trials.
 * @returns Trials in the good group.
 * @example
 * ```ts
 * import { defaultGamma } from 'omai-engine/tune'
 * [defaultGamma(10), defaultGamma(11), defaultGamma(1000)] // => [1, 2, 25]
 * ```
 */
export function defaultGamma(n: number): number {
  return Math.min(Math.ceil(0.1 * n), 25)
}

/**
 * Optuna's gamma for several objectives: ceil(0.1 n), uncapped.
 *
 * @param n - Finished trials.
 * @returns Trials in the good group.
 * @example
 * ```ts
 * import { defaultGammaMultiObjective } from 'omai-engine/tune'
 * defaultGammaMultiObjective(1000) // => 100
 * ```
 */
export function defaultGammaMultiObjective(n: number): number {
  return Math.ceil(0.1 * n)
}

/**
 * hyperopt's gamma, min(ceil(0.25 sqrt(n)), 25).
 *
 * @param n - Finished trials.
 * @returns Trials in the good group.
 * @example
 * ```ts
 * import { hyperoptGamma } from 'omai-engine/tune'
 * hyperoptGamma(100) // => 3
 * ```
 */
export function hyperoptGamma(n: number): number {
  return Math.min(Math.ceil(0.25 * Math.sqrt(n)), 25)
}

/**
 * Optuna's default kernel weights of n observations (oldest first): all ones below 25, else a ramp from 1/n to 1 over
 * the oldest n - 25 and ones for the newest 25.
 *
 * @param n - Observations.
 * @returns The weights.
 * @example
 * ```ts
 * import { defaultWeights } from 'omai-engine/tune'
 * defaultWeights(3) // => [1, 1, 1]
 * defaultWeights(27).slice(0, 3) // => [0.037037, 1, 1]
 * ```
 */
export function defaultWeights(n: number): number[] {
  if (n === 0) return []
  if (n < 25) return new Array<number>(n).fill(1)
  const m = n - 25
  const ramp = Array.from({ length: m }, (_, i) => (m === 1 ? 1 / n : 1 / n + ((1 - 1 / n) * i) / (m - 1)))
  return [...ramp, ...new Array<number>(25).fill(1)]
}

/** Options of a Parzen estimator. */
export interface ParzenOptions {
  /** Weight of the prior kernel (default 1). */
  readonly priorWeight?: number
  /** Floor bandwidths at range / min(100, 1 + kernels) (default true). */
  readonly considerMagicClip?: boolean
  /** Let the outermost kernels reach the bounds (default false). */
  readonly considerEndpoints?: boolean
  /** Kernel weights for n observations (default {@link defaultWeights}). */
  readonly weights?: (n: number) => readonly number[]
  /** Kernel weights given directly (one per observation), instead of `weights`. */
  readonly predeterminedWeights?: readonly number[]
}

/** The kernels of one leaf: truncated normals (natural or log units), discretised on a step, or categorical rows. */
export type ParzenKernels =
  | {
      readonly kind: 'numerical'
      readonly mus: Float64Array
      readonly sigmas: Float64Array
      readonly low: number
      readonly high: number
      readonly step: number
      readonly log: boolean
    }
  | { readonly kind: 'categorical'; readonly probs: readonly Float64Array[] }

/** A mixture of products of per-leaf kernels, one component per observation plus the prior (Optuna's TPE). */
export interface ParzenEstimator {
  readonly leaves: readonly Leaf[]
  /** Normalised component weights. */
  readonly weights: Float64Array
  readonly kernels: readonly ParzenKernels[]
}

function stableArgsort(xs: ArrayLike<number>): number[] {
  return Array.from({ length: xs.length }, (_, i) => i).sort((a, b) => (xs[a] ?? 0) - (xs[b] ?? 0) || a - b)
}

function numericalKernels(
  leaf: Leaf,
  observations: readonly number[],
  o: Required<Omit<ParzenOptions, 'predeterminedWeights' | 'weights'>>,
): ParzenKernels {
  const step = leaf.kind === 'int' ? 1 : (leaf.step ?? 0)
  let low = leaf.low - step / 2
  let high = leaf.high + step / 2
  let mus = Float64Array.from(observations)
  if (leaf.log) {
    mus = mus.map(Math.log)
    low = Math.log(low)
    high = Math.log(high)
  }
  const n = mus.length
  const order = stableArgsort(mus)
  const ends = [low, ...order.map((i) => mus[i] as number), high]
  const sorted = Float64Array.from({ length: n }, (_, i) =>
    Math.max((ends[i + 1] as number) - (ends[i] as number), (ends[i + 2] as number) - (ends[i + 1] as number)),
  )
  if (!o.considerEndpoints && ends.length >= 4) {
    sorted[0] = (ends[2] as number) - (ends[1] as number)
    sorted[n - 1] = (ends.at(-2) as number) - (ends.at(-3) as number)
  }
  const maxSigma = high - low
  const minSigma = o.considerMagicClip ? (high - low) / Math.min(100, 1 + n + 1) : EPS
  const sigmas = new Float64Array(n + 1)
  for (const [rank, i] of order.entries()) sigmas[i] = Math.min(maxSigma, Math.max(minSigma, sorted[rank] as number))
  sigmas[n] = high - low
  const withPrior = new Float64Array(n + 1)
  withPrior.set(mus)
  withPrior[n] = 0.5 * (low + high)
  return { kind: 'numerical', mus: withPrior, sigmas, low, high, step, log: leaf.log }
}

function categoricalKernels(leaf: Leaf, observations: readonly number[], priorWeight: number): ParzenKernels {
  const n = leaf.n
  if (observations.length === 0) return { kind: 'categorical', probs: [new Float64Array(n).fill(1 / n)] }
  const k = observations.length + 1
  const probs = Array.from({ length: k }, (_, row) => {
    const w = new Float64Array(n).fill(priorWeight / k)
    if (row < observations.length) w[observations[row] as number] = (w[observations[row] as number] ?? 0) + 1
    let total = 0
    for (const v of w) total += v
    return total === 0 ? w : w.map((v) => v / total)
  })
  return { kind: 'categorical', probs }
}

/**
 * The Parzen estimator of Optuna's TPE over some leaves of a space: one product kernel per observation (truncated
 * normals for real and integer leaves, discretised on their step, in log units for log leaves; smoothed indicator rows
 * for options), plus a prior kernel (centred, as wide as the range, weight `priorWeight`). Bandwidths are the larger
 * gap to the neighbouring observations (the bounds count as neighbours), clipped to [range / min(100, kernels + 1),
 * range].
 *
 * @param leaves - The leaves the estimator covers.
 * @param observations - One row per observation, each the flat numbers of the leaves (option indices for options).
 * @param options - Prior weight, clipping, endpoints and weights.
 * @returns The estimator.
 * @throws OmaiError with code `value` for a negative prior weight or invalid weights.
 * @concept tree-structured-parzen-estimator
 * @reference Bergstra et al. (2011), Algorithms for hyper-parameter optimization; Watanabe (2023), Tree-structured
 *   Parzen estimator: understanding its algorithm components.
 * @example
 * ```ts
 * import { leavesOfSpace, parzenEstimator, parzenLogPdf } from 'omai-engine/tune'
 * import { real } from 'omai-engine/space'
 * const pe = parzenEstimator(leavesOfSpace(real(0, 1)), [[0.1], [0.15], [0.2]])
 * pe.weights.length // => 4
 * (parzenLogPdf(pe, [[0.15]])[0] ?? 0) > (parzenLogPdf(pe, [[0.9]])[0] ?? 0) // => true
 * ```
 */
export function parzenEstimator(
  leaves: readonly Leaf[],
  observations: readonly (readonly number[])[],
  options: ParzenOptions = {},
): ParzenEstimator {
  const priorWeight = options.priorWeight ?? 1
  if (!(priorWeight >= 0)) throw new OmaiError('value', `parzenEstimator: priorWeight must be non-negative`)
  const n = observations.length
  const raw = options.predeterminedWeights ?? (options.weights ?? defaultWeights)(n).slice(0, n)
  if (raw.length !== n) throw new OmaiError('value', `parzenEstimator: ${raw.length} weights for ${n} observations`)
  if (raw.some((w) => !(w >= 0) || !Number.isFinite(w)))
    throw new OmaiError('value', 'parzenEstimator: weights must be finite and non-negative')
  const w = n === 0 ? [1] : [...raw, priorWeight]
  const total = w.reduce((a, b) => a + b, 0)
  if (!(total > 0)) throw new OmaiError('value', 'parzenEstimator: the weights sum to zero')
  const o = {
    priorWeight,
    considerMagicClip: options.considerMagicClip ?? true,
    considerEndpoints: options.considerEndpoints ?? false,
  }
  const kernels = leaves.map((leaf, j) => {
    const column = observations.map((row) => row[j] as number)
    return leaf.kind === 'categorical'
      ? categoricalKernels(leaf, column, priorWeight)
      : numericalKernels(leaf, column, o)
  })
  return { leaves, weights: Float64Array.from(w, (v) => v / total), kernels }
}

function numericalLogPdf(k: Extract<ParzenKernels, { kind: 'numerical' }>, xs: readonly number[]): Value {
  const c = xs.length
  const tr = (x: number): number => (k.log ? Math.log(x) : x)
  if (k.step === 0) {
    const d = truncatedNormal(k.mus, k.sigmas, k.low, k.high)
    return d.logProb(reshape(asTensor(xs.map(tr)), [c, 1]))
  }
  const left = reshape(asTensor(xs.map((x) => tr(x - k.step / 2))), [c, 1])
  const right = reshape(asTensor(xs.map((x) => tr(x + k.step / 2))), [c, 1])
  const mass = logIntervalProbability(div(sub(left, k.mus), k.sigmas), div(sub(right, k.mus), k.sigmas))
  const z = logIntervalProbability(div(sub(k.low, k.mus), k.sigmas), div(sub(k.high, k.mus), k.sigmas))
  return sub(mass, z)
}

/**
 * The log density of a Parzen estimator at points (Optuna's `_ParzenEstimator.log_pdf`): log-sum-exp over components
 * of the log weight plus the per-leaf log kernels (log-unit densities for log leaves, log masses for stepped leaves).
 *
 * @param pe - The estimator.
 * @param points - One row per point, the flat numbers of its leaves.
 * @returns The log densities.
 * @concept tree-structured-parzen-estimator
 * @example
 * ```ts
 * import { parzenEstimator, parzenLogPdf, leavesOfSpace } from 'omai-engine/tune'
 * import { choice } from 'omai-engine/space'
 * const pe = parzenEstimator(leavesOfSpace(choice(['a', 'b'])), [[0]])
 * Math.exp(parzenLogPdf(pe, [[0]])[0] ?? 0) // => 0.625
 * ```
 */
export function parzenLogPdf(pe: ParzenEstimator, points: readonly (readonly number[])[]): number[] {
  const c = points.length
  const kCount = pe.weights.length
  const total = Array.from({ length: c }, () => Float64Array.from(pe.weights, (w) => Math.log(w)))
  for (const [j, k] of pe.kernels.entries()) {
    const xs = points.map((p) => p[j] as number)
    const lp = k.kind === 'categorical' ? undefined : toFlat(numericalLogPdf(k, xs))
    for (const [i, row] of total.entries()) {
      for (let m = 0; m < kCount; m++) {
        const term =
          lp === undefined
            ? Math.log(k.kind === 'categorical' ? (k.probs[m]?.[xs[i] as number] ?? 0) : 0)
            : (lp[i * kCount + m] ?? 0)
        row[m] = (row[m] ?? 0) + term
      }
    }
  }
  return total.map((row) => {
    const v = logSumExpOf(row)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  })
}

/**
 * Draw points from a Parzen estimator: a component by weight, then each leaf from its kernel (truncated normals
 * rounded to their step, categorical rows).
 *
 * @param pe - The estimator.
 * @param key - The key.
 * @param size - Number of points.
 * @returns One row of leaf numbers per point.
 * @concept tree-structured-parzen-estimator
 * @example
 * ```ts
 * import { parzenEstimator, parzenSample, leavesOfSpace } from 'omai-engine/tune'
 * import { int } from 'omai-engine/space'
 * import { key } from 'omai-engine/random'
 * parzenSample(parzenEstimator(leavesOfSpace(int(1, 3)), [[2]]), key(0), 5).every(([v]) => [1, 2, 3].includes(v ?? 0)) // => true
 * ```
 */
export function parzenSample(pe: ParzenEstimator, key: Key, size: number): number[][] {
  const active = Array.from(categorical(foldIn(key, 0), Array.from(pe.weights), [size]).data)
  const out = Array.from({ length: size }, () => new Array<number>(pe.kernels.length).fill(0))
  for (const [j, k] of pe.kernels.entries()) {
    const kj = foldIn(key, j + 1)
    if (k.kind === 'categorical') {
      const u = uniform(kj, [size]).data
      for (let i = 0; i < size; i++) {
        const row = k.probs[active[i] as number] ?? k.probs[0] ?? new Float64Array(0)
        let acc = 0
        let pick = row.length - 1
        for (let c = 0; c < row.length; c++) {
          acc += row[c] ?? 0
          if ((u[i] ?? 0) < acc) {
            pick = c
            break
          }
        }
        ;(out[i] as number[])[j] = pick
      }
      continue
    }
    const mus = active.map((m) => k.mus[m] as number)
    const sigmas = active.map((m) => k.sigmas[m] as number)
    const draws = toFlat(truncatedNormal(mus, sigmas, k.low, k.high).sample(kj))
    const leaf = pe.leaves[j] as Leaf
    for (let i = 0; i < size; i++) {
      let x = draws[i] ?? 0
      if (k.log) x = Math.exp(x)
      if (k.step > 0)
        x = Math.min(leaf.high, Math.max(leaf.low, leaf.low + Math.round((x - leaf.low) / k.step) * k.step))
      ;(out[i] as number[])[j] = x
    }
  }
  return out
}

/**
 * The scalar parameters of a space as TPE and the importance evaluators see them: one leaf per real, integer, option,
 * box element or binary coordinate, with its path, kind, bounds, log scale and step.
 *
 * @param space - The space.
 * @returns The leaves in flattening order.
 * @example
 * ```ts
 * import { leavesOfSpace } from 'omai-engine/tune'
 * import { dict, real, choice } from 'omai-engine/space'
 * leavesOfSpace(dict({ lr: real(1e-4, 1, { log: true }), act: choice(['relu', 'tanh']) })).map((l) => [l.path, l.kind])
 * // => [['lr', 'float'], ['act', 'categorical']]
 * ```
 */
export function leavesOfSpace(space: Space): Leaf[] {
  return leavesOf(space)
}

/**
 * TPE's acquisition at candidate points: log l(x) - log g(x), the log density ratio of the good and bad groups
 * (maximising it maximises expected improvement under the TPE model).
 *
 * @param below - The good group's estimator l.
 * @param above - The bad group's estimator g.
 * @param points - Candidate rows of leaf numbers.
 * @returns The log ratios.
 * @concept tree-structured-parzen-estimator
 * @example
 * ```ts
 * import { parzenEstimator, tpeAcquisition, leavesOfSpace } from 'omai-engine/tune'
 * import { real } from 'omai-engine/space'
 * const leaves = leavesOfSpace(real(0, 1))
 * const [good, bad] = tpeAcquisition(parzenEstimator(leaves, [[0.1]]), parzenEstimator(leaves, [[0.9]]), [[0.1], [0.9]])
 * (good ?? 0) > 0 && (bad ?? 0) < 0 // => true
 * ```
 */
export function tpeAcquisition(
  below: ParzenEstimator,
  above: ParzenEstimator,
  points: readonly (readonly number[])[],
): number[] {
  const l = parzenLogPdf(below, points)
  const g = parzenLogPdf(above, points)
  return l.map((v, i) => v - (g[i] ?? 0))
}
