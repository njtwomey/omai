import { argmaxOf, type Json, logSumExpOf, WhiteboxError } from '../base/index.ts'
import { truncatedNormal } from '../dists/index.ts'
import { logIntervalProbability } from '../numerics/special/index.ts'
import { hypervolume, nonDominatedSort } from '../optim/index.ts'
import { categorical, foldIn, type Key, uniform } from '../random/index.ts'
import { type Space, sample } from '../space/index.ts'
import { asTensor, div, reshape, sub, toFlat, type Value } from '../tensor/index.ts'
import { assemble, type Leaf, leavesOf, nativeOf } from './_leaves.ts'

export type { Leaf } from './_leaves.ts'

import type { Sampler } from './samplers.ts'
import { lossMatrix, lossOf, type Study, type StudyDirection, type Trial } from './study.ts'

const EPS = 1e-12

/**
 * Optuna's default gamma: the number of "good" trials, min(ceil(0.1 n), 25).
 *
 * @param n - Finished trials.
 * @returns Trials in the good group.
 * @example
 * ```ts
 * import { defaultGamma } from 'whitebox-engine/tune'
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
 * import { defaultGammaMultiObjective } from 'whitebox-engine/tune'
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
 * import { hyperoptGamma } from 'whitebox-engine/tune'
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
 * import { defaultWeights } from 'whitebox-engine/tune'
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
 * @throws WhiteboxError with code `value` for a negative prior weight or invalid weights.
 * @concept tree-structured-parzen-estimator
 * @reference Bergstra et al. (2011), Algorithms for hyper-parameter optimization; Watanabe (2023), Tree-structured
 *   Parzen estimator: understanding its algorithm components.
 * @example
 * ```ts
 * import { leavesOfSpace, parzenEstimator, parzenLogPdf } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
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
  if (!(priorWeight >= 0)) throw new WhiteboxError('value', `parzenEstimator: priorWeight must be non-negative`)
  const n = observations.length
  const raw = options.predeterminedWeights ?? (options.weights ?? defaultWeights)(n).slice(0, n)
  if (raw.length !== n) throw new WhiteboxError('value', `parzenEstimator: ${raw.length} weights for ${n} observations`)
  if (raw.some((w) => !(w >= 0) || !Number.isFinite(w)))
    throw new WhiteboxError('value', 'parzenEstimator: weights must be finite and non-negative')
  const w = n === 0 ? [1] : [...raw, priorWeight]
  const total = w.reduce((a, b) => a + b, 0)
  if (!(total > 0)) throw new WhiteboxError('value', 'parzenEstimator: the weights sum to zero')
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
 * import { parzenEstimator, parzenLogPdf, leavesOfSpace } from 'whitebox-engine/tune'
 * import { choice } from 'whitebox-engine/space'
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
 * import { parzenEstimator, parzenSample, leavesOfSpace } from 'whitebox-engine/tune'
 * import { int } from 'whitebox-engine/space'
 * import { key } from 'whitebox-engine/random'
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

/** The good and bad groups of trials for TPE. */
export interface TpeSplit {
  readonly below: readonly Trial[]
  readonly above: readonly Trial[]
}

function prunedScore(t: Trial, d: StudyDirection): [number, number] {
  if (t.intermediate.length === 0) return [1, 0]
  const last = t.intermediate.reduce((a, b) => (b.step > a.step ? b : a))
  if (Number.isNaN(last.value)) return [-last.step, Number.POSITIVE_INFINITY]
  return [-last.step, lossOf(d, last.value)]
}

/**
 * The greedy hypervolume subset selection Optuna uses to break ties in a non-dominated front: repeatedly take the
 * point adding the most hypervolume (first in lexicographic order on ties), over the distinct points.
 *
 * @param losses - The front's loss vectors.
 * @param size - How many to select.
 * @param reference - The reference point.
 * @returns Indices into `losses`.
 * @concept hypervolume
 * @example
 * ```ts
 * import { hypervolumeSubsetSelection } from 'whitebox-engine/tune'
 * hypervolumeSubsetSelection([[1, 4], [2, 2], [4, 1]], 1, [5, 5]) // => [1]
 * ```
 */
export function hypervolumeSubsetSelection(
  losses: readonly (readonly number[])[],
  size: number,
  reference: readonly number[],
): number[] {
  if (size >= losses.length) return losses.map((_, i) => i)
  // Distinct rows in lexicographic order, each by its first index (numpy.unique(axis=0, return_index=True)).
  const keyOf = (r: readonly number[]): string => JSON.stringify(r)
  const first = new Map<string, number>()
  for (const [i, r] of losses.entries()) if (!first.has(keyOf(r))) first.set(keyOf(r), i)
  const unique = [...first.values()].sort((a, b) => {
    const ra = losses[a] as readonly number[]
    const rb = losses[b] as readonly number[]
    for (let j = 0; j < ra.length; j++) if (ra[j] !== rb[j]) return (ra[j] ?? 0) - (rb[j] ?? 0)
    return 0
  })
  if (unique.length < size) {
    const chosen = new Set(unique)
    for (const [i] of losses.entries()) if (chosen.size < size && !chosen.has(i)) chosen.add(i)
    return losses.map((_, i) => i).filter((i) => chosen.has(i))
  }
  if (!reference.every(Number.isFinite)) return unique.slice(0, size)
  const selected: number[] = []
  let remaining = [...unique]
  let hv = 0
  while (selected.length < size) {
    const gains = remaining.map(
      (i) =>
        hypervolume(
          [...selected, i].map((s) => losses[s] as number[]),
          reference,
        ) - hv,
    )
    const best = argmaxOf(gains)
    hv += gains[best] ?? 0
    selected.push(remaining[best] as number)
    remaining = remaining.filter((_, j) => j !== best)
  }
  return selected
}

/**
 * Optuna's split of finished trials into the good group (`nBelow` of them) and the rest: complete trials by value
 * (for several objectives by non-dominated rank, ties in the last front broken by hypervolume subset selection), then
 * pruned trials by the step they reached and their last value.
 *
 * @param study - The study.
 * @param trials - Complete and pruned trials.
 * @param nBelow - Size of the good group.
 * @returns Both groups, each in trial order.
 * @concept tree-structured-parzen-estimator
 * @example
 * ```ts
 * import { createStudy, splitTrials } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
 * const trial = (number: number, value: number) =>
 *   ({ number, params: value, state: 'complete', values: [value], intermediate: [], source: 'x' }) as const
 * const study = createStudy(real(0, 9), { trials: [trial(0, 5), trial(1, 1), trial(2, 3)] })
 * splitTrials(study, study.trials, 2).below.map((t) => t.number) // => [1, 2]
 * ```
 */
export function splitTrials(study: Study, trials: readonly Trial[], nBelow: number): TpeSplit {
  const complete = trials.filter((t) => t.state === 'complete')
  const pruned = trials.filter((t) => t.state === 'pruned')
  const nComplete = Math.min(nBelow, complete.length)
  let belowComplete: Trial[]
  if (study.directions.length === 1) {
    const d = study.directions[0] as StudyDirection
    const order = complete
      .map((t, i) => ({ t, i, v: lossOf(d, t.values[0] ?? Number.NaN) }))
      .sort((a, b) => a.v - b.v || a.i - b.i)
    belowComplete = order.slice(0, nComplete).map((o) => o.t)
  } else belowComplete = splitMultiObjective(study, complete, nComplete)
  let rest = Math.max(0, nBelow - belowComplete.length)
  const d = study.directions[0] as StudyDirection
  const prunedOrder = pruned
    .map((t, i) => ({ t, i, s: prunedScore(t, d) }))
    .sort((a, b) => a.s[0] - b.s[0] || a.s[1] - b.s[1] || a.i - b.i)
  const belowPruned =
    study.directions.length === 1 ? prunedOrder.slice(0, Math.min(rest, pruned.length)).map((o) => o.t) : []
  rest = Math.max(0, rest - belowPruned.length)
  const below = new Set([...belowComplete, ...belowPruned])
  const byNumber = (a: Trial, b: Trial): number => a.number - b.number
  return { below: [...below].sort(byNumber), above: trials.filter((t) => !below.has(t)).sort(byNumber) }
}

function splitMultiObjective(study: Study, trials: readonly Trial[], nBelow: number): Trial[] {
  if (nBelow === 0) return []
  if (nBelow >= trials.length) return [...trials]
  const losses = lossMatrix(study, trials)
  const chosen: number[] = []
  for (const front of nonDominatedSort(losses)) {
    if (chosen.length + front.length <= nBelow) {
      chosen.push(...front)
      if (chosen.length === nBelow) break
      continue
    }
    const rows = front.map((i) => losses[i] as number[])
    const worst = rows[0]?.map((_, j) => rows.reduce((m, r) => Math.max(m, r[j] ?? 0), Number.NEGATIVE_INFINITY)) ?? []
    const reference = worst.map((w) => {
      const r = Math.max(1.1 * w, 0.9 * w)
      return r === 0 ? EPS : r
    })
    chosen.push(...hypervolumeSubsetSelection(rows, nBelow - chosen.length, reference).map((k) => front[k] as number))
    break
  }
  const keep = new Set(chosen)
  return trials.filter((_, i) => keep.has(i))
}

/** Options of {@link tpeSampler}. */
export interface TpeOptions extends Omit<ParzenOptions, 'predeterminedWeights'> {
  /** Random trials before the model is used (default 10). */
  readonly nStartupTrials?: number
  /** Candidates drawn from the good density per trial (default 24). */
  readonly nEiCandidates?: number
  /** The size of the good group for n finished trials (default {@link defaultGamma}, or ceil(0.1 n) for several). */
  readonly gamma?: (n: number) => number
  /** Model the leaves jointly (default: for one objective; independently for several, as Optuna). */
  readonly multivariate?: boolean
}

function observationsOf(study: Study, trials: readonly Trial[], columns: readonly number[]): number[][] {
  return trials.flatMap((t) => {
    const native = nativeOf(study.space, t.params)
    const row = columns.map((c) => native[c] as number)
    return row.some(Number.isNaN) ? [] : [row]
  })
}

/**
 * The tree-structured Parzen estimator (Optuna's `TPESampler`): after `nStartupTrials` random trials, split the
 * finished trials into a good group (gamma(n) best) and the rest, fit a Parzen estimator l(x) to the good and g(x) to
 * the rest, draw `nEiCandidates` candidates from l and take the one maximising log l(x) - log g(x) (the expected
 * improvement ratio). Multivariate by default for one objective (one joint estimator over the leaves every trial
 * has; conditional leaves independently), independent per leaf for several objectives (MOTPE, with
 * hypervolume-based tie breaking).
 *
 * @param options - Startup trials, candidates, gamma, weights, prior weight, bandwidth clipping and multivariate.
 * @returns The sampler.
 * @concept tree-structured-parzen-estimator
 * @reference Bergstra et al. (2011); Ozaki et al. (2020), Multiobjective tree-structured Parzen estimator.
 * @example
 * ```ts
 * import { tpeSampler, functionObjective, tune } from 'whitebox-engine/tune'
 * import { dict, real } from 'whitebox-engine/space'
 * const f = functionObjective((p) => { const { x, y } = p as { x: number; y: number }; return (x - 1) ** 2 + (y + 2) ** 2 })
 * (tune(f, dict({ x: real(-5, 5), y: real(-5, 5) }), { sampler: tpeSampler(), nTrials: 80, seed: 0 }).best?.values[0] ?? 9) < 0.5 // => true
 * ```
 */
export function tpeSampler(options: TpeOptions = {}): Sampler {
  const nStartup = options.nStartupTrials ?? 10
  const nCandidates = options.nEiCandidates ?? 24
  const parzen: ParzenOptions = {
    ...(options.priorWeight === undefined ? {} : { priorWeight: options.priorWeight }),
    ...(options.considerMagicClip === undefined ? {} : { considerMagicClip: options.considerMagicClip }),
    ...(options.considerEndpoints === undefined ? {} : { considerEndpoints: options.considerEndpoints }),
    ...(options.weights === undefined ? {} : { weights: options.weights }),
  }
  const config: Json = {
    nStartupTrials: nStartup,
    nEiCandidates: nCandidates,
    priorWeight: options.priorWeight ?? 1,
    considerMagicClip: options.considerMagicClip ?? true,
    considerEndpoints: options.considerEndpoints ?? false,
    multivariate: options.multivariate ?? null,
  }
  return {
    name: 'tpe',
    config,
    multiObjective: true,
    ask: (study, key) => {
      const finished = study.trials.filter((t) => t.state !== 'failed')
      if (finished.length < nStartup) return sample(study.space, key)
      const multi = study.directions.length > 1
      const gamma = options.gamma ?? (multi ? defaultGammaMultiObjective : defaultGamma)
      const split = splitTrials(study, finished, gamma(finished.length))
      const leaves = leavesOf(study.space)
      const natives = finished.map((t) => nativeOf(study.space, t.params))
      const varying = leaves.map((l) => l.low < l.high)
      const everywhere = leaves.map(
        (_, j) => varying[j] === true && natives.every((r) => !Number.isNaN(r[j] ?? Number.NaN)),
      )
      const multivariate = options.multivariate ?? !multi
      const groups: number[][] = []
      const joint = leaves.flatMap((_, j) => (multivariate && everywhere[j] === true ? [j] : []))
      if (joint.length > 0) groups.push(joint)
      for (const [j] of leaves.entries()) if (varying[j] === true && !joint.includes(j)) groups.push([j])
      const out = leaves.map((l) => (l.low < l.high ? Number.NaN : l.low))
      for (const [g, columns] of groups.entries()) {
        const covered = columns.map((c) => leaves[c] as Leaf)
        const below = observationsOf(study, split.below, columns)
        const above = observationsOf(study, split.above, columns)
        const l = parzenEstimator(
          covered,
          below,
          multi ? { ...parzen, predeterminedWeights: below.map(() => 1) } : parzen,
        )
        const gEst = parzenEstimator(covered, above, parzen)
        const candidates = parzenSample(l, foldIn(key, g), nCandidates)
        const best = candidates[argmaxOf(tpeAcquisition(l, gEst, candidates))] as number[]
        for (const [i, c] of columns.entries()) out[c] = best[i] as number
      }
      return assemble(study.space, leaves, out)
    },
  }
}

/**
 * The scalar parameters of a space as TPE and the importance evaluators see them: one leaf per real, integer, option,
 * box element or binary coordinate, with its path, kind, bounds, log scale and step.
 *
 * @param space - The space.
 * @returns The leaves in flattening order.
 * @example
 * ```ts
 * import { leavesOfSpace } from 'whitebox-engine/tune'
 * import { dict, real, choice } from 'whitebox-engine/space'
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
 * import { parzenEstimator, tpeAcquisition, leavesOfSpace } from 'whitebox-engine/tune'
 * import { real } from 'whitebox-engine/space'
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
