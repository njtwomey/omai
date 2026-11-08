/**
 * Modifiers that change the labels or class balance of a dataset: label noise from a known transition matrix (symmetric
 * or pair-flip), subsampling to exact prevalences, label shift that resamples towards new class priors while `p(x | y)`
 * stays fixed, and long-tailed or step class imbalance. Clean labels and the transition are kept in the truth, and each
 * modifier is also available as a recipe modifier.
 *
 * @index function Label noise and class-imbalance modifiers.
 */

import { OmaiError } from '../../base/index.ts'
import { categorical, dirichlet, foldIn, type Key, permutation, uniform } from '../../random/index.ts'
import { choice, dict, real } from '../../space/index.ts'
import { asTensor, cast, concreteOrThrow, fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import type { Generated } from '../generator.ts'
import type { Modifier } from '../recipes.ts'
import { take } from '../rows.ts'
import type { ClassificationData } from '../schema/tabular.ts'
import {
  classCounts,
  classPriors,
  posteriorTimes,
  priorShifted,
  requireClassification,
  withPosterior,
} from './_shared.ts'

/** Options of {@link withLabelNoise}. */
export interface LabelNoiseOptions {
  /** The probability that a label is corrupted (default 0.1). */
  readonly rate?: number
  /**
   * `symmetric` (default): to a uniformly chosen other class; `pairflip`: to the next class; or give `transition`.
   */
  readonly mode?: 'symmetric' | 'pairflip'
  /**
   * A clean-to-noisy transition matrix `T[clean][noisy]` (rows sum to 1), overriding `rate` and `mode`: a `[k, k]`
   * tensor or nested array (migration, D-009: was `readonly (readonly number[])[]`).
   */
  readonly transition?: TensorLike
  readonly key: Key
}

/**
 * The clean-to-noisy transition matrix of symmetric or pair-flip label noise. Returns a tensor (migration, D-009: was a
 * `number[][]`; {@link LabelNoiseOptions} `transition` takes it back).
 *
 * @param k - The number of classes.
 * @param rate - The corruption probability.
 * @param mode - `symmetric` or `pairflip`.
 * @returns A float64 tensor of shape `[k, k]`, `T[clean, noisy]`.
 * @example
 * ```ts
 * import { noiseTransition } from 'omai-engine/data'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(noiseTransition(2, 0.2, 'symmetric')) // => [[0.8, 0.2], [0.2, 0.8]]
 * ```
 */
export function noiseTransition(
  k: number,
  rate: number,
  mode: 'symmetric' | 'pairflip' = 'symmetric',
): Tensor<'float64'> {
  const out = new Float64Array(k * k)
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      out[i * k + j] =
        i === j ? (k === 1 ? 1 : 1 - rate) : mode === 'pairflip' ? (j === (i + 1) % k ? rate : 0) : rate / (k - 1)
    }
  }
  return fromStorage(out, [k, k], 'float64')
}

/**
 * Corrupt class labels with known noise (symmetric, pair-flip or a given transition matrix), keeping the clean labels
 * in the truth. The truth's posterior becomes the noisy-label posterior `p(noisy | x) = p(clean | x) T`, and its Bayes
 * error is re-estimated on the rows.
 *
 * @param input - Classification data and its truth.
 * @param options - Rate, mode or transition matrix, and key.
 * @returns The noisy data; `truth.latents.cleanLabels`, `truth.latents.flipped` and `truth.extra.transition` (a `[k, k]`
 * float64 tensor).
 * @throws OmaiError with code `value` for non-classification data or a malformed transition matrix.
 * @example
 * ```ts
 * import { withLabelNoise, asGenerated, xyClasses } from 'omai-engine/data'
 * import { key } from 'omai-engine/random'
 * const noisy = withLabelNoise(asGenerated(xyClasses([[0], [1], [2], [3]], [0, 1, 0, 1])), { rate: 1, key: key(0) })
 * Array.from(noisy.data.y.data) // => [1, 0, 1, 0]
 * ```
 */
export function withLabelNoise<D extends ClassificationData>(
  input: Generated<D>,
  options: LabelNoiseOptions,
): Generated<D> {
  const { data, truth } = requireClassification(input, 'withLabelNoise') as Generated<D>
  const k = data.classes.length
  const given =
    options.transition === undefined ? undefined : concreteOrThrow(asTensor(options.transition), 'withLabelNoise')
  const t = cast(
    given ?? noiseTransition(k, options.rate ?? 0.1, options.mode ?? 'symmetric'),
    'float64',
  ) as Tensor<'float64'>
  const rowSum = (i: number): number => {
    let total = 0
    for (let j = 0; j < k; j++) total += t.data[i * k + j] ?? 0
    return total
  }
  if (
    t.shape.length !== 2 ||
    t.shape[0] !== k ||
    t.shape[1] !== k ||
    Array.from({ length: k }, (_, i) => i).some((i) => Math.abs(rowSum(i) - 1) > 1e-9)
  )
    throw new OmaiError('value', `withLabelNoise: the transition matrix must be ${k} by ${k} with rows summing to 1`)
  const n = data.y.shape[0] ?? 0
  const y = new Int32Array(n)
  const flipped = new Uint8Array(n)
  const u = uniform(options.key, [n]).data
  for (let i = 0; i < n; i++) {
    const c = data.y.data[i] ?? 0
    // Invert the row's cumulative distribution at u_i.
    const valid = c >= 0 && c < k
    let v = valid ? k - 1 : -1
    let acc = 0
    for (let j = 0; valid && j < k; j++) {
      const p = t.data[c * k + j] ?? 0
      acc += p
      if ((u[i] ?? 0) < acc && p > 0) {
        v = j
        break
      }
    }
    y[i] = v
    flipped[i] = v === c ? 0 : 1
  }
  const noisy = { ...data, y: fromStorage(y, [n], 'int32') }
  const posterior = truth.posterior === undefined ? undefined : posteriorTimes(truth.posterior, t)
  return {
    data: noisy,
    truth: {
      ...withPosterior(truth, posterior, data.x, { transition: t, cleanBayesError: truth.bayesError }),
      latents: { ...(truth.latents ?? {}), cleanLabels: data.y, flipped: fromStorage(flipped, [n], 'bool') },
    },
  }
}

function subsampleByClass<D extends ClassificationData>(data: D, counts: readonly number[], key: Key): D {
  const byClass: number[][] = data.classes.map(() => [])
  for (const [i, c] of data.y.data.entries()) byClass[c]?.push(i)
  const rows: number[] = []
  for (const [c, members] of byClass.entries()) {
    const want = counts[c] ?? 0
    if (want > members.length)
      throw new OmaiError('value', `class ${c} has ${members.length} rows, ${want} were asked for`)
    const p = permutation(foldIn(key, c), members.length).data
    for (let i = 0; i < want; i++) rows.push(members[p[i] ?? 0] ?? 0)
  }
  const order = permutation(foldIn(key, -2 >>> 0), rows.length).data
  return take(
    data,
    Array.from(order, (i) => rows[i] ?? 0),
  )
}

function largestRemainder(total: number, props: readonly number[]): number[] {
  const exact = props.map((p) => p * total)
  const out = exact.map(Math.floor)
  let left = total - out.reduce((a, b) => a + b, 0)
  const order = exact.map((v, j) => [v - Math.floor(v), j] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1])
  for (const [, j] of order) {
    if (left <= 0) break
    out[j] = (out[j] ?? 0) + 1
    left--
  }
  return out
}

/** Options of {@link withPrevalence}. */
export interface PrevalenceOptions {
  /** The positive-class (class 1) prevalence for binary data, or one proportion per class. */
  readonly prevalence: number | readonly number[]
  /** Rows to keep (default the most that the class counts allow). */
  readonly n?: number
  readonly key: Key
}

/**
 * Subsample rows (without replacement) to reach exact class proportions, as in quantification and prevalence-shift
 * benchmarks. The posterior is adjusted for the new priors (`p'(k|x) ∝ p(k|x) π'_k / π_k`).
 *
 * @param input - Classification data and its truth.
 * @param options - Target prevalence, size and key.
 * @returns The resampled data, with `truth.extra.priors` set to the new proportions.
 * @throws OmaiError with code `value` when a class has too few rows.
 * @example
 * ```ts
 * import { asGenerated, withPrevalence, xyClasses } from 'omai-engine/data'
 * import { key } from 'omai-engine/random'
 * const d = asGenerated(xyClasses([[0], [1], [2], [3], [4], [5]], [0, 0, 0, 0, 1, 1]))
 * Array.from(withPrevalence(d, { prevalence: 0.5, key: key(0) }).data.y.data).sort() // => [0, 0, 1, 1]
 * ```
 */
export function withPrevalence<D extends ClassificationData>(
  input: Generated<D>,
  options: PrevalenceOptions,
): Generated<D> {
  const { data, truth } = requireClassification(input, 'withPrevalence') as Generated<D>
  const k = data.classes.length
  const props =
    typeof options.prevalence === 'number' ? [1 - options.prevalence, options.prevalence] : [...options.prevalence]
  if (props.length !== k || Math.abs(props.reduce((a, b) => a + b, 0) - 1) > 1e-9)
    throw new OmaiError('value', `withPrevalence: need ${k} proportions summing to 1`)
  const counts = classCounts(data)
  let nMax = Number.POSITIVE_INFINITY
  for (const [c, p] of props.entries()) if (p > 0) nMax = Math.min(nMax, Math.floor((counts[c] ?? 0) / p))
  const n = options.n ?? nMax
  const out = subsampleByClass(data, largestRemainder(n, props), options.key)
  const old = classPriors({ data, truth })
  const posterior =
    truth.posterior === undefined
      ? undefined
      : priorShifted(
          truth.posterior,
          props.map((p, c) => p / Math.max(old[c] ?? 1, 1e-300)),
        )
  return { data: out, truth: withPosterior(truth, posterior, out.x, { priors: props, sourcePriors: old }) }
}

/** Options of {@link withLabelShift}. */
export interface LabelShiftOptions {
  /** Target class priors; default drawn from a symmetric Dirichlet with concentration `alpha`. */
  readonly targetPriors?: readonly number[]
  /** Dirichlet concentration for random target priors (default 1; smaller is a stronger shift). */
  readonly alpha?: number
  /** Rows of the shifted sample (default the input size). */
  readonly n?: number
  readonly key: Key
}

/**
 * Label shift (prior probability shift, as in black-box shift estimation benchmarks): resample rows with
 * replacement so the class priors become `targetPriors` (or a Dirichlet draw), while `p(x | y)` stays fixed. The truth
 * records both priors and the class importance weights `π'_k / π_k`.
 *
 * @param input - Classification data and its truth.
 * @param options - Target priors or Dirichlet concentration, size and key.
 * @returns The shifted sample.
 * @throws OmaiError with code `value` when a class with positive target prior has no rows.
 * @example
 * ```ts
 * import { asGenerated, withLabelShift, xyClasses } from 'omai-engine/data'
 * import { key } from 'omai-engine/random'
 * const d = asGenerated(xyClasses([[0], [1], [2], [3]], [0, 0, 1, 1]))
 * withLabelShift(d, { targetPriors: [0.25, 0.75], key: key(0) }).truth.extra?.['weights'] // => [0.5, 1.5]
 * ```
 */
export function withLabelShift<D extends ClassificationData>(
  input: Generated<D>,
  options: LabelShiftOptions,
): Generated<D> {
  const { data, truth } = requireClassification(input, 'withLabelShift') as Generated<D>
  const k = data.classes.length
  const target =
    options.targetPriors ?? Array.from(dirichlet(foldIn(options.key, 0), new Array(k).fill(options.alpha ?? 1), 1).data)
  const source = classPriors({ data, truth })
  const counts = classCounts(data)
  const weights = target.map((p, c) => p / Math.max(source[c] ?? 0, 1e-300))
  const byClass: number[][] = data.classes.map(() => [])
  for (const [i, c] of data.y.data.entries()) byClass[c]?.push(i)
  for (const [c, p] of target.entries()) {
    if (p > 0 && (counts[c] ?? 0) === 0) throw new OmaiError('value', `withLabelShift: class ${c} has no rows`)
  }
  const n = options.n ?? data.y.shape[0] ?? 0
  const classes = categorical(foldIn(options.key, 1), target as number[], [n]).data
  const u = uniform(foldIn(options.key, 2), [n]).data
  const rows = Array.from(classes, (c, i) => {
    const members = byClass[c] ?? []
    return members[Math.floor((u[i] ?? 0) * members.length)] ?? 0
  })
  const out = take(data, rows)
  const posterior = truth.posterior === undefined ? undefined : priorShifted(truth.posterior, weights)
  return {
    data: out,
    truth: withPosterior(truth, posterior, out.x, { priors: target, sourcePriors: source, weights }),
  }
}

/** Options of {@link withClassImbalance}. */
export interface ClassImbalanceOptions {
  /** The ratio of the largest to the smallest class (at least 1). */
  readonly ratio: number
  /**
   * `exponential` (default): class `c` keeps `n_max ratio^(-c / (k - 1))` rows (long-tailed); `step`: the second half
   * of the classes keep `n_max / ratio`.
   */
  readonly profile?: 'exponential' | 'step'
  readonly key: Key
}

/**
 * Make classes imbalanced by subsampling (the long-tailed and step profiles of imbalanced-learning benchmarks), the
 * class order deciding which classes shrink. The posterior is adjusted for the new priors.
 *
 * @param input - Classification data and its truth.
 * @param options - Imbalance ratio, profile and key.
 * @returns The imbalanced data, with `truth.extra.priors`.
 * @throws OmaiError with code `value` for a ratio below 1.
 * @example
 * ```ts
 * import { asGenerated, withClassImbalance, xyClasses } from 'omai-engine/data'
 * import { key } from 'omai-engine/random'
 * const d = asGenerated(xyClasses(Array.from({ length: 8 }, (_, i) => [i]), [0, 0, 0, 0, 1, 1, 1, 1]))
 * Array.from(withClassImbalance(d, { ratio: 4, key: key(0) }).data.y.data).sort() // => [0, 0, 0, 0, 1]
 * ```
 */
export function withClassImbalance<D extends ClassificationData>(
  input: Generated<D>,
  options: ClassImbalanceOptions,
): Generated<D> {
  const { data, truth } = requireClassification(input, 'withClassImbalance') as Generated<D>
  if (!(options.ratio >= 1)) throw new OmaiError('value', 'withClassImbalance: the ratio must be at least 1')
  const counts = classCounts(data)
  const k = counts.length
  let base = Number.POSITIVE_INFINITY
  for (const c of counts) base = Math.min(base, c)
  const keep = counts.map((_, c) => {
    if (options.profile === 'step') return c >= Math.ceil(k / 2) ? Math.max(1, Math.floor(base / options.ratio)) : base
    return Math.max(1, Math.floor(base * options.ratio ** (-c / Math.max(1, k - 1))))
  })
  const out = subsampleByClass(data, keep, options.key)
  const total = keep.reduce((a, b) => a + b, 0)
  const props = keep.map((c) => c / total)
  const old = classPriors({ data, truth })
  const posterior =
    truth.posterior === undefined
      ? undefined
      : priorShifted(
          truth.posterior,
          props.map((p, c) => p / Math.max(old[c] ?? 1, 1e-300)),
        )
  return { data: out, truth: withPosterior(truth, posterior, out.x, { priors: props, sourcePriors: old }) }
}

/**
 * {@link withLabelNoise} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { labelNoiseModifier } from 'omai-engine/data'
 * labelNoiseModifier.name // => 'labelNoise'
 * ```
 */
export const labelNoiseModifier: Modifier = {
  name: 'labelNoise',
  description: 'Corrupt labels with symmetric or pair-flip noise.',
  knobs: dict({
    rate: real(0, 1, { default: 0.1, label: 'noise rate' }),
    mode: choice(['symmetric', 'pairflip'], { default: 'symmetric' }),
  }),
  apply: (key, input, knobs) =>
    withLabelNoise(input as Generated<ClassificationData>, {
      rate: knobs['rate'] as number,
      mode: knobs['mode'] as 'symmetric' | 'pairflip',
      key,
    }),
}

/**
 * {@link withPrevalence} for binary data as a recipe modifier.
 *
 * @example
 * ```ts
 * import { prevalenceModifier } from 'omai-engine/data'
 * prevalenceModifier.name // => 'prevalence'
 * ```
 */
export const prevalenceModifier: Modifier = {
  name: 'prevalence',
  description: 'Subsample to an exact positive-class prevalence.',
  knobs: dict({ prevalence: real(0.01, 0.99, { default: 0.1, label: 'positive prevalence' }) }),
  apply: (key, input, knobs) =>
    withPrevalence(input as Generated<ClassificationData>, { prevalence: knobs['prevalence'] as number, key }),
}

/**
 * {@link withLabelShift} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { labelShiftModifier } from 'omai-engine/data'
 * labelShiftModifier.name // => 'labelShift'
 * ```
 */
export const labelShiftModifier: Modifier = {
  name: 'labelShift',
  description: 'Resample to class priors drawn from a Dirichlet.',
  knobs: dict({ alpha: real(0.01, 100, { default: 1, log: true, label: 'Dirichlet concentration' }) }),
  apply: (key, input, knobs) =>
    withLabelShift(input as Generated<ClassificationData>, { alpha: knobs['alpha'] as number, key }),
}

/**
 * {@link withClassImbalance} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { classImbalanceModifier } from 'omai-engine/data'
 * classImbalanceModifier.name // => 'classImbalance'
 * ```
 */
export const classImbalanceModifier: Modifier = {
  name: 'classImbalance',
  description: 'Subsample classes to a long-tailed or step imbalance.',
  knobs: dict({
    ratio: real(1, 1000, { default: 10, log: true, label: 'imbalance ratio' }),
    profile: choice(['exponential', 'step'], { default: 'exponential' }),
  }),
  apply: (key, input, knobs) =>
    withClassImbalance(input as Generated<ClassificationData>, {
      ratio: knobs['ratio'] as number,
      profile: knobs['profile'] as 'exponential' | 'step',
      key,
    }),
}
