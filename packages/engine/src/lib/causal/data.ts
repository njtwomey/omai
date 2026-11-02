/**
 * The shapes the causal estimators share: `CausalData` (covariates, a binary treatment and an outcome), the split of a
 * generated dataset whose treatment is one of its feature columns, and `EffectEstimate`, an effect with its standard
 * error, interval and p-value, built either from the per-unit influence values of the estimator (the sample variance of
 * the influence over n is its variance) or from a standard error given directly.
 *
 * @index structure CausalData: covariates, binary treatment and outcome shared by estimators.
 */

import { type Estimate, estimate, MdmlError } from '../base/index.ts'
import type { LearnData } from '../learn/estimator.ts'
import { matrixView } from '../learn/validation.ts'
import { fromStorage, type TensorLike } from '../tensor/index.ts'
import { tPpf, twoSided, variance } from './_dense.ts'

/** What an estimator of a treatment effect is given: covariates `[n, d]`, a 0/1 treatment `[n]` and an outcome `[n]`. */
export interface CausalData {
  readonly x: TensorLike
  readonly treatment: TensorLike
  readonly outcome: TensorLike
}

/** The effects an estimator may target. */
export type Estimand = 'ate' | 'att' | 'atc' | 'late' | 'effect-at-cutoff' | 'did'

/** An estimated effect: the estimate with its interval, a p-value against zero, and the influence values when known. */
export interface EffectEstimate {
  readonly estimand: Estimand
  readonly estimate: Estimate
  /** Two-sided p-value of the null that the effect is zero. */
  readonly pValue: number
  /** The number of units. */
  readonly n: number
  /** The estimator's influence value per unit (the estimate minus the truth is their mean), when it has one. */
  readonly influence?: Float64Array
}

/**
 * Split a dataset whose treatment is one of the feature columns (the layout of the `treatmentEffects` generator: the
 * covariates, then the treatment) into covariates, treatment and outcome.
 *
 * @param data - Features `x` and the outcome `y`.
 * @param treatmentColumn - The treatment's column (default the last).
 * @returns The causal data; the covariates are the other columns.
 * @example
 * ```ts
 * import { splitTreatment } from 'mdml-engine/causal'
 * import { toArray } from 'mdml-engine/tensor'
 * const d = splitTreatment({ x: [[0.5, 1], [1.5, 0]], y: [3, 4] })
 * [toArray(d.x), toArray(d.treatment)] // => [[[0.5], [1.5]], [1, 0]]
 * ```
 */
export function splitTreatment(data: LearnData, treatmentColumn = -1): CausalData {
  if (data.y === undefined) throw new MdmlError('value', 'splitTreatment: the data need an outcome y')
  const m = matrixView(data.x, 'splitTreatment')
  const c = treatmentColumn < 0 ? m.d + treatmentColumn : treatmentColumn
  if (c < 0 || c >= m.d) throw new MdmlError('value', `splitTreatment: no column ${treatmentColumn}`)
  const x = new Float64Array(m.n * (m.d - 1))
  const t = new Float64Array(m.n)
  for (let i = 0; i < m.n; i++) {
    t[i] = m.data[i * m.d + c] ?? 0
    let k = 0
    for (let j = 0; j < m.d; j++) if (j !== c) x[i * (m.d - 1) + k++] = m.data[i * m.d + j] ?? 0
  }
  return { x: fromStorage(x, [m.n, m.d - 1], 'float64'), treatment: fromStorage(t, [m.n], 'float64'), outcome: data.y }
}

/**
 * An effect from a point estimate and a standard error: the normal interval and p-value (or Student's t when `df` is
 * given).
 *
 * @param estimand - The target.
 * @param value - The point estimate.
 * @param standardError - Its standard error.
 * @param n - The number of units.
 * @param options - `level` (default 0.95), `df` for a t interval, `method` label.
 * @returns The estimate.
 * @example
 * ```ts
 * import { effectFromSe } from 'mdml-engine/causal'
 * effectFromSe('ate', 2, 0.5, 100).estimate.interval // => [1.020018, 2.979982]
 * ```
 */
export function effectFromSe(
  estimand: Estimand,
  value: number,
  standardError: number,
  n: number,
  options: { readonly level?: number; readonly df?: number; readonly method?: string } = {},
): EffectEstimate {
  const level = options.level ?? 0.95
  let est: Estimate = estimate(value, standardError, level)
  if (options.df !== undefined) {
    const q = tPpf(options.df, 0.5 + level / 2)
    est = { ...est, interval: [value - q * standardError, value + q * standardError] }
  }
  if (options.method !== undefined) est = { ...est, method: options.method }
  const stat = value / standardError
  return { estimand, estimate: est, pValue: twoSided(stat, options.df), n }
}

/**
 * An effect from the estimator's influence values: the standard error is the standard deviation of the influence over
 * the square root of n (the estimate's variance under the usual regularity conditions).
 *
 * @param estimand - The target.
 * @param value - The point estimate.
 * @param influence - One centred influence value per unit.
 * @param options - `level` and `method` label.
 * @returns The estimate, keeping the influence values.
 * @example
 * ```ts
 * import { effectFromInfluence } from 'mdml-engine/causal'
 * effectFromInfluence('ate', 1, [1, -1, 1, -1]).estimate.standardError // => 0.5773502691896257
 * ```
 */
export function effectFromInfluence(
  estimand: Estimand,
  value: number,
  influence: ArrayLike<number>,
  options: { readonly level?: number; readonly method?: string } = {},
): EffectEstimate {
  const n = influence.length
  const inf = Float64Array.from(influence)
  const se = Math.sqrt(variance(inf, 1) / n)
  const base = effectFromSe(estimand, value, se, n, options)
  return { ...base, influence: inf }
}
