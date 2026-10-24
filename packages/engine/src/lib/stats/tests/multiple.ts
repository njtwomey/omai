import { type TestResult, WhiteboxError } from '../../base/index.ts'
import type { Tensor, TensorLike } from '../../tensor/index.ts'
import { chi2Cdf, chi2Sf, normPpf, normSf, sample, tSf, vec } from '../_util.ts'
import { testResult } from './result.ts'

/** A multiple-testing correction (statsmodels.stats.multitest.multipletests names, spelled out). */
export type CorrectionMethod =
  | 'bonferroni'
  | 'sidak'
  | 'holm'
  | 'holm-sidak'
  | 'hochberg'
  | 'hommel'
  | 'benjamini-hochberg'
  | 'benjamini-yekutieli'

function pValuesOf(name: string, p: TensorLike): Float64Array {
  const a = sample(name, p)
  for (const v of a) if (!(v >= 0 && v <= 1)) throw new WhiteboxError('value', `${name}: p-values must lie in [0, 1]`)
  return a
}

/**
 * Adjusted p-values for multiple testing (statsmodels multipletests' pvals_corrected): family-wise error control by
 * Bonferroni, Sidak, Holm (step-down), Holm-Sidak, Hochberg (step-up) and Hommel, or false-discovery-rate control by
 * Benjamini-Hochberg and Benjamini-Yekutieli; adjusted values are capped at one.
 *
 * @param p - The p-values.
 * @param method - The correction.
 * @returns The adjusted p-values, in the original order.
 * @example
 * ```ts
 * import { adjustPValues } from 'whitebox-engine/stats/tests'
 * adjustPValues([0.01, 0.04, 0.03], 'holm') // => [0.03, 0.06, 0.06]
 * adjustPValues([0.01, 0.04, 0.03], 'benjamini-hochberg') // => [0.03, 0.04, 0.04]
 * ```
 */
export function adjustPValues(p: TensorLike, method: CorrectionMethod): Tensor {
  const pv = pValuesOf('adjustPValues', p)
  const m = pv.length
  const order = Array.from({ length: m }, (_, i) => i).sort((a, b) => (pv[a] ?? 0) - (pv[b] ?? 0) || a - b)
  const s = Float64Array.from(order, (i) => pv[i] ?? 0)
  const out = new Float64Array(m)
  let adjusted: Float64Array
  switch (method) {
    case 'bonferroni':
      adjusted = s.map((v) => v * m)
      break
    case 'sidak':
      adjusted = s.map((v) => -Math.expm1(m * Math.log1p(-v)))
      break
    case 'holm':
    case 'holm-sidak': {
      adjusted = s.map((v, i) => (method === 'holm' ? v * (m - i) : -Math.expm1((m - i) * Math.log1p(-v))))
      for (let i = 1; i < m; i++) adjusted[i] = Math.max(adjusted[i] ?? 0, adjusted[i - 1] ?? 0)
      break
    }
    case 'hochberg': {
      adjusted = s.map((v, i) => v * (m - i))
      for (let i = m - 2; i >= 0; i--) adjusted[i] = Math.min(adjusted[i] ?? 0, adjusted[i + 1] ?? 0)
      break
    }
    case 'hommel': {
      adjusted = Float64Array.from(s)
      for (let k = m; k > 1; k--) {
        let cim = Number.POSITIVE_INFINITY
        for (let j = 0; j < k; j++) cim = Math.min(cim, (k * (s[m - k + j] ?? 0)) / (j + 1))
        for (let j = m - k; j < m; j++) adjusted[j] = Math.max(adjusted[j] ?? 0, cim)
        for (let j = 0; j < m - k; j++) adjusted[j] = Math.max(adjusted[j] ?? 0, Math.min(k * (s[j] ?? 0), cim))
      }
      break
    }
    case 'benjamini-hochberg':
    case 'benjamini-yekutieli': {
      let cm = 1
      if (method === 'benjamini-yekutieli') {
        cm = 0
        for (let i = 1; i <= m; i++) cm += 1 / i
      }
      adjusted = s.map((v, i) => (v * m * cm) / (i + 1))
      for (let i = m - 2; i >= 0; i--) adjusted[i] = Math.min(adjusted[i] ?? 0, adjusted[i + 1] ?? 0)
      break
    }
    default:
      throw new WhiteboxError('value', `adjustPValues: unknown method ${String(method)}`)
  }
  order.forEach((i, r) => {
    out[i] = Math.min(1, adjusted[r] ?? 1)
  })
  return vec(out)
}

/** The result of {@link multipleTests}. */
export interface MultipleTestsResult {
  /** Whether each hypothesis is rejected at the family level alpha. */
  readonly reject: readonly boolean[]
  readonly pValuesCorrected: Tensor
  /** The per-test levels that Sidak's and Bonferroni's corrections imply. */
  readonly alphaSidak: number
  readonly alphaBonferroni: number
}

/**
 * Multiple testing (statsmodels multipletests): adjusted p-values and rejections at a family-wise or false-discovery
 * level.
 *
 * @param p - The p-values.
 * @param options - method (default `'holm'`) and alpha (default 0.05).
 * @returns Rejections, adjusted p-values and the implied per-test levels.
 * @example
 * ```ts
 * import { multipleTests } from 'whitebox-engine/stats/tests'
 * multipleTests([0.01, 0.04, 0.03], { method: 'bonferroni' }).reject // => [true, false, false]
 * ```
 */
export function multipleTests(
  p: TensorLike,
  options: { readonly method?: CorrectionMethod; readonly alpha?: number } = {},
): MultipleTestsResult {
  const alpha = options.alpha ?? 0.05
  const adjusted = adjustPValues(p, options.method ?? 'holm')
  const m = adjusted.data.length
  return {
    reject: Array.from(adjusted.data as Float64Array, (v) => v <= alpha),
    pValuesCorrected: adjusted,
    alphaSidak: 1 - (1 - alpha) ** (1 / m),
    alphaBonferroni: alpha / m,
  }
}

/**
 * Bonferroni's adjusted p-values, min(1, m p).
 *
 * @param p - The p-values.
 * @returns The adjusted p-values.
 * @example
 * ```ts
 * import { bonferroni } from 'whitebox-engine/stats/tests'
 * bonferroni([0.01, 0.2]) // => [0.02, 0.4]
 * ```
 */
export function bonferroni(p: TensorLike): Tensor {
  return adjustPValues(p, 'bonferroni')
}

/**
 * Holm's step-down adjusted p-values (uniformly more powerful than Bonferroni, same family-wise control).
 *
 * @param p - The p-values.
 * @returns The adjusted p-values.
 * @example
 * ```ts
 * import { holm } from 'whitebox-engine/stats/tests'
 * holm([0.01, 0.2]) // => [0.02, 0.2]
 * ```
 */
export function holm(p: TensorLike): Tensor {
  return adjustPValues(p, 'holm')
}

/**
 * Hochberg's step-up adjusted p-values (valid under independence or positive dependence).
 *
 * @param p - The p-values.
 * @returns The adjusted p-values.
 * @example
 * ```ts
 * import { hochberg } from 'whitebox-engine/stats/tests'
 * hochberg([0.03, 0.04]) // => [0.04, 0.04]
 * ```
 */
export function hochberg(p: TensorLike): Tensor {
  return adjustPValues(p, 'hochberg')
}

/**
 * Benjamini and Hochberg's false-discovery-rate adjusted p-values (q-values).
 *
 * @param p - The p-values.
 * @returns The adjusted p-values.
 * @example
 * ```ts
 * import { benjaminiHochberg } from 'whitebox-engine/stats/tests'
 * benjaminiHochberg([0.01, 0.02, 0.03, 0.5]) // => [0.04, 0.04, 0.04, 0.5]
 * ```
 */
export function benjaminiHochberg(p: TensorLike): Tensor {
  return adjustPValues(p, 'benjamini-hochberg')
}

/**
 * Benjamini and Yekutieli's false-discovery-rate adjusted p-values, valid under any dependence (the BH values times
 * the harmonic number of m).
 *
 * @param p - The p-values.
 * @returns The adjusted p-values.
 * @example
 * ```ts
 * import { benjaminiYekutieli } from 'whitebox-engine/stats/tests'
 * benjaminiYekutieli([0.01, 0.02]) // => [0.03, 0.03]
 * ```
 */
export function benjaminiYekutieli(p: TensorLike): Tensor {
  return adjustPValues(p, 'benjamini-yekutieli')
}

/** A method of {@link combinePValues} (scipy's names). */
export type CombineMethod = 'fisher' | 'pearson' | 'tippett' | 'stouffer' | 'mudholkar-george'

/**
 * Combine independent p-values of the same null hypothesis (scipy.stats.combine_pvalues): Fisher's -2 sum log p,
 * Pearson's, Tippett's minimum, Stouffer's (optionally weighted) Z, or Mudholkar and George's logit method.
 *
 * @param p - The p-values.
 * @param options - method (default `'fisher'`) and weights (Stouffer only).
 * @returns The combined statistic and p-value.
 * @example
 * ```ts
 * import { combinePValues } from 'whitebox-engine/stats/tests'
 * combinePValues([0.1, 0.1]).pValue // => 0.056052
 * ```
 */
export function combinePValues(
  p: TensorLike,
  options: { readonly method?: CombineMethod; readonly weights?: TensorLike } = {},
): TestResult {
  const pv = pValuesOf('combinePValues', p)
  const k = pv.length
  const method = options.method ?? 'fisher'
  let statistic: number
  let pValue: number
  switch (method) {
    case 'fisher':
      statistic = -2 * pv.reduce((s, v) => s + Math.log(v), 0)
      pValue = chi2Sf(2 * k, statistic)
      break
    case 'pearson':
      statistic = 2 * pv.reduce((s, v) => s + Math.log1p(-v), 0)
      pValue = chi2Cdf(2 * k, -statistic)
      break
    case 'tippett':
      statistic = pv.reduce((s, v) => Math.min(s, v), 1)
      pValue = -Math.expm1(k * Math.log1p(-statistic))
      break
    case 'stouffer': {
      const w = options.weights === undefined ? new Float64Array(k).fill(1) : sample('combinePValues', options.weights)
      if (w.length !== k) throw new WhiteboxError('shape', 'combinePValues: weights must match the p-values')
      let dot = 0
      let norm = 0
      for (let i = 0; i < k; i++) {
        dot -= (w[i] ?? 0) * normPpf(pv[i] ?? 0)
        norm += (w[i] ?? 0) ** 2
      }
      statistic = dot / Math.sqrt(norm)
      pValue = normSf(statistic)
      break
    }
    case 'mudholkar-george': {
      statistic = -pv.reduce((s, v) => s + Math.log(v), 0) + pv.reduce((s, v) => s + Math.log1p(-v), 0)
      const nu = 5 * k + 4
      pValue = tSf(nu, statistic * (Math.sqrt(3 / k) / Math.PI) * Math.sqrt(nu / (nu - 2)))
      break
    }
    default:
      throw new WhiteboxError('value', `combinePValues: unknown method ${String(method)}`)
  }
  return testResult({ test: 'combined p-values', statistic, pValue, method })
}
