/**
 * The log-rank family of tests that survival curves differ between two or more groups, with right-censored data: at
 * every distinct event time the observed events of each group are compared with their share of the pooled events
 * (the hypergeometric mean and covariance), and the differences are summed with a weight that sets which part of the
 * follow-up the test is sensitive to (the log-rank test weighs all times alike, Gehan-Breslow and Tarone-Ware weigh
 * early times more, Peto-Peto and Fleming-Harrington choose by the pooled survival curve). Strata are summed over.
 */

import { MdmlError, type TestResult } from '../base/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { chi2Sf, inverseSpd, normCdf } from './_util.ts'
import { type SurvivalSample, survivalSample } from './data.ts'

/** The weight of the log-rank family. */
export type LogRankWeight =
  | 'logrank'
  | 'gehan-breslow'
  | 'wilcoxon'
  | 'tarone-ware'
  | 'peto-peto'
  | 'fleming-harrington'

/** Options of {@link logRankTest}. */
export interface LogRankOptions {
  /**
   * The weight at each event time: `'logrank'` (1, the default), `'gehan-breslow'` or `'wilcoxon'` (the number at
   * risk), `'tarone-ware'` (its square root), `'peto-peto'` (the Peto-Prentice survival `prod (1 - d / (n + 1))`) or
   * `'fleming-harrington'` (`S(t-)^p (1 - S(t-))^q` of the pooled Kaplan-Meier curve).
   */
  readonly weight?: LogRankWeight
  /** Fleming-Harrington's p (default 0). */
  readonly p?: number
  /** Fleming-Harrington's q (default 0). */
  readonly q?: number
  /** A stratum for every subject: the statistic sums the within-stratum differences. */
  readonly strata?: TensorLike
}

/** The result of {@link logRankTest}: the test and what it was computed from. */
export interface LogRankResult extends TestResult {
  /** The distinct group labels, ascending. */
  readonly groups: readonly number[]
  /** Subjects per group. */
  readonly sizes: readonly number[]
  /** Observed events per group. */
  readonly observed: readonly number[]
  /** Expected events per group under the null of equal hazards. */
  readonly expected: readonly number[]
  /** The weighted observed minus expected events per group (the statistic's numerators). */
  readonly score: readonly number[]
  /** The covariance of the scores, row-major `K x K`. */
  readonly covariance: readonly number[]
  readonly weight: LogRankWeight
}

function weightOf(
  weight: LogRankWeight,
  n: number,
  survivalBefore: number,
  petoSurvival: number,
  p: number,
  q: number,
): number {
  switch (weight) {
    case 'logrank':
      return 1
    case 'gehan-breslow':
    case 'wilcoxon':
      return n
    case 'tarone-ware':
      return Math.sqrt(n)
    case 'peto-peto':
      return petoSurvival
    case 'fleming-harrington':
      return survivalBefore ** p * (1 - survivalBefore) ** q
  }
}

/**
 * The (weighted) log-rank test of equal survival in two or more groups (statsmodels' `survdiff`, R's `survdiff` and
 * `coin::logrank_test`): scores `U_g = sum_t w_t (d_gt - d_t n_gt / n_t)`, their hypergeometric covariance
 * `sum_t w_t^2 d_t (n_t - d_t) / (n_t - 1) (n_gt / n_t)(delta_gh - n_ht / n_t)`, and the chi-square statistic
 * `U' V^-1 U` on `K - 1` degrees of freedom (the last group's score is redundant).
 *
 * @param time - Observed times.
 * @param event - Event indicators (omitted: all events).
 * @param group - The group of every subject (numbers).
 * @param options - `weight`, Fleming-Harrington's `p` and `q`, and `strata`.
 * @returns The test, with observed and expected events per group.
 * @throws MdmlError with code `value` for fewer than two groups or lengths that differ.
 * @example
 * ```ts
 * import { logRankTest } from 'mdml-engine/survival'
 * const r = logRankTest([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1])
 * r.df // => 1
 * r.observed // => [4, 4]
 * r.pValue < 0.1 // => true
 * ```
 */
export function logRankTest(
  time: TensorLike,
  event: TensorLike | undefined,
  group: TensorLike,
  options: LogRankOptions = {},
): LogRankResult {
  const sample: SurvivalSample = survivalSample(time, event, 'logRankTest')
  const n = sample.n
  const g = toFlat(asTensor(group))
  if (g.length !== n) throw new MdmlError('shape', `logRankTest: ${g.length} groups for ${n} times`)
  const labels = Array.from(new Set(g)).sort((a, b) => a - b)
  const K = labels.length
  if (K < 2) throw new MdmlError('value', 'logRankTest: needs at least two groups')
  const gi = Int32Array.from(g, (v) => labels.indexOf(v))
  const strataRaw = options.strata === undefined ? new Array<number>(n).fill(0) : toFlat(asTensor(options.strata))
  if (strataRaw.length !== n) throw new MdmlError('shape', `logRankTest: ${strataRaw.length} strata for ${n} times`)
  const weight = options.weight ?? 'logrank'
  const p = options.p ?? 0
  const q = options.q ?? 0

  const observed = new Float64Array(K)
  const expected = new Float64Array(K)
  const score = new Float64Array(K)
  const cov = new Float64Array(K * K)
  const sizes = new Float64Array(K)
  for (let i = 0; i < n; i++) sizes[gi[i] ?? 0] = (sizes[gi[i] ?? 0] ?? 0) + 1

  for (const stratum of new Set(strataRaw)) {
    const idx = Array.from({ length: n }, (_, i) => i)
      .filter((i) => strataRaw[i] === stratum)
      .sort((a, b) => (sample.time[a] ?? 0) - (sample.time[b] ?? 0))
    const atRisk = new Float64Array(K)
    for (const i of idx) atRisk[gi[i] ?? 0] = (atRisk[gi[i] ?? 0] ?? 0) + 1
    let survival = 1
    let peto = 1
    let pos = 0
    while (pos < idx.length) {
      const t = sample.time[idx[pos] ?? 0] ?? 0
      const d = new Float64Array(K)
      const c = new Float64Array(K)
      let end = pos
      while (end < idx.length && (sample.time[idx[end] ?? 0] ?? 0) === t) {
        const k = gi[idx[end] ?? 0] ?? 0
        if (sample.event[idx[end] ?? 0] === 1) d[k] = (d[k] ?? 0) + 1
        else c[k] = (c[k] ?? 0) + 1
        end++
      }
      let nt = 0
      let dt = 0
      for (let k = 0; k < K; k++) {
        nt += atRisk[k] ?? 0
        dt += d[k] ?? 0
      }
      if (dt > 0) {
        const survivalBefore = survival
        survival *= 1 - dt / nt
        peto *= 1 - dt / (nt + 1)
        const w = weightOf(weight, nt, survivalBefore, peto, p, q)
        const factor = nt > 1 ? (dt * (nt - dt)) / (nt - 1) : 0
        for (let a = 0; a < K; a++) {
          const na = atRisk[a] ?? 0
          const ea = (dt * na) / nt
          observed[a] = (observed[a] ?? 0) + (d[a] ?? 0)
          expected[a] = (expected[a] ?? 0) + ea
          score[a] = (score[a] ?? 0) + w * ((d[a] ?? 0) - ea)
          for (let b = 0; b < K; b++) {
            const nb = atRisk[b] ?? 0
            cov[a * K + b] = (cov[a * K + b] ?? 0) + w * w * factor * (na / nt) * ((a === b ? 1 : 0) - nb / nt)
          }
        }
      }
      for (let k = 0; k < K; k++) atRisk[k] = (atRisk[k] ?? 0) - (d[k] ?? 0) - (c[k] ?? 0)
      pos = end
    }
  }

  const m = K - 1
  const sub = new Float64Array(m * m)
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) sub[a * m + b] = cov[a * K + b] ?? 0
  const inv = inverseSpd(sub, m)
  let chi2 = Number.NaN
  if (inv !== undefined) {
    chi2 = 0
    for (let a = 0; a < m; a++)
      for (let b = 0; b < m; b++) chi2 += (score[a] ?? 0) * (inv[a * m + b] ?? 0) * (score[b] ?? 0)
  }
  const names: Readonly<Record<LogRankWeight, string>> = {
    logrank: 'Log-rank test',
    'gehan-breslow': 'Gehan-Breslow test',
    wilcoxon: 'Gehan-Breslow (Wilcoxon) test',
    'tarone-ware': 'Tarone-Ware test',
    'peto-peto': 'Peto-Peto test',
    'fleming-harrington': `Fleming-Harrington(${p}, ${q}) test`,
  }
  return {
    test: names[weight],
    statistic: chi2,
    pValue: Number.isNaN(chi2) ? Number.NaN : chi2Sf(m, chi2),
    alternative: 'two-sided',
    df: m,
    method: 'weighted log-rank',
    groups: labels,
    sizes: Array.from(sizes),
    observed: Array.from(observed),
    expected: Array.from(expected),
    score: Array.from(score),
    covariance: Array.from(cov),
    weight,
  }
}

/**
 * The signed z statistic of a two-group (weighted) log-rank test, `U_1 / sqrt(V_11)`, positive when the first group
 * has more events than expected, with its two-sided normal p-value.
 *
 * @param result - A {@link logRankTest} of two groups.
 * @returns The z statistic and p-value.
 * @throws MdmlError with code `value` for more than two groups.
 * @example
 * ```ts
 * import { logRankTest, logRankZ } from 'mdml-engine/survival'
 * const r = logRankTest([1, 2, 3, 4, 5, 6, 7, 8], undefined, [0, 0, 0, 0, 1, 1, 1, 1])
 * logRankZ(r).z > 0 // => true
 * ```
 */
export function logRankZ(result: LogRankResult): { readonly z: number; readonly pValue: number } {
  if (result.groups.length !== 2) throw new MdmlError('value', 'logRankZ: needs a test of two groups')
  const z = (result.score[0] ?? 0) / Math.sqrt(result.covariance[0] ?? Number.NaN)
  return { z, pValue: 2 * (1 - normCdf(Math.abs(z))) }
}
