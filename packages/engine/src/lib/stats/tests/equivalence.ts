import { meanOf, type TestResult, varianceOf, WhiteboxError } from '../../base/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { normCdf, normPpf, normSf, paired, sample, tCdf, tPpf, tSf } from '../_util.ts'
import { binomTest, proportionConfint } from './proportions.ts'
import { intervalEstimate, testResult } from './result.ts'

/** The equivalence bounds of a TOST procedure: the effect must lie strictly inside (low, upp). */
export interface EquivalenceBounds {
  readonly low: number
  readonly upp: number
}

function checkBounds(name: string, b: EquivalenceBounds): void {
  if (!(b.low < b.upp)) throw new WhiteboxError('value', `${name}: low must be below upp`)
}

function tostResult(
  name: string,
  t1: number,
  p1: number,
  t2: number,
  p2: number,
  estimate: number,
  interval: [number, number],
  level: number,
  df?: number,
): TestResult {
  const upperBinds = p2 > p1
  return testResult({
    test: name,
    statistic: upperBinds ? t2 : t1,
    pValue: Math.max(p1, p2),
    alternative: 'two-sided',
    df,
    estimate: intervalEstimate(estimate, interval, level, { method: `${Math.round(level * 100)}% interval` }),
    assumptions: { statisticLower: t1, pLower: p1, statisticUpper: t2, pUpper: p2 },
    method: 'two one-sided tests',
  })
}

/** Options of {@link tost}. */
export interface TostOptions extends EquivalenceBounds {
  /** Paired samples (statsmodels ttost_paired). */
  readonly paired?: boolean
  /** Pooled variance (default true) or Welch for independent samples (statsmodels' usevar 'unequal'). */
  readonly equalVar?: boolean
  /** The significance level each one-sided test uses; the reported interval has level 1 - 2 alpha (default 0.05). */
  readonly alpha?: number
}

/**
 * Two one-sided t tests of equivalence for means (statsmodels ttost_ind, ttost_paired and DescrStatsW.ttost_mean):
 * equivalence is shown when both H0: delta <= low and H0: delta >= upp are rejected; the p-value is the larger of the
 * two, and the estimate carries the (1 - 2 alpha) interval that lies inside the bounds exactly when TOST rejects.
 *
 * @param x - First sample (or the only sample, testing its mean).
 * @param y - Second sample, or undefined.
 * @param options - low, upp, paired, equalVar and alpha.
 * @returns The binding t statistic, the TOST p-value and the difference with its interval.
 * @example
 * ```ts
 * import { tost } from 'whitebox-engine/stats/tests'
 * tost([0.1, -0.2, 0.05, 0.15, -0.1, 0.0, 0.2, -0.05], undefined, { low: -0.5, upp: 0.5 }).pValue < 0.01 // => true
 * ```
 */
export function tost(x: TensorLike, y: TensorLike | undefined, options: TostOptions): TestResult {
  checkBounds('tost', options)
  const alpha = options.alpha ?? 0.05
  let diff: number
  let se: number
  let df: number
  if (y === undefined || options.paired === true) {
    let d: Float64Array
    if (y === undefined) d = sample('tost', x, 2)
    else {
      const [a, b] = paired('tost', x, y, 2)
      d = Float64Array.from(a, (v, i) => v - (b[i] ?? 0))
    }
    diff = meanOf(d)
    se = Math.sqrt(varianceOf(d, 1) / d.length)
    df = d.length - 1
  } else {
    const a = sample('tost', x, 2)
    const b = sample('tost', y, 2)
    const nx = a.length
    const ny = b.length
    const vx = varianceOf(a, 1)
    const vy = varianceOf(b, 1)
    diff = meanOf(a) - meanOf(b)
    if (options.equalVar === false) {
      se = Math.sqrt(vx / nx + vy / ny)
      df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    } else {
      df = nx + ny - 2
      se = Math.sqrt((((nx - 1) * vx + (ny - 1) * vy) / df) * (1 / nx + 1 / ny))
    }
  }
  const t1 = (diff - options.low) / se
  const t2 = (diff - options.upp) / se
  const c = tPpf(df, 1 - alpha)
  return tostResult(
    'TOST equivalence test of means',
    t1,
    tSf(df, t1),
    t2,
    tCdf(df, t2),
    diff,
    [diff - c * se, diff + c * se],
    1 - 2 * alpha,
    df,
  )
}

/**
 * The exact binomial TOST of equivalence for one proportion (statsmodels binom_tost): the larger of the exact one-sided
 * p-values of H0: p <= low and H0: p >= upp, with the 90% Clopper-Pearson interval.
 *
 * @param k - Successes.
 * @param n - Trials.
 * @param bounds - low and upp.
 * @returns The TOST p-value with both one-sided results.
 * @example
 * ```ts
 * import { tostProportion } from 'whitebox-engine/stats/tests'
 * tostProportion(50, 100, { low: 0.35, upp: 0.65 }).pValue < 0.01 // => true
 * ```
 */
export function tostProportion(k: number, n: number, bounds: EquivalenceBounds): TestResult {
  checkBounds('tostProportion', bounds)
  const p1 = binomTest(k, n, bounds.low, { alternative: 'greater' }).pValue
  const p2 = binomTest(k, n, bounds.upp, { alternative: 'less' }).pValue
  const ci = proportionConfint(k, n, { method: 'clopper-pearson', level: 0.9 }).interval ?? [0, 1]
  return tostResult('TOST equivalence test of a proportion', k / n, p1, k / n, p2, k / n, [ci[0], ci[1]], 0.9)
}

/**
 * The TOST of equivalence for the difference of two independent proportions (statsmodels tost_proportions_2indep with
 * compare 'diff'): two one-sided z tests by the Agresti-Caffo (default) or Wald statistic.
 *
 * @param k1 - Successes in the first sample.
 * @param n1 - Its size.
 * @param k2 - Successes in the second sample.
 * @param n2 - Its size.
 * @param options - low, upp, method and alpha.
 * @returns The binding z statistic and the TOST p-value.
 * @example
 * ```ts
 * import { tostProportions } from 'whitebox-engine/stats/tests'
 * tostProportions(60, 120, 58, 120, { low: -0.15, upp: 0.15 }).pValue < 0.05 // => true
 * ```
 */
export function tostProportions(
  k1: number,
  n1: number,
  k2: number,
  n2: number,
  options: EquivalenceBounds & { readonly method?: 'agresti-caffo' | 'wald'; readonly alpha?: number },
): TestResult {
  checkBounds('tostProportions', options)
  const add = (options.method ?? 'agresti-caffo') === 'agresti-caffo' ? 1 : 0
  const a = (k1 + add) / (n1 + 2 * add)
  const b = (k2 + add) / (n2 + 2 * add)
  const se = Math.sqrt((a * (1 - a)) / (n1 + 2 * add) + (b * (1 - b)) / (n2 + 2 * add))
  const z1 = (a - b - options.low) / se
  const z2 = (a - b - options.upp) / se
  const alpha = options.alpha ?? 0.05
  const c = -normPpf(alpha)
  const diff = k1 / n1 - k2 / n2
  return tostResult(
    'TOST equivalence test of two proportions',
    z1,
    normSf(z1),
    z2,
    normCdf(z2),
    diff,
    [a - b - c * se, a - b + c * se],
    1 - 2 * alpha,
  )
}
