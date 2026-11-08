/**
 * Inference on binomial proportions: confidence intervals for one proportion (Wald, Wilson, Agresti-Coull,
 * Clopper-Pearson and Jeffreys, as statsmodels proportion_confint), the exact binomial test (scipy.stats.binomtest), z
 * tests of one or two proportions, and the difference of two proportions with its interval.
 */

import { type Alternative, type Estimate, OmaiError, type TestResult } from '../../base/index.ts'
import { binomial } from '../../dists/index.ts'
import { betaincinv } from '../../numerics/special/index.ts'
import { arange, item, type TensorLike, toFlat, toVector } from '../../tensor/index.ts'
import { normCdf, normPpf, normSf } from '../_util.ts'
import { cohensH } from '../effect-size.ts'
import { intervalEstimate, pValueFromTails, resolveTestOptions, type TestOptions, testResult } from './result.ts'

/** A method for the confidence interval of one proportion (statsmodels.stats.proportion.proportion_confint names). */
export type ProportionInterval = 'wald' | 'wilson' | 'agresti-coull' | 'clopper-pearson' | 'jeffreys'

function counts(name: string, k: number, n: number): void {
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('value', `${name}: n must be a positive integer`)
  if (!(Number.isInteger(k) && k >= 0 && k <= n))
    throw new OmaiError('value', `${name}: k must be an integer in [0, n]`)
}

const betaPpf = (a: number, b: number, p: number): number => item(betaincinv(a, b, p))

/**
 * The Wilson score interval of a proportion at a normal quantile, unclamped: the arithmetic of `proportionConfint`'s
 * `'wilson'` method for a proportion `p` of `n` trials, which `metrics` shares for weighted proportions.
 *
 * @param p - The proportion.
 * @param n - The number of trials (the effective number for weighted data).
 * @param z - The normal quantile of the coverage, `ndtri(1 - alpha)`.
 * @returns The lower and upper bound.
 * @example
 * ```ts
 * import { wilsonBounds } from 'omai-engine/stats/tests'
 * wilsonBounds(0.5, 4, 1.96).map((v) => Number(v.toFixed(4))) // => [0.15, 0.85]
 * ```
 */
export function wilsonBounds(p: number, n: number, z: number): readonly [number, number] {
  const z2 = z * z
  const denom = 1 + z2 / n
  const center = (p + z2 / (2 * n)) / denom
  const d = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom
  return [center - d, center + d]
}

/**
 * A confidence interval for a binomial proportion (statsmodels proportion_confint): Wald (`'wald'`, statsmodels'
 * `normal`), Wilson's score interval, Agresti-Coull, the exact Clopper-Pearson interval (`'clopper-pearson'`,
 * statsmodels' `beta`) and Jeffreys' equal-tailed Bayesian interval; Wald, Wilson and Agresti-Coull are clipped to [0,
 * 1]. One-sided alternatives give one-sided bounds.
 *
 * @param k - Successes.
 * @param n - Trials.
 * @param options - method (default `'wilson'`), level and alternative.
 * @returns The proportion with its interval.
 * @example
 * ```ts
 * import { proportionConfint } from 'omai-engine/stats/tests'
 * proportionConfint(5, 10, { method: 'wald' }).interval // => [0.190102, 0.809898]
 * ```
 */
export function proportionConfint(
  k: number,
  n: number,
  options: TestOptions & { readonly method?: ProportionInterval } = {},
): Estimate {
  counts('proportionConfint', k, n)
  const { alternative, level } = resolveTestOptions(options)
  const method = options.method ?? 'wilson'
  const alpha = alternative === 'two-sided' ? (1 - level) / 2 : 1 - level
  const q = k / n
  const z = normPpf(1 - alpha)
  let lo: number
  let hi: number
  switch (method) {
    case 'wald': {
      const d = z * Math.sqrt((q * (1 - q)) / n)
      ;[lo, hi] = [q - d, q + d]
      break
    }
    case 'agresti-coull': {
      const nc = n + z * z
      const qc = (k + (z * z) / 2) / nc
      const d = z * Math.sqrt((qc * (1 - qc)) / nc)
      ;[lo, hi] = [qc - d, qc + d]
      break
    }
    case 'wilson':
      ;[lo, hi] = wilsonBounds(q, n, z)
      break
    case 'clopper-pearson':
      lo = k === 0 ? 0 : betaPpf(k, n - k + 1, alpha)
      hi = k === n ? 1 : betaPpf(k + 1, n - k, 1 - alpha)
      break
    case 'jeffreys':
      lo = betaPpf(k + 0.5, n - k + 0.5, alpha)
      hi = betaPpf(k + 0.5, n - k + 0.5, 1 - alpha)
      break
    default:
      throw new OmaiError('value', `proportionConfint: unknown method ${String(method)}`)
  }
  if (method === 'wald' || method === 'agresti-coull' || method === 'wilson') {
    lo = Math.min(1, Math.max(0, lo))
    hi = Math.min(1, Math.max(0, hi))
  }
  if (alternative === 'greater') hi = 1
  if (alternative === 'less') lo = 0
  return intervalEstimate(q, [lo, hi], level, { method })
}

/**
 * The exact binomial test (scipy.stats.binomtest): one-sided tails, or for two-sided the total probability of every
 * outcome no more likely than the observed one (with scipy's relative tolerance 1e-7); the interval is Clopper-Pearson
 * by default.
 *
 * @param k - Successes.
 * @param n - Trials.
 * @param p - Hypothesised success probability (default 0.5).
 * @param options - alternative, level and the interval method.
 * @returns The proportion k / n as the statistic, the exact p-value, the interval and Cohen's h against p.
 * @example
 * ```ts
 * import { binomTest } from 'omai-engine/stats/tests'
 * binomTest(7, 10).pValue // => 0.34375
 * binomTest(7, 10, 0.5, { alternative: 'greater' }).pValue // => 0.171875
 * ```
 */
export function binomTest(
  k: number,
  n: number,
  p = 0.5,
  options: TestOptions & { readonly method?: ProportionInterval } = {},
): TestResult {
  counts('binomTest', k, n)
  if (!(p >= 0 && p <= 1)) throw new OmaiError('value', 'binomTest: p must lie in [0, 1]')
  const { alternative, level } = resolveTestOptions(options)
  const pmf = toFlat(binomial(n, p).prob(arange(0, n + 1)))
  let pValue: number
  if (alternative === 'less') pValue = pmf.slice(0, k + 1).reduce((s, v) => s + v, 0)
  else if (alternative === 'greater') pValue = pmf.slice(k).reduce((s, v) => s + v, 0)
  else if (k === p * n) pValue = 1
  else {
    const d = (pmf[k] ?? 0) * (1 + 1e-7)
    pValue = 0
    for (const v of pmf) if (v <= d) pValue += v
  }
  pValue = Math.min(1, pValue)
  return testResult({
    test: 'exact binomial test',
    statistic: k / n,
    pValue,
    alternative,
    estimate: proportionConfint(k, n, { alternative, level, method: options.method ?? 'clopper-pearson' }),
    effectSize: cohensH(k / n, p),
    assumptions: { n },
  })
}

/**
 * The z test of one proportion against a value, or of the difference of two proportions
 * (statsmodels proportions_ztest): z = (p_hat - value) / sqrt(p(1 - p) sum(1 / n_i)) with p the pooled sample
 * proportion (or `propVar` when given), referred to the standard normal. Takes tensors (migration, D-009: `count` and
 * `nobs` were `number | readonly number[]`; nested arrays still work).
 *
 * @param count - Successes: one count, or a vector of two.
 * @param nobs - Trials: one number (for every count) or a vector of one or two.
 * @param options - value (default 0 for two samples; required for one), propVar, alternative and level.
 * @returns z, its p-value and the proportion (or difference) with its Wald interval.
 * @throws OmaiError with code `shape` unless `count` and `nobs` are numbers or vectors, and `value` for counts and
 * sizes that do not match.
 * @example
 * ```ts
 * import { proportionsZtest } from 'omai-engine/stats/tests'
 * proportionsZtest([15, 10], [50, 50]).statistic // => 1.154701
 * ```
 */
export function proportionsZtest(
  count: number | TensorLike,
  nobs: number | TensorLike,
  options: TestOptions & { readonly value?: number; readonly propVar?: number } = {},
): TestResult {
  const { alternative, level } = resolveTestOptions(options)
  const c = typeof count === 'number' ? [count] : [...toVector(count, 'proportionsZtest')]
  const nn = typeof nobs === 'number' ? c.map(() => nobs) : [...toVector(nobs, 'proportionsZtest')]
  if (c.length !== nn.length || c.length < 1 || c.length > 2)
    throw new OmaiError('value', 'proportionsZtest: one or two samples of matching counts and sizes')
  for (let i = 0; i < c.length; i++) counts('proportionsZtest', c[i] ?? 0, nn[i] ?? 1)
  const props = c.map((v, i) => v / (nn[i] ?? 1))
  if (c.length === 1 && options.value === undefined)
    throw new OmaiError('value', 'proportionsZtest: value must be given for one sample')
  const value = options.value ?? 0
  const diff = c.length === 1 ? (props[0] ?? 0) - value : (props[0] ?? 0) - (props[1] ?? 0) - value
  const pooled = options.propVar ?? c.reduce((s, v) => s + v, 0) / nn.reduce((s, v) => s + v, 0)
  const fact = nn.reduce((s, v) => s + 1 / v, 0)
  const se = Math.sqrt(pooled * (1 - pooled) * fact)
  const z = diff / se
  const est = c.length === 1 ? (props[0] ?? 0) : (props[0] ?? 0) - (props[1] ?? 0)
  const seWald = Math.sqrt(props.reduce((s, v, i) => s + (v * (1 - v)) / (nn[i] ?? 1), 0))
  const zc = normPpf(0.5 + level / 2)
  return testResult({
    test: c.length === 1 ? 'one-proportion z test' : 'two-proportion z test',
    statistic: z,
    pValue: pValueFromTails(alternative, normCdf(z), normSf(z)),
    alternative,
    estimate: intervalEstimate(est, [est - zc * seWald, est + zc * seWald], level, {
      standardError: seWald,
      method: 'wald',
    }),
    ...(c.length === 2
      ? { effectSize: cohensH(props[0] ?? 0, props[1] ?? 0, { n1: nn[0] ?? 1, n2: nn[1] ?? 1, level }) }
      : {}),
  })
}

/** A method for the interval of a difference of proportions (statsmodels confint_proportions_2indep). */
export type DifferenceInterval = 'wald' | 'agresti-caffo' | 'newcombe'

/**
 * The difference of two independent proportions p1 - p2: the Wald z test of no difference (unpooled variance) and its
 * interval by Wald, Agresti-Caffo (one success and one failure added to each sample) or Newcombe's hybrid score method
 * from the two Wilson intervals (statsmodels confint_proportions_2indep with compare 'diff').
 *
 * @param k1 - Successes in the first sample.
 * @param n1 - Its size.
 * @param k2 - Successes in the second sample.
 * @param n2 - Its size.
 * @param options - method (default `'newcombe'`), alternative and level.
 * @returns z, its p-value and the difference with its interval.
 * @example
 * ```ts
 * import { proportionsDifference } from 'omai-engine/stats/tests'
 * proportionsDifference(30, 50, 20, 50).estimate?.value // => 0.2
 * ```
 */
export function proportionsDifference(
  k1: number,
  n1: number,
  k2: number,
  n2: number,
  options: TestOptions & { readonly method?: DifferenceInterval } = {},
): TestResult {
  counts('proportionsDifference', k1, n1)
  counts('proportionsDifference', k2, n2)
  const { alternative, level } = resolveTestOptions(options)
  const p1 = k1 / n1
  const p2 = k2 / n2
  const diff = p1 - p2
  const se = Math.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
  const z = diff / se
  const zc = normPpf(0.5 + level / 2)
  const method = options.method ?? 'newcombe'
  let interval: [number, number]
  if (method === 'wald') interval = [diff - zc * se, diff + zc * se]
  else if (method === 'agresti-caffo') {
    const a1 = (k1 + 1) / (n1 + 2)
    const a2 = (k2 + 1) / (n2 + 2)
    const s = Math.sqrt((a1 * (1 - a1)) / (n1 + 2) + (a2 * (1 - a2)) / (n2 + 2))
    interval = [a1 - a2 - zc * s, a1 - a2 + zc * s]
  } else {
    const [l1, u1] = proportionConfint(k1, n1, { method: 'wilson', level }).interval ?? [0, 1]
    const [l2, u2] = proportionConfint(k2, n2, { method: 'wilson', level }).interval ?? [0, 1]
    interval = [diff - Math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), diff + Math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]
  }
  return testResult({
    test: 'difference of proportions',
    statistic: z,
    pValue: pValueFromTails(alternative as Alternative, normCdf(z), normSf(z)),
    alternative,
    estimate: intervalEstimate(diff, interval, level, { standardError: se, method }),
    effectSize: cohensH(p1, p2, { n1, n2, level }),
  })
}
