/**
 * Stein and empirical Bayes shrinkage estimators for the normal means problem: classic and positive-part James-Stein
 * shrinkage towards zero, a fixed vector, or the sample grand mean (with Stein's Unbiased Risk Estimate and empirical
 * Bayes interpretation), and Robbins / Efron-Morris normal-normal empirical Bayes posterior means and intervals under
 * homoscedastic or heteroscedastic sampling variances.
 */

import { type ConfidenceLevel, type Estimate, meanOf, OmaiError, resolveConfidenceLevel } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { sample, vec, zCritical } from './_util.ts'

/**
 * Options for {@link jamesStein} and {@link positivePartJamesStein}.
 */
export interface JamesSteinOptions {
  /** Known sampling standard deviation $\sigma > 0$ of each coordinate (default `1`, unless `variance` is given). */
  readonly sigma?: number
  /** Known sampling variance $\sigma^2 > 0$ of each coordinate (overrides `sigma` when supplied). */
  readonly variance?: number
  /**
   * Shrinkage target: `'zero'` (default, requires $p \ge 3$ with $df = p - 2$), `'grand-mean'` (shrinks towards
   * $\bar{y}\mathbf{1}$, requires $p \ge 4$ with $df = p - 3$), a scalar constant, or a 1-D target vector of length $p$.
   */
  readonly target?: 'zero' | 'grand-mean' | number | TensorLike
  /** When true (default `true`), applies Baranchik's positive-part truncation $\max(0, 1 - c)$ so the estimator never flips signs past the target. */
  readonly positivePart?: boolean
}

/**
 * Result of James-Stein shrinkage on a normal means vector $y \sim \mathcal{N}_p(\theta, \sigma^2 I_p)$.
 */
export interface JamesSteinResult {
  /** Shrunk point estimates $\hat{\theta}^{\text{JS}}$ of shape `[p]`. */
  readonly estimates: Tensor
  /** Original unshrunk observations $y$ of shape `[p]`. */
  readonly raw: Tensor
  /** Shrinkage target vector $\mu_0$ of shape `[p]`. */
  readonly target: Tensor
  /** Raw James-Stein shrinkage factor $c = \frac{\text{df}\cdot\sigma^2}{\|y - \mu_0\|^2}$ before positive-part clamping. */
  readonly rawShrinkageFactor: number
  /** Effective shrinkage factor applied ($c$ for classic, $\min(1, c)$ for positive-part). */
  readonly shrinkageFactor: number
  /** Retention multiplier $1 - \text{shrinkageFactor}$ applied to $y - \mu_0$. */
  readonly multiplier: number
  /** Effective degrees of freedom ($p - 2$ towards a fixed target, $p - 3$ towards the sample grand mean). */
  readonly dfShrinkage: number
  /** Squared Euclidean norm $\|y - \mu_0\|^2$ around the target. */
  readonly squaredNorm: number
  /** Stein's Unbiased Risk Estimate $\hat{R}_{\text{SURE}} = p\sigma^2 - \text{df}^2 \sigma^4 / \|y - \mu_0\|^2$ of total squared error risk. */
  readonly sureRiskEstimate: number
  /** Total squared error risk $p \sigma^2$ of the maximum-likelihood estimator $\hat{\theta}^{\text{MLE}} = y$. */
  readonly mleRisk: number
  /** Empirical Bayes prior variance estimate $\hat{\tau}^2 = \max(0, \|y - \mu_0\|^2 / \text{df} - \sigma^2)$. */
  readonly priorVarianceEstimate: number
}

/**
 * James-Stein shrinkage estimator for $y \sim \mathcal{N}_p(\theta, \sigma^2 I_p)$:
 * $\hat{\theta}^{\text{JS}} = \mu_0 + \left(1 - \frac{\text{df}\cdot\sigma^2}{\|y - \mu_0\|^2}\right)_+ (y - \mu_0)$,
 * where $\text{df} = p - 2$ when shrinking towards `'zero'` or a fixed target ($p \ge 3$), and $\text{df} = p - 3$
 * when shrinking towards `'grand-mean'` ($p \ge 4$, Lindley's modification). By default (`positivePart: true`),
 * Baranchik's positive-part truncation $(1 - c)_+ = \max(0, 1 - c)$ is applied, strictly dominating the classic
 * James-Stein estimator as well as the MLE $y$ under squared error loss.
 *
 * @param y - 1-D vector of observed normal means ($p \ge 3$ for a fixed target, $p \ge 4$ for `'grand-mean'`).
 * @param options - `sigma` or `variance` (default $\sigma^2 = 1$), `target` (default `'zero'`), and `positivePart` (default `true`).
 * @returns Shrunk estimates, target vector, shrinkage factor, SURE risk estimate, MLE risk, and empirical Bayes prior variance.
 * @throws OmaiError with code `value` if $p$ is too small for Stein domination or $\sigma^2 \le 0$, `shape` if the target length differs.
 * @example
 * ```ts
 * import { jamesStein } from 'omai-engine/stats'
 * const res = jamesStein([2, 2, 2, 2], { sigma: 1, target: 'zero' })
 * Number(res.shrinkageFactor.toFixed(4)) // => 0.125
 * Array.from(res.estimates.data) // => [1.75, 1.75, 1.75, 1.75]
 * res.sureRiskEstimate < res.mleRisk // => true
 * ```
 */
export function jamesStein(y: TensorLike, options: JamesSteinOptions = {}): JamesSteinResult {
  const ys = sample('jamesStein', y, 1)
  const p = ys.length
  const sigma2 =
    options.variance !== undefined ? options.variance : options.sigma !== undefined ? options.sigma * options.sigma : 1
  if (!(sigma2 > 0) || !Number.isFinite(sigma2)) {
    throw new OmaiError('value', `jamesStein: variance must be finite and > 0, got ${sigma2}`)
  }

  const targetOpt = options.target ?? 'zero'
  const targetArr = new Float64Array(p)
  let dfShrinkage: number

  if (targetOpt === 'grand-mean') {
    if (p < 4) {
      throw new OmaiError(
        'value',
        `jamesStein: shrinking towards 'grand-mean' requires dimension p >= 4 (so p - 3 >= 1), got p=${p}`,
      )
    }
    const gm = meanOf(ys)
    targetArr.fill(gm)
    dfShrinkage = p - 3
  } else {
    if (p < 3) {
      throw new OmaiError(
        'value',
        `jamesStein: shrinking towards a fixed target requires dimension p >= 3 (so p - 2 >= 1), got p=${p}`,
      )
    }
    dfShrinkage = p - 2
    if (targetOpt === 'zero') {
      targetArr.fill(0)
    } else if (typeof targetOpt === 'number') {
      if (!Number.isFinite(targetOpt)) {
        throw new OmaiError('value', `jamesStein: scalar target must be finite, got ${targetOpt}`)
      }
      targetArr.fill(targetOpt)
    } else {
      const tVec = sample('jamesStein.target', targetOpt, p)
      if (tVec.length !== p) {
        throw new OmaiError('shape', `jamesStein: target length ${tVec.length} must equal y length ${p}`)
      }
      targetArr.set(tVec)
    }
  }

  let squaredNorm = 0
  for (let i = 0; i < p; i++) {
    const d = (ys[i] ?? 0) - (targetArr[i] ?? 0)
    squaredNorm += d * d
  }

  const positivePart = options.positivePart ?? true
  const rawShrinkageFactor = squaredNorm > 0 ? (dfShrinkage * sigma2) / squaredNorm : Number.POSITIVE_INFINITY
  const shrinkageFactor = positivePart ? Math.min(1, rawShrinkageFactor) : rawShrinkageFactor
  const multiplier = positivePart ? Math.max(0, 1 - rawShrinkageFactor) : 1 - rawShrinkageFactor

  const est = new Float64Array(p)
  for (let i = 0; i < p; i++) {
    const mu0 = targetArr[i] ?? 0
    est[i] = Number.isFinite(multiplier) ? mu0 + multiplier * ((ys[i] ?? 0) - mu0) : mu0
  }

  const mleRisk = p * sigma2
  const sureRiskEstimate =
    squaredNorm > 0 ? mleRisk - (dfShrinkage * dfShrinkage * sigma2 * sigma2) / squaredNorm : -Number.POSITIVE_INFINITY
  const priorVarianceEstimate = Math.max(0, squaredNorm / dfShrinkage - sigma2)

  return {
    estimates: vec(est),
    raw: vec(ys),
    target: vec(targetArr),
    rawShrinkageFactor,
    shrinkageFactor,
    multiplier,
    dfShrinkage,
    squaredNorm,
    sureRiskEstimate,
    mleRisk,
    priorVarianceEstimate,
  }
}

/**
 * Positive-part James-Stein estimator $\hat{\theta}^{\text{JS}+} = \mu_0 + \max\!\left(0, 1 - \frac{\text{df}\cdot\sigma^2}{\|y - \mu_0\|^2}\right)(y - \mu_0)$
 * (Baranchik 1970), which strictly dominates the classic James-Stein estimator by clamping the multiplier at zero when
 * $\|y - \mu_0\|^2 < \text{df}\cdot\sigma^2$.
 *
 * @param y - 1-D vector of observed normal means.
 * @param options - Options forwarded to {@link jamesStein} with `positivePart: true`.
 * @returns James-Stein result with positive-part truncation enforced.
 * @example
 * ```ts
 * import { positivePartJamesStein } from 'omai-engine/stats'
 * const res = positivePartJamesStein([0.1, -0.1, 0.05], { sigma: 1 })
 * Array.from(res.estimates.data) // => [0, 0, 0]
 * res.multiplier // => 0
 * ```
 */
export function positivePartJamesStein(
  y: TensorLike,
  options: Omit<JamesSteinOptions, 'positivePart'> = {},
): JamesSteinResult {
  return jamesStein(y, { ...options, positivePart: true })
}

/**
 * Options for {@link empiricalBayesNormalMeans}.
 */
export interface EmpiricalBayesOptions {
  /** Common sampling standard deviation $\sigma > 0$ when all units share the same noise scale (default `1`). */
  readonly sigma?: number
  /** Common sampling variance $\sigma^2 > 0$ when all units share the same noise scale. */
  readonly variance?: number
  /** Per-unit sampling variances $\sigma_i^2 > 0$ of length $p$ for heteroscedastic normal means. */
  readonly variances?: TensorLike
  /** Per-unit standard errors $\sigma_i > 0$ of length $p$ for heteroscedastic normal means. */
  readonly se?: TensorLike
  /** Fixed prior mean $\mu_0$ (when omitted, estimated from the marginal distribution of $y$). */
  readonly priorMean?: number
  /** Fixed prior variance $\tau^2 \ge 0$ (when omitted, estimated by moments). */
  readonly priorVariance?: number
  /** Credible interval confidence level in `(0, 1)` (default `0.95`). */
  readonly level?: ConfidenceLevel
}

/**
 * Result of empirical Bayes normal-means shrinkage (Robbins / Efron-Morris).
 */
export interface EmpiricalBayesNormalMeansResult {
  /** Empirical Bayes posterior mean estimates $\hat{\theta}_i = (1 - B_i) y_i + B_i \hat{\mu}_0$ of shape `[p]`. */
  readonly posteriorMeans: Tensor
  /** Posterior variances $V_i = (1 - B_i)\sigma_i^2 = \frac{\sigma_i^2 \hat{\tau}^2}{\sigma_i^2 + \hat{\tau}^2}$ of shape `[p]`. */
  readonly posteriorVariances: Tensor
  /** Posterior standard deviations $\sqrt{V_i}$ of shape `[p]`. */
  readonly posteriorSe: Tensor
  /** Per-unit shrinkage factors $B_i = \frac{\sigma_i^2}{\sigma_i^2 + \hat{\tau}^2} \in [0, 1]$ towards $\hat{\mu}_0$. */
  readonly shrinkageFactors: Tensor
  /** Estimated (or fixed) prior mean $\hat{\mu}_0$. */
  readonly priorMean: number
  /** Estimated (or fixed) prior variance $\hat{\tau}^2 \ge 0$. */
  readonly priorVariance: number
  /** Per-unit empirical Bayes posterior credible intervals at `level`. */
  readonly intervals: readonly Estimate[]
}

/**
 * Empirical Bayes normal-means estimator (Robbins 1956; Efron & Morris 1973) for the hierarchical model
 * $\theta_i \sim \mathcal{N}(\mu_0, \tau^2)$, $y_i \mid \theta_i \sim \mathcal{N}(\theta_i, \sigma_i^2)$:
 * estimates hyperparameters $(\hat{\mu}_0, \hat{\tau}^2)$ from the marginal distribution
 * $y_i \sim \mathcal{N}(\mu_0, \sigma_i^2 + \tau^2)$ (the Stein / Efron-Morris unbiased estimate of the shrinkage factor in the homoscedastic case, the
 * DerSimonian-Laird moment estimator in the heteroscedastic case), then computes posterior means
 * $\hat{\theta}_i = (1 - B_i) y_i + B_i \hat{\mu}_0$ with shrinkage weights $B_i = \sigma_i^2 / (\sigma_i^2 + \hat{\tau}^2)$.
 *
 * @param y - 1-D vector of observed normal means $y_1, \dots, y_p$.
 * @param options - Sampling noise (`sigma`, `variance`, `variances`, or `se`), optional fixed `priorMean` / `priorVariance`, and `level`.
 * @returns Posterior means, variances, standard errors, shrinkage factors $B_i$, estimated hyperparameters, and credible intervals.
 * @throws OmaiError with code `shape` or `value` for invalid inputs.
 * @example
 * ```ts
 * import { empiricalBayesNormalMeans } from 'omai-engine/stats'
 * const eb = empiricalBayesNormalMeans([-3, -1, 1, 3], { sigma: 1, priorMean: 0 })
 * Number(eb.priorVariance.toFixed(4)) // => 9
 * Number((eb.shrinkageFactors.data[0] ?? 0).toFixed(4)) // => 0.1
 * ```
 */
export function empiricalBayesNormalMeans(
  y: TensorLike,
  options: EmpiricalBayesOptions = {},
): EmpiricalBayesNormalMeansResult {
  const ys = sample('empiricalBayesNormalMeans', y, 2)
  const p = ys.length
  const level = resolveConfidenceLevel(options.level)
  const z = zCritical(level)

  const sig2 = new Float64Array(p)
  let homoscedastic = true
  if (options.variances !== undefined) {
    const v = sample('empiricalBayesNormalMeans.variances', options.variances, p)
    if (v.length !== p) {
      throw new OmaiError('shape', `empiricalBayesNormalMeans: variances length ${v.length} must equal y length ${p}`)
    }
    for (let i = 0; i < p; i++) {
      const vi = v[i] ?? 0
      if (!(vi > 0) || !Number.isFinite(vi)) {
        throw new OmaiError('value', `empiricalBayesNormalMeans: variances[${i}] must be > 0, got ${vi}`)
      }
      sig2[i] = vi
      if (i > 0 && vi !== sig2[0]) homoscedastic = false
    }
  } else if (options.se !== undefined) {
    const s = sample('empiricalBayesNormalMeans.se', options.se, p)
    if (s.length !== p) {
      throw new OmaiError('shape', `empiricalBayesNormalMeans: se length ${s.length} must equal y length ${p}`)
    }
    for (let i = 0; i < p; i++) {
      const si = s[i] ?? 0
      if (!(si > 0) || !Number.isFinite(si)) {
        throw new OmaiError('value', `empiricalBayesNormalMeans: se[${i}] must be > 0, got ${si}`)
      }
      sig2[i] = si * si
      if (i > 0 && sig2[i] !== sig2[0]) homoscedastic = false
    }
  } else {
    const v0 =
      options.variance !== undefined
        ? options.variance
        : options.sigma !== undefined
          ? options.sigma * options.sigma
          : 1
    if (!(v0 > 0) || !Number.isFinite(v0)) {
      throw new OmaiError('value', `empiricalBayesNormalMeans: variance must be finite and > 0, got ${v0}`)
    }
    sig2.fill(v0)
  }

  // Estimate or use fixed priorMean and priorVariance.
  let priorMean: number
  let priorVariance: number

  if (options.priorVariance !== undefined) {
    if (!(options.priorVariance >= 0) || !Number.isFinite(options.priorVariance)) {
      throw new OmaiError(
        'value',
        `empiricalBayesNormalMeans: priorVariance must be >= 0, got ${options.priorVariance}`,
      )
    }
    priorVariance = options.priorVariance
    if (options.priorMean !== undefined) {
      priorMean = options.priorMean
    } else {
      let wSum = 0
      let wySum = 0
      for (let i = 0; i < p; i++) {
        const w = 1 / ((sig2[i] ?? 1) + priorVariance)
        wSum += w
        wySum += w * (ys[i] ?? 0)
      }
      priorMean = wySum / wSum
    }
  } else if (homoscedastic) {
    const s2 = sig2[0] ?? 1
    const fixedMean = options.priorMean !== undefined
    priorMean = fixedMean ? (options.priorMean ?? 0) : meanOf(ys)
    let ss = 0
    for (let i = 0; i < p; i++) {
      const d = (ys[i] ?? 0) - priorMean
      ss += d * d
    }
    // Efron-Morris / Stein marginal moment estimator matching (p - 2) for fixed mean or (p - 3) for estimated mean
    // when p is large enough, falling back to (p - 1) / p for small p:
    const df = fixedMean ? (p >= 3 ? p - 2 : p) : p >= 4 ? p - 3 : p - 1
    priorVariance = Math.max(0, ss / df - s2)
  } else {
    // Heteroscedastic: the DerSimonian-Laird moment estimator, then the precision-weighted mean at that tau^2.
    const fixedMean = options.priorMean !== undefined
    let wSum = 0
    let wySum = 0
    let w2Sum = 0
    for (let i = 0; i < p; i++) {
      const w = 1 / (sig2[i] ?? 1)
      wSum += w
      wySum += w * (ys[i] ?? 0)
      w2Sum += w * w
    }
    priorMean = fixedMean ? (options.priorMean ?? 0) : wySum / wSum
    let q = 0
    for (let i = 0; i < p; i++) {
      const w = 1 / (sig2[i] ?? 1)
      const d = (ys[i] ?? 0) - priorMean
      q += w * d * d
    }
    const df = fixedMean ? p : p - 1
    const c = fixedMean ? wSum : wSum - w2Sum / wSum
    priorVariance = c > 0 ? Math.max(0, (q - df) / c) : 0

    // Refine precision-weighted priorMean once tau^2 is updated.
    if (!fixedMean && priorVariance > 0) {
      let rwSum = 0
      let rwySum = 0
      for (let i = 0; i < p; i++) {
        const w = 1 / ((sig2[i] ?? 1) + priorVariance)
        rwSum += w
        rwySum += w * (ys[i] ?? 0)
      }
      priorMean = rwySum / rwSum
    }
  }

  const postMeans = new Float64Array(p)
  const postVars = new Float64Array(p)
  const postSe = new Float64Array(p)
  const shrink = new Float64Array(p)
  const intervals: Estimate[] = []

  for (let i = 0; i < p; i++) {
    const s2 = sig2[i] ?? 1
    const b = priorVariance > 0 ? s2 / (s2 + priorVariance) : 1
    const pm = (1 - b) * (ys[i] ?? 0) + b * priorMean
    const pv = (1 - b) * s2
    const se = Math.sqrt(pv)
    shrink[i] = b
    postMeans[i] = pm
    postVars[i] = pv
    postSe[i] = se
    intervals.push({
      value: pm,
      standardError: se,
      interval: [pm - z * se, pm + z * se],
      level,
      method: 'empirical Bayes posterior',
    })
  }

  return {
    posteriorMeans: vec(postMeans),
    posteriorVariances: vec(postVars),
    posteriorSe: vec(postSe),
    shrinkageFactors: vec(shrink),
    priorMean,
    priorVariance,
    intervals,
  }
}
