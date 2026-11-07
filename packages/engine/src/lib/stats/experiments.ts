/**
 * Online experimentation and variance-reduction estimators: CUPED pre-experiment covariate adjustment (`cuped`),
 * post-stratification (`postStratify`), Lin's demeaned covariate-interaction regression adjustment
 * (`regressionAdjustment`), the delta method for user-clustered ratio metrics (`deltaMethodRatio`), cluster-randomised
 * inference with intracluster correlation and design effect (`clusterRandomisedTest`), and switchback experiment
 * analysis with period clustering and Newey-West HAC standard errors (`switchbackAnalysis`).
 */

import { type Estimate, MdmlError, meanOf, type TestResult, varianceOf } from '../base/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { ols } from './_linear.ts'
import { normCdf, normPpf, normSf, paired, rowsOf, sample, tCdf, tPpf, tSf, vec } from './_util.ts'
import { type TtestIndOptions, ttestInd } from './tests/means.ts'
import {
  intervalEstimate,
  pValueFromTails,
  resolveTestOptions,
  type TestOptions,
  testResult,
  waldInterval,
} from './tests/result.ts'

/** A Wald-type inference on one estimate: the estimate with its interval and the test of `value = nullValue`. */
function waldInference(
  test: string,
  value: number,
  se: number,
  df: number,
  options: TestOptions,
  nullValue = 0,
  method = test,
): { readonly estimate: Estimate; readonly test: TestResult } {
  const { alternative, level } = resolveTestOptions(options)
  const normal = !Number.isFinite(df)
  const quantile = (p: number): number => (normal ? normPpf(p) : tPpf(df, p))
  const interval = waldInterval(value, se, alternative, level, quantile)
  const estimate = intervalEstimate(value, interval, level, { standardError: se, method })
  const stat = se > 0 ? (value - nullValue) / se : value === nullValue ? 0 : Math.sign(value - nullValue) * Infinity
  const cdf = normal ? normCdf(stat) : tCdf(df, stat)
  const sf = normal ? normSf(stat) : tSf(df, stat)
  const result = testResult({
    test,
    statistic: stat,
    pValue: pValueFromTails(alternative, cdf, sf),
    alternative,
    ...(normal ? {} : { df }),
    estimate,
    method,
  })
  return { estimate, test: result }
}

/**
 * Options for {@link cuped}.
 */
export interface CupedOptions extends TestOptions {
  /**
   * Optional binary treatment assignment vector of length $n$ (`0` for control, `1` for treatment). When provided,
   * `cuped` also computes the CUPED-adjusted and unadjusted average treatment effect (ATE) and two-sample test.
   */
  readonly treatment?: TensorLike
  /** Optional fixed adjustment coefficient $\theta$; when omitted, $\hat{\theta} = \operatorname{Cov}(Y, X) / \operatorname{Var}(X)$ is estimated from the pooled sample. */
  readonly theta?: number
}

/**
 * Result of CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.
 */
export interface CupedResult {
  /** Optimal (or supplied) linear adjustment coefficient $\hat{\theta} = \operatorname{Cov}(Y, X_{\text{pre}}) / \operatorname{Var}(X_{\text{pre}})$. */
  readonly theta: number
  /** Sample Pearson correlation $\rho = \operatorname{Corr}(Y, X_{\text{pre}})$. */
  readonly correlation: number
  /** Sample mean $\bar{X}_{\text{pre}}$ of the pre-experiment covariate used for centering. */
  readonly covariateMean: number
  /** Unadjusted sample variance $\operatorname{Var}(Y)$. */
  readonly rawVariance: number
  /** Variance $\operatorname{Var}(Y_{\text{CUPED}})$ of the adjusted outcomes. */
  readonly adjustedVariance: number
  /** Variance reduction fraction $1 - \operatorname{Var}(Y_{\text{CUPED}}) / \operatorname{Var}(Y) \approx \rho^2$. */
  readonly varianceReduction: number
  /** CUPED-adjusted outcomes $Y_i^{\text{CUPED}} = Y_i - \hat{\theta}(X_i - \bar{X})$ of shape `[n]`. */
  readonly adjustedOutcomes: Tensor
  /** CUPED-adjusted average treatment effect (treatment minus control) when `options.treatment` is provided. */
  readonly ate?: Estimate
  /** Unadjusted average treatment effect (treatment minus control) when `options.treatment` is provided. */
  readonly unadjustedAte?: Estimate
  /** Welch two-sample t-test on the CUPED-adjusted outcomes when `options.treatment` is provided. */
  readonly test?: TestResult
}

function parseBinaryTreatment(
  name: string,
  treatment: TensorLike,
  n: number,
): { readonly mask: Uint8Array; readonly n1: number; readonly n0: number } {
  const t = sample(`${name}.treatment`, treatment, n)
  if (t.length !== n) {
    throw new MdmlError('shape', `${name}: treatment length ${t.length} must equal sample length ${n}`)
  }
  const mask = new Uint8Array(n)
  let n1 = 0
  let n0 = 0
  for (let i = 0; i < n; i++) {
    const v = t[i]
    if (v === 1) {
      mask[i] = 1
      n1++
    } else if (v === 0) {
      mask[i] = 0
      n0++
    } else {
      throw new MdmlError('value', `${name}: treatment must contain only 0 and 1, got ${v} at index ${i}`)
    }
  }
  if (n1 < 2 || n0 < 2) {
    throw new MdmlError('value', `${name}: need at least 2 treated and 2 control observations, got n1=${n1}, n0=${n0}`)
  }
  return { mask, n1, n0 }
}

/**
 * CUPED (Controlled-experiment Using Pre-Experiment Data; Deng et al. 2013): replaces outcomes $Y_i$ with covariate-adjusted
 * outcomes $Y_i^{\text{CUPED}} = Y_i - \hat{\theta}(X_i - \bar{X})$ where
 * $\hat{\theta} = \operatorname{Cov}(Y, X) / \operatorname{Var}(X)$ minimizes $\operatorname{Var}(Y^{\text{CUPED}})$
 * and reduces variance by a factor of $1 - \rho_{YX}^2$ without biasing the average treatment effect whenever $X$ is
 * measured pre-randomization.
 *
 * @param y - 1-D vector of post-randomization outcomes $Y_1, \dots, Y_n$.
 * @param xPre - 1-D vector of pre-randomization covariate values $X_1, \dots, X_n$.
 * @param options - Optional binary `treatment` vector, fixed `theta`, `alternative`, and confidence `level`.
 * @returns CUPED coefficient `theta`, `correlation`, `varianceReduction`, `adjustedOutcomes`, and (when `treatment` is given) `ate`, `unadjustedAte`, and `test`.
 * @throws MdmlError with code `shape` or `value` if inputs have mismatched lengths or fewer than 2 observations.
 * @example
 * ```ts
 * import { cuped } from 'mdml-engine/stats'
 * const x = [1, 2, 3, 4, 5, 6, 7, 8]
 * const t = [0, 1, 0, 1, 0, 1, 0, 1]
 * const y = x.map((xi, i) => 2 * xi + (t[i] === 1 ? 1.5 : 0))
 * const res = cuped(y, x, { treatment: t, theta: 2 })
 * Number((res.ate?.value ?? 0).toFixed(4)) // => 1.5
 * res.varianceReduction > 0.9 // => true
 * ```
 */
export function cuped(y: TensorLike, xPre: TensorLike, options: CupedOptions = {}): CupedResult {
  const [ys, xs] = paired('cuped', y, xPre, 2)
  const n = ys.length
  const meanY = meanOf(ys)
  const meanX = meanOf(xs)

  let sxx = 0
  let syy = 0
  let sxy = 0
  for (let i = 0; i < n; i++) {
    const dx = (xs[i] ?? 0) - meanX
    const dy = (ys[i] ?? 0) - meanY
    sxx += dx * dx
    syy += dy * dy
    sxy += dx * dy
  }

  const varX = sxx / (n - 1)
  const rawVariance = syy / (n - 1)
  const covYX = sxy / (n - 1)

  const theta = options.theta !== undefined ? options.theta : varX > 0 ? covYX / varX : 0
  const correlation = varX > 0 && rawVariance > 0 ? covYX / Math.sqrt(varX * rawVariance) : 0

  const adj = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    adj[i] = (ys[i] ?? 0) - theta * ((xs[i] ?? 0) - meanX)
  }
  const adjustedVariance = varianceOf(adj, 1)
  const varianceReduction = rawVariance > 0 ? Math.max(0, 1 - adjustedVariance / rawVariance) : 0

  if (options.treatment === undefined) {
    return {
      theta,
      correlation,
      covariateMean: meanX,
      rawVariance,
      adjustedVariance,
      varianceReduction,
      adjustedOutcomes: vec(adj),
    }
  }

  const { mask, n1, n0 } = parseBinaryTreatment('cuped', options.treatment, n)
  const y1Adj = new Float64Array(n1)
  const y0Adj = new Float64Array(n0)
  const y1Raw = new Float64Array(n1)
  const y0Raw = new Float64Array(n0)
  let i1 = 0
  let i0 = 0
  for (let i = 0; i < n; i++) {
    if (mask[i] === 1) {
      y1Adj[i1] = adj[i] ?? 0
      y1Raw[i1] = ys[i] ?? 0
      i1++
    } else {
      y0Adj[i0] = adj[i] ?? 0
      y0Raw[i0] = ys[i] ?? 0
      i0++
    }
  }

  const testOptions: TtestIndOptions = {
    equalVar: false,
    ...(options.alternative === undefined ? {} : { alternative: options.alternative }),
    ...(options.level === undefined ? {} : { level: options.level }),
  }
  const adjTest = ttestInd(y1Adj, y0Adj, testOptions)
  const rawTest = ttestInd(y1Raw, y0Raw, testOptions)
  const ate: Estimate = adjTest.estimate ?? { value: 0 }
  const unadjustedAte: Estimate = rawTest.estimate ?? { value: 0 }

  return {
    theta,
    correlation,
    covariateMean: meanX,
    rawVariance,
    adjustedVariance,
    varianceReduction,
    adjustedOutcomes: vec(adj),
    ate,
    unadjustedAte,
    test: adjTest,
  }
}

/**
 * Options for {@link postStratify}.
 */
export interface PostStratifyOptions extends TestOptions {
  /**
   * Optional population stratum weights aligned with sorted unique values of `strata`. When omitted, empirical sample
   * shares $w_s = n_s / n$ are used.
   */
  readonly weights?: TensorLike
}

/**
 * Result of post-stratification ATE estimation across discrete strata.
 */
export interface PostStratifyResult extends TestResult {
  /** Post-stratified average treatment effect estimate and confidence interval. */
  readonly ate: Estimate
  /**Sorted unique stratum labels $s_1, \dots, s_S$. */
  readonly strataLabels: readonly number[]
  /** Within-stratum ATE estimates $\hat{\tau}_s = \bar{Y}_{1, s} - \bar{Y}_{0, s}$ of shape `[S]`. */
  readonly stratumEffects: Tensor
  /** Normalized stratum weights $w_s$ of shape `[S]` summing to `1`. */
  readonly stratumWeights: Tensor
  /** Within-stratum sampling variances $V_s = s_{1, s}^2 / n_{1, s} + s_{0, s}^2 / n_{0, s}$ of shape `[S]`. */
  readonly stratumVariances: Tensor
  /** Unstratified difference-in-means standard error for comparison. */
  readonly unadjustedSe: number
  /** Relative variance reduction $1 - \operatorname{SE}_{\text{PS}}^2 / \operatorname{SE}_{\text{unadj}}^2$. */
  readonly varianceReduction: number
}

/**
 * Post-stratification average treatment effect estimator across discrete pre-experiment strata:
 * computes $\hat{\tau}_s = \bar{Y}_{1, s} - \bar{Y}_{0, s}$ within each stratum $s$ and combines them as
 * $\hat{\tau}_{\text{PS}} = \sum_{s=1}^S w_s \hat{\tau}_s$ with variance $\sum_{s=1}^S w_s^2 (s_{1, s}^2 / n_{1, s} + s_{0, s}^2 / n_{0, s})$,
 * eliminating between-stratum outcome variance.
 *
 * @param y - 1-D vector of outcomes $Y_1, \dots, Y_n$.
 * @param treatment - Binary treatment indicator (`0` or `1`) of length $n$.
 * @param strata - 1-D vector of discrete integer or numeric stratum labels of length $n$.
 * @param options - Optional population `weights` across sorted unique strata, `alternative`, and `level`.
 * @returns `PostStratifyResult` (`TestResult` plus per-stratum breakdown and variance reduction).
 * @throws MdmlError with code `shape` or `value` if any stratum lacks treated or control observations.
 * @example
 * ```ts
 * import { postStratify } from 'mdml-engine/stats'
 * const y = [10, 10.2, 12, 12.2, 50, 50.2, 52, 52.2]
 * const t = [0, 0, 1, 1, 0, 0, 1, 1]
 * const s = [0, 0, 0, 0, 1, 1, 1, 1]
 * const res = postStratify(y, t, s)
 * Number(res.ate.value.toFixed(4)) // => 2
 * res.varianceReduction > 0.95 // => true
 * ```
 */
export function postStratify(
  y: TensorLike,
  treatment: TensorLike,
  strata: TensorLike,
  options: PostStratifyOptions = {},
): PostStratifyResult {
  const ys = sample('postStratify.y', y, 4)
  const n = ys.length
  const { mask } = parseBinaryTreatment('postStratify', treatment, n)
  const st = sample('postStratify.strata', strata, n)
  if (st.length !== n) {
    throw new MdmlError('shape', `postStratify: strata length ${st.length} must equal y length ${n}`)
  }

  const labelSet = new Set<number>()
  for (let i = 0; i < n; i++) labelSet.add(st[i] ?? 0)
  const strataLabels = Array.from(labelSet).sort((a, b) => a - b)
  const S = strataLabels.length

  const weightsArr = new Float64Array(S)
  if (options.weights !== undefined) {
    const wIn = sample('postStratify.weights', options.weights, S)
    if (wIn.length !== S) {
      throw new MdmlError('shape', `postStratify: weights length ${wIn.length} must equal number of strata ${S}`)
    }
    let wSum = 0
    for (let s = 0; s < S; s++) {
      const ws = wIn[s] ?? 0
      if (!(ws > 0) || !Number.isFinite(ws)) {
        throw new MdmlError('value', `postStratify: weights[${s}] must be > 0, got ${ws}`)
      }
      weightsArr[s] = ws
      wSum += ws
    }
    for (let s = 0; s < S; s++) weightsArr[s] = (weightsArr[s] ?? 0) / wSum
  }

  const stratumEffects = new Float64Array(S)
  const stratumVariances = new Float64Array(S)
  let totalDf = 0

  for (let s = 0; s < S; s++) {
    const label = strataLabels[s] ?? 0
    const y1: number[] = []
    const y0: number[] = []
    for (let i = 0; i < n; i++) {
      if (st[i] === label) {
        if (mask[i] === 1) y1.push(ys[i] ?? 0)
        else y0.push(ys[i] ?? 0)
      }
    }
    if (y1.length < 1 || y0.length < 1) {
      throw new MdmlError(
        'value',
        `postStratify: stratum ${label} must contain at least 1 treated and 1 control unit (got n1=${y1.length}, n0=${y0.length})`,
      )
    }
    if (options.weights === undefined) {
      weightsArr[s] = (y1.length + y0.length) / n
    }
    const m1 = meanOf(y1)
    const m0 = meanOf(y0)
    const v1 = y1.length > 1 ? varianceOf(y1, 1) / y1.length : 0
    const v0 = y0.length > 1 ? varianceOf(y0, 1) / y0.length : 0
    stratumEffects[s] = m1 - m0
    stratumVariances[s] = v1 + v0
    totalDf += Math.max(1, y1.length + y0.length - 2)
  }

  let tauHat = 0
  let varHat = 0
  for (let s = 0; s < S; s++) {
    const w = weightsArr[s] ?? 0
    tauHat += w * (stratumEffects[s] ?? 0)
    varHat += w * w * (stratumVariances[s] ?? 0)
  }
  const se = Math.sqrt(Math.max(0, varHat))

  // Compute unadjusted difference in means SE for variance reduction comparison.
  const y1All: number[] = []
  const y0All: number[] = []
  for (let i = 0; i < n; i++) {
    if (mask[i] === 1) y1All.push(ys[i] ?? 0)
    else y0All.push(ys[i] ?? 0)
  }
  const unadjVar = varianceOf(y1All, 1) / y1All.length + varianceOf(y0All, 1) / y0All.length
  const unadjustedSe = Math.sqrt(Math.max(0, unadjVar))
  const varianceReduction = unadjVar > 0 ? Math.max(0, 1 - varHat / unadjVar) : 0

  const df = Math.max(1, totalDf)
  const { estimate: ate, test: base } = waldInference('Post-stratified difference in means', tauHat, se, df, options)

  return {
    ...base,
    ate,
    strataLabels,
    stratumEffects: vec(stratumEffects),
    stratumWeights: vec(weightsArr),
    stratumVariances: vec(stratumVariances),
    unadjustedSe,
    varianceReduction,
  }
}

/**
 * Alias for {@link postStratify}.
 *
 * @param y - 1-D vector of outcomes.
 * @param treatment - Binary treatment indicator (`0` or `1`).
 * @param strata - 1-D vector of discrete stratum labels.
 * @param options - Post-stratification options.
 * @returns Post-stratified ATE test and breakdown.
 * @example
 * ```ts
 * import { postStratification } from 'mdml-engine/stats'
 * const res = postStratification([1, 1.1, 3, 3.1, 10, 10.1, 12, 12.1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1])
 * Number(res.ate.value.toFixed(4)) // => 2
 * ```
 */
export function postStratification(
  y: TensorLike,
  treatment: TensorLike,
  strata: TensorLike,
  options: PostStratifyOptions = {},
): PostStratifyResult {
  return postStratify(y, treatment, strata, options)
}

/**
 * Options for {@link regressionAdjustment}.
 */
export interface RegressionAdjustmentOptions extends TestOptions {
  /**
   * When true (default), includes treatment-by-demeaned-covariate interactions $T_i (X_i - \bar{X})$ as in Lin (2013),
   * guaranteeing asymptotic efficiency at least as good as unadjusted difference in means even under model
   * misspecification and unequal treatment shares.
   */
  readonly interaction?: boolean
}

/**
 * Result of Lin's (2013) demeaned covariate-adjusted regression estimator for randomized experiments.
 */
export interface RegressionAdjustmentResult extends TestResult {
  /** Regression-adjusted ATE estimate and confidence interval (using HC2 robust SE). */
  readonly ate: Estimate
  /** Intercept $\hat{\alpha}$ (estimated control potential outcome mean $\bar{Y}(0)$). */
  readonly intercept: number
  /** Main covariate slope coefficients $\hat{\beta}_0$ of shape `[p]`. */
  readonly controlSlopes: Tensor
  /** Treatment-by-demeaned-covariate interaction slopes $\hat{\gamma}$ of shape `[p]` (zeros when `interaction: false`). */
  readonly interactionSlopes: Tensor
  /** Sample covariate means $\bar{X}$ used to center covariates (shape `[p]`). */
  readonly covariateMeans: Tensor
  /** Huber-White HC2 heteroscedasticity-robust standard error on the treatment effect $\hat{\tau}$. */
  readonly robustSe: number
  /** Classical homoscedastic OLS standard error on $\hat{\tau}$. */
  readonly olsSe: number
  /** Multiple $R^2$ of the regression fit. */
  readonly rSquared: number
}

/**
 * Lin's (2013) regression adjustment estimator for randomized experiments: centers covariates at the full-sample mean
 * $\tilde{X}_i = X_i - \bar{X}$ and fits
 * $Y_i = \alpha + \tau T_i + \tilde{X}_i^\top \beta_0 + T_i \tilde{X}_i^\top \gamma + \varepsilon_i$
 * by ordinary least squares with Huber-White HC2 heteroscedasticity-robust standard errors.
 *
 * @param y - 1-D vector of outcomes $Y_1, \dots, Y_n$.
 * @param treatment - Binary treatment indicator (`0` or `1`) of length $n$.
 * @param covariates - 1-D vector (length $n$) or 2-D matrix (shape `[n, p]`) of pre-treatment covariates.
 * @param options - `interaction` (default `true`), `alternative`, and confidence `level`.
 * @returns `RegressionAdjustmentResult` (`TestResult` plus `ate`, slopes, `robustSe`, `olsSe`, and `rSquared`).
 * @throws MdmlError with code `shape` or `value` for mismatched dimensions or singular design.
 * @example
 * ```ts
 * import { regressionAdjustment } from 'mdml-engine/stats'
 * const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
 * const t = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
 * const y = x.map((xi, i) => 1 + 3 * xi + (t[i] === 1 ? 2.5 : 0))
 * const res = regressionAdjustment(y, t, x)
 * Number(res.ate.value.toFixed(4)) // => 2.5
 * ```
 */
export function regressionAdjustment(
  y: TensorLike,
  treatment: TensorLike,
  covariates: TensorLike,
  options: RegressionAdjustmentOptions = {},
): RegressionAdjustmentResult {
  const ys = sample('regressionAdjustment.y', y, 4)
  const n = ys.length
  const { mask } = parseBinaryTreatment('regressionAdjustment', treatment, n)
  const covMat = rowsOf('regressionAdjustment.covariates', covariates)
  if (covMat.rows !== n) {
    throw new MdmlError('shape', `regressionAdjustment: covariates has ${covMat.rows} rows, expected ${n} to match y`)
  }
  const p = covMat.cols
  const interaction = options.interaction ?? true
  const kCols = interaction ? 2 + 2 * p : 2 + p
  if (n <= kCols) {
    throw new MdmlError(
      'shape',
      `regressionAdjustment: need n > ${kCols} observations for p=${p} covariates (interaction=${interaction}), got n=${n}`,
    )
  }

  const covMeans = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += covMat.data[i * p + j] ?? 0
    covMeans[j] = s / n
  }

  // Design columns: [1, T_i, (X_i - Xbar), T_i * (X_i - Xbar)]
  const X = new Float64Array(n * kCols)
  for (let i = 0; i < n; i++) {
    const ti = mask[i] ?? 0
    const rowOff = i * kCols
    X[rowOff] = 1
    X[rowOff + 1] = ti
    for (let j = 0; j < p; j++) {
      const xCent = (covMat.data[i * p + j] ?? 0) - (covMeans[j] ?? 0)
      X[rowOff + 2 + j] = xCent
      if (interaction) {
        X[rowOff + 2 + p + j] = ti * xCent
      }
    }
  }

  const fit = ols(X, n, kCols, ys)
  const tauHat = fit.beta[1] ?? 0
  const olsSe = fit.se[1] ?? 0

  // Compute HC2 leverage h_ii = x_i^T (X^T X)^-1 x_i and sandwich variance for column 1 (treatment).
  // Let w_i = ((X^T X)^-1 x_i)_1; then Var_HC2(tauHat) = sum_i w_i^2 * e_i^2 / (1 - h_ii).
  let hc2Var = 0
  for (let i = 0; i < n; i++) {
    const rowOff = i * kCols
    let w1 = 0
    let hii = 0
    for (let a = 0; a < kCols; a++) {
      const xa = X[rowOff + a] ?? 0
      w1 += (fit.xtxInv[1 * kCols + a] ?? 0) * xa
      for (let b = 0; b < kCols; b++) {
        hii += xa * (fit.xtxInv[a * kCols + b] ?? 0) * (X[rowOff + b] ?? 0)
      }
    }
    const denom = Math.max(1e-6, 1 - Math.min(0.999999, hii))
    const ei = fit.residuals[i] ?? 0
    hc2Var += (w1 * w1 * ei * ei) / denom
  }
  const robustSe = Math.sqrt(Math.max(0, hc2Var))

  const controlSlopes = new Float64Array(p)
  const interactionSlopes = new Float64Array(p)
  for (let j = 0; j < p; j++) {
    controlSlopes[j] = fit.beta[2 + j] ?? 0
    interactionSlopes[j] = interaction ? (fit.beta[2 + p + j] ?? 0) : 0
  }

  const meanY = meanOf(ys)
  let tss = 0
  for (let i = 0; i < n; i++) {
    const d = (ys[i] ?? 0) - meanY
    tss += d * d
  }
  const rSquared = tss > 0 ? Math.max(0, 1 - fit.rss / tss) : 1

  const { estimate: ate, test: base } = waldInference(
    'Regression-adjusted difference in means',
    tauHat,
    robustSe,
    fit.dfResid,
    options,
  )

  return {
    ...base,
    ate,
    intercept: fit.beta[0] ?? 0,
    controlSlopes: vec(controlSlopes),
    interactionSlopes: vec(interactionSlopes),
    covariateMeans: vec(covMeans),
    robustSe,
    olsSe,
    rSquared,
  }
}

/**
 * Options for {@link deltaMethodRatio}.
 */
export interface DeltaMethodRatioOptions extends TestOptions {
  /**
   * Optional binary treatment assignment vector (`0` for control, `1` for treatment) of length $n$. When provided,
   * `deltaMethodRatio` also computes the two-sample delta-method test for the difference in ratios $\hat{R}_1 - \hat{R}_0$.
   */
  readonly treatment?: TensorLike
  /** Null hypothesis value for the ratio (single-sample, default `0`) or difference in ratios (two-sample, default `0`). */
  readonly nullValue?: number
}

/**
 * Result of delta-method estimation and inference for a user-clustered ratio metric $\hat{R} = \bar{Y} / \bar{X}$.
 */
export interface DeltaMethodRatioResult {
  /** Ratio estimate $\hat{R} = \bar{Y} / \bar{X}$ (or difference $\hat{R}_1 - \hat{R}_0$ when `treatment` is given) with delta-method SE and CI. */
  readonly estimate: Estimate
  /** Overall ratio $\bar{Y} / \bar{X}$ across the sample. */
  readonly ratio: number
  /** Delta-method variance $\widehat{\operatorname{Var}}(\hat{R}) = \frac{1}{n \bar{X}^2}(s_Y^2 - 2\hat{R} s_{YX} + \hat{R}^2 s_X^2)$ of the ratio. */
  readonly variance: number
  /** Delta-method standard error $\sqrt{\widehat{\operatorname{Var}}(\hat{R})}$. */
  readonly se: number
  /** Sample mean $\bar{Y}$ of the user-level numerators. */
  readonly numeratorMean: number
  /** Sample mean $\bar{X}$ of the user-level denominators. */
  readonly denominatorMean: number
  /** User-level linearized values $L_i = \hat{R} + (Y_i - \hat{R} X_i) / \bar{X}$ whose sample mean and variance equal the delta-method moments. */
  readonly linearized: Tensor
  /** Treatment group ratio estimate when `options.treatment` is supplied. */
  readonly treatmentRatio?: Estimate
  /** Control group ratio estimate when `options.treatment` is supplied. */
  readonly controlRatio?: Estimate
  /** Relative lift $(\hat{R}_1 - \hat{R}_0) / \hat{R}_0$ when `options.treatment` is supplied. */
  readonly relativeLift?: number
  /** Asymptotic z-test for the ratio (or difference in ratios $\hat{R}_1 - \hat{R}_0$ when `options.treatment` is supplied). */
  readonly test: TestResult
}

function singleGroupRatioMoments(
  num: Float64Array,
  den: Float64Array,
): {
  readonly ratio: number
  readonly variance: number
  readonly se: number
  readonly meanY: number
  readonly meanX: number
  readonly linearized: Float64Array
} {
  const n = num.length
  const meanY = meanOf(num)
  const meanX = meanOf(den)
  if (Math.abs(meanX) < 1e-15) {
    throw new MdmlError('value', 'deltaMethodRatio: denominator mean must be non-zero')
  }
  const ratio = meanY / meanX
  let syy = 0
  let sxx = 0
  let syx = 0
  const linearized = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const yi = num[i] ?? 0
    const xi = den[i] ?? 0
    const dy = yi - meanY
    const dx = xi - meanX
    syy += dy * dy
    sxx += dx * dx
    syx += dy * dx
    linearized[i] = ratio + (yi - ratio * xi) / meanX
  }
  const varY = syy / (n - 1)
  const varX = sxx / (n - 1)
  const covYX = syx / (n - 1)
  const variance = Math.max(0, (varY - 2 * ratio * covYX + ratio * ratio * varX) / (n * meanX * meanX))
  return { ratio, variance, se: Math.sqrt(variance), meanY, meanX, linearized }
}

/**
 * Delta-method estimator for user-clustered ratio metrics $\hat{R} = \sum_{i=1}^n Y_i / \sum_{i=1}^n X_i = \bar{Y} / \bar{X}$
 * (such as click-through rate per pageview or revenue per session where each user $i$ contributes a random number of
 * denominator events $X_i$ and numerator events $Y_i$):
 * computes the first-order Taylor variance
 * $\widehat{\operatorname{Var}}(\hat{R}) = \frac{1}{n \bar{X}^2}\left(s_Y^2 - 2\hat{R} s_{YX} + \hat{R}^2 s_X^2\right)$
 * and user-level linearized scores $L_i = \hat{R} + (Y_i - \hat{R} X_i) / \bar{X}$ (Deng et al. 2018). When
 * `options.treatment` is provided, also performs the two-sample delta-method test for $\hat{R}_1 - \hat{R}_0$.
 *
 * @param numerator - 1-D vector of user-level numerator totals $Y_1, \dots, Y_n$.
 * @param denominator - 1-D vector of user-level denominator totals $X_1, \dots, X_n$ with $\bar{X} \ne 0$.
 * @param options - Optional binary `treatment` vector, `nullValue`, `alternative`, and `level`.
 * @returns Ratio estimate, delta-method variance and SE, user-level linearized scores, and hypothesis test.
 * @throws MdmlError with code `shape` or `value` if $\bar{X} = 0$ or inputs have fewer than 2 users.
 * @example
 * ```ts
 * import { deltaMethodRatio } from 'mdml-engine/stats'
 * const clicks = [2, 4, 6, 8]
 * const views = [10, 20, 30, 40]
 * const res = deltaMethodRatio(clicks, views)
 * Number(res.ratio.toFixed(4)) // => 0.2
 * Number(res.se.toFixed(6)) // => 0
 * ```
 */
export function deltaMethodRatio(
  numerator: TensorLike,
  denominator: TensorLike,
  options: DeltaMethodRatioOptions = {},
): DeltaMethodRatioResult {
  const [num, den] = paired('deltaMethodRatio', numerator, denominator, 2)
  const n = num.length
  const nullValue = options.nullValue ?? 0
  const overall = singleGroupRatioMoments(num, den)

  if (options.treatment === undefined) {
    const { estimate: est, test } = waldInference(
      'Delta-method ratio z test',
      overall.ratio,
      overall.se,
      Number.POSITIVE_INFINITY,
      options,
      nullValue,
      'delta method',
    )
    return {
      estimate: est,
      ratio: overall.ratio,
      variance: overall.variance,
      se: overall.se,
      numeratorMean: overall.meanY,
      denominatorMean: overall.meanX,
      linearized: vec(overall.linearized),
      test,
    }
  }

  const { mask, n1, n0 } = parseBinaryTreatment('deltaMethodRatio', options.treatment, n)
  const num1 = new Float64Array(n1)
  const den1 = new Float64Array(n1)
  const num0 = new Float64Array(n0)
  const den0 = new Float64Array(n0)
  let i1 = 0
  let i0 = 0
  for (let i = 0; i < n; i++) {
    if (mask[i] === 1) {
      num1[i1] = num[i] ?? 0
      den1[i1] = den[i] ?? 0
      i1++
    } else {
      num0[i0] = num[i] ?? 0
      den0[i0] = den[i] ?? 0
      i0++
    }
  }

  const m1 = singleGroupRatioMoments(num1, den1)
  const m0 = singleGroupRatioMoments(num0, den0)
  const diff = m1.ratio - m0.ratio
  const diffVar = m1.variance + m0.variance
  const diffSe = Math.sqrt(diffVar)

  const { estimate: est, test } = waldInference(
    'Delta-method difference of ratios z test',
    diff,
    diffSe,
    Number.POSITIVE_INFINITY,
    options,
    nullValue,
    'delta method',
  )
  const wald = (r: number, se: number): Estimate =>
    waldInference(
      'ratio',
      r,
      se,
      Number.POSITIVE_INFINITY,
      options.level === undefined ? {} : { level: options.level },
      0,
      'delta method',
    ).estimate
  const treatEst = wald(m1.ratio, m1.se)
  const ctrlEst = wald(m0.ratio, m0.se)

  return {
    estimate: est,
    ratio: overall.ratio,
    variance: diffVar,
    se: diffSe,
    numeratorMean: overall.meanY,
    denominatorMean: overall.meanX,
    linearized: vec(overall.linearized),
    treatmentRatio: treatEst,
    controlRatio: ctrlEst,
    relativeLift: m0.ratio !== 0 ? diff / m0.ratio : Number.NaN,
    test,
  }
}

/**
 * Result of a cluster-randomised experiment analysis (`clusterRandomisedTest`).
 */
export interface ClusterRandomisedResult extends TestResult {
  /** Cluster-robust ATE estimate and confidence interval. */
  readonly ate: Estimate
  /** Number of clusters $K$ ($K_1$ treated + $K_0$ control). */
  readonly nClusters: number
  /** Number of treated clusters $K_1$. */
  readonly nTreatedClusters: number
  /** Number of control clusters $K_0$. */
  readonly nControlClusters: number
  /** Average cluster size $\bar{m} = n / K$. */
  readonly meanClusterSize: number
  /** Estimated intracluster correlation coefficient $\text{ICC} \in [0, 1]$ from one-way random-effects ANOVA within arms. */
  readonly icc: number
  /** Kish / Donner design effect $\text{DEFF} = 1 + (\bar{m} - 1)\cdot\text{ICC}$. */
  readonly designEffect: number
  /** Effective sample size $n_{\text{eff}} = n / \text{DEFF}$. */
  readonly effectiveSampleSize: number
  /** Liang-Zeger cluster-robust standard error on the individual-level ATE. */
  readonly clusterRobustSe: number
  /** Naive individual-level standard error ignoring clustering. */
  readonly naiveSe: number
}

/**
 * Analyze a cluster-randomised experiment where entire clusters $c \in \{1, \dots, K\}$ are assigned to treatment or
 * control: estimates the intracluster correlation coefficient (ICC), design effect $\text{DEFF} = 1 + (\bar{m} - 1)\text{ICC}$,
 * effective sample size $n / \text{DEFF}$, and Liang-Zeger CR1 cluster-robust standard error with $K - 2$ degrees of
 * freedom.
 *
 * @param y - 1-D vector of unit-level outcomes $Y_1, \dots, Y_n$.
 * @param treatment - Binary unit-level treatment vector (`0` or `1`) of length $n$ (constant within each cluster).
 * @param clusterIds - 1-D vector of numeric cluster identifiers of length $n$.
 * @param options - `alternative` and confidence `level`.
 * @returns `ClusterRandomisedResult` (`TestResult` plus `ate`, `icc`, `designEffect`, `effectiveSampleSize`, `clusterRobustSe`, and `naiveSe`).
 * @throws MdmlError with code `shape` or `value` if fewer than 2 clusters per arm or treatment varies within a cluster.
 * @example
 * ```ts
 * import { clusterRandomisedTest } from 'mdml-engine/stats'
 * const y = [1, 1.1, 2, 2.1, 5, 5.1, 6, 6.1]
 * const t = [0, 0, 0, 0, 1, 1, 1, 1]
 * const c = [1, 1, 2, 2, 3, 3, 4, 4]
 * const res = clusterRandomisedTest(y, t, c)
 * Number(res.ate.value.toFixed(4)) // => 4
 * res.designEffect > 1 // => true
 * ```
 */
export function clusterRandomisedTest(
  y: TensorLike,
  treatment: TensorLike,
  clusterIds: TensorLike,
  options: TestOptions = {},
): ClusterRandomisedResult {
  const ys = sample('clusterRandomisedTest.y', y, 4)
  const n = ys.length
  const { mask } = parseBinaryTreatment('clusterRandomisedTest', treatment, n)
  const cIds = sample('clusterRandomisedTest.clusterIds', clusterIds, n)
  if (cIds.length !== n) {
    throw new MdmlError('shape', `clusterRandomisedTest: clusterIds length ${cIds.length} must equal y length ${n}`)
  }

  const clusters = new Map<number, { readonly treat: number; readonly values: number[] }>()
  for (let i = 0; i < n; i++) {
    const cid = cIds[i] ?? 0
    const tr = mask[i] ?? 0
    const existing = clusters.get(cid)
    if (existing === undefined) {
      clusters.set(cid, { treat: tr, values: [ys[i] ?? 0] })
    } else {
      if (existing.treat !== tr) {
        throw new MdmlError(
          'value',
          `clusterRandomisedTest: cluster ${cid} has mixed treatment assignments (must be cluster-level assigned)`,
        )
      }
      existing.values.push(ys[i] ?? 0)
    }
  }

  const clusterList = Array.from(clusters.values())
  const K = clusterList.length
  const treatedClusters = clusterList.filter((c) => c.treat === 1)
  const controlClusters = clusterList.filter((c) => c.treat === 0)
  const K1 = treatedClusters.length
  const K0 = controlClusters.length
  if (K1 < 2 || K0 < 2) {
    throw new MdmlError(
      'shape',
      `clusterRandomisedTest: need at least 2 treated and 2 control clusters, got K1=${K1}, K0=${K0}`,
    )
  }

  const y1All: number[] = []
  const y0All: number[] = []
  for (let i = 0; i < n; i++) {
    if (mask[i] === 1) y1All.push(ys[i] ?? 0)
    else y0All.push(ys[i] ?? 0)
  }
  const n1 = y1All.length
  const n0 = y0All.length
  const mean1 = meanOf(y1All)
  const mean0 = meanOf(y0All)
  const tauHat = mean1 - mean0
  const naiveSe = Math.sqrt(varianceOf(y1All, 1) / n1 + varianceOf(y0All, 1) / n0)

  // Estimate residual within-arm ICC via one-way random-effects ANOVA on arm-centered outcomes.
  let ssb = 0
  let ssw = 0
  let sumM2 = 0
  for (let k = 0; k < K; k++) {
    const cl = clusterList[k]
    if (cl === undefined) continue
    const armMean = cl.treat === 1 ? mean1 : mean0
    const m = cl.values.length
    const cMean = meanOf(cl.values)
    ssb += m * (cMean - armMean) * (cMean - armMean)
    for (let j = 0; j < m; j++) {
      const d = (cl.values[j] ?? 0) - cMean
      ssw += d * d
    }
    sumM2 += m * m
  }
  const dfB = K - 2
  const dfW = n - K
  let icc = 0
  if (dfB > 0 && dfW > 0) {
    const msb = ssb / dfB
    const msw = ssw / dfW
    const m0Bar = (n - sumM2 / n) / dfB
    const denom = msb + (m0Bar - 1) * msw
    if (denom > 0) {
      icc = Math.max(0, Math.min(1, (msb - msw) / denom))
    }
  }

  const meanClusterSize = n / K
  const designEffect = 1 + Math.max(0, meanClusterSize - 1) * icc
  const effectiveSampleSize = n / designEffect

  // Liang-Zeger CR1 cluster-robust variance on the individual-level difference in means:
  // For each cluster c in arm a, cluster residual sum is S_c = sum_{j in c} (Y_{cj} - mean_a) = m_c (Ybar_c - mean_a).
  let sumSq1 = 0
  for (let k = 0; k < K1; k++) {
    const cl = treatedClusters[k]
    if (cl === undefined) continue
    const sc = cl.values.length * (meanOf(cl.values) - mean1)
    sumSq1 += sc * sc
  }
  let sumSq0 = 0
  for (let k = 0; k < K0; k++) {
    const cl = controlClusters[k]
    if (cl === undefined) continue
    const sc = cl.values.length * (meanOf(cl.values) - mean0)
    sumSq0 += sc * sc
  }
  const crScale1 = K1 / (K1 - 1)
  const crScale0 = K0 / (K0 - 1)
  const crVar = (crScale1 * sumSq1) / (n1 * n1) + (crScale0 * sumSq0) / (n0 * n0)
  const clusterRobustSe = Math.sqrt(Math.max(0, crVar))

  const { estimate: ate, test: base } = waldInference(
    'Cluster-randomised difference in means',
    tauHat,
    clusterRobustSe,
    K - 2,
    options,
  )

  return {
    ...base,
    ate,
    nClusters: K,
    nTreatedClusters: K1,
    nControlClusters: K0,
    meanClusterSize,
    icc,
    designEffect,
    effectiveSampleSize,
    clusterRobustSe,
    naiveSe,
  }
}

/**
 * Alias for {@link clusterRandomisedTest}.
 *
 * @param y - 1-D vector of unit-level outcomes.
 * @param treatment - Binary unit-level treatment vector (`0` or `1`).
 * @param clusterIds - 1-D vector of cluster identifiers.
 * @param options - Test options.
 * @returns Cluster-randomised test result and design effect diagnostics.
 * @example
 * ```ts
 * import { clusterRandomisation } from 'mdml-engine/stats'
 * const res = clusterRandomisation([1, 1, 2, 2, 5, 5, 6, 6], [0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 2, 2, 3, 3, 4, 4])
 * Number(res.ate.value.toFixed(4)) // => 4
 * ```
 */
export function clusterRandomisation(
  y: TensorLike,
  treatment: TensorLike,
  clusterIds: TensorLike,
  options: TestOptions = {},
): ClusterRandomisedResult {
  return clusterRandomisedTest(y, treatment, clusterIds, options)
}

/**
 * Options for {@link switchbackAnalysis}.
 */
export interface SwitchbackOptions extends TestOptions {
  /** Maximum Newey-West autocorrelation lag $L \ge 0$ across switchback periods (default $\lfloor T^{1/4} \rfloor$). */
  readonly maxLag?: number
  /** Optional PRNG key to run a randomization (permutation) test across period treatment assignments. */
  readonly key?: Key
  /** Number of random period permutations when `key` is supplied (default `1000`). */
  readonly nPermutations?: number
}

/**
 * Result of a time-period switchback experiment analysis (`switchbackAnalysis`).
 */
export interface SwitchbackResult extends TestResult {
  /** Period-level ATE estimate and confidence interval (using HAC standard error). */
  readonly ate: Estimate
  /** Total number of switchback time periods $T$. */
  readonly nPeriods: number
  /** Sorted time-period identifiers. */
  readonly periodIds: readonly number[]
  /** Period-level average outcomes $\bar{Y}_t$ of shape `[T]`. */
  readonly periodMeans: Tensor
  /** Period-level treatment assignments $W_t \in \{0, 1\}$ of shape `[T]`. */
  readonly periodTreatments: Tensor
  /** Newey-West HAC standard error accounting for cross-period temporal autocorrelation up to `maxLag`. */
  readonly hacSe: number
  /** Period-clustered standard error assuming zero cross-period autocorrelation ($L = 0$). */
  readonly periodSe: number
  /** First-order autocorrelation $\hat{\rho}_1$ of period-level residuals. */
  readonly lag1Autocorrelation: number
  /** Exact/Monte-Carlo randomization p-value when `options.key` is provided. */
  readonly permutationPValue?: number
}

/**
 * Analyze a time-period switchback experiment (Bojinov, Simchi-Levi & Zhao 2023) where treatment $W_t \in \{0, 1\}$ is
 * toggled across sequential time periods $t = 1, \dots, T$:
 * aggregates observations to period-level means $\bar{Y}_t$, estimates the period-level average treatment effect,
 * computes both period-clustered ($L = 0$) and Newey-West HAC Bartlett-kernel standard errors up to lag `maxLag` to
 * account for temporal carryover and serial correlation, and optionally computes a randomization inference p-value.
 *
 * @param y - 1-D vector of outcomes (either unit-level observations or already aggregated period-level means).
 * @param treatment - Binary treatment indicator (`0` or `1`) of length $n$ (constant within each period).
 * @param periodIds - 1-D vector of sequential time-period identifiers of length $n$.
 * @param options - `maxLag` (default $\lfloor T^{1/4} \rfloor$), optional `key` and `nPermutations` for randomization inference, `alternative`, and `level`.
 * @returns `SwitchbackResult` (`TestResult` plus period series, `hacSe`, `periodSe`, `lag1Autocorrelation`, and optional `permutationPValue`).
 * @throws MdmlError with code `shape` or `value` if fewer than 2 treated or 2 control periods exist.
 * @example
 * ```ts
 * import { switchbackAnalysis } from 'mdml-engine/stats'
 * const y = [10, 12, 10.2, 12.2, 9.8, 11.8, 10.1, 12.1]
 * const t = [0, 1, 0, 1, 0, 1, 0, 1]
 * const p = [1, 2, 3, 4, 5, 6, 7, 8]
 * const res = switchbackAnalysis(y, t, p)
 * Number(res.ate.value.toFixed(4)) // => 2
 * res.nPeriods // => 8
 * ```
 */
export function switchbackAnalysis(
  y: TensorLike,
  treatment: TensorLike,
  periodIds: TensorLike,
  options: SwitchbackOptions = {},
): SwitchbackResult {
  const ys = sample('switchbackAnalysis.y', y, 4)
  const n = ys.length
  const { mask } = parseBinaryTreatment('switchbackAnalysis', treatment, n)
  const pIds = sample('switchbackAnalysis.periodIds', periodIds, n)
  if (pIds.length !== n) {
    throw new MdmlError('shape', `switchbackAnalysis: periodIds length ${pIds.length} must equal y length ${n}`)
  }

  const periodMap = new Map<number, { readonly treat: number; sum: number; count: number }>()
  for (let i = 0; i < n; i++) {
    const pid = pIds[i] ?? 0
    const tr = mask[i] ?? 0
    const existing = periodMap.get(pid)
    if (existing === undefined) {
      periodMap.set(pid, { treat: tr, sum: ys[i] ?? 0, count: 1 })
    } else {
      if (existing.treat !== tr) {
        throw new MdmlError(
          'value',
          `switchbackAnalysis: period ${pid} has mixed treatment assignments (must be constant within each period)`,
        )
      }
      existing.sum += ys[i] ?? 0
      existing.count += 1
    }
  }

  const sortedIds = Array.from(periodMap.keys()).sort((a, b) => a - b)
  const T = sortedIds.length
  const periodMeans = new Float64Array(T)
  const periodTreat = new Float64Array(T)
  const y1: number[] = []
  const y0: number[] = []

  for (let t = 0; t < T; t++) {
    const pid = sortedIds[t] ?? 0
    const entry = periodMap.get(pid)
    const meanVal = entry !== undefined ? entry.sum / entry.count : 0
    const tr = entry?.treat ?? 0
    periodMeans[t] = meanVal
    periodTreat[t] = tr
    if (tr === 1) y1.push(meanVal)
    else y0.push(meanVal)
  }

  if (y1.length < 2 || y0.length < 2) {
    throw new MdmlError(
      'shape',
      `switchbackAnalysis: need at least 2 treated and 2 control periods, got T1=${y1.length}, T0=${y0.length}`,
    )
  }

  const mean1 = meanOf(y1)
  const mean0 = meanOf(y0)
  const tauHat = mean1 - mean0

  // OLS / Neyman score for slope tau in Ybar_t = mean0 + tau * W_t + u_t:
  // wBar = T1 / T, score_t = ((W_t - wBar) / sum_s (W_s - wBar)^2) * u_t.
  const wBar = y1.length / T
  let sww = 0
  const u = new Float64Array(T)
  const score = new Float64Array(T)
  for (let t = 0; t < T; t++) {
    const wt = periodTreat[t] ?? 0
    const dw = wt - wBar
    sww += dw * dw
    u[t] = (periodMeans[t] ?? 0) - (wt === 1 ? mean1 : mean0)
  }
  for (let t = 0; t < T; t++) {
    const dw = (periodTreat[t] ?? 0) - wBar
    score[t] = (dw / sww) * (u[t] ?? 0)
  }

  // Lag-1 residual autocorrelation
  let uVarSum = 0
  let uCov1Sum = 0
  for (let t = 0; t < T; t++) {
    const ut = u[t] ?? 0
    uVarSum += ut * ut
    if (t > 0) uCov1Sum += ut * (u[t - 1] ?? 0)
  }
  const lag1Autocorrelation = uVarSum > 0 ? uCov1Sum / uVarSum : 0

  const maxLag = options.maxLag !== undefined ? options.maxLag : Math.floor(T ** 0.25)
  if (!Number.isInteger(maxLag) || maxLag < 0) {
    throw new MdmlError('value', `switchbackAnalysis: maxLag must be a non-negative integer, got ${maxLag}`)
  }

  // Newey-West variance of the OLS slope: the Bartlett-weighted long-run variance of the scores, with a T / (T - 2)
  // small-sample factor. Lag 0 is the period-clustered (HC1) sandwich.
  const hacVariance = (lags: number): number => {
    let omega = 0
    for (let t = 0; t < T; t++) omega += (score[t] ?? 0) ** 2
    for (let lag = 1; lag <= Math.min(lags, T - 1); lag++) {
      let gammaLag = 0
      for (let t = lag; t < T; t++) gammaLag += (score[t] ?? 0) * (score[t - lag] ?? 0)
      omega += 2 * (1 - lag / (lags + 1)) * gammaLag
    }
    return Math.max(0, (T / (T - 2)) * omega)
  }
  const periodSe = Math.sqrt(hacVariance(0))
  const hacSe = Math.sqrt(hacVariance(maxLag))

  const { alternative } = resolveTestOptions(options)
  const { estimate: ate, test: base } = waldInference(
    'Switchback difference in means',
    tauHat,
    hacSe,
    T - 2,
    options,
    0,
    'Newey-West HAC',
  )

  let permutationPValue: number | undefined
  if (options.key !== undefined) {
    const nPerm = options.nPermutations ?? 1000
    const T1 = y1.length
    const T0 = y0.length
    let totalSum = 0
    for (let t = 0; t < T; t++) totalSum += periodMeans[t] ?? 0
    let extreme = 0
    for (let b = 0; b < nPerm; b++) {
      const perm = permutation(foldIn(options.key, b), T).data
      let sum1 = 0
      for (let j = 0; j < T1; j++) {
        sum1 += periodMeans[perm[j] ?? 0] ?? 0
      }
      const permDiff = sum1 / T1 - (totalSum - sum1) / T0
      if (alternative === 'two-sided' && Math.abs(permDiff) >= Math.abs(tauHat) - 1e-12) extreme++
      else if (alternative === 'greater' && permDiff >= tauHat - 1e-12) extreme++
      else if (alternative === 'less' && permDiff <= tauHat + 1e-12) extreme++
    }
    permutationPValue = (extreme + 1) / (nPerm + 1)
  }

  return {
    ...base,
    ate,
    nPeriods: T,
    periodIds: sortedIds,
    periodMeans: vec(periodMeans),
    periodTreatments: vec(periodTreat),
    hacSe,
    periodSe,
    lag1Autocorrelation,
    ...(permutationPValue !== undefined ? { permutationPValue } : {}),
  }
}

/**
 * Alias for {@link switchbackAnalysis}.
 *
 * @param y - 1-D vector of outcomes.
 * @param treatment - Binary treatment indicator (`0` or `1`).
 * @param periodIds - 1-D vector of sequential time-period identifiers.
 * @param options - Switchback analysis options.
 * @returns Switchback experiment analysis result.
 * @example
 * ```ts
 * import { switchbackDesign } from 'mdml-engine/stats'
 * const res = switchbackDesign([1, 3, 1, 3, 1, 3], [0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6])
 * Number(res.ate.value.toFixed(4)) // => 2
 * ```
 */
export function switchbackDesign(
  y: TensorLike,
  treatment: TensorLike,
  periodIds: TensorLike,
  options: SwitchbackOptions = {},
): SwitchbackResult {
  return switchbackAnalysis(y, treatment, periodIds, options)
}
