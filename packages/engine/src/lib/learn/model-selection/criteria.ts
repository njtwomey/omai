import { meanOf, WhiteboxError } from '../../base/index.ts'
import { bootstrapIndices, foldIn } from '../../random/index.ts'
import { asTensor, isTensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { innerFitOptions, keyFor, runOptionsFor } from '../_shared.ts'
import { type Estimator, type FitOptions, type LearnData, requireVerb } from '../estimator.ts'
import { rowsOf, takeData, targetVector } from '../validation.ts'

/**
 * Akaike's information criterion, 2k - 2 log L.
 *
 * @param logLikelihood - The maximised log-likelihood.
 * @param k - The number of estimated parameters.
 * @returns AIC (lower is better).
 * @concept information-criteria
 * @example
 * ```ts
 * import { aic } from 'whitebox-engine/learn/model-selection'
 * aic(-10, 3) // => 26
 * ```
 */
export function aic(logLikelihood: number, k: number): number {
  return 2 * k - 2 * logLikelihood
}

/**
 * AIC with the small-sample correction (Hurvich and Tsai 1989): AIC + 2k(k + 1) / (n - k - 1).
 *
 * @param logLikelihood - The maximised log-likelihood.
 * @param k - The number of parameters.
 * @param n - The number of observations (more than k + 1).
 * @returns AICc.
 * @throws WhiteboxError with code `domain` when n <= k + 1.
 * @example
 * ```ts
 * import { aicc } from 'whitebox-engine/learn/model-selection'
 * aicc(-10, 3, 10) // => 30
 * ```
 */
export function aicc(logLikelihood: number, k: number, n: number): number {
  if (n <= k + 1) throw new WhiteboxError('domain', `aicc: needs n > k + 1 (n = ${n}, k = ${k})`)
  return aic(logLikelihood, k) + (2 * k * (k + 1)) / (n - k - 1)
}

/**
 * The Bayesian (Schwarz) information criterion, k log n - 2 log L.
 *
 * @param logLikelihood - The maximised log-likelihood.
 * @param k - The number of parameters.
 * @param n - The number of observations.
 * @returns BIC (lower is better).
 * @example
 * ```ts
 * import { bic } from 'whitebox-engine/learn/model-selection'
 * bic(-10, 2, Math.E ** 2) // => 24
 * ```
 */
export function bic(logLikelihood: number, k: number, n: number): number {
  return k * Math.log(n) - 2 * logLikelihood
}

/**
 * The maximised Gaussian log-likelihood of a least-squares fit, -n/2 (log(2 pi RSS / n) + 1) (statsmodels' OLS llf).
 *
 * @param rss - The residual sum of squares.
 * @param n - The number of observations.
 * @returns The log-likelihood.
 * @example
 * ```ts
 * import { gaussianLogLikelihood } from 'whitebox-engine/learn/model-selection'
 * gaussianLogLikelihood(10, 10) === -5 * (Math.log(2 * Math.PI) + 1) // => true
 * ```
 */
export function gaussianLogLikelihood(rss: number, n: number): number {
  return (-n / 2) * (Math.log((2 * Math.PI * rss) / n) + 1)
}

/**
 * Mallows' Cp, RSS / sigma^2 - n + 2p, with sigma^2 estimated from the largest model.
 *
 * @param rss - The residual sum of squares of the candidate.
 * @param sigma2 - The error variance estimate.
 * @param n - The number of observations.
 * @param p - The number of parameters of the candidate (with the intercept).
 * @returns Cp (close to p for an unbiased model).
 * @example
 * ```ts
 * import { mallowsCp } from 'whitebox-engine/learn/model-selection'
 * mallowsCp(20, 2, 12, 3) // => 4
 * ```
 */
export function mallowsCp(rss: number, sigma2: number, n: number, p: number): number {
  return rss / sigma2 - n + 2 * p
}

/**
 * Generalised cross-validation (Craven and Wahba 1979; Golub, Heath and Wahba): n RSS / (n - df)^2, the
 * rotation-invariant approximation to leave-one-out for a linear smoother with df = trace(S).
 *
 * @param rss - The residual sum of squares.
 * @param n - The number of observations.
 * @param df - The effective degrees of freedom (trace of the hat matrix).
 * @returns The GCV score (lower is better).
 * @throws WhiteboxError with code `domain` when df >= n.
 * @example
 * ```ts
 * import { gcv } from 'whitebox-engine/learn/model-selection'
 * gcv(8, 10, 2) // => 1.25
 * ```
 */
export function gcv(rss: number, n: number, df: number): number {
  if (df >= n) throw new WhiteboxError('domain', 'gcv: the degrees of freedom must be below n')
  return (n * rss) / (n - df) ** 2
}

/** Leave-one-out quantities of a linear smoother. */
export interface SmootherLoo {
  /** Leave-one-out residuals (y_i - yhat_i) / (1 - S_ii). */
  readonly residuals: readonly number[]
  /** Their mean square: the leave-one-out mean squared error. */
  readonly mse: number
  /** The generalised cross-validation score. */
  readonly gcv: number
  /** The effective degrees of freedom, trace(S). */
  readonly df: number
}

/**
 * Exact leave-one-out for a linear smoother yhat = S y (least squares, ridge, splines, kernel smoothers) without
 * refitting: the residual of a left-out point is its ordinary residual over 1 - S_ii.
 *
 * @param hat - The smoother (hat) matrix `[n, n]`.
 * @param y - The targets `[n]`.
 * @returns The LOO residuals, LOO MSE, GCV and degrees of freedom.
 * @throws WhiteboxError with code `domain` when some S_ii is 1 (a point fitted exactly by itself).
 * @example
 * ```ts
 * import { linearSmootherLoo } from 'whitebox-engine/learn/model-selection'
 * // The mean as a smoother: S = J / n, so the LOO residual is (y_i - mean) n / (n - 1).
 * linearSmootherLoo([[0.5, 0.5], [0.5, 0.5]], [1, 3]).residuals // => [-2, 2]
 * ```
 */
export function linearSmootherLoo(hat: TensorLike, y: TensorLike): SmootherLoo {
  const s = asTensor(hat)
  if (!isTensor(s) || s.shape.length !== 2 || s.shape[0] !== s.shape[1]) {
    throw new WhiteboxError('shape', 'linearSmootherLoo: the hat matrix must be square')
  }
  const n = s.shape[0] ?? 0
  const v = targetVector(y, 'linearSmootherLoo', n)
  const sv = toFlat(s)
  const residuals: number[] = []
  let rss = 0
  let df = 0
  for (let i = 0; i < n; i++) {
    let fit = 0
    for (let j = 0; j < n; j++) fit += (sv[i * n + j] ?? 0) * (v[j] ?? 0)
    const h = sv[i * n + i] ?? 0
    if (Math.abs(1 - h) < 1e-12) throw new WhiteboxError('domain', `linearSmootherLoo: leverage of point ${i} is 1`)
    const r = (v[i] ?? 0) - fit
    rss += r * r
    df += h
    residuals.push(r / (1 - h))
  }
  return { residuals, mse: meanOf(residuals.map((r) => r * r)), gcv: gcv(rss, n, df), df }
}

/** Options of {@link oneStandardErrorRule}. */
export interface OneStandardErrorOptions {
  /** The complexity of each candidate (default its index): the simplest qualifying candidate is chosen. */
  readonly complexity?: readonly number[]
}

/**
 * The one-standard-error rule (Breiman et al. 1984; Hastie, Tibshirani and Friedman): among candidates whose mean
 * score is within one standard error of the best, choose the simplest.
 *
 * @param means - Mean scores (higher is better).
 * @param standardErrors - Their standard errors.
 * @param options - Complexities (default the candidate order, simplest first).
 * @returns The index of the chosen candidate.
 * @example
 * ```ts
 * import { oneStandardErrorRule } from 'whitebox-engine/learn/model-selection'
 * oneStandardErrorRule([0.70, 0.78, 0.80, 0.79], [0.03, 0.03, 0.03, 0.03]) // => 1
 * ```
 */
export function oneStandardErrorRule(
  means: readonly number[],
  standardErrors: readonly number[],
  options: OneStandardErrorOptions = {},
): number {
  if (means.length === 0 || means.length !== standardErrors.length) {
    throw new WhiteboxError('shape', 'oneStandardErrorRule: needs one standard error per mean')
  }
  let best = 0
  means.forEach((m, i) => {
    if (m > (means[best] ?? Number.NEGATIVE_INFINITY)) best = i
  })
  const threshold = (means[best] ?? 0) - (standardErrors[best] ?? 0)
  const complexity = options.complexity ?? means.map((_, i) => i)
  let chosen = best
  means.forEach((m, i) => {
    if (m >= threshold && (complexity[i] ?? i) < (complexity[chosen] ?? chosen)) chosen = i
  })
  return chosen
}

/** Options of {@link bootstrap632}. */
export interface Bootstrap632Options extends FitOptions {
  /** Bootstrap replicates (default 200). */
  readonly nBootstraps?: number
  /** The pointwise loss (default squared error for regressors, 0-1 for classifiers). */
  readonly loss?: (yTrue: number, yPred: number) => number
  /** Use Efron and Tibshirani's .632+ correction for overfitting (default false). */
  readonly plus?: boolean
}

/** The result of {@link bootstrap632}. */
export interface Bootstrap632Result {
  /** The .632 (or .632+) estimate of the prediction error. */
  readonly error: number
  /** The training (resubstitution) error. */
  readonly trainingError: number
  /** The leave-one-out bootstrap error (each sample scored only by models not trained on it). */
  readonly outOfBagError: number
  /** The no-information error rate (used by .632+). */
  readonly noInformationError: number
  /** The weight on the out-of-bag error (0.632, or the .632+ weight). */
  readonly weight: number
}

/**
 * The .632 and .632+ bootstrap estimates of prediction error (Efron 1983; Efron and Tibshirani 1997): a weighted mix
 * of the optimistic training error and the pessimistic leave-one-out bootstrap error, with .632+ shifting weight to
 * the latter as the model overfits. A traced algorithm with one step per replicate.
 *
 * @param estimator - The estimator (it must predict).
 * @param data - The data.
 * @param options - nBootstraps, loss, plus, key and tracing.
 * @returns The estimate and its parts.
 * @concept bootstrap
 * @example
 * ```ts
 * import { bootstrap632 } from 'whitebox-engine/learn/model-selection'
 * import { defineEstimator, rowsOf } from 'whitebox-engine/learn'
 * import { full } from 'whitebox-engine/tensor'
 * const zero = defineEstimator({ name: 'zero', kind: 'regressor', fit: () => ({}), methods: () => ({ predict: (x) => full([rowsOf(x)], 0) }) }, {})
 * bootstrap632(zero, { x: [[0], [1], [2], [3]], y: [1, 1, 1, 1] }, { nBootstraps: 20 }).error // => 1
 * ```
 */
export function bootstrap632(
  estimator: Estimator,
  data: LearnData,
  options: Bootstrap632Options = {},
): Bootstrap632Result {
  const n = rowsOf(data.x)
  const y = targetVector(data.y ?? [], 'bootstrap632', n)
  const nBoot = options.nBootstraps ?? 200
  if (!(Number.isInteger(nBoot) && nBoot >= 1))
    throw new WhiteboxError('value', 'bootstrap632: nBootstraps must be >= 1')
  const loss =
    options.loss ??
    (estimator.kind === 'classifier'
      ? (a: number, b: number) => (a === b ? 0 : 1)
      : (a: number, b: number) => (a - b) ** 2)
  const key = keyFor(options, 'bootstrap632')
  const full = estimator.fit(data, innerFitOptions(options, foldIn(key, 0), options.parent))
  const fullPred = toFlat(requireVerb(full, 'predict', 'bootstrap632')(data.x))
  const trainingError = meanOf(Array.from(y, (v, i) => loss(v, fullPred[i] ?? 0)))
  const sum = new Float64Array(n)
  const count = new Float64Array(n)
  const algorithm: Algorithm<number> = {
    name: 'bootstrap-632',
    config: { estimator: estimator.name, nBootstraps: nBoot, plus: options.plus === true },
    schema: channels({
      metrics: { outOfBag: { description: 'the running leave-one-out bootstrap error', direction: 'lower' } },
    }),
    init: () => 0,
    step: (b, ctx) => {
      const rows = toFlat(bootstrapIndices(foldIn(key, 1 + b), n))
      const inBag = new Uint8Array(n)
      for (const r of rows) inBag[r] = 1
      const oob = Array.from({ length: n }, (_, i) => i).filter((i) => inBag[i] === 0)
      if (oob.length > 0) {
        const fitted = estimator.fit(takeData(data, rows), innerFitOptions(options, foldIn(ctx.key, 0), ctx.run))
        const pred = toFlat(requireVerb(fitted, 'predict', 'bootstrap632')(takeData(data, oob).x))
        oob.forEach((i, r) => {
          sum[i] = (sum[i] ?? 0) + loss(y[i] ?? 0, pred[r] ?? 0)
          count[i] = (count[i] ?? 0) + 1
        })
      }
      return b + 1
    },
    done: (b) => (b >= nBoot ? 'completed' : undefined),
    observe: () => {
      const seen = Array.from(count.keys()).filter((i) => (count[i] ?? 0) > 0)
      return {
        metrics: seen.length === 0 ? {} : { outOfBag: meanOf(seen.map((i) => (sum[i] ?? 0) / (count[i] ?? 1))) },
      }
    },
  }
  runAlgorithm(algorithm, { ...runOptionsFor(options, 'bootstrap632'), maxSteps: nBoot })
  const seen = Array.from(count.keys()).filter((i) => (count[i] ?? 0) > 0)
  if (seen.length === 0)
    throw new WhiteboxError('value', 'bootstrap632: no sample was ever out of bag; use more replicates')
  const outOfBagError = meanOf(seen.map((i) => (sum[i] ?? 0) / (count[i] ?? 1)))
  let gamma = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) gamma += loss(y[i] ?? 0, fullPred[j] ?? 0)
  gamma /= n * n
  let weight = 0.632
  if (options.plus === true) {
    const oob = Math.min(outOfBagError, gamma)
    const r = oob > trainingError && gamma > trainingError ? (oob - trainingError) / (gamma - trainingError) : 0
    weight = 0.632 / (1 - 0.368 * r)
    return {
      error: (1 - weight) * trainingError + weight * oob,
      trainingError,
      outOfBagError,
      noInformationError: gamma,
      weight,
    }
  }
  return {
    error: 0.368 * trainingError + 0.632 * outOfBagError,
    trainingError,
    outOfBagError,
    noInformationError: gamma,
    weight,
  }
}
