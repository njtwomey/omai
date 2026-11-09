/**
 * Automatic relevance determination regression (MacKay; Tipping), scikit-learn's `ARDRegression`: a linear model with
 * a Gaussian prior of its own precision on each weight, fitted by maximising the evidence. Each step re-estimates the
 * posterior of the weights (their covariance `sigma` and mean) and then the precisions `lambda` of the weights and
 * `alpha` of the noise by the fixed-point updates with Gamma hyperpriors, pruning the weights whose precision passes a
 * threshold: irrelevant features are switched off. `ardAlgorithm` is that loop as a traced `Algorithm`; `ardRegression`
 * is the estimator on the iterative base, with the posterior predictive standard deviation as a verb.
 */

import { type IterativeEstimator, iterative, type LearnData, matrixView, targetVector } from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { dgemm, dgemv, eigh, slogdet } from '../numerics/linalg/index.ts'
import { dict, real } from '../space/index.ts'
import { fromStorage, item, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { centre, type Design, designOf, interceptFor, linearPredictor, vectorTensor } from './design.ts'

/** The hyperpriors and stopping rule of {@link ardAlgorithm} (scikit-learn's names and defaults). */
export interface ArdOptions {
  /** Shape of the Gamma prior on the noise precision (default 1e-6). */
  readonly alpha1?: number
  /** Rate of the Gamma prior on the noise precision (default 1e-6). */
  readonly alpha2?: number
  /** Shape of the Gamma prior on the weight precisions (default 1e-6). */
  readonly lambda1?: number
  /** Rate of the Gamma prior on the weight precisions (default 1e-6). */
  readonly lambda2?: number
  /** Weights whose precision reaches this are pruned to zero (default 1e4). */
  readonly thresholdLambda?: number
  /** Stop when the coefficients' L1 change falls below this (default 1e-3). */
  readonly tol?: number
  /** Fit an intercept by centring (default true). */
  readonly fitIntercept?: boolean
}

/**
 * The state of {@link ardAlgorithm}: the weights, the precisions of the weights and of the noise, which weights are
 * kept, the last log evidence and how far the weights moved. Before the first step `change` is `Number.MAX_VALUE`.
 */
export type ArdState = {
  readonly coef: number[]
  /** The noise precision. */
  readonly alpha: number
  /** The precision of each weight's prior. */
  readonly lambda: number[]
  readonly keep: boolean[]
  /** The weights of the previous step, for the stopping rule. */
  readonly previous: number[]
  /** The L1 distance between the last two steps' weights. */
  readonly change: number
  /** The objective scikit-learn calls `scores_` (the log evidence up to a constant) at the last step. */
  readonly score: number
  readonly iteration: number
}

/** Symmetric pseudo-inverse (scipy's `pinvh`) of a row-major `[m, m]` matrix, and the sum of the logs of its kept eigenvalues. */
function pinvh(a: Float64Array, m: number): Float64Array {
  const { eigenvalues, eigenvectors } = eigh(fromStorage(a, [m, m], 'float64'))
  const w = toFlat(eigenvalues)
  const v = toFlat(eigenvectors)
  const cutoff = m * Number.EPSILON * Math.max(...w.map(Math.abs), 0)
  const out = new Float64Array(m * m)
  for (let k = 0; k < m; k++) {
    const wk = w[k] ?? 0
    if (Math.abs(wk) <= cutoff) continue
    for (let i = 0; i < m; i++) {
      const vi = (v[i * m + k] ?? 0) / wk
      for (let j = 0; j < m; j++) out[i * m + j] = (out[i * m + j] ?? 0) + vi * (v[j * m + k] ?? 0)
    }
  }
  return out
}

/** The columns of a row-major `[n, d]` matrix that are kept. */
function keptColumns(x: Float64Array, n: number, d: number, keep: readonly boolean[]): { x: Float64Array; m: number } {
  const cols = keep.flatMap((k, j) => (k ? [j] : []))
  const out = new Float64Array(n * cols.length)
  for (let i = 0; i < n; i++) for (const [c, j] of cols.entries()) out[i * cols.length + c] = x[i * d + j] ?? 0
  return { x: out, m: cols.length }
}

/**
 * The posterior covariance of the kept weights, `(diag(lambda) + alpha X^T X)^-1`, row-major `[m, m]`: inverted
 * directly when there are at least as many rows as features, and through the Woodbury identity (one `[n, n]` inverse)
 * when there are fewer, as scikit-learn does.
 */
function posteriorCovariance(
  x: Float64Array,
  n: number,
  d: number,
  alpha: number,
  lambda: readonly number[],
  keep: readonly boolean[],
): { sigma: Float64Array; m: number; xk: Float64Array } {
  const { x: xk, m } = keptColumns(x, n, d, keep)
  const kept = lambda.filter((_, j) => keep[j])
  if (n >= d) {
    const gram = dgemm(xk, xk, m, n, m, { transposeA: true })
    const a = new Float64Array(m * m)
    for (let i = 0; i < m * m; i++) a[i] = alpha * (gram[i] ?? 0)
    for (let i = 0; i < m; i++) a[i * m + i] = (a[i * m + i] ?? 0) + (kept[i] ?? 0)
    return { sigma: pinvh(a, m), m, xk }
  }
  // Woodbury: Sigma = L^-1 - L^-1 X^T (I / alpha + X L^-1 X^T)^-1 X L^-1, with L = diag(lambda).
  const scaled = Float64Array.from(xk, (v, i) => v / (kept[i % m] ?? 1))
  const inner = dgemm(scaled, xk, n, m, n, { transposeB: true })
  for (let i = 0; i < n; i++) inner[i * n + i] = (inner[i * n + i] ?? 0) + 1 / alpha
  const middle = dgemm(pinvh(inner, n), scaled, n, n, m)
  const sigma = dgemm(xk, middle, m, n, m, { transposeA: true })
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) sigma[i * m + j] = -(sigma[i * m + j] ?? 0) / (kept[i] ?? 1)
    sigma[i * m + i] = (sigma[i * m + i] ?? 0) + 1 / (kept[i] ?? 1)
  }
  return { sigma, m, xk }
}

/** The posterior mean of the kept weights placed among zeros: `alpha sigma X^T y`. */
function posteriorMean(
  sigma: Float64Array,
  xk: Float64Array,
  y: Float64Array,
  n: number,
  m: number,
  alpha: number,
  keep: readonly boolean[],
): number[] {
  const xty = dgemv(xk, n, m, y, true)
  const mu = dgemv(sigma, m, m, xty)
  const coef = new Array<number>(keep.length).fill(0)
  let c = 0
  for (const [j, k] of keep.entries()) if (k) coef[j] = alpha * (mu[c++] ?? 0)
  return coef
}

const ardSchema = (d: number): ReturnType<typeof channels> =>
  channels({
    metrics: {
      alpha: { description: 'the noise precision' },
      change: { description: 'the L1 distance the weights moved this step' },
      active: { description: 'the weights not yet pruned' },
      logEvidence: { description: "scikit-learn's objective: the log evidence up to a constant", direction: 'higher' },
    },
    tensors: {
      coefficients: { role: 'param', description: 'the posterior mean of the weights', shape: [d] },
      precisions: { role: 'param', description: 'the precision of the prior of each weight', shape: [d] },
    },
  })

/**
 * Evidence maximisation for automatic relevance determination as a traced algorithm, one iteration of
 * scikit-learn's loop per step: the posterior of the kept weights under the current precisions, then the updates
 * `lambda_j = (gamma_j + 2 lambda1) / (w_j^2 + 2 lambda2)` and `alpha = (n - sum gamma + 2 alpha1) / (rss + 2 alpha2)`
 * with `gamma_j = 1 - lambda_j sigma_jj`, then the pruning of weights whose precision reaches `thresholdLambda`. The
 * run ends when the weights move by less than `tol` in L1 or all are pruned. The state keeps no past.
 *
 * @param design - Features, targets and weights.
 * @param options - The hyperpriors, the pruning threshold, the tolerance and the intercept.
 * @returns The algorithm. The fitted weights are those of {@link ardCoefficients} on its final state.
 * @index algorithm Automatic relevance determination by evidence maximisation, one iteration per step.
 * @example
 * ```ts
 * import { ardAlgorithm, designOf } from 'omai-engine/linear'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), (i % 5) - 2])
 * const y = x.map((r) => 2 * r[0] + 0.01 * r[2])
 * const end = runAlgorithm(ardAlgorithm(designOf({ x, y }, 'demo')), { maxSteps: 300 }).state
 * end.keep // => [true, false, false]
 * ```
 */
export function ardAlgorithm(design: Design, options: ArdOptions = {}): Algorithm<ArdState> {
  const { alpha1 = 1e-6, alpha2 = 1e-6, lambda1 = 1e-6, lambda2 = 1e-6, thresholdLambda = 1e4, tol = 1e-3 } = options
  const fitIntercept = options.fitIntercept ?? true
  const { n, d } = design.x
  const c = centre(design, fitIntercept)
  const y = c.y
  const x = c.x
  let mean = 0
  for (const v of y) mean += v / n
  let variance = 0
  for (const v of y) variance += (v - mean) ** 2 / n
  return {
    name: 'ard-regression',
    schema: ardSchema(d),
    config: { alpha1, alpha2, lambda1, lambda2, thresholdLambda, tol, fitIntercept },
    init: () => ({
      coef: new Array<number>(d).fill(0),
      alpha: 1 / (variance + Number.EPSILON),
      lambda: new Array<number>(d).fill(1),
      keep: new Array<boolean>(d).fill(true),
      previous: new Array<number>(d).fill(0),
      change: Number.MAX_VALUE,
      score: Number.NEGATIVE_INFINITY,
      iteration: 0,
    }),
    step: (s) => {
      const { sigma, m, xk } = posteriorCovariance(x, n, d, s.alpha, s.lambda, s.keep)
      const coef = posteriorMean(sigma, xk, y, n, m, s.alpha, s.keep)
      const fitted = dgemv(x, n, d, coef)
      let sse = 0
      for (let i = 0; i < n; i++) sse += ((y[i] ?? 0) - (fitted[i] ?? 0)) ** 2
      const lambda = [...s.lambda]
      let gammaSum = 0
      let c2 = 0
      for (const [j, k] of s.keep.entries()) {
        if (!k) continue
        const gamma = 1 - (s.lambda[j] ?? 0) * (sigma[c2 * m + c2] ?? 0)
        gammaSum += gamma
        lambda[j] = (gamma + 2 * lambda1) / ((coef[j] ?? 0) ** 2 + 2 * lambda2)
        c2++
      }
      const alpha = (n - gammaSum + 2 * alpha1) / (sse + 2 * alpha2)
      const keep = lambda.map((l) => l < thresholdLambda)
      const pruned = coef.map((v, j) => (keep[j] ? v : 0))
      let score = 0
      for (const l of lambda) score += lambda1 * Math.log(l) - lambda2 * l
      score += alpha1 * Math.log(alpha) - alpha2 * alpha
      const logdet = m === 0 ? 0 : item(slogdet(fromStorage(sigma, [m, m], 'float64')).logabsdet)
      score += 0.5 * (logdet + n * Math.log(alpha) + lambda.reduce((a, l) => a + Math.log(l), 0))
      score -= 0.5 * (alpha * sse + lambda.reduce((a, l, j) => a + l * (pruned[j] ?? 0) ** 2, 0))
      const change =
        s.iteration === 0 ? Number.MAX_VALUE : pruned.reduce((a, v, j) => a + Math.abs((s.previous[j] ?? 0) - v), 0)
      return { coef: pruned, alpha, lambda, keep, previous: pruned, change, score, iteration: s.iteration + 1 }
    },
    done: (s) => (s.change < tol || !s.keep.some(Boolean) ? 'converged' : undefined),
    observe: (s) => ({
      metrics:
        s.iteration === 0
          ? { alpha: s.alpha, active: d }
          : {
              alpha: s.alpha,
              change: s.change === Number.MAX_VALUE ? -1 : s.change,
              active: s.keep.filter(Boolean).length,
              logEvidence: s.score,
            },
      tensors: {
        coefficients: { role: 'param', value: s.coef, shape: [d] },
        precisions: { role: 'param', value: s.lambda, shape: [d] },
      },
    }),
    serialise: (s) => stateToJson({ ...s, score: Number.isFinite(s.score) ? s.score : -Number.MAX_VALUE }),
    deserialise: (j) => {
      const s = stateFromJson(j) as ArdState
      return { ...s, score: s.score === -Number.MAX_VALUE ? Number.NEGATIVE_INFINITY : s.score }
    },
  }
}

/** The weights, the posterior covariance of the kept ones and the noise precision after a final update, as scikit-learn ends. */
export type ArdPosterior = {
  readonly coef: number[]
  readonly intercept: number
  readonly alpha: number
  readonly lambda: number[]
  /** The posterior covariance of the kept weights `[m][m]` (`m` the number kept). */
  readonly sigma: number[][]
  readonly keep: boolean[]
  readonly xMean: number[]
  readonly yMean: number
  /** The iterations run (scikit-learn's `n_iter_`). */
  readonly iterations: number
  /** The objective scikit-learn calls `scores_` at the last iteration (the log evidence up to a constant). */
  readonly logEvidence: number
}

/**
 * The model a state gives, as scikit-learn ends a fit: one more posterior update under the final precisions (so the
 * weights are the posterior mean for them, not those of the last iteration) and the intercept.
 *
 * @param state - A state of {@link ardAlgorithm}.
 * @param design - The design it ran on.
 * @param fitIntercept - Whether it centred.
 * @returns The weights, intercept, precisions and posterior covariance.
 * @example
 * ```ts
 * import { ardAlgorithm, ardCoefficients, designOf } from 'omai-engine/linear'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = Array.from({ length: 20 }, (_, i) => [Math.sin(i), Math.cos(3 * i)])
 * const design = designOf({ x, y: x.map((r) => 3 * r[0] + 1) }, 'demo')
 * const end = runAlgorithm(ardAlgorithm(design), { maxSteps: 300 }).state
 * ardCoefficients(end, design, true).coef[0].toFixed(4) // => '3.0000'
 * ```
 */
export function ardCoefficients(state: ArdState, design: Design, fitIntercept: boolean): ArdPosterior {
  const { n, d } = design.x
  const c = centre(design, fitIntercept)
  let coef = [...state.coef]
  let sigma: number[][] = []
  if (state.keep.some(Boolean)) {
    const post = posteriorCovariance(c.x, n, d, state.alpha, state.lambda, state.keep)
    coef = posteriorMean(post.sigma, post.xk, c.y, n, post.m, state.alpha, state.keep)
    sigma = Array.from({ length: post.m }, (_, i) => Array.from(post.sigma.subarray(i * post.m, (i + 1) * post.m)))
  }
  return {
    coef,
    intercept: fitIntercept ? interceptFor(c, coef) : 0,
    alpha: state.alpha,
    lambda: [...state.lambda],
    sigma,
    keep: [...state.keep],
    xMean: Array.from(c.xMean),
    yMean: c.yMean,
    iterations: state.iteration,
    logEvidence: Number.isFinite(state.score) ? state.score : 0,
  }
}

/** Hyperparameters of {@link ardRegression}. */
export interface ArdHyper {
  readonly alpha1: number
  readonly alpha2: number
  readonly lambda1: number
  readonly lambda2: number
  readonly thresholdLambda: number
  readonly tol: number
  /** The step budget (default 300). */
  readonly maxIter: number
  readonly fitIntercept: boolean
}

/** The verbs of a fitted ARD regression. */
export interface ArdVerbs {
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  /** The standard deviation of the posterior predictive distribution of each row: `sqrt(1 / alpha + x S x^T)` over the kept weights. */
  readonly predictStd: (x: TensorLike) => Tensor
  /** One importance per feature: the absolute weight. */
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

const ARD_DEFAULTS: ArdHyper = {
  alpha1: 1e-6,
  alpha2: 1e-6,
  lambda1: 1e-6,
  lambda2: 1e-6,
  thresholdLambda: 1e4,
  tol: 1e-3,
  maxIter: 300,
  fitIntercept: true,
}

/**
 * Bayesian linear regression with automatic relevance determination (`sklearn.linear_model.ARDRegression`) on the
 * iterative base: weights with a precision each, learnt by maximising the evidence, so features that do not help are
 * pruned (`params.keep`, `params.lambda`). It runs scikit-learn's updates, so the fit matches it; `predictStd` is the
 * posterior predictive standard deviation (`predict(X, return_std=True)`). A session adds one iteration per step.
 *
 * @param hyper - The hyperpriors `alpha1`, `alpha2`, `lambda1`, `lambda2`, `thresholdLambda`, `tol`, `maxIter`, `fitIntercept`.
 * @returns The estimator.
 * @index estimator Linear regression with a learnt precision per weight; irrelevant features are pruned.
 * @example
 * ```ts
 * import { ardRegression } from 'omai-engine/linear'
 * import { toFlat } from 'omai-engine/tensor'
 * const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(2 * i), Math.sin(5 * i + 1)])
 * const f = ardRegression().fit({ x, y: x.map((r) => 2 * r[0] + 1) })
 * f.params.keep // => [true, false, false]
 * Math.round(toFlat(f.predict([[1, 0, 0]]))[0] ?? 0) // => 3
 * ```
 */
export function ardRegression(
  hyper: Partial<ArdHyper> = {},
): IterativeEstimator<ArdHyper, ArdState, ArdPosterior, ArdVerbs> {
  return iterative<ArdHyper, ArdState, ArdPosterior, ArdVerbs>(
    {
      name: 'ard-regression',
      kind: 'regressor',
      defaults: ARD_DEFAULTS,
      hyperSpace: (h) => dict({ thresholdLambda: real(10, 1e6, { default: h.thresholdLambda, log: true }) }),
      algorithm: (data, h) =>
        ardAlgorithm(designOf(data, 'ardRegression'), {
          alpha1: h.alpha1,
          alpha2: h.alpha2,
          lambda1: h.lambda1,
          lambda2: h.lambda2,
          thresholdLambda: h.thresholdLambda,
          tol: h.tol,
          fitIntercept: h.fitIntercept,
        }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => ardCoefficients(s, designOf(data, 'ardRegression'), h.fitIntercept),
      verbs: (p) => {
        const predict = (x: TensorLike): Tensor =>
          vectorTensor(linearPredictor(x, p.coef, p.intercept, 'ardRegression'))
        const cols = p.keep.flatMap((k, j) => (k ? [j] : []))
        return {
          nFeaturesIn: p.coef.length,
          predict,
          predictStd: (x) => {
            const m = matrixView(x, 'ardRegression', { nFeatures: p.coef.length })
            const out = new Float64Array(m.n)
            for (let i = 0; i < m.n; i++) {
              let q = 0
              for (const [a, ja] of cols.entries())
                for (const [b, jb] of cols.entries())
                  q +=
                    ((m.data[i * m.d + ja] ?? 0) - (p.xMean[ja] ?? 0)) *
                    (p.sigma[a]?.[b] ?? 0) *
                    ((m.data[i * m.d + jb] ?? 0) - (p.xMean[jb] ?? 0))
              out[i] = Math.sqrt(q + 1 / p.alpha)
            }
            return vectorTensor(out)
          },
          featureImportances: p.coef.map(Math.abs),
          score: (data) =>
            r2Score(targetVector(data.y ?? [], 'ardRegression'), predict(data.x), {
              ...(data.weights === undefined ? {} : { sampleWeight: data.weights }),
            }).estimate.value,
        }
      },
    },
    { ...ARD_DEFAULTS, ...hyper },
  )
}
