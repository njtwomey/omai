/**
 * The relevance vector machine (Tipping) as an estimator on the iterative base: sparse Bayesian regression over a basis
 * of kernels centred on the training points, each weight with its own precision, re-estimated by MacKay's evidence
 * updates until most diverge and their basis functions are pruned. The updates are those of scikit-learn's
 * `ARDRegression` on the kernel matrix; each step is one pass, logging the precisions, the number of relevance vectors
 * left and the log evidence, so the pruning can be watched.
 */

import { MdmlError } from '../base/index.ts'
import { marginRecord } from '../exposition/index.ts'
import { type Kernel, rbfKernel } from '../kernels/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView, targetVector } from '../learn/index.ts'
import { cholesky, choSolve } from '../numerics/linalg/index.ts'
import { dict, real } from '../space/index.ts'
import {
  eye,
  fromStorage,
  matmul,
  primalOf,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import { channels, derived, Log, stateFromJson, stateToJson } from '../trace/index.ts'

/** The hyperparameters of {@link relevanceVectorMachine}. */
export interface RvmHyper {
  readonly kernel: Kernel
  /** Whether to fit an intercept (the targets and basis columns are centred, as scikit-learn's ARD does). */
  readonly fitIntercept: boolean
  /** Basis functions with a precision above this are pruned (scikit-learn's threshold_lambda; Tipping used 1e9). */
  readonly threshold: number
  /** The Gamma hyperpriors on the noise precision (`alpha1`, `alpha2`) and on the weight precisions (`lambda1`, `lambda2`). */
  readonly alpha1: number
  readonly alpha2: number
  readonly lambda1: number
  readonly lambda2: number
  /** Stop when the weights move less than this in L1 norm (scikit-learn's tol). */
  readonly tol: number
  readonly maxIter: number
}

/** The state of {@link relevanceVectorMachine}. */
export interface RvmState {
  /** The precision of every basis function's weight. */
  readonly lambda: Float64Array
  /** The noise precision. */
  readonly noise: number
  /** The weights (zero for pruned basis functions). */
  readonly coef: Float64Array
  /** Which basis functions are still in the model. */
  readonly keep: Uint8Array
  readonly iteration: number
  /** The log evidence log p(t | precisions) of the centred targets at this iterate. */
  readonly logEvidence: number
  readonly converged: boolean
}

/** The fitted parameters of {@link relevanceVectorMachine}: plain JSON. */
export type RvmParams = {
  /** The training rows kept as relevance vectors, their indices and weights. */
  readonly vectors: number[][]
  readonly relevant: number[]
  readonly coef: number[]
  readonly intercept: number
  /** The noise precision, and the covariance of the kept weights. */
  readonly noise: number
  readonly covariance: number[][]
  /** The column means of the training basis, which prediction centres by when there is an intercept. */
  readonly basisMean: number[]
}

/** What a fitted relevance vector machine offers. */
export interface RvmVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** The predictive mean and standard deviation (the noise included). */
  readonly predictWithStd: (x: TensorLike) => { readonly mean: Tensor; readonly std: Tensor }
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

const WHO = 'relevanceVectorMachine'

interface Prepared {
  readonly n: number
  readonly d: number
  readonly rows: number[][]
  /** The centred basis `[n][n]` (flat) and centred targets. */
  readonly phi: Float64Array
  readonly t: Float64Array
  readonly yMean: number
  readonly basisMean: Float64Array
}

function prepare(data: LearnData, h: RvmHyper): Prepared {
  const m = matrixView(data.x, WHO)
  const y = targetVector(data.y ?? [], WHO, m.n)
  const x = fromStorage(m.data, [m.n, m.d], 'float64')
  const K = Float64Array.from(toFlat(primalOf(h.kernel.k(x))))
  const n = m.n
  const basisMean = new Float64Array(n)
  let yMean = 0
  if (h.fitIntercept) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let i = 0; i < n; i++) s += K[i * n + j] ?? 0
      basisMean[j] = s / n
    }
    yMean = y.reduce((a, b) => a + b, 0) / n
  }
  const phi = Float64Array.from(K, (v, idx) => v - (basisMean[idx % n] ?? 0))
  return {
    n,
    d: m.d,
    rows: Array.from({ length: n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
    phi,
    t: Float64Array.from(y, (v) => v - yMean),
    yMean,
    basisMean,
  }
}

const matrix = (a: ArrayLike<number>, r: number, c: number): Value =>
  fromStorage(Float64Array.from(a), [r, c], 'float64')

/** The posterior of the kept weights: covariance Sigma = (beta Phi^T Phi + diag(lambda))^-1 and mean beta Sigma Phi^T t. */
function posterior(
  p: Prepared,
  s: { lambda: Float64Array; noise: number; keep: Uint8Array },
): {
  readonly kept: number[]
  readonly sigma: Float64Array
  readonly mean: Float64Array
  readonly logDet: number
} {
  const kept = Array.from(s.keep).flatMap((v, j) => (v === 1 ? [j] : []))
  const k = kept.length
  if (k === 0) return { kept, sigma: new Float64Array(0), mean: new Float64Array(0), logDet: 0 }
  const P = Float64Array.from(
    { length: p.n * k },
    (_, idx) => p.phi[Math.floor(idx / k) * p.n + (kept[idx % k] ?? 0)] ?? 0,
  )
  const Pm = matrix(P, p.n, k)
  const precision = matmul(transpose(Pm), Pm)
  const A = Float64Array.from(
    toFlat(primalOf(precision)),
    (v, idx) => s.noise * v + (idx % k === Math.floor(idx / k) ? (s.lambda[kept[idx % k] ?? 0] ?? 0) : 0),
  )
  const L = cholesky(matrix(A, k, k))
  const sigma = Float64Array.from(toFlat(primalOf(choSolve(L, eye(k)))))
  const Lf = toFlat(primalOf(L))
  let logDet = 0
  for (let i = 0; i < k; i++) logDet += 2 * Math.log(Lf[i * k + i] ?? 1)
  const Ptt = Float64Array.from(toFlat(primalOf(matmul(transpose(Pm), matrix(p.t, p.n, 1)))))
  const mean = Float64Array.from(toFlat(primalOf(matmul(matrix(sigma, k, k), matrix(Ptt, k, 1)))), (v) => v * s.noise)
  return { kept, sigma, mean, logDet }
}

function verbsOf(params: RvmParams, h: RvmHyper): RvmVerbs {
  const d = params.vectors[0]?.length ?? 0
  const vectors =
    params.vectors.length === 0
      ? null
      : fromStorage(Float64Array.from(params.vectors.flat()), [params.vectors.length, d], 'float64')
  const k = params.relevant.length
  const design = (x: TensorLike): { rows: number; phi: Float64Array } => {
    const m = matrixView(x, WHO, d > 0 ? { nFeatures: d } : {})
    if (vectors === null) return { rows: m.n, phi: new Float64Array(0) }
    const K = toFlat(primalOf(h.kernel.k(fromStorage(m.data, [m.n, m.d], 'float64'), vectors)))
    return { rows: m.n, phi: Float64Array.from(K) }
  }
  const meanOf = (x: TensorLike): Tensor => {
    const { rows, phi } = design(x)
    return fromStorage(
      Float64Array.from({ length: rows }, (_, i) => {
        let s = params.intercept
        for (let j = 0; j < k; j++) s += (phi[i * k + j] ?? 0) * (params.coef[j] ?? 0)
        return s
      }),
      [rows],
      'float64',
    )
  }
  return {
    predict: meanOf,
    predictWithStd: (x) => {
      const { rows, phi } = design(x)
      const std = Float64Array.from({ length: rows }, (_, i) => {
        let v = 1 / params.noise
        // The covariance is that of the weights on the centred basis.
        const centred = Array.from(
          { length: k },
          (_, a) => (phi[i * k + a] ?? 0) - (params.basisMean[params.relevant[a] ?? 0] ?? 0),
        )
        for (let a = 0; a < k; a++)
          for (let b = 0; b < k; b++) v += (centred[a] ?? 0) * (params.covariance[a]?.[b] ?? 0) * (centred[b] ?? 0)
        return Math.sqrt(Math.max(v, 0))
      })
      return { mean: meanOf(x), std: fromStorage(std, [rows], 'float64') }
    },
    score: (data) => {
      const truth = targetVector(data.y ?? [], WHO)
      const fit = toFlat(meanOf(data.x))
      const mean = truth.reduce((a, b) => a + b, 0) / Math.max(1, truth.length)
      let rss = 0
      let tss = 0
      truth.forEach((v, i) => {
        rss += (v - (fit[i] ?? 0)) ** 2
        tss += (v - mean) ** 2
      })
      return tss === 0 ? 0 : 1 - rss / tss
    },
    nFeaturesIn: d,
  }
}

/**
 * The relevance vector machine for regression. The model is a sum of kernels centred on the training points, with a
 * Gaussian prior on each weight of its own precision; the evidence updates (MacKay, as `ARDRegression`) raise the precisions
 * of the useless basis functions until they pass `threshold` and are pruned, leaving the relevance vectors (how many
 * depends on the noise, the kernel width and how long the updates run: with the default threshold and tolerance, which are
 * scikit-learn's, pruning is mild on nearly noise-free data). Each step is one update of the weights, the precisions and the noise precision.
 *
 * @concept relevance-vector-machines
 * @reference M. E. Tipping, Sparse Bayesian learning and the relevance vector machine (2001).
 * @example
 * ```ts
 * import { relevanceVectorMachine } from 'mdml-engine/gp'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 25 }, (_, i) => [i / 4])
 * const fitted = relevanceVectorMachine.fit({ x, y: x.map(([v]) => Math.sin(v as number)) })
 * fitted.params.relevant.length < 25 // => true
 * ```
 */
export const relevanceVectorMachine: IterativeEstimator<RvmHyper, RvmState, RvmParams, RvmVerbs> = iterative<
  RvmHyper,
  RvmState,
  RvmParams,
  RvmVerbs
>({
  name: 'gp/relevance-vector-machine',
  kind: 'regressor',
  defaults: {
    kernel: rbfKernel({ lengthScale: 1 }),
    fitIntercept: true,
    threshold: 1e4,
    alpha1: 1e-6,
    alpha2: 1e-6,
    lambda1: 1e-6,
    lambda2: 1e-6,
    tol: 1e-3,
    maxIter: 300,
  },
  hyperSpace: (h) =>
    dict({
      threshold: real(1e2, 1e10, {
        log: true,
        default: Math.min(Math.max(h.threshold, 1e2), 1e10),
        label: 'pruning threshold',
      }),
    }),
  algorithm: (data, h) => {
    const p = prepare(data, h)
    if (!(h.threshold > 0)) throw new MdmlError('domain', `${WHO}: threshold must be positive`)
    const variance = p.t.reduce((a, b) => a + b * b, 0) / p.n - (p.t.reduce((a, b) => a + b, 0) / p.n) ** 2
    const evidence = (s: RvmState): number => {
      const post = posterior(p, { lambda: s.lambda, noise: s.noise, keep: s.keep })
      if (post.kept.length === 0) return Number.NEGATIVE_INFINITY
      let tt = 0
      for (let i = 0; i < p.n; i++) tt += (p.t[i] ?? 0) ** 2
      let fit = 0
      for (let i = 0; i < p.n; i++) {
        let f = 0
        for (let j = 0; j < post.kept.length; j++)
          f += (p.phi[i * p.n + (post.kept[j] ?? 0)] ?? 0) * (post.mean[j] ?? 0)
        fit += (p.t[i] ?? 0) * f
      }
      let logLambda = 0
      for (const j of post.kept) logLambda += Math.log(s.lambda[j] ?? 1)
      const logDetC = post.logDet - logLambda - p.n * Math.log(s.noise)
      const quad = s.noise * (tt - fit)
      return -0.5 * (p.n * Math.log(2 * Math.PI) + logDetC + quad)
    }
    return {
      name: 'rvm',
      config: { fitIntercept: h.fitIntercept, threshold: h.threshold, tol: h.tol },
      schema: channels({
        metrics: {
          logEvidence: {
            description: 'the log evidence of the targets under the current precisions',
            direction: 'higher',
          },
          relevant: { description: 'the relevance vectors left' },
          noise: { description: 'the noise precision' },
          change: { description: 'the L1 change of the weights in the last step' },
        },
        tensors: {
          precisions: { role: 'param', description: 'the precision of every weight (large: pruned)', shape: ['n'] },
        },
      }),
      init: () => ({
        lambda: new Float64Array(p.n).fill(1),
        noise: 1 / (variance + Number.EPSILON),
        coef: new Float64Array(p.n),
        keep: new Uint8Array(p.n).fill(1),
        iteration: 0,
        logEvidence: Number.NEGATIVE_INFINITY,
        converged: false,
      }),
      step: (s) => {
        const post = posterior(p, s)
        const coef = new Float64Array(p.n)
        for (const [a, j] of post.kept.entries()) coef[j] = post.mean[a] ?? 0
        let rss = 0
        for (let i = 0; i < p.n; i++) {
          let f = 0
          for (const j of post.kept) f += (p.phi[i * p.n + j] ?? 0) * (coef[j] ?? 0)
          rss += ((p.t[i] ?? 0) - f) ** 2
        }
        const k = post.kept.length
        const gamma = post.kept.map((j, a) => 1 - (s.lambda[j] ?? 0) * (post.sigma[a * k + a] ?? 0))
        const lambda = Float64Array.from(s.lambda)
        for (const [a, j] of post.kept.entries())
          lambda[j] = ((gamma[a] ?? 0) + 2 * h.lambda1) / ((coef[j] ?? 0) ** 2 + 2 * h.lambda2)
        const noise = (p.n - gamma.reduce((a, b) => a + b, 0) + 2 * h.alpha1) / (rss + 2 * h.alpha2)
        const keep = Uint8Array.from(lambda, (v, j) => ((s.keep[j] ?? 0) === 1 && v < h.threshold ? 1 : 0))
        for (let j = 0; j < p.n; j++) if (keep[j] === 0) coef[j] = 0
        let change = 0
        for (let j = 0; j < p.n; j++) change += Math.abs((coef[j] ?? 0) - (s.coef[j] ?? 0))
        const next: RvmState = {
          lambda,
          noise,
          coef,
          keep,
          iteration: s.iteration + 1,
          logEvidence: Number.NEGATIVE_INFINITY,
          converged: false,
        }
        const converged = (s.iteration !== 0 && change < h.tol) || keep.every((v) => v === 0)
        return { ...next, logEvidence: evidence(next), converged }
      },
      done: (s) => (s.converged ? 'converged' : undefined),
      observe: (s) => ({
        metrics: {
          ...(Number.isFinite(s.logEvidence) ? { logEvidence: s.logEvidence } : {}),
          relevant: s.keep.reduce((a, b) => a + b, 0),
          noise: s.noise,
        },
        tensors: { precisions: { role: 'param', value: s.lambda } },
      }),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as RvmState,
    }
  },
  maxSteps: (h) => h.maxIter,
  params: (s, data, h) => {
    const p = prepare(data, h)
    const post = posterior(p, { lambda: s.lambda, noise: s.noise, keep: s.keep })
    const k = post.kept.length
    return {
      vectors: post.kept.map((j) => p.rows[j] ?? []),
      relevant: post.kept,
      coef: Array.from(post.mean),
      intercept: p.yMean - post.kept.reduce((acc, j, a) => acc + (p.basisMean[j] ?? 0) * (post.mean[a] ?? 0), 0),
      noise: s.noise,
      covariance: Array.from({ length: k }, (_, a) => Array.from({ length: k }, (_, b) => post.sigma[a * k + b] ?? 0)),
      basisMean: Array.from(p.basisMean),
    }
  },
  verbs: verbsOf,
  contract: (data, h) => {
    const p = prepare(data, h)
    return {
      'model/relevance-vectors': derived<RvmState>({
        description:
          'the relevance vectors so far with their weights, the intercept and the residual of every training row',
        compute: (s) => {
          const post = posterior(p, { lambda: s.lambda, noise: s.noise, keep: s.keep })
          const residuals = Array.from({ length: p.n }, (_, i) => {
            let f = 0
            for (const [a, j] of post.kept.entries()) f += (p.phi[i * p.n + j] ?? 0) * (post.mean[a] ?? 0)
            return (p.t[i] ?? 0) - f
          })
          return marginRecord({
            support: post.kept,
            dualCoefficients: Array.from(post.mean),
            intercept: p.yMean,
            margins: residuals,
          })
        },
      }),
    }
  },
  logs: { 'solver/logEvidence': Log(), 'solver/relevant': Log() },
})
