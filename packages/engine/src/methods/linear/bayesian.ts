import { linspaceNumbers, maxOf, minOf } from '../../lib/base/index.ts'
import { normal } from '../../lib/dists/index.ts'
import { coefficientTable, type Exposition, hyperparameterSurface, posteriorGrid } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { ddot, pinv } from '../../lib/numerics/linalg/index.ts'
import { dict, int, real, type Space } from '../../lib/space/index.ts'
import { primalOf, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import { centre, designOf, interceptFor, matrixTensor, vectorTensor } from './_design.ts'
import { svdOf } from './solvers.ts'

// TODO: conjugate Bayesian linear regression as a probabilistic program solved by the conjugate engine.

/** Gamma hyperpriors on the noise precision alpha and the weight precision lambda (scikit-learn's defaults 1e-6). */
export interface EvidencePriors {
  readonly alpha1: number
  readonly alpha2: number
  readonly lambda1: number
  readonly lambda2: number
}

/** Hyperparameters of {@link bayesianRidge}. */
export interface BayesianRidgeHyper extends EvidencePriors {
  readonly maxIter: number
  readonly tol: number
  readonly fitIntercept: boolean
  /** Grid points per axis of the evidence surface (default 25; 0 for none). */
  readonly surfaceResolution: number
}

/** Parameters of a fitted Bayesian linear model. */
export type BayesianLinearParams = {
  readonly coef: number[]
  readonly intercept: number
  /** Noise precision. */
  readonly alpha: number
  /** Weight precision (one per feature for ARD). */
  readonly lambda: number[]
  /** Posterior covariance of the coefficients, row-major. */
  readonly sigma: number[]
  /** Means subtracted before fitting (zeros without an intercept). */
  readonly xMean: number[]
  readonly iterations: number
}

/** Verbs of a fitted Bayesian linear model. */
export interface BayesianLinearVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** Predictive standard deviations sqrt(x Sigma x^T + 1 / alpha). */
  readonly predictStd: (x: TensorLike) => Tensor
  /** Normal predictive distributions. */
  readonly predictDistribution: (x: TensorLike) => ReturnType<typeof normal>
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted Bayesian linear model. */
export type FittedBayesianLinear = Fitted<BayesianLinearParams> & BayesianLinearVerbs

function bayesianVerbs(p: BayesianLinearParams, who: string): BayesianLinearVerbs {
  const d = p.coef.length
  const moments = (x: TensorLike): { mean: Float64Array; sd: Float64Array } => {
    const m = matrixView(x, who, { nFeatures: d })
    const mean = new Float64Array(m.n)
    const sd = new Float64Array(m.n)
    for (let i = 0; i < m.n; i++) {
      const row = m.data.subarray(i * d, (i + 1) * d)
      mean[i] = p.intercept + ddot(row, p.coef)
      let v = 0
      for (let a = 0; a < d; a++) {
        const xa = (row[a] ?? 0) - (p.xMean[a] ?? 0)
        for (let b = 0; b < d; b++) v += xa * (p.sigma[a * d + b] ?? 0) * ((row[b] ?? 0) - (p.xMean[b] ?? 0))
      }
      sd[i] = Math.sqrt(Math.max(v, 0) + 1 / p.alpha)
    }
    return { mean, sd }
  }
  return {
    predict: (x) => vectorTensor(moments(x).mean),
    predictStd: (x) => vectorTensor(moments(x).sd),
    predictDistribution: (x) => {
      const { mean, sd } = moments(x)
      return normal(vectorTensor(mean), vectorTensor(sd))
    },
    nFeaturesIn: d,
    featureImportances: p.coef.map(Math.abs),
  }
}

/** The state of the evidence iteration of {@link bayesianRidge}. */
export interface EvidenceState {
  readonly coef: Float64Array
  readonly alpha: number
  readonly lambda: number
  readonly logEvidence: number
  readonly change: number
}

/** The data of a centred evidence problem with the SVD of its design. */
interface EvidenceProblem {
  readonly n: number
  readonly d: number
  readonly x: Float64Array
  readonly y: Float64Array
  readonly u: number[]
  readonly s: number[]
  readonly vh: number[]
  readonly priors: EvidencePriors
}

function ridgeAt(e: EvidenceProblem, alpha: number, lambda: number): { coef: Float64Array; rss: number } {
  const k = e.s.length
  const coef = new Float64Array(e.d)
  for (let a = 0; a < k; a++) {
    const sa = e.s[a] ?? 0
    let uty = 0
    for (let i = 0; i < e.n; i++) uty += (e.u[i * k + a] ?? 0) * (e.y[i] ?? 0)
    const f = (sa / (sa * sa + lambda / alpha)) * uty
    for (let j = 0; j < e.d; j++) coef[j] = (coef[j] ?? 0) + f * (e.vh[a * e.d + j] ?? 0)
  }
  let rss = 0
  for (let i = 0; i < e.n; i++) rss += ((e.y[i] ?? 0) - ddot(e.x.subarray(i * e.d, (i + 1) * e.d), coef)) ** 2
  return { coef, rss }
}

/** scikit-learn's log marginal likelihood of Bayesian ridge at (alpha, lambda), with the gamma hyperpriors. */
function logEvidence(e: EvidenceProblem, alpha: number, lambda: number): number {
  const { coef, rss } = ridgeAt(e, alpha, lambda)
  const eig = Array.from({ length: e.d }, (_, j) => (e.s[j] ?? 0) ** 2)
  let logdet = 0
  for (const v of eig) logdet -= Math.log(lambda + alpha * v)
  const p = e.priors
  return (
    p.lambda1 * Math.log(lambda) -
    p.lambda2 * lambda +
    p.alpha1 * Math.log(alpha) -
    p.alpha2 * alpha +
    0.5 *
      (e.d * Math.log(lambda) +
        e.n * Math.log(alpha) -
        alpha * rss -
        lambda * ddot(coef, coef) +
        logdet -
        e.n * Math.log(2 * Math.PI))
  )
}

/**
 * The evidence (type II maximum likelihood) iteration of Bayesian ridge as a traced algorithm: MacKay's fixed-point
 * updates of the noise precision alpha and weight precision lambda, from the SVD of the centred design (scikit-learn's
 * BayesianRidge). Records the log evidence, alpha, lambda and the coefficient change.
 */
function evidenceAlgorithm(e: EvidenceProblem, alpha0: number, tol: number): Algorithm<EvidenceState> {
  const eig = e.s.map((v) => v * v)
  return {
    name: 'bayesianRidgeEvidence',
    config: { n: e.n, d: e.d },
    schema: channels({
      metrics: {
        logEvidence: { description: 'log marginal likelihood', direction: 'higher' },
        alpha: { description: 'noise precision' },
        lambda: { description: 'weight precision' },
        change: { description: 'sum of absolute coefficient changes' },
      },
    }),
    init: () => ({
      coef: new Float64Array(e.d),
      alpha: alpha0,
      lambda: 1,
      logEvidence: logEvidence(e, alpha0, 1),
      change: Infinity,
    }),
    step: (st, ctx) => {
      const { coef, rss } = ridgeAt(e, st.alpha, st.lambda)
      let gamma = 0
      for (const v of eig) gamma += (st.alpha * v) / (st.lambda + st.alpha * v)
      const p = e.priors
      const lambda = (gamma + 2 * p.lambda1) / (ddot(coef, coef) + 2 * p.lambda2)
      const alpha = (e.n - gamma + 2 * p.alpha1) / (rss + 2 * p.alpha2)
      let change = 0
      for (let j = 0; j < e.d; j++) change += Math.abs((coef[j] ?? 0) - (st.coef[j] ?? 0))
      return {
        coef,
        alpha,
        lambda,
        logEvidence: logEvidence(e, alpha, lambda),
        change: ctx.t === 0 ? Infinity : change,
      }
    },
    done: (st) => (st.change < tol ? 'converged' : undefined),
    observe: (st) => ({
      metrics: {
        logEvidence: st.logEvidence,
        alpha: st.alpha,
        lambda: st.lambda,
        change: Number.isFinite(st.change) ? st.change : -1,
      },
    }),
  }
}

function posteriorRecords(fitted: BayesianLinearVerbs, data: LearnData): Exposition[] {
  const m = matrixView(data.x, 'posteriorGrid')
  if (m.d !== 1) return []
  const xs = Array.from(m.data)
  const lo = minOf(xs)
  const hi = maxOf(xs)
  const pad = 0.1 * (hi - lo || 1)
  const grid = linspaceNumbers(lo - pad, hi + pad, 60).map((v) => [v])
  const sd = toFlat(fitted.predictStd(grid))
  return [
    posteriorGrid({
      inputs: grid,
      mean: toFlat(fitted.predict(grid)),
      variance: sd.map((v) => v * v),
      observed: { inputs: xs.map((v) => [v]), targets: toFlat(data.y as Tensor) },
    }),
  ]
}

const priorsOf = (h: Partial<EvidencePriors>): EvidencePriors => ({
  alpha1: h.alpha1 ?? 1e-6,
  alpha2: h.alpha2 ?? 1e-6,
  lambda1: h.lambda1 ?? 1e-6,
  lambda2: h.lambda2 ?? 1e-6,
})

/**
 * Bayesian ridge regression (sklearn.linear_model.BayesianRidge): a Gaussian prior on the coefficients with precision
 * lambda, Gaussian noise with precision alpha, both chosen by maximising the evidence with MacKay's fixed-point
 * iteration (traced). `predictDistribution` gives the normal posterior predictive. Exposition: the coefficient table
 * (posterior means and standard deviations), the log-evidence surface over (alpha, lambda) with the iteration's path,
 * and for one feature the posterior predictive on a grid.
 *
 * @param hyper - Iterations (default 300), tolerance (default 1e-3), intercept, gamma hyperpriors and surface size.
 * @returns The estimator.
 */
export function bayesianRidge(
  hyper: Partial<BayesianRidgeHyper> = {},
): ExplainedEstimator<BayesianRidgeHyper, FittedBayesianLinear> {
  return defineExplainedEstimator(
    {
      name: 'bayesian-ridge',
      kind: 'regressor',
      hyperSpace: (h: BayesianRidgeHyper): Space => dict({ maxIter: int(1, 300, { default: h.maxIter }) }),
      fit: (data: LearnData, h: BayesianRidgeHyper, options: FitOptions) => {
        const design = designOf(data, 'bayesianRidge')
        const { n, d } = design.x
        const c = centre(design, h.fitIntercept)
        const { u, s, vh } = svdOf(c.x, n, d)
        const e: EvidenceProblem = { n, d, x: c.x, y: c.y, u, s, vh, priors: h }
        let mean = 0
        for (const v of c.y) mean += v / n
        let variance = 0
        for (const v of c.y) variance += (v - mean) ** 2 / n
        const algorithm = evidenceAlgorithm(e, 1 / (variance + 2.220446049250313e-16), h.tol)
        const path: [number, number][] = []
        const r = runAlgorithm(
          {
            ...algorithm,
            step: (st, ctx) => {
              const next = algorithm.step(st, ctx)
              path.push([next.alpha, next.lambda])
              return next
            },
          },
          {
            maxSteps: h.maxIter,
            ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
            ...(options.policy === undefined ? {} : { policy: options.policy }),
          },
        )
        const { alpha, lambda } = r.state
        const { coef } = ridgeAt(e, alpha, lambda)
        // Sigma = Vh^T diag(1 / (alpha s^2 + lambda)) Vh (plus 1 / lambda along the null space when d > n).
        const k = s.length
        const sigma = new Float64Array(d * d)
        for (let a = 0; a < d; a++)
          for (let b = 0; b < d; b++) {
            let v = 0
            for (let q = 0; q < k; q++)
              v += ((vh[q * d + a] ?? 0) * (vh[q * d + b] ?? 0)) / (alpha * (s[q] ?? 0) ** 2 + lambda)
            sigma[a * d + b] = v
          }
        const params: BayesianLinearParams = {
          coef: Array.from(coef),
          intercept: h.fitIntercept ? interceptFor(c, coef) : 0,
          alpha,
          lambda: [lambda],
          sigma: Array.from(sigma),
          xMean: Array.from(c.xMean),
          iterations: r.steps,
        }
        const verbs = bayesianVerbs(params, 'bayesianRidge')
        const exposition: Exposition[] = [
          coefficientTable({
            names: featureNamesOf(data, d),
            estimates: params.coef,
            standardErrors: params.coef.map((_, j) => Math.sqrt(sigma[j * d + j] ?? 0)),
          }),
        ]
        if (h.surfaceResolution > 0) {
          const span = (v: number): number[] =>
            linspaceNumbers(Math.log10(v) - 2, Math.log10(v) + 2, h.surfaceResolution).map((t) => 10 ** t)
          const xs = span(alpha)
          const ys = span(lambda)
          exposition.push(
            hyperparameterSurface({
              parameters: ['alpha', 'lambda'],
              x: xs,
              y: ys,
              values: ys.map((l) => xs.map((a) => logEvidence(e, a, l))),
              path,
              logScale: [true, true],
            }),
          )
        }
        exposition.push(...posteriorRecords(verbs, data))
        return { params, exposition }
      },
      methods: (p: BayesianLinearParams) => bayesianVerbs(p, 'bayesianRidge'),
    },
    {
      ...priorsOf(hyper),
      maxIter: hyper.maxIter ?? 300,
      tol: hyper.tol ?? 1e-3,
      fitIntercept: hyper.fitIntercept ?? true,
      surfaceResolution: hyper.surfaceResolution ?? 25,
    },
  )
}

/** Hyperparameters of {@link ardRegression}. */
export interface ArdHyper extends EvidencePriors {
  readonly maxIter: number
  readonly tol: number
  readonly fitIntercept: boolean
  /** Features whose precision exceeds this are pruned (default 1e4). */
  readonly thresholdLambda: number
}

/**
 * Automatic relevance determination regression (sklearn.linear_model.ARDRegression): one prior precision per
 * coefficient, updated with the noise precision by evidence maximisation (traced), pruning features whose precision
 * exceeds `thresholdLambda`. Exposition: the coefficient table (posterior means and standard deviations, zero for
 * pruned features) and, for one feature, the posterior predictive on a grid.
 *
 * @param hyper - Iterations (default 300), tolerance (default 1e-3), intercept, pruning threshold and hyperpriors.
 * @returns The estimator.
 */
export function ardRegression(hyper: Partial<ArdHyper> = {}): ExplainedEstimator<ArdHyper, FittedBayesianLinear> {
  return defineExplainedEstimator(
    {
      name: 'ard-regression',
      kind: 'regressor',
      hyperSpace: (h: ArdHyper): Space =>
        dict({
          thresholdLambda: real(10, 1e6, {
            label: 'pruning threshold λ',
            description: 'Coefficients whose precision λ exceeds this are pruned to zero.',
            log: true,
            default: h.thresholdLambda,
          }),
        }),
      fit: (data: LearnData, h: ArdHyper, options: FitOptions) => {
        const design = designOf(data, 'ardRegression')
        const { n, d } = design.x
        const c = centre(design, h.fitIntercept)
        let mean = 0
        for (const v of c.y) mean += v / n
        let variance = 0
        for (const v of c.y) variance += (v - mean) ** 2 / n
        interface ArdState {
          readonly coef: Float64Array
          readonly alpha: number
          readonly lambda: Float64Array
          readonly keep: readonly number[]
          readonly sigma: Float64Array
          readonly change: number
        }
        const sigmaFor = (alpha: number, lambda: Float64Array, keep: readonly number[]): Float64Array => {
          const k = keep.length
          const g = new Float64Array(k * k)
          for (let i = 0; i < n; i++)
            for (const [a, ja] of keep.entries())
              for (const [b, jb] of keep.entries())
                g[a * k + b] = (g[a * k + b] ?? 0) + alpha * (c.x[i * d + ja] ?? 0) * (c.x[i * d + jb] ?? 0)
          for (const [a, ja] of keep.entries()) g[a * k + a] = (g[a * k + a] ?? 0) + (lambda[ja] ?? 0)
          return k === 0 ? new Float64Array(0) : Float64Array.from(toFlat(primalOf(pinv(matrixTensor(g, k, k)))))
        }
        const coefFor = (alpha: number, keep: readonly number[], sigma: Float64Array): Float64Array => {
          const k = keep.length
          const xty = keep.map((j) => {
            let t = 0
            for (let i = 0; i < n; i++) t += (c.x[i * d + j] ?? 0) * (c.y[i] ?? 0)
            return t
          })
          const coef = new Float64Array(d)
          for (const [a, ja] of keep.entries()) {
            let v = 0
            for (let b = 0; b < k; b++) v += (sigma[a * k + b] ?? 0) * (xty[b] ?? 0)
            coef[ja] = alpha * v
          }
          return coef
        }
        const algorithm: Algorithm<ArdState> = {
          name: 'ardEvidence',
          config: { n, d, thresholdLambda: h.thresholdLambda },
          schema: channels({
            metrics: { alpha: { description: 'noise precision' }, kept: { description: 'features kept' } },
          }),
          init: () => ({
            coef: new Float64Array(d),
            alpha: 1 / (variance + 2.220446049250313e-16),
            lambda: new Float64Array(d).fill(1),
            keep: Array.from({ length: d }, (_, j) => j),
            sigma: new Float64Array(0),
            change: Infinity,
          }),
          step: (st, ctx) => {
            const sigma = sigmaFor(st.alpha, st.lambda, st.keep)
            const coef = coefFor(st.alpha, st.keep, sigma)
            let rss = 0
            for (let i = 0; i < n; i++) rss += ((c.y[i] ?? 0) - ddot(c.x.subarray(i * d, (i + 1) * d), coef)) ** 2
            const k = st.keep.length
            const lambda = Float64Array.from(st.lambda)
            let gammaSum = 0
            for (const [a, j] of st.keep.entries()) {
              const gamma = 1 - (st.lambda[j] ?? 0) * (sigma[a * k + a] ?? 0)
              gammaSum += gamma
              lambda[j] = (gamma + 2 * h.lambda1) / ((coef[j] ?? 0) ** 2 + 2 * h.lambda2)
            }
            const alpha = (n - gammaSum + 2 * h.alpha1) / (rss + 2 * h.alpha2)
            const keep = Array.from({ length: d }, (_, j) => j).filter((j) => (lambda[j] ?? 0) < h.thresholdLambda)
            for (let j = 0; j < d; j++) if (!keep.includes(j)) coef[j] = 0
            let change = 0
            for (let j = 0; j < d; j++) change += Math.abs((coef[j] ?? 0) - (st.coef[j] ?? 0))
            return { coef, alpha, lambda, keep, sigma, change: ctx.t === 0 ? Infinity : change }
          },
          done: (st) => (st.change < h.tol || st.keep.length === 0 ? 'converged' : undefined),
          observe: (st) => ({ metrics: { alpha: st.alpha, kept: st.keep.length } }),
        }
        const r = runAlgorithm(algorithm, {
          maxSteps: h.maxIter,
          ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        })
        const { alpha, lambda, keep } = r.state
        const sk = sigmaFor(alpha, lambda, keep)
        const coef = keep.length === 0 ? new Float64Array(d) : coefFor(alpha, keep, sk)
        const sigma = new Float64Array(d * d)
        for (const [a, ja] of keep.entries())
          for (const [b, jb] of keep.entries()) sigma[ja * d + jb] = sk[a * keep.length + b] ?? 0
        const params: BayesianLinearParams = {
          coef: Array.from(coef),
          intercept: h.fitIntercept ? interceptFor(c, coef) : 0,
          alpha,
          lambda: Array.from(lambda),
          sigma: Array.from(sigma),
          xMean: Array.from(c.xMean),
          iterations: r.steps,
        }
        const verbs = bayesianVerbs(params, 'ardRegression')
        return {
          params,
          exposition: [
            coefficientTable({
              names: featureNamesOf(data, d),
              estimates: params.coef,
              standardErrors: params.coef.map((_, j) => Math.sqrt(sigma[j * d + j] ?? 0)),
            }),
            ...posteriorRecords(verbs, data),
          ],
        }
      },
      methods: (p: BayesianLinearParams) => bayesianVerbs(p, 'ardRegression'),
    },
    {
      ...priorsOf(hyper),
      maxIter: hyper.maxIter ?? 300,
      tol: hyper.tol ?? 1e-3,
      fitIntercept: hyper.fitIntercept ?? true,
      thresholdLambda: hyper.thresholdLambda ?? 1e4,
    },
  )
}

/** {@link bayesianRidge} as a class. */
export class BayesianRidge extends ExplainedEstimatorClass<BayesianRidgeHyper, FittedBayesianLinear> {
  /** @param hyper - As {@link bayesianRidge}. */
  constructor(hyper: Partial<BayesianRidgeHyper> = {}) {
    super(bayesianRidge(hyper))
  }
}

/** {@link ardRegression} as a class. */
export class ArdRegression extends ExplainedEstimatorClass<ArdHyper, FittedBayesianLinear> {
  /** @param hyper - As {@link ardRegression}. */
  constructor(hyper: Partial<ArdHyper> = {}) {
    super(ardRegression(hyper))
  }
}
