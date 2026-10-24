import { linspaceNumbers, maxOf, minOf, WhiteboxError } from '../../lib/base/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { coefficientTable, type Exposition, hyperparameterSurface, posteriorGrid } from '../../lib/exposition/index.ts'
import { hyperparameters, type Kernel, theta as thetaOf, withTheta } from '../../lib/kernels/index.ts'
import {
  classNamesOf,
  collecting,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { cholesky, dgemv, solveTriangular } from '../../lib/numerics/linalg/index.ts'
import { erf } from '../../lib/numerics/special/index.ts'
import { nelderMeadAlgorithm } from '../../lib/optim/index.ts'
import { bool, dict, type Space } from '../../lib/space/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat, type Value } from '../../lib/tensor/index.ts'
import { runAlgorithm } from '../../lib/trace/index.ts'
import { defaultGpKernel } from './regression.ts'

// TODO: expectation propagation, multinomial (softmax) Laplace.

const flat = (v: Value): number[] => toFlat(primalOf(v))
const sigmoid = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))

/** The Laplace approximation of a binary GP classifier at fixed hyperparameters (Rasmussen and Williams, Alg. 3.1). */
interface LaplaceFit {
  readonly f: Float64Array
  readonly pi: Float64Array
  readonly sqrtW: Float64Array
  /** Lower Cholesky factor of B = I + sqrt(W) K sqrt(W), row-major. */
  readonly l: number[]
  readonly logMarginalLikelihood: number
}

const matVec = (a: readonly number[], n: number, v: ArrayLike<number>): Float64Array => dgemv(a, n, n, v)

function triSolve(l: readonly number[], n: number, b: ArrayLike<number>, transpose: boolean): Float64Array {
  const lt = fromStorage(Float64Array.from(l), [n, n], 'float64')
  return Float64Array.from(
    flat(
      solveTriangular(lt, fromStorage(Float64Array.from(b), [n], 'float64'), {
        lower: true,
        ...(transpose ? { transpose: true } : {}),
      }),
    ),
  )
}

function laplace(K: readonly number[], n: number, y: Float64Array, maxIter = 100): LaplaceFit {
  let f: Float64Array = new Float64Array(n)
  let objective = Number.NEGATIVE_INFINITY
  let fit: LaplaceFit | undefined
  for (let it = 0; it < maxIter; it++) {
    const pi = Float64Array.from(f, sigmoid)
    const W = Float64Array.from(pi, (p) => p * (1 - p))
    const sqrtW = Float64Array.from(W, Math.sqrt)
    const B = Array.from({ length: n * n }, (_, idx) => {
      const i = Math.floor(idx / n)
      const j = idx % n
      return (i === j ? 1 : 0) + (sqrtW[i] ?? 0) * (K[idx] ?? 0) * (sqrtW[j] ?? 0)
    })
    const l = flat(cholesky(fromStorage(Float64Array.from(B), [n, n], 'float64')))
    const b = Float64Array.from(f, (v, i) => (W[i] ?? 0) * v + (y[i] ?? 0) - (pi[i] ?? 0))
    const kb = matVec(K, n, b)
    const inner = triSolve(
      l,
      n,
      triSolve(
        l,
        n,
        Float64Array.from(kb, (v, i) => v * (sqrtW[i] ?? 0)),
        false,
      ),
      true,
    )
    const a = Float64Array.from(b, (v, i) => v - (sqrtW[i] ?? 0) * (inner[i] ?? 0))
    f = matVec(K, n, a)
    let loglik = 0
    for (let i = 0; i < n; i++) loglik -= Math.log1p(Math.exp(-(2 * (y[i] ?? 0) - 1) * (f[i] ?? 0)))
    let af = 0
    for (let i = 0; i < n; i++) af += (a[i] ?? 0) * (f[i] ?? 0)
    const next = -0.5 * af + loglik
    let logdet = 0
    for (let i = 0; i < n; i++) logdet += Math.log(l[i * n + i] ?? 1)
    fit = { f, pi, sqrtW, l, logMarginalLikelihood: next - logdet }
    if (next - objective < 1e-10) break
    objective = next
  }
  // Recompute the factor at the final mode (as scikit-learn's posterior mode).
  const pi = Float64Array.from(f, sigmoid)
  const sqrtW = Float64Array.from(pi, (p) => Math.sqrt(p * (1 - p)))
  const B = Array.from({ length: n * n }, (_, idx) => {
    const i = Math.floor(idx / n)
    const j = idx % n
    return (i === j ? 1 : 0) + (sqrtW[i] ?? 0) * (K[idx] ?? 0) * (sqrtW[j] ?? 0)
  })
  const l = flat(cholesky(fromStorage(Float64Array.from(B), [n, n], 'float64')))
  return { ...(fit as LaplaceFit), f, pi, sqrtW, l }
}

const LAMBDAS = [0.41, 0.4, 0.37, 0.44, 0.39]
const COEFS = [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654]

/** The predictive probability of class 1: the logistic integrated against N(mean, variance) (scikit-learn's expansion). */
function averagedSigmoid(mean: number, variance: number): number {
  const alpha = 1 / (2 * variance)
  let s = 0
  for (const [k, lam] of LAMBDAS.entries()) {
    const gamma = lam * mean
    const integral =
      (Math.sqrt(Math.PI / alpha) * (toFlat(erf(gamma * Math.sqrt(alpha / (alpha + lam * lam))))[0] ?? 0)) /
      (2 * Math.sqrt(variance * 2 * Math.PI))
    s += (COEFS[k] ?? 0) * integral
  }
  return s + 0.5 * COEFS.reduce((a, b) => a + b, 0)
}

/** Hyperparameters of {@link gaussianProcessClassifier}. */
export interface GpcHyper {
  readonly kernel: Kernel
  /** Fit the hyperparameters by maximising the Laplace approximate marginal likelihood (Nelder-Mead in log space). */
  readonly optimize: boolean
  readonly maxIterPredict: number
  /** Nelder-Mead steps when optimising (default 100). */
  readonly maxIter: number
  /** Grid points per axis of the approximate marginal-likelihood surface (default 12; 0 for none). */
  readonly surfaceResolution: number
}

/** One binary problem's fit: class `positive` against the rest. */
export type GpcBinary = { readonly theta: number[]; readonly f: number[]; readonly logMarginalLikelihood: number }

/** Parameters of a fitted GP classifier. */
export type GpcParams = {
  readonly x: number[][]
  readonly classes: number[]
  readonly problems: GpcBinary[]
  readonly targets: number[][]
}

/** Verbs of a fitted GP classifier. */
export interface GpcVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The latent posterior mean and variance of each binary problem at some points. */
  readonly latent: (x: TensorLike) => { readonly mean: number[][]; readonly variance: number[][] }
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted GP classifier. */
export type FittedGpc = Fitted<GpcParams> & GpcVerbs

function gpcVerbs(p: GpcParams, h: GpcHyper): GpcVerbs {
  const n = p.x.length
  const d = p.x[0]?.length ?? 0
  const prepared = p.problems.map((prob, s) => {
    const kernel = withTheta(h.kernel, prob.theta)
    const K = flat(kernel.k(p.x))
    const y = Float64Array.from(p.targets[s] ?? [])
    const fit = laplace(K, n, y, h.maxIterPredict)
    return { kernel, fit, y }
  })
  const latent = (x: TensorLike): { mean: number[][]; variance: number[][] } => {
    const m = matrixView(x, 'gaussianProcessClassifier', { nFeatures: d }).n
    const mean: number[][] = []
    const variance: number[][] = []
    for (const { kernel, fit, y } of prepared) {
      const ks = flat(kernel.k(p.x, x))
      const diag = flat(kernel.diag(x))
      const mu: number[] = []
      const va: number[] = []
      for (let j = 0; j < m; j++) {
        const col = Array.from({ length: n }, (_, i) => ks[i * m + j] ?? 0)
        let fm = 0
        for (let i = 0; i < n; i++) fm += (col[i] ?? 0) * ((y[i] ?? 0) - (fit.pi[i] ?? 0))
        const v = triSolve(
          fit.l,
          n,
          Float64Array.from(col, (c, i) => c * (fit.sqrtW[i] ?? 0)),
          false,
        )
        let vv = 0
        for (const t of v) vv += t * t
        mu.push(fm)
        va.push(Math.max((diag[j] ?? 0) - vv, 1e-12))
      }
      mean.push(mu)
      variance.push(va)
    }
    return { mean, variance }
  }
  const proba = (x: TensorLike): Tensor => {
    const { mean, variance } = latent(x)
    const m = mean[0]?.length ?? 0
    const k = p.classes.length
    const out = new Float64Array(m * k)
    for (let i = 0; i < m; i++) {
      if (k === 2) {
        const q = averagedSigmoid(mean[0]?.[i] ?? 0, variance[0]?.[i] ?? 1)
        out[i * 2] = 1 - q
        out[i * 2 + 1] = q
      } else {
        const qs = p.classes.map((_, c) => averagedSigmoid(mean[c]?.[i] ?? 0, variance[c]?.[i] ?? 1))
        const total = qs.reduce((a, b) => a + b, 0)
        for (let c = 0; c < k; c++) out[i * k + c] = (qs[c] ?? 0) / total
      }
    }
    return fromStorage(out, [m, k], 'float64')
  }
  return {
    predict: (x) => {
      const pr = proba(x)
      const k = p.classes.length
      const m = pr.shape[0] ?? 0
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => {
          let best = 0
          for (let c = 1; c < k; c++) if ((pr.data[i * k + c] ?? 0) > (pr.data[i * k + best] ?? 0)) best = c
          return p.classes[best] ?? 0
        }),
        [m],
        'float64',
      )
    },
    predictProba: proba,
    predictDistribution: (x) => categorical({ probs: proba(x) }),
    latent,
    classes: p.classes,
    nFeaturesIn: d,
  }
}

/**
 * Gaussian-process classification by the Laplace approximation (sklearn.gaussian_process.GaussianProcessClassifier):
 * Newton's iteration for the posterior mode of the latent function under a logistic likelihood, predictive
 * probabilities by integrating the logistic against the latent Gaussian, one-versus-rest beyond two classes, and the
 * hyperparameters fitted by maximising the approximate marginal likelihood (traced Nelder-Mead in log space).
 * Exposition: the hyperparameters, the approximate marginal-likelihood surface over the first two log hyperparameters
 * with the optimiser's path (binary problems), the latent posterior on a grid for one feature, and for two features
 * the decision surface.
 *
 * @param hyper - Kernel (default ConstantKernel * RBF), whether to optimise, and Newton iterations.
 * @returns The estimator.
 */
export function gaussianProcessClassifier(hyper: Partial<GpcHyper> = {}): ExplainedEstimator<GpcHyper, FittedGpc> {
  const est: ExplainedEstimator<GpcHyper, FittedGpc> = defineExplainedEstimator(
    {
      name: 'gaussian-process-classifier',
      kind: 'classifier',
      hyperSpace: (h: GpcHyper): Space => dict({ optimize: bool({ default: h.optimize }) }),
      fit: (data: LearnData, h: GpcHyper, options: FitOptions) => {
        const m = matrixView(data.x, 'gaussianProcessClassifier')
        const { classes, codes } = encodeLabels(data.y ?? [], 'gaussianProcessClassifier', m.n)
        if (classes.length < 2) throw new WhiteboxError('value', 'gaussianProcessClassifier: needs two classes')
        const x = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const targets = (classes.length === 2 ? [1] : classes.map((_, c) => c)).map((c) =>
          Array.from(codes, (v) => (v === c ? 1 : 0)),
        )
        const t0 = Array.from(thetaOf(h.kernel).data)
        const approxLml = (t: readonly number[], y: Float64Array): number => {
          try {
            return laplace(flat(withTheta(h.kernel, t).k(x)), m.n, y).logMarginalLikelihood
          } catch {
            return Number.NEGATIVE_INFINITY
          }
        }
        const paths: [number, number][][] = []
        const problems = targets.map((tg) => {
          const y = Float64Array.from(tg)
          let theta = t0
          if (h.optimize && t0.length > 0) {
            const objective = (v: Value): Value => {
              const value = -approxLml(flat(v), y)
              return fromStorage(Float64Array.of(Number.isFinite(value) ? value : 1e300), [], 'float64')
            }
            const tracked = collecting(
              nelderMeadAlgorithm(objective, Float64Array.from(t0), {
                bounds: t0.map((): [number, number] => [Math.log(1e-5), Math.log(1e5)]),
              }),
              (s) => Array.from(s.x),
            )
            const r = runAlgorithm(tracked.algorithm, {
              maxSteps: h.maxIter,
              ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
              ...(options.policy === undefined ? {} : { policy: options.policy }),
            })
            theta = Array.from(r.state.x)
            paths.push(tracked.history.map((xs) => [Math.exp(xs[0] ?? 0), Math.exp(xs[1] ?? xs[0] ?? 0)]))
          }
          const fit = laplace(flat(withTheta(h.kernel, theta).k(x)), m.n, y)
          return { theta, f: Array.from(fit.f), logMarginalLikelihood: fit.logMarginalLikelihood }
        })
        const params: GpcParams = { x, classes: [...classes], problems, targets }
        const fitted = est.revive(params)
        const hp = hyperparameters(withTheta(h.kernel, problems[0]?.theta ?? t0)).filter((p) => p.size > 0)
        const names = hp.flatMap((p) => (p.size === 1 ? [p.name] : p.value.map((_, i) => `${p.name}[${i}]`)))
        const exposition: Exposition[] = [coefficientTable({ names, estimates: hp.flatMap((p) => [...p.value]) })]
        if (classes.length === 2 && t0.length >= 1 && h.surfaceResolution > 0) {
          const centre = problems[0]?.theta ?? t0
          const axis = (i: number): number[] =>
            linspaceNumbers((centre[i] ?? 0) - 2.5, (centre[i] ?? 0) + 2.5, h.surfaceResolution).map(Math.exp)
          const xs = axis(0)
          const ys = centre.length >= 2 ? axis(1) : [Math.exp(centre[0] ?? 0)]
          const y = Float64Array.from(targets[0] ?? [])
          const values = ys.map((b) =>
            xs.map((a) => {
              const t = [...centre]
              t[0] = Math.log(a)
              if (t.length >= 2) t[1] = Math.log(b)
              return approxLml(t, y)
            }),
          )
          exposition.push(
            hyperparameterSurface({
              parameters: [names[0] ?? 'theta0', names[1] ?? names[0] ?? 'theta1'],
              x: xs,
              y: ys,
              values,
              ...(paths[0] === undefined ? {} : { path: paths[0] }),
              logScale: [true, true],
            }),
          )
        }
        const features = featureNamesOf(data, m.d)
        if (m.d === 1) {
          const xsData = x.map((r) => r[0] ?? 0)
          const lo = minOf(xsData)
          const hi = maxOf(xsData)
          const pad = 0.1 * (hi - lo || 1)
          const grid = linspaceNumbers(lo - pad, hi + pad, 60).map((v) => [v])
          const lat = fitted.latent(grid)
          exposition.push(
            posteriorGrid({
              inputs: grid,
              mean: lat.mean[0] ?? [],
              variance: lat.variance[0] ?? [],
              observed: { inputs: x, targets: targets[0] ?? [] },
            }),
          )
        }
        if (m.d === 2)
          exposition.push(
            decisionSurfaceOf(fitted, data.x, {
              featureNames: features,
              classNames: classNamesOf(data, classes),
              resolution: 20,
            }),
          )
        return { params, exposition }
      },
      methods: (p: GpcParams, h: GpcHyper) => gpcVerbs(p, h),
    },
    {
      kernel: hyper.kernel ?? defaultGpKernel(),
      optimize: hyper.optimize ?? true,
      maxIterPredict: hyper.maxIterPredict ?? 100,
      maxIter: hyper.maxIter ?? 100,
      surfaceResolution: hyper.surfaceResolution ?? 12,
    },
  )
  return est
}

/** {@link gaussianProcessClassifier} as a class. */
export class GaussianProcessClassifier extends ExplainedEstimatorClass<GpcHyper, FittedGpc> {
  /** @param hyper - As {@link gaussianProcessClassifier}. */
  constructor(hyper: Partial<GpcHyper> = {}) {
    super(gaussianProcessClassifier(hyper))
  }
}
