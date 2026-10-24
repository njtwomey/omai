import { linspaceNumbers, maxOf, minOf } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { type Distribution, multivariateNormal, normal } from '../../lib/dists/index.ts'
import { coefficientTable, type Exposition, hyperparameterSurface, posteriorGrid } from '../../lib/exposition/index.ts'
import {
  hyperparameters,
  type Kernel,
  rbfKernel,
  scaleKernel,
  theta as thetaOf,
  withTheta,
} from '../../lib/kernels/index.ts'
import {
  collecting,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  type Fitted,
  type LearnData,
  matrixView,
  targetVector,
} from '../../lib/learn/index.ts'
import { cholesky, choleskyLogdet, choSolve, solveTriangular } from '../../lib/numerics/linalg/index.ts'
import { lbfgsbAlgorithm } from '../../lib/optim/index.ts'
import { foldIn, uniform } from '../../lib/random/index.ts'
import { bool, dict, int, real, type Space } from '../../lib/space/index.ts'
import {
  add,
  eye,
  fromStorage,
  matmul,
  mul,
  primalOf,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../../lib/tensor/index.ts'
import { runAlgorithm } from '../../lib/trace/index.ts'

const LOG_2PI = Math.log(2 * Math.PI)
const flat = (v: Value): number[] => toFlat(primalOf(v))

/**
 * The log marginal likelihood log p(y | X, theta) = -y^T K^{-1} y / 2 - log|L| - n log(2 pi) / 2 of a zero-mean GP
 * with K = k(X) + alpha I, differentiable in the kernel's hyperparameters (for `grad` with `withTheta`).
 *
 * @param kernel - The kernel (possibly traced).
 * @param x - Inputs `[n, d]`.
 * @param y - Targets `[n]`.
 * @param alpha - Noise added to the diagonal.
 * @returns The log marginal likelihood (a scalar value).
 */
export function logMarginalLikelihood(kernel: Kernel, x: TensorLike, y: TensorLike, alpha: number): Value {
  const k = kernel.k(x)
  const n = k.shape[0] ?? 0
  const l = cholesky(add(k, mul(eye(n), alpha)))
  const a = choSolve(l, y)
  return sub(sub(mul(sum(mul(y, a)), -0.5), mul(choleskyLogdet(l), 0.5)), (n * LOG_2PI) / 2)
}

/** Hyperparameters of {@link gaussianProcessRegressor}. */
export interface GprHyper {
  /** The kernel (default ConstantKernel(1) * RBF(1), as scikit-learn). */
  readonly kernel: Kernel
  /** Noise variance added to the diagonal (default 1e-10). */
  readonly alpha: number
  /** Fit the hyperparameters by maximising the log marginal likelihood (default true). */
  readonly optimize: boolean
  /** Extra optimiser starts from keyed log-uniform points within the bounds (default 0). */
  readonly nRestarts: number
  /** Standardise the targets before fitting (default false). */
  readonly normalizeY: boolean
  /** log-scale bounds of every positive hyperparameter (default 1e-5 to 1e5, as scikit-learn). */
  readonly bounds: readonly [number, number]
  /** Grid points per axis of the marginal-likelihood surface (default 20; 0 for none). */
  readonly surfaceResolution: number
}

/** Parameters of a fitted GP regressor. */
export type GprParams = {
  readonly theta: number[]
  readonly x: number[][]
  readonly y: number[]
  readonly yMean: number
  readonly yScale: number
  readonly logMarginalLikelihood: number
}

/** Verbs of a fitted GP regressor. */
export interface GprVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** Posterior standard deviations of the latent function. */
  readonly predictStd: (x: TensorLike) => Tensor
  /** Normal predictives with the latent variance plus alpha. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** The joint posterior of the latent function at some points (for samples). */
  readonly posterior: (x: TensorLike) => { readonly mean: Tensor; readonly covariance: Tensor }
  /** The fitted kernel. */
  readonly kernel: Kernel
  readonly nFeaturesIn: number
}

/** A fitted GP regressor. */
export type FittedGpr = Fitted<GprParams> & GprVerbs

function gprVerbs(p: GprParams, h: GprHyper): GprVerbs {
  const kernel = withTheta(h.kernel, p.theta)
  const n = p.x.length
  const d = p.x[0]?.length ?? 0
  const kxx = add(kernel.k(p.x), mul(eye(n), h.alpha))
  const l = cholesky(kxx)
  const alphaVec = choSolve(l, p.y)
  const posterior = (x: TensorLike): { mean: Tensor; covariance: Tensor } => {
    const m = matrixView(x, 'gaussianProcessRegressor', { nFeatures: d })
    const ks = kernel.k(p.x, x)
    const mean = flat(matmul(alphaVec, ks)).map((v) => v * p.yScale + p.yMean)
    const v = solveTriangular(l, ks, { lower: true })
    const cov = flat(sub(kernel.k(x), matmul(transpose(v), v))).map((c) => c * p.yScale * p.yScale)
    return {
      mean: fromStorage(Float64Array.from(mean), [m.n], 'float64'),
      covariance: fromStorage(Float64Array.from(cov), [m.n, m.n], 'float64'),
    }
  }
  const moments = (x: TensorLike): { mean: Float64Array; sd: Float64Array } => {
    const m = matrixView(x, 'gaussianProcessRegressor', { nFeatures: d })
    const ks = kernel.k(p.x, x)
    const mean = Float64Array.from(flat(matmul(alphaVec, ks)), (v) => v * p.yScale + p.yMean)
    const v = flat(solveTriangular(l, ks, { lower: true }))
    const diag = flat(kernel.diag(x))
    const sd = Float64Array.from({ length: m.n }, (_, j) => {
      let s = 0
      for (let i = 0; i < n; i++) s += (v[i * m.n + j] ?? 0) ** 2
      return Math.sqrt(Math.max((diag[j] ?? 0) - s, 0)) * p.yScale
    })
    return { mean, sd }
  }
  return {
    predict: (x) => {
      const { mean } = moments(x)
      return fromStorage(mean, [mean.length], 'float64')
    },
    predictStd: (x) => {
      const { sd } = moments(x)
      return fromStorage(sd, [sd.length], 'float64')
    },
    predictDistribution: (x) => {
      const { mean, sd } = moments(x)
      const scale = Float64Array.from(sd, (s) => Math.sqrt(s * s + h.alpha * p.yScale * p.yScale) || 1e-12)
      return normal(fromStorage(mean, [mean.length], 'float64'), fromStorage(scale, [scale.length], 'float64'))
    },
    posterior,
    kernel,
    nFeaturesIn: d,
  }
}

/** The default kernel: ConstantKernel(1) * RBF(1). */
export function defaultGpKernel(): Kernel {
  return scaleKernel(rbfKernel({ lengthScale: 1 }), 1)
}

/**
 * Gaussian-process regression (sklearn.gaussian_process.GaussianProcessRegressor): the exact posterior given a kernel
 * from the core's kernels, with the hyperparameters fitted by maximising the log marginal likelihood (gradients by
 * automatic differentiation of the Cholesky-based likelihood; traced L-BFGS-B in log coordinates within bounds, with
 * keyed restarts). Exposition: the hyperparameters as a table, the log-marginal-likelihood surface over the first two
 * log hyperparameters with the optimiser's path, and for one feature the posterior on a grid (mean, variance and
 * three samples) with the observations.
 *
 * @param hyper - Kernel, noise alpha, whether to optimise, restarts, target normalisation, bounds and surface size.
 * @returns The estimator.
 */
export function gaussianProcessRegressor(hyper: Partial<GprHyper> = {}): ExplainedEstimator<GprHyper, FittedGpr> {
  const est: ExplainedEstimator<GprHyper, FittedGpr> = defineExplainedEstimator(
    {
      name: 'gaussian-process-regressor',
      kind: 'regressor',
      hyperSpace: (h: GprHyper): Space =>
        dict({
          alpha: real(1e-10, 1, { log: true, default: h.alpha, label: 'noise' }),
          optimize: bool({
            label: 'optimise kernel',
            description: 'Fit the kernel hyperparameters by maximising the marginal likelihood.',
            default: h.optimize,
          }),
          nRestarts: int(0, 10, {
            label: 'optimiser restarts',
            description: 'Extra starts of the kernel hyperparameter optimiser from random points.',
            default: h.nRestarts,
          }),
        }),
      fit: (data: LearnData, h: GprHyper, options: FitOptions) => {
        const m = matrixView(data.x, 'gaussianProcessRegressor')
        const yRaw = targetVector(data.y ?? [], 'gaussianProcessRegressor', m.n)
        const x = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        let yMean = 0
        let yScale = 1
        if (h.normalizeY) {
          yMean = yRaw.reduce((a, b) => a + b, 0) / m.n
          yScale = Math.sqrt(yRaw.reduce((a, b) => a + (b - yMean) ** 2, 0) / m.n) || 1
        }
        const y = Array.from(yRaw, (v) => (v - yMean) / yScale)
        const yT = fromStorage(Float64Array.from(y), [m.n], 'float64')
        const lml = (t: Value): Value => logMarginalLikelihood(withTheta(h.kernel, t), x, yT, h.alpha)
        const t0 = Array.from(thetaOf(h.kernel).data)
        const [lo, hi] = [Math.log(h.bounds[0]), Math.log(h.bounds[1])]
        const path: [number, number][] = []
        let best = {
          theta: t0,
          value:
            toFlat(primalOf(lml(fromStorage(Float64Array.from(t0), [t0.length], 'float64'))))[0] ??
            Number.NEGATIVE_INFINITY,
        }
        if (h.optimize && t0.length > 0) {
          const key = resolveKey(options, 'gaussianProcessRegressor')
          const starts = [
            t0,
            ...Array.from({ length: h.nRestarts }, (_, r) =>
              Array.from(toFlat(uniform(foldIn(key, r), [t0.length], { low: lo, high: hi }))),
            ),
          ]
          for (const [s, start] of starts.entries()) {
            const tracked = collecting(
              lbfgsbAlgorithm((t: Value) => mul(lml(t), -1), Float64Array.from(start), {
                bounds: start.map((): [number, number] => [lo, hi]),
              }),
              (state) => Array.from(state.x),
            )
            const r = runAlgorithm(tracked.algorithm, {
              maxSteps: 200,
              ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
              ...(options.policy === undefined ? {} : { policy: options.policy }),
            })
            const value = -r.state.value
            if (s === 0)
              for (const xs of tracked.history) path.push([Math.exp(xs[0] ?? 0), Math.exp(xs[1] ?? xs[0] ?? 0)])
            if (value > best.value) best = { theta: Array.from(r.state.x), value }
          }
        }
        const params: GprParams = { theta: best.theta, x, y, yMean, yScale, logMarginalLikelihood: best.value }
        const fitted = est.revive(params)
        const hp = hyperparameters(withTheta(h.kernel, best.theta)).filter((p) => p.size > 0)
        const exposition: Exposition[] = [
          coefficientTable({
            names: hp.flatMap((p) => (p.size === 1 ? [p.name] : p.value.map((_, i) => `${p.name}[${i}]`))),
            estimates: hp.flatMap((p) => [...p.value]),
          }),
        ]
        if (h.surfaceResolution > 0 && best.theta.length >= 1) {
          const centre = best.theta
          const axis = (i: number): number[] =>
            linspaceNumbers((centre[i] ?? 0) - 3, (centre[i] ?? 0) + 3, h.surfaceResolution).map(Math.exp)
          const xs = axis(0)
          const ys = best.theta.length >= 2 ? axis(1) : [Math.exp(centre[0] ?? 0)]
          const values = ys.map((b) =>
            xs.map((a) => {
              const t = [...centre]
              t[0] = Math.log(a)
              if (t.length >= 2) t[1] = Math.log(b)
              try {
                return toFlat(primalOf(lml(fromStorage(Float64Array.from(t), [t.length], 'float64'))))[0] ?? Number.NaN
              } catch {
                return Number.NaN
              }
            }),
          )
          const names = hp.flatMap((p) => (p.size === 1 ? [p.name] : p.value.map((_, i) => `${p.name}[${i}]`)))
          exposition.push(
            hyperparameterSurface({
              parameters: [names[0] ?? 'theta0', names[1] ?? names[0] ?? 'theta1'],
              x: xs,
              y: ys,
              values,
              path,
              logScale: [true, true],
            }),
          )
        }
        if (m.d === 1) {
          const xsData = x.map((r) => r[0] ?? 0)
          const lo2 = minOf(xsData)
          const hi2 = maxOf(xsData)
          const pad = 0.1 * (hi2 - lo2 || 1)
          const grid = linspaceNumbers(lo2 - pad, hi2 + pad, 60).map((v) => [v])
          const post = fitted.posterior(grid)
          const cov = Array.from(post.covariance.data)
          const jitter = cov.map((v, i) => (Math.floor(i / 60) === i % 60 ? v + 1e-8 * (1 + Math.abs(v)) : v))
          const samples = toFlat(
            multivariateNormal(post.mean, {
              covariance: fromStorage(Float64Array.from(jitter), [60, 60], 'float64'),
            }).sample(foldIn(resolveKey(options, 'gaussianProcessRegressor'), 1000), [3]),
          )
          exposition.push(
            posteriorGrid({
              inputs: grid,
              mean: Array.from(post.mean.data),
              variance: grid.map((_, i) => Math.max(cov[i * 60 + i] ?? 0, 0)),
              samples: [0, 1, 2].map((s) => samples.slice(s * 60, (s + 1) * 60)),
              observed: { inputs: x, targets: Array.from(yRaw) },
            }),
          )
        }
        return { params, exposition }
      },
      methods: (p: GprParams, h: GprHyper) => gprVerbs(p, h),
    },
    {
      kernel: hyper.kernel ?? defaultGpKernel(),
      alpha: hyper.alpha ?? 1e-10,
      optimize: hyper.optimize ?? true,
      nRestarts: hyper.nRestarts ?? 0,
      normalizeY: hyper.normalizeY ?? false,
      bounds: hyper.bounds ?? [1e-5, 1e5],
      surfaceResolution: hyper.surfaceResolution ?? 20,
    },
  )
  return est
}

/** {@link gaussianProcessRegressor} as a class. */
export class GaussianProcessRegressor extends ExplainedEstimatorClass<GprHyper, FittedGpr> {
  /** @param hyper - As {@link gaussianProcessRegressor}. */
  constructor(hyper: Partial<GprHyper> = {}) {
    super(gaussianProcessRegressor(hyper))
  }
}
