import { linspaceNumbers, maxOf, minOf } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { type Distribution, normal } from '../../lib/dists/index.ts'
import { coefficientTable, type Exposition, posteriorGrid } from '../../lib/exposition/index.ts'
import { hyperparameters, type Kernel, theta as thetaOf, withTheta } from '../../lib/kernels/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  type Fitted,
  type LearnData,
  matrixView,
  targetVector,
} from '../../lib/learn/index.ts'
import { cholesky, solveTriangular } from '../../lib/numerics/linalg/index.ts'
import { lbfgsb } from '../../lib/optim/index.ts'
import { subsample } from '../../lib/random/index.ts'
import { bool, choice, dict, int, type Space } from '../../lib/space/index.ts'
import {
  add,
  diagonal,
  div,
  exp,
  eye,
  fromStorage,
  log,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../../lib/tensor/index.ts'
import { defaultGpKernel } from './regression.ts'

// TODO: stochastic variational GPs, inducing-point optimisation, multi-output and latent-variable GPs.

/** The sparse approximations: subset of regressors, deterministic training conditional, FITC, and Titsias' VFE. */
export type SparseMethod = 'sor' | 'dtc' | 'fitc' | 'vfe'

/** Hyperparameters of {@link sparseGpRegressor}. */
export interface SparseGprHyper {
  readonly kernel: Kernel
  readonly method: SparseMethod
  /** Inducing points, a keyed subset of the training rows (default 20). */
  readonly nInducing: number
  /** The starting noise variance (default 0.1). */
  readonly noise: number
  /** Fit the kernel hyperparameters and the noise by maximising the method's objective (default true). */
  readonly optimize: boolean
}

/** Parameters of a fitted sparse GP. */
export type SparseGprParams = {
  readonly theta: number[]
  readonly noise: number
  readonly inducing: number[][]
  readonly x: number[][]
  readonly y: number[]
  readonly objective: number
}

/** Verbs of a fitted sparse GP. */
export interface SparseGprVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictStd: (x: TensorLike) => Tensor
  /** Normal predictives with the latent variance plus the noise. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly nFeaturesIn: number
}

/** A fitted sparse GP. */
export type FittedSparseGpr = Fitted<SparseGprParams> & SparseGprVerbs

interface Factors {
  readonly l: Value
  readonly lb: Value
  readonly lambda: Value
  readonly c: Value
  readonly a: Value
}

/** The Woodbury factors shared by the objective and the predictions (all differentiable). */
function factors(kernel: Kernel, z: Value, x: Value, y: Value, noise: Value, method: SparseMethod): Factors {
  const m = z.shape[0] ?? 0
  const kuu = add(kernel.k(z), mul(eye(m), 1e-8))
  const l = cholesky(kuu)
  const a = solveTriangular(l, kernel.k(z, x), { lower: true })
  const qdiag = sum(square(a), 0)
  // The noise variance on every row (a vector, so FITC can add its diagonal correction).
  const base = mul(add(mul(qdiag, 0), 1), noise)
  const lambda = method === 'fitc' ? add(base, sub(kernel.diag(x), qdiag)) : base
  const aScaled = div(a, lambda)
  const b = add(eye(m), matmul(aScaled, transpose(a)))
  const lb = cholesky(b)
  const c = solveTriangular(lb, matmul(aScaled, reshape(y, [-1, 1])), { lower: true })
  return { l, lb, lambda, c, a }
}

/** The method's log objective: the log marginal likelihood (SoR, DTC, FITC) or Titsias' lower bound (VFE). */
function objective(kernel: Kernel, z: Value, x: Value, y: Value, noise: Value, method: SparseMethod): Value {
  const f = factors(kernel, z, x, y, noise, method)
  const n = x.shape[0] ?? 0
  const logdet = add(sum(log(f.lambda)), mul(sum(log(diagonal(f.lb))), 2))
  const quad = sub(sum(div(square(y), f.lambda)), sum(square(f.c)))
  let value = mul(add(add(logdet, quad), n * Math.log(2 * Math.PI)), -0.5)
  if (method === 'vfe') value = sub(value, div(mul(sum(sub(kernel.diag(x), sum(square(f.a), 0))), 0.5), noise))
  return value
}

function sparseVerbs(p: SparseGprParams, h: SparseGprHyper): SparseGprVerbs {
  const kernel = withTheta(h.kernel, p.theta)
  const z = fromStorage(
    Float64Array.from(p.inducing.flat()),
    [p.inducing.length, p.inducing[0]?.length ?? 0],
    'float64',
  )
  const xs = fromStorage(Float64Array.from(p.x.flat()), [p.x.length, p.x[0]?.length ?? 0], 'float64')
  const ys = fromStorage(Float64Array.from(p.y), [p.y.length], 'float64')
  const noise = fromStorage(Float64Array.of(p.noise), [], 'float64')
  const f = factors(kernel, z, xs, ys, noise, h.method)
  const d = p.x[0]?.length ?? 0
  const moments = (q: TensorLike): { mean: number[]; sd: number[] } => {
    matrixView(q, 'sparseGpRegressor', { nFeatures: d })
    const v = solveTriangular(f.l, kernel.k(z, q), { lower: true })
    const w = solveTriangular(f.lb, v, { lower: true })
    const mean = toFlat(primalOf(reshape(matmul(transpose(w), f.c), [-1])))
    const vv = toFlat(primalOf(sum(square(v), 0)))
    const ww = toFlat(primalOf(sum(square(w), 0)))
    const kss = toFlat(primalOf(kernel.diag(q)))
    const sd = mean.map((_, j) => {
      const variance = h.method === 'sor' ? (ww[j] ?? 0) : (kss[j] ?? 0) - (vv[j] ?? 0) + (ww[j] ?? 0)
      return Math.sqrt(Math.max(variance, 0))
    })
    return { mean, sd }
  }
  return {
    predict: (q) => {
      const { mean } = moments(q)
      return fromStorage(Float64Array.from(mean), [mean.length], 'float64')
    },
    predictStd: (q) => {
      const { sd } = moments(q)
      return fromStorage(Float64Array.from(sd), [sd.length], 'float64')
    },
    predictDistribution: (q) => {
      const { mean, sd } = moments(q)
      return normal(
        fromStorage(Float64Array.from(mean), [mean.length], 'float64'),
        fromStorage(
          Float64Array.from(sd, (s) => Math.sqrt(s * s + p.noise)),
          [sd.length],
          'float64',
        ),
      )
    },
    nFeaturesIn: d,
  }
}

/**
 * Sparse Gaussian-process regression with m inducing points (Quinonero-Candela and Rasmussen 2005; Titsias 2009):
 * the subset-of-regressors, DTC and FITC approximate priors, or the variational free energy bound, all through the
 * Woodbury identity in O(n m^2), with the kernel hyperparameters and the noise fitted by maximising the method's
 * objective (gradients by automatic differentiation, L-BFGS-B in log space). Inducing points are a keyed subset of the
 * training rows. Exposition: the hyperparameters, noise and objective as a table (with the inducing inputs for one
 * feature), and for one feature the posterior on a grid with the observations.
 *
 * @param hyper - Kernel (default ConstantKernel * RBF), method (default VFE), inducing points (20), starting noise and
 * whether to optimise.
 * @returns The estimator.
 */
export function sparseGpRegressor(
  hyper: Partial<SparseGprHyper> = {},
): ExplainedEstimator<SparseGprHyper, FittedSparseGpr> {
  const est: ExplainedEstimator<SparseGprHyper, FittedSparseGpr> = defineExplainedEstimator(
    {
      name: 'sparse-gp-regressor',
      kind: 'regressor',
      hyperSpace: (h: SparseGprHyper): Space =>
        dict({
          method: choice(['sor', 'dtc', 'fitc', 'vfe'], {
            label: 'approximation',
            description: 'Subset of regressors, DTC, FITC or variational free energy (Titsias).',
            default: h.method,
          }),
          nInducing: int(2, 100, { default: h.nInducing, label: 'inducing points' }),
          optimize: bool({
            label: 'optimise kernel',
            description: 'Fit the kernel hyperparameters by maximising the marginal likelihood.',
            default: h.optimize,
          }),
        }),
      fit: (data: LearnData, h: SparseGprHyper, options: FitOptions) => {
        const m = matrixView(data.x, 'sparseGpRegressor')
        const yv = targetVector(data.y ?? [], 'sparseGpRegressor', m.n)
        const x = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const count = Math.min(h.nInducing, m.n)
        const picks = Array.from(toFlat(subsample(resolveKey(options, 'sparseGpRegressor'), m.n, count))).sort(
          (a, b) => a - b,
        )
        const inducing = picks.map((i) => x[i] as number[])
        const z = fromStorage(Float64Array.from(inducing.flat()), [count, m.d], 'float64')
        const xs = fromStorage(m.data, [m.n, m.d], 'float64')
        const ys = fromStorage(Float64Array.from(yv), [m.n], 'float64')
        const t0 = [...Array.from(thetaOf(h.kernel).data), Math.log(h.noise)]
        const k = t0.length - 1
        const negObjective = (t: Value): Value =>
          mul(
            objective(
              withTheta(h.kernel, sliceAxis(t, 0, 0, k)),
              z,
              xs,
              ys,
              reshape(exp(sliceAxis(t, 0, k, k + 1)), []),
              h.method,
            ),
            -1,
          )
        let t = t0
        if (h.optimize) {
          const r = lbfgsb(negObjective, Float64Array.from(t0), {
            bounds: t0.map((): [number, number] => [Math.log(1e-6), Math.log(1e5)]),
            maxIterations: 200,
            ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
            ...(options.policy === undefined ? {} : { policy: options.policy }),
          })
          t = Array.from(toFlat(r.x as Tensor))
        }
        const value = -(
          toFlat(primalOf(negObjective(fromStorage(Float64Array.from(t), [t.length], 'float64'))))[0] ?? Number.NaN
        )
        const params: SparseGprParams = {
          theta: t.slice(0, k),
          noise: Math.exp(t[k] ?? Math.log(h.noise)),
          inducing,
          x,
          y: Array.from(yv),
          objective: value,
        }
        const hp = hyperparameters(withTheta(h.kernel, params.theta)).filter((p) => p.size > 0)
        const exposition: Exposition[] = [
          coefficientTable({
            names: [
              ...hp.flatMap((p) => (p.size === 1 ? [p.name] : p.value.map((_, i) => `${p.name}[${i}]`))),
              'noise',
              h.method === 'vfe' ? 'evidence lower bound' : 'log marginal likelihood',
              ...(m.d === 1 ? inducing.map((_, i) => `inducing[${i}]`) : []),
            ],
            estimates: [
              ...hp.flatMap((p) => [...p.value]),
              params.noise,
              value,
              ...(m.d === 1 ? inducing.map((r) => r[0] ?? 0) : []),
            ],
          }),
        ]
        if (m.d === 1) {
          const fitted = est.revive(params)
          const xsData = x.map((r) => r[0] ?? 0)
          const lo = minOf(xsData)
          const hi = maxOf(xsData)
          const pad = 0.1 * (hi - lo || 1)
          const grid = linspaceNumbers(lo - pad, hi + pad, 60).map((v) => [v])
          const sd = toFlat(fitted.predictStd(grid))
          exposition.push(
            posteriorGrid({
              inputs: grid,
              mean: toFlat(fitted.predict(grid)),
              variance: sd.map((s) => s * s),
              observed: { inputs: x, targets: Array.from(yv) },
            }),
          )
        }
        return { params, exposition }
      },
      methods: (p: SparseGprParams, h: SparseGprHyper) => sparseVerbs(p, h),
    },
    {
      kernel: hyper.kernel ?? defaultGpKernel(),
      method: hyper.method ?? 'vfe',
      nInducing: hyper.nInducing ?? 20,
      noise: hyper.noise ?? 0.1,
      optimize: hyper.optimize ?? true,
    },
  )
  return est
}

/** {@link sparseGpRegressor} as a class. */
export class SparseGpRegressor extends ExplainedEstimatorClass<SparseGprHyper, FittedSparseGpr> {
  /** @param hyper - As {@link sparseGpRegressor}. */
  constructor(hyper: Partial<SparseGprHyper> = {}) {
    super(sparseGpRegressor(hyper))
  }
}
