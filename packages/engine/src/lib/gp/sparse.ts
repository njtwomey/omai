/**
 * Sparse Gaussian-process regression with inducing points, as an estimator on the iterative base: the subset of
 * regressors (SoR), the deterministic training conditional (DTC), the fully independent training conditional (FITC) and
 * the variational free energy (VFE, Titsias) approximations, which replace the n x n kernel matrix by one through m
 * inducing inputs. The kernel's hyperparameters, the noise and the inducing inputs are fitted together on the
 * approximation's marginal likelihood (the VFE one is a lower bound on the exact likelihood), one L-BFGS-B iteration per
 * step with the bound and the inducing points on the tape.
 */

import { MdmlError } from '../base/index.ts'
import { type PosteriorGrid, posteriorGrid } from '../exposition/index.ts'
import {
  asRows,
  hyperparameters,
  type Kernel,
  rbfKernel,
  scaleKernel,
  theta as thetaOf,
  withTheta,
} from '../kernels/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView, targetVector } from '../learn/index.ts'
import { cholesky, solveTriangular } from '../numerics/linalg/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import {
  add,
  diagonal,
  div,
  exp,
  expandDims,
  eye,
  fromStorage,
  log,
  matmul,
  maximum,
  mul,
  ones,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { currentOf, lbfgsbSearch, type SearchState } from './_search.ts'

/** The approximations: they differ in the diagonal correction of the training covariance and in the predictive variance. */
export type SparseApproximation = 'sor' | 'dtc' | 'fitc' | 'vfe'

/** The hyperparameters of {@link sparseGp}. */
export interface SparseGpHyper {
  readonly kernel: Kernel
  readonly approximation: SparseApproximation
  /** The number of inducing points (a random subset of the training inputs to start), or the starting points themselves. */
  readonly inducing: number | readonly (readonly number[])[]
  /** The starting noise variance. */
  readonly noise: number
  /** Whether to fit the kernel's hyperparameters, the noise and (if `optimiseInducing`) the inducing inputs. */
  readonly optimise: boolean
  /** Whether the inducing inputs move under the optimisation (otherwise they stay where they start). */
  readonly optimiseInducing: boolean
  /** Added to the diagonal of the inducing kernel matrix, relative to its mean diagonal. */
  readonly jitter: number
  readonly normalizeY: boolean
  /** Extra runs from other random subsets of inducing points (when `inducing` is a count). */
  readonly restarts: number
  readonly bounds: readonly [number, number]
  readonly gradTol: number
  readonly maxIter: number
}

/** The state of {@link sparseGp}: a multi-start L-BFGS-B search over (kernel theta, log noise, inducing inputs). */
export type SparseState = SearchState

/** The fitted parameters of {@link sparseGp}: plain JSON. */
export type SparseParams = {
  readonly theta: number[]
  readonly noise: number
  /** The inducing inputs `[m][d]`. */
  readonly z: number[][]
  /** The bound (or approximate log marginal likelihood) at these parameters. */
  readonly bound: number
  readonly x: number[][]
  readonly y: number[]
  readonly yMean: number
  readonly yStd: number
}

/** What a fitted sparse Gaussian process offers. */
export interface SparseGpVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** The posterior mean and standard deviation `[m]` of the latent function. */
  readonly predictWithStd: (x: TensorLike) => { readonly mean: Tensor; readonly std: Tensor }
  readonly score: (data: LearnData) => number
  readonly kernel: Kernel
  readonly nFeaturesIn: number
}

const WHO = 'sparseGp'

interface Prepared {
  readonly n: number
  readonly d: number
  readonly rows: number[][]
  readonly y: Float64Array
  readonly yn: Float64Array
  readonly yMean: number
  readonly yStd: number
  readonly xs: Tensor
  readonly m: number
}

function prepare(data: LearnData, h: SparseGpHyper): Prepared {
  const mat = matrixView(data.x, WHO)
  const y = targetVector(data.y ?? [], WHO, mat.n)
  const rows = Array.from({ length: mat.n }, (_, i) => Array.from(mat.data.subarray(i * mat.d, (i + 1) * mat.d)))
  let yMean = 0
  let yStd = 1
  if (h.normalizeY) {
    yMean = y.reduce((a, b) => a + b, 0) / mat.n
    const sd = Math.sqrt(y.reduce((a, b) => a + (b - yMean) ** 2, 0) / mat.n)
    yStd = sd > 0 ? sd : 1
  }
  const m = typeof h.inducing === 'number' ? Math.min(Math.max(1, Math.round(h.inducing)), mat.n) : h.inducing.length
  return {
    n: mat.n,
    d: mat.d,
    rows,
    y,
    yn: Float64Array.from(y, (v) => (v - yMean) / yStd),
    yMean,
    yStd,
    xs: fromStorage(mat.data, [mat.n, mat.d], 'float64'),
    m,
  }
}

interface Pieces {
  /** log N(y | 0, Q + Lambda). */
  readonly logLikelihood: Value
  /** The trace term of the variational bound, sum(diag K) - sum(diag Q), over 2 sigma^2 (zero but for VFE). */
  readonly penalty: Value
}

/** The pieces of the marginal likelihood by the Woodbury identity through the m x m factor, differentiable. */
function marginal(
  approximation: SparseApproximation,
  kernel: Kernel,
  x: Value,
  y: Value,
  z: Value,
  sigma2: Value,
  jitter: number,
): Pieces {
  const n = x.shape[0] ?? 0
  const m = z.shape[0] ?? 0
  const Kmm = kernel.k(z)
  const scale = mul(sum(diagonal(Kmm)), 1 / m)
  const L = cholesky(add(Kmm, mul(eye(m), mul(scale, jitter))))
  const Knm = kernel.k(x, z)
  const V = solveTriangular(L, transpose(Knm), { lower: true })
  const diagQ = sum(square(V), 0)
  const diagK = kernel.diag(x)
  const lambda = approximation === 'fitc' ? add(sigma2, maximum(sub(diagK, diagQ), 0)) : mul(ones([n]), sigma2)
  const Vl = div(V, expandDims(lambda, 0))
  const B = add(eye(m), matmul(Vl, transpose(V)))
  const LB = cholesky(B)
  const c = solveTriangular(LB, matmul(Vl, expandDims(y, 1)), { lower: true })
  const quad = sub(sum(div(square(y), lambda)), sum(square(c)))
  const logdet = add(sum(log(lambda)), mul(sum(log(diagonal(LB))), 2))
  const logLikelihood = sub(mul(add(quad, logdet), -0.5), 0.5 * n * Math.log(2 * Math.PI))
  const penalty = approximation === 'vfe' ? div(sub(sum(diagK), sum(diagQ)), mul(sigma2, 2)) : mul(sigma2, 0)
  return { logLikelihood, penalty }
}

/** The objective, minus the bound, as a function of the packed parameters. */
function objectiveOf(h: SparseGpHyper, p: Prepared, p0: number, fixedZ: Tensor | null): (params: Value) => Value {
  const y = fromStorage(p.yn, [p.n], 'float64')
  return (params) => {
    const kernel = p0 === 0 ? h.kernel : withTheta(h.kernel, sliceAxis(params, 0, 0, p0))
    const sigma2 = exp(reshape(sliceAxis(params, 0, p0, p0 + 1), []))
    const z = fixedZ !== null ? fixedZ : reshape(sliceAxis(params, 0, p0 + 1, p0 + 1 + p.m * p.d), [p.m, p.d])
    const { logLikelihood, penalty } = marginal(h.approximation, kernel, p.xs, y, z, sigma2, h.jitter)
    return mul(sub(logLikelihood, penalty), -1)
  }
}

function startsFor(h: SparseGpHyper, p: Prepared, k: Key, p0: number): number[][] {
  const base = [...toFlat(thetaOf(h.kernel)), Math.log(h.noise)]
  const out: number[][] = []
  const runs = typeof h.inducing === 'number' && h.optimise ? h.restarts + 1 : 1
  for (let r = 0; r < runs; r++) {
    let z: number[]
    if (typeof h.inducing === 'number') {
      const order = Array.from(permutation(foldIn(k, r), p.n).data)
      z = order.slice(0, p.m).flatMap((i) => p.rows[i] ?? [])
    } else z = h.inducing.flatMap((row) => [...row])
    out.push([...base, ...z])
  }
  void p0
  return out
}

function packedBox(h: SparseGpHyper, p: Prepared): { lower: number[]; upper: number[] } {
  const lower: number[] = []
  const upper: number[] = []
  for (const hp of hyperparameters(h.kernel))
    for (let i = 0; i < hp.size; i++) {
      lower.push(hp.constraint === 'positive' ? Math.log(h.bounds[0]) : -Infinity)
      upper.push(hp.constraint === 'positive' ? Math.log(h.bounds[1]) : Infinity)
    }
  lower.push(Math.log(1e-8))
  upper.push(Math.log(1e5))
  for (let i = 0; i < p.m * p.d; i++) {
    lower.push(-Infinity)
    upper.push(Infinity)
  }
  return { lower, upper }
}

function unpack(
  h: SparseGpHyper,
  p: Prepared,
  x: readonly number[],
): { theta: number[]; noise: number; z: number[][] } {
  const p0 = thetaOf(h.kernel).shape[0] ?? 0
  const z = Array.from({ length: p.m }, (_, i) => x.slice(p0 + 1 + i * p.d, p0 + 1 + (i + 1) * p.d))
  return { theta: x.slice(0, p0), noise: Math.exp(x[p0] ?? 0), z }
}

function verbsOf(params: SparseParams, h: SparseGpHyper): SparseGpVerbs {
  const kernel = params.theta.length === 0 ? h.kernel : withTheta(h.kernel, params.theta)
  const d = params.x[0]?.length ?? 0
  const n = params.y.length
  const m = params.z.length
  const x = fromStorage(Float64Array.from(params.x.flat()), [n, d], 'float64')
  const z = fromStorage(Float64Array.from(params.z.flat()), [m, d], 'float64')
  const y = fromStorage(
    Float64Array.from(params.y, (v) => (v - params.yMean) / params.yStd),
    [n],
    'float64',
  )
  let cache: { readonly L: Value; readonly LB: Value; readonly weights: Value } | undefined
  const factors = (): { readonly L: Value; readonly LB: Value; readonly weights: Value } => {
    if (cache !== undefined) return cache
    const Kmm = kernel.k(z)
    const scale = mul(sum(diagonal(Kmm)), 1 / m)
    const L = cholesky(add(Kmm, mul(eye(m), mul(scale, h.jitter))))
    const V = solveTriangular(L, transpose(kernel.k(x, z)), { lower: true })
    const diagQ = sum(square(V), 0)
    const lambda =
      h.approximation === 'fitc'
        ? add(params.noise, maximum(sub(kernel.diag(x), diagQ), 0))
        : mul(ones([n]), params.noise)
    const Vl = div(V, expandDims(lambda, 0))
    const LB = cholesky(add(eye(m), matmul(Vl, transpose(V))))
    const weights = solveTriangular(LB, matmul(Vl, expandDims(y, 1)), { lower: true })
    cache = { L, LB, weights }
    return cache
  }
  const at = (xnew: TensorLike): { mean: Value; variance: Value } => {
    const mat = matrixView(xnew, WHO, { nFeatures: d })
    const xs = fromStorage(mat.data, [mat.n, mat.d], 'float64')
    const { L, LB, weights } = factors()
    const A = solveTriangular(L, transpose(kernel.k(xs, z)), { lower: true })
    const mean = reshape(matmul(transpose(solveTriangular(LB, A, { lower: true })), weights), [-1])
    const explained = sum(square(solveTriangular(LB, A, { lower: true })), 0)
    const projected = sum(square(A), 0)
    const variance = h.approximation === 'sor' ? explained : add(sub(kernel.diag(xs), projected), explained)
    return { mean, variance: maximum(variance, 0) }
  }
  const meanOf = (xnew: TensorLike): Tensor => primalOf(add(mul(at(xnew).mean, params.yStd), params.yMean))
  return {
    predict: meanOf,
    predictWithStd: (xnew) => {
      const r = at(xnew)
      return {
        mean: primalOf(add(mul(r.mean, params.yStd), params.yMean)),
        std: primalOf(mul(sqrt(r.variance), params.yStd)),
      }
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
    kernel,
    nFeaturesIn: d,
  }
}

/**
 * The bound (or approximate log marginal likelihood) of a sparse approximation at given parameters.
 *
 * @param approximation - `sor`, `dtc`, `fitc` or `vfe`.
 * @param kernel - The kernel.
 * @param x - Inputs `[n, d]`.
 * @param y - Targets `[n]`.
 * @param z - Inducing inputs `[m, d]`.
 * @param noise - The noise variance.
 * @param jitter - Relative jitter on the inducing kernel matrix (default 1e-6).
 * @returns The value: log N(y | 0, Q + Lambda), less the trace term for `vfe`; for m = n and Z = X it is the exact log
 *   marginal likelihood.
 * @concept sparse-gaussian-processes
 * @reference M. Titsias, Variational learning of inducing variables in sparse Gaussian processes (2009); J. Quinonero-Candela
 * and C. E. Rasmussen, A unifying view of sparse approximate Gaussian process regression (2005).
 * @example
 * ```ts
 * import { sparseBound } from 'mdml-engine/gp'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * const x = [[0], [1], [2]]
 * const exact = sparseBound('vfe', rbfKernel(), x, [0, 1, 0], x, 0.1, 1e-9)
 * Math.round(exact * 1e4) / 1e4 // => -3.4936
 * ```
 */
export function sparseBound(
  approximation: SparseApproximation,
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  z: TensorLike,
  noise: number,
  jitter = 1e-6,
): number {
  const { logLikelihood, penalty } = marginal(
    approximation,
    kernel,
    asRows(x, 'sparseBound').rows,
    promoteFloat(y)[0] as Value,
    asRows(z, 'sparseBound').rows,
    fromStorage(Float64Array.of(noise), [], 'float64'),
    jitter,
  )
  return toFlat(sub(logLikelihood, penalty))[0] ?? Number.NaN
}

/**
 * The paths of the sparse Gaussian process: the bound, the hyperparameters, the inducing inputs and the derived `model/posterior`.
 *
 * @example
 * ```ts
 * import type { SparseGpPath } from 'mdml-engine/gp'
 * const path: SparseGpPath = 'gp/inducing'
 * path // => 'gp/inducing'
 * ```
 */
export type SparseGpPath = 'gp/bound' | 'gp/hyper' | 'gp/inducing' | 'model/posterior'

/**
 * Sparse Gaussian-process regression with inducing points. `approximation` picks the model: `sor`, `dtc` (the same
 * likelihood, different predictive variance), `fitc` (a diagonal correction of the training covariance) or `vfe` (the
 * variational bound of Titsias, which never overfits the inducing points). The fit maximises the bound over the kernel's
 * hyperparameters, the noise and, unless `optimiseInducing` is false, the inducing inputs (starting from a random subset of
 * the data), one L-BFGS-B iteration per step, logging `gp/bound`, `gp/hyper` (the kernel hyperparameters, then the noise
 * variance) and `gp/inducing` (the inducing inputs): the points can be watched moving. The probe `model/posterior` is the
 * posterior over the first feature, with the inducing points' values on the mean.
 *
 * @concept sparse-gaussian-processes
 * @example
 * ```ts
 * import { sparseGp } from 'mdml-engine/gp'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = Array.from({ length: 30 }, (_, i) => [i / 5])
 * const y = x.map(([v]) => Math.sin(v as number))
 * const fitted = sparseGp.with({ inducing: 6, approximation: 'vfe' }).fit({ x, y })
 * Math.abs((toFlat(fitted.predict([[3]]))[0] as number) - Math.sin(3)) < 0.1 // => true
 * ```
 */
export const sparseGp: IterativeEstimator<SparseGpHyper, SparseState, SparseParams, SparseGpVerbs, SparseGpPath> =
  iterative<SparseGpHyper, SparseState, SparseParams, SparseGpVerbs, SparseGpPath>({
    name: 'gp/sparse',
    kind: 'regressor',
    defaults: {
      kernel: scaleKernel(rbfKernel(), 1),
      approximation: 'vfe',
      inducing: 10,
      noise: 0.1,
      optimise: true,
      optimiseInducing: true,
      jitter: 1e-6,
      normalizeY: false,
      restarts: 0,
      bounds: [1e-5, 1e5],
      gradTol: 1e-5,
      maxIter: 300,
    },
    hyperSpace: (h) =>
      dict({
        ...(typeof h.inducing === 'number'
          ? { inducing: int(2, 200, { default: Math.min(Math.max(h.inducing, 2), 200), label: 'inducing points' }) }
          : {}),
        noise: real(1e-6, 10, { log: true, default: Math.min(Math.max(h.noise, 1e-6), 10), label: 'noise variance' }),
      }),
    algorithm: (data, h): Algorithm<SparseState, SparseGpPath> => {
      const p = prepare(data, h)
      if (
        h.approximation !== 'sor' &&
        h.approximation !== 'dtc' &&
        h.approximation !== 'fitc' &&
        h.approximation !== 'vfe'
      )
        throw new MdmlError('value', `${WHO}: unknown approximation ${String(h.approximation)}`)
      const p0 = thetaOf(h.kernel).shape[0] ?? 0
      const box = packedBox(h, p)
      const objective = objectiveOf(h, p, p0, null)
      const frozen = !h.optimiseInducing
      // With the inducing inputs held, they are boxed to where they start (a degenerate box that L-BFGS-B never moves).
      const search = lbfgsbSearch({
        objective,
        lower: box.lower,
        upper: box.upper,
        optimise: h.optimise,
        gradTol: h.gradTol,
        maxIter: h.maxIter,
        starts: (k) => {
          const starts = startsFor(h, p, k, p0)
          if (frozen) {
            const first = starts[0] as number[]
            for (let i = p0 + 1; i < first.length; i++) {
              box.lower[i] = first[i] as number
              box.upper[i] = first[i] as number
            }
          }
          return starts
        },
      })
      return {
        name: 'sparse-gp',
        config: {
          approximation: h.approximation,
          inducing: p.m,
          optimise: h.optimise,
          optimiseInducing: h.optimiseInducing,
        },
        schema: channels({
          metrics: {
            gradNorm: { description: 'the infinity norm of the projected gradient of the bound' },
            run: { description: 'the optimiser run in progress' },
          },
        }),
        init: (k) => search.init(k),
        step: (s, ctx) => {
          const next = search.step(s, ctx)
          if (s.inner !== null) {
            const now = currentOf(next)
            const un = unpack(h, p, now.x)
            ctx.log('gp/bound', -now.value)
            ctx.log('gp/hyper', [
              ...hyperparameters(un.theta.length === 0 ? h.kernel : withTheta(h.kernel, un.theta)).flatMap((hp) => [
                ...hp.value,
              ]),
              un.noise,
            ])
            ctx.log('gp/inducing', un.z)
          }
          return next
        },
        done: (s) => (s.finished ? 'converged' : undefined),
        observe: (s) => ({
          metrics: { run: s.run, ...(s.inner === null ? {} : { gradNorm: s.inner.projectedGradNorm }) },
        }),
        serialise: (s) => stateToJson(s),
        deserialise: (j) => stateFromJson(j) as SparseState,
      }
    },
    maxSteps: (h) => (h.restarts + 1) * h.maxIter + 1,
    params: (s, data, h) => {
      const p = prepare(data, h)
      const model = currentOf(s)
      const un = unpack(h, p, model.x)
      return { ...un, bound: -model.value, x: p.rows, y: Array.from(p.y), yMean: p.yMean, yStd: p.yStd }
    },
    verbs: verbsOf,
    contract: (data, h): Contract<SparseState, SparseGpPath> => {
      const p = prepare(data, h)
      return defineContract({
        'gp/bound': scalar({
          description: 'the bound (or approximate log marginal likelihood) being maximised',
          direction: 'higher',
        }),
        'gp/hyper': tensor({
          role: 'param',
          description: 'the kernel hyperparameters in natural units, then the noise variance',
        }),
        'gp/inducing': tensor({ role: 'param', description: 'the inducing inputs [m, d]' }),
        'model/posterior': derived<SparseState>({
          description: 'the posterior over the first feature with the training data (args: resolution)',
          cost: 'dear',
          compute: (s, args) => gridOf(h, p, s, (args as { resolution?: number } | undefined)?.resolution ?? 100),
        }),
      })
    },
    logs: { 'gp/bound': Log(), 'gp/hyper': Log() },
  })

function gridOf(h: SparseGpHyper, p: Prepared, s: SparseState, resolution: number): PosteriorGrid {
  const model = currentOf(s)
  const un = unpack(h, p, model.x)
  const verbs = verbsOf({ ...un, bound: -model.value, x: p.rows, y: Array.from(p.y), yMean: p.yMean, yStd: p.yStd }, h)
  const column = p.rows.map((r) => r[0] ?? 0)
  const lo = Math.min(...column)
  const hi = Math.max(...column)
  const pad = 0.15 * (hi - lo || 1)
  const centre = Array.from({ length: p.d }, (_, j) => p.rows.reduce((a, r) => a + (r[j] ?? 0), 0) / p.n)
  const m = Math.max(2, Math.min(resolution, 400))
  const inputs = Array.from({ length: m }, (_, i) => {
    const row = [...centre]
    row[0] = lo - pad + ((hi - lo + 2 * pad) * i) / (m - 1)
    return row
  })
  const { mean, std } = verbs.predictWithStd(inputs)
  return posteriorGrid({
    inputs,
    mean: Array.from(toFlat(mean)),
    variance: Array.from(toFlat(std), (v) => v * v),
    observed: { inputs: p.rows, targets: Array.from(p.y) },
  })
}
