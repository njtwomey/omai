/**
 * Gaussian-process classification (scikit-learn's `GaussianProcessClassifier`, and expectation propagation as in GPML) as
 * an estimator on the iterative base: two classes, a logistic or probit likelihood on a latent Gaussian process, the
 * posterior approximated by the Laplace method or by expectation propagation. The fit has two phases, both stepped: the
 * kernel's hyperparameters on the approximate log marginal likelihood by L-BFGS-B (each step logs the likelihood and the
 * hyperparameters), then the approximation at the fitted hyperparameters one Newton iteration, or one EP site update, at a
 * time.
 */

import { MdmlError } from '../base/index.ts'
import { type PosteriorGrid, posteriorGrid } from '../exposition/index.ts'
import { hyperparameters, type Kernel, rbfKernel, scaleKernel, theta as thetaOf, withTheta } from '../kernels/index.ts'
import {
  encodeLabels,
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  targetVector,
} from '../learn/index.ts'
import { cholesky, choSolve, solveTriangular } from '../numerics/linalg/index.ts'
import { gaussHermite } from '../numerics/quadrature/index.ts'
import { erf, ndtr } from '../numerics/special/index.ts'
import { boundedSolvers } from '../optim/index.ts'
import { foldIn, uniform } from '../random/index.ts'
import { dict, int, type SpaceValue } from '../space/index.ts'
import {
  add,
  expandDims,
  fromStorage,
  matmul,
  maximum,
  mul,
  primalOf,
  reshape,
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
import {
  currentOf,
  defaultSearchSolver,
  gradNormOf,
  lbfgsbSearch,
  negatedObjective,
  type SearchState,
} from './_search.ts'
import {
  type ClassLikelihood,
  type EpState,
  epAlgorithm,
  epMarginal,
  type LaplaceState,
  laplaceAlgorithm,
  laplaceMarginal,
  latentTerms,
} from './approximations.ts'

/** The hyperparameters of {@link gpClassifier}. */
export interface GpClassifierHyper {
  readonly kernel: Kernel
  /** `laplace` (a Newton iteration on the latent function) or `ep` (expectation propagation, probit only). */
  readonly method: 'laplace' | 'ep'
  /** The likelihood of a label given the latent function: the logistic sigmoid or the normal CDF. */
  readonly likelihood: ClassLikelihood
  /** Whether to optimise the kernel's hyperparameters on the approximate log marginal likelihood. */
  readonly optimise: boolean
  /** Extra optimiser runs from random starting points inside the bounds. */
  readonly restarts: number
  /** The bounds of every positive hyperparameter, in natural units. */
  readonly bounds: readonly [number, number]
  /** The solver stops when its gradient (the projected one, for L-BFGS-B) falls below this. */
  readonly gradTol: number
  /** The most solver iterations of one run. */
  readonly maxIter: number
  /** How the approximate likelihood is maximised: a value of `boundedSolvers.space` (default L-BFGS-B); `gradTol` is its tolerance. */
  readonly solver?: SpaceValue
  /** The most Newton iterations (Laplace) or sweeps over the sites (EP) of one posterior. */
  readonly maxInner: number
  /** The tolerance of the posterior: the log marginal likelihood's gain (Laplace) or the largest change of a site (EP). */
  readonly tol: number
  /** The class probability from the latent Gaussian: scikit-learn's five-erf approximation or Gauss-Hermite quadrature (logit only). */
  readonly predictive: 'approximate' | 'quadrature'
}

/** The state of {@link gpClassifier}. */
export interface GpcState {
  readonly phase: 'hyper' | 'posterior' | 'done'
  /** The multi-start search over the hyperparameters. */
  readonly search: SearchState
  /** The Laplace iteration at the fitted hyperparameters (when `method` is `laplace` and the phase has reached it). */
  readonly laplace: LaplaceState | null
  /** The EP sweeps at the fitted hyperparameters. */
  readonly ep: EpState | null
}

/** The fitted parameters of {@link gpClassifier}: plain JSON. */
export type GpcParams = {
  /** The two class labels: the negative, then the positive. */
  readonly classes: number[]
  readonly theta: number[]
  /** The approximate log marginal likelihood. */
  readonly logMarginalLikelihood: number
  readonly x: number[][]
  /** The training labels as 0 or 1. */
  readonly t: number[]
  /** Laplace: the latent function at the mode (`f`); EP: the site precisions (`tau`) and precision-means (`nu`). */
  readonly posterior: { readonly f: number[] } | { readonly tau: number[]; readonly nu: number[] }
}

/** What a fitted Gaussian-process classifier offers. */
export interface GpcVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** `[m, 2]`, columns in the order of `classes`. */
  readonly predictProba: (x: TensorLike) => Tensor
  /** The mean of the latent function `[m]`, positive for the second class. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** The mean and variance of the latent function at the rows of x. */
  readonly latent: (x: TensorLike) => { readonly mean: Tensor; readonly variance: Tensor }
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly kernel: Kernel
  readonly nFeaturesIn: number
}

const WHO = 'gpClassifier'

interface Prepared {
  readonly n: number
  readonly d: number
  readonly rows: number[][]
  readonly t: Float64Array
  readonly classes: number[]
  readonly xs: Tensor
}

function prepare(data: LearnData, h: GpClassifierHyper): Prepared {
  if (h.method === 'ep' && h.likelihood !== 'probit')
    throw new MdmlError('value', `${WHO}: expectation propagation is implemented for the probit likelihood`)
  const m = matrixView(data.x, WHO)
  const { classes, codes } = encodeLabels(data.y ?? [], WHO, m.n)
  if (classes.length !== 2)
    throw new MdmlError(
      'value',
      `${WHO}: needs exactly two classes, got ${classes.length}; fit one classifier per class`,
    )
  const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  return {
    n: m.n,
    d: m.d,
    rows,
    t: Float64Array.from(codes),
    classes: [...classes],
    xs: fromStorage(m.data, [m.n, m.d], 'float64'),
  }
}

const natural = (kernel: Kernel, theta: readonly number[]): number[] =>
  hyperparameters(theta.length === 0 ? kernel : withTheta(kernel, theta)).flatMap((hp) => [...hp.value])

function thetaBox(kernel: Kernel, bounds: readonly [number, number]): { lower: number[]; upper: number[] } {
  const lower: number[] = []
  const upper: number[] = []
  for (const hp of hyperparameters(kernel))
    for (let i = 0; i < hp.size; i++) {
      lower.push(hp.constraint === 'positive' ? Math.log(bounds[0]) : -Infinity)
      upper.push(hp.constraint === 'positive' ? Math.log(bounds[1]) : Infinity)
    }
  return { lower, upper }
}

/** The posterior at theta, run to convergence (no gradient). */
function converged(
  h: GpClassifierHyper,
  p: Prepared,
  theta: readonly number[],
): { posterior: GpcParams['posterior']; lml: number } {
  const kernel = theta.length === 0 ? h.kernel : withTheta(h.kernel, theta)
  const K = kernel.k(p.xs)
  if (h.method === 'laplace') {
    const alg = laplaceAlgorithm(K, p.t, { likelihood: h.likelihood, tol: h.tol })
    let s = alg.init({} as never)
    for (let i = 0; i < h.maxInner && !s.converged; i++) s = alg.step(s, {} as never)
    return { posterior: { f: Array.from(s.f) }, lml: s.logMarginalLikelihood }
  }
  const alg = epAlgorithm(K, p.t, { tol: h.tol })
  let s = alg.init({} as never)
  for (let i = 0; i < h.maxInner * p.n && !s.converged; i++) s = alg.step(s, {} as never)
  return { posterior: { tau: Array.from(s.tau), nu: Array.from(s.nu) }, lml: s.logMarginalLikelihood }
}

const COEFS = [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654] as const
const LAMBDAS = [0.41, 0.4, 0.37, 0.44, 0.39] as const

/** Williams and Barber's five-erf approximation of E[sigmoid(f)] for f ~ N(mean, variance), as scikit-learn uses. */
function logisticIntegral(mean: number, variance: number): number {
  const alpha = 1 / (2 * variance)
  let total = 0.5 * COEFS.reduce((a, b) => a + b, 0)
  for (let i = 0; i < 5; i++) {
    const lambda = LAMBDAS[i] as number
    const integral =
      (Math.sqrt(Math.PI / alpha) * erfNumber(lambda * mean * Math.sqrt(alpha / (alpha + lambda * lambda)))) /
      (2 * Math.sqrt(variance * 2 * Math.PI))
    total += (COEFS[i] as number) * integral
  }
  return total
}

const erfNumber = (x: number): number => toFlat(erf(x))[0] ?? Number.NaN

function verbsOf(params: GpcParams, h: GpClassifierHyper): GpcVerbs {
  const kernel = params.theta.length === 0 ? h.kernel : withTheta(h.kernel, params.theta)
  const d = params.x[0]?.length ?? 0
  const n = params.t.length
  const xs = fromStorage(Float64Array.from(params.x.flat()), [n, d], 'float64')
  const t = Float64Array.from(params.t, (v) => 2 * v - 1)
  const inputs = (x: TensorLike): Value => {
    const m = matrixView(x, WHO, { nFeatures: d })
    return fromStorage(m.data, [m.n, m.d], 'float64')
  }
  // The latent predictive: mean k*^T w and variance k** - |L^-1 S k*|^2, with (w, S, L) from the approximation.
  let cache: { readonly weights: Value; readonly scale: Value; readonly factor: Value } | undefined
  const factors = (): { readonly weights: Value; readonly scale: Value; readonly factor: Value } => {
    if (cache !== undefined) return cache
    const K = kernel.k(xs)
    const eye = Float64Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0))
    const post = params.posterior
    if ('f' in post) {
      const f = Float64Array.from(post.f)
      const terms = latentTerms(h.likelihood, f, t)
      const sw = Float64Array.from(terms.w, Math.sqrt)
      const swValue = fromStorage(sw, [n], 'float64')
      const B = add(fromStorage(eye, [n, n], 'float64'), mul(mul(expandDims(swValue, 1), K), expandDims(swValue, 0)))
      cache = { weights: fromStorage(terms.grad, [n], 'float64'), scale: swValue, factor: cholesky(B) }
    } else {
      const tau = Float64Array.from(post.tau, (v) => Math.max(v, 1e-12))
      const A = add(
        K,
        mul(
          fromStorage(eye, [n, n], 'float64'),
          fromStorage(
            Float64Array.from(tau, (v) => 1 / v),
            [1, n],
            'float64',
          ),
        ),
      )
      const L = cholesky(A)
      const muTilde = Float64Array.from(post.nu, (v, i) => v / (tau[i] ?? 1))
      cache = {
        weights: choSolve(L, fromStorage(muTilde, [n], 'float64')),
        scale: fromStorage(new Float64Array(n).fill(1), [n], 'float64'),
        factor: L,
      }
    }
    return cache
  }
  const latent = (x: TensorLike): { mean: Tensor; variance: Tensor } => {
    const xv = inputs(x)
    const { weights, scale, factor } = factors()
    const Ks = kernel.k(xv, xs)
    const mean = reshape(matmul(Ks, expandDims(weights, -1)), [-1])
    const V = solveTriangular(factor, mul(expandDims(scale, 1), transpose(Ks)), { lower: true })
    const variance = maximum(sub(kernel.diag(xv), sum(square(V), 0)), 1e-12)
    return { mean: primalOf(mean), variance: primalOf(variance) }
  }
  const probability = (x: TensorLike): Float64Array => {
    const { mean, variance } = latent(x)
    const mu = toFlat(mean)
    const v = toFlat(variance)
    if (h.likelihood === 'probit')
      return Float64Array.from(mu, (m, i) => toFlat(ndtr(m / Math.sqrt(1 + (v[i] ?? 0))))[0] ?? 0)
    if (h.predictive === 'approximate') return Float64Array.from(mu, (m, i) => logisticIntegral(m, v[i] ?? 1))
    const rule = gaussHermite(40)
    const nodes = toFlat(rule.nodes)
    const weights = toFlat(rule.weights)
    return Float64Array.from(mu, (m, i) => {
      let total = 0
      for (let k = 0; k < nodes.length; k++)
        total += (weights[k] ?? 0) / (1 + Math.exp(-(m + Math.SQRT2 * Math.sqrt(v[i] ?? 0) * (nodes[k] ?? 0))))
      return total / Math.sqrt(Math.PI)
    })
  }
  const predict = (x: TensorLike): Tensor => {
    const p = probability(x)
    return fromStorage(
      Float64Array.from(p, (v) => (v > 0.5 ? (params.classes[1] as number) : (params.classes[0] as number))),
      [p.length],
      'float64',
    )
  }
  return {
    predict,
    predictProba: (x) => {
      const p = probability(x)
      const out = new Float64Array(2 * p.length)
      p.forEach((v, i) => {
        out[2 * i] = 1 - v
        out[2 * i + 1] = v
      })
      return fromStorage(out, [p.length, 2], 'float64')
    },
    decisionFunction: (x) => latent(x).mean,
    latent,
    score: (data) => {
      const truth = targetVector(data.y ?? [], WHO)
      const fit = toFlat(predict(data.x))
      return fit.reduce((s, v, i) => s + (v === truth[i] ? 1 : 0), 0) / Math.max(1, fit.length)
    },
    classes: params.classes,
    kernel,
    nFeaturesIn: d,
  }
}

/**
 * The paths of the Gaussian-process classifier: the approximate log marginal likelihood, the hyperparameters, the latent function at the training points and the derived `model/posterior`.
 *
 * @example
 * ```ts
 * import type { GpcPath } from 'mdml-engine/gp'
 * const path: GpcPath = 'gp/latent'
 * path // => 'gp/latent'
 * ```
 */
export type GpcPath = 'gp/logMarginalLikelihood' | 'gp/hyper' | 'gp/latent' | 'model/posterior'

/**
 * Gaussian-process classification over any kernel, with the Laplace approximation or expectation propagation. The fit
 * first maximises the approximate log marginal likelihood over the kernel's hyperparameters (L-BFGS-B, the gradient by
 * Rasmussen and Williams' Algorithm 5.1 or its EP counterpart, restarts as in the regressor), each step logging
 * `gp/logMarginalLikelihood` and `gp/hyper`; then it runs the approximation itself one Newton iteration (Laplace) or one
 * site update (EP) per step at the fitted hyperparameters, logging the likelihood and the latent function
 * (`gp/latent`), so the posterior can be watched settling. The probe `model/posterior` is the latent function's mean and
 * variance over the first feature.
 *
 * @concept gaussian-process-classification
 * @reference C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning (2006), chapter 3.
 * @example
 * ```ts
 * import { gpClassifier } from 'mdml-engine/gp'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[-2], [-1.5], [-1], [1], [1.5], [2]]
 * const fitted = gpClassifier.fit({ x, y: [0, 0, 0, 1, 1, 1] })
 * toFlat(fitted.predict([[-1.8], [1.8]])) // => [0, 1]
 * ```
 */
export const gpClassifier: IterativeEstimator<GpClassifierHyper, GpcState, GpcParams, GpcVerbs, GpcPath> = iterative<
  GpClassifierHyper,
  GpcState,
  GpcParams,
  GpcVerbs,
  GpcPath
>({
  name: 'gp/classifier',
  kind: 'classifier',
  defaults: {
    kernel: scaleKernel(rbfKernel(), 1),
    method: 'laplace',
    likelihood: 'logit',
    optimise: true,
    restarts: 0,
    bounds: [1e-5, 1e5],
    gradTol: 1e-5,
    solver: defaultSearchSolver,
    maxIter: 200,
    maxInner: 100,
    tol: 1e-10,
    predictive: 'approximate',
  },
  hyperSpace: (h) =>
    dict({
      restarts: int(0, 10, { default: Math.min(Math.max(h.restarts, 0), 10), label: 'restarts' }),
      solver: boundedSolvers.space,
    }),
  algorithm: (data, h): Algorithm<GpcState, GpcPath> => {
    const p = prepare(data, h)
    const p0 = thetaOf(h.kernel).shape[0] ?? 0
    const box = thetaBox(h.kernel, h.bounds)
    const search = lbfgsbSearch({
      objective: negatedObjective(`gpc-${h.method}`, (theta) => {
        const r =
          h.method === 'laplace'
            ? laplaceMarginal(h.kernel, p.xs, p.t, {
                likelihood: h.likelihood,
                theta,
                maxIterations: h.maxInner,
                tol: h.tol,
              })
            : epMarginal(h.kernel, p.xs, p.t, { theta, maxSweeps: h.maxInner })
        return r
      }),
      lower: box.lower,
      upper: box.upper,
      optimise: h.optimise,
      gradTol: h.gradTol,
      ...(h.solver === undefined ? {} : { solver: h.solver }),
      maxIter: h.maxIter,
      starts: (k) => {
        const first = Array.from(toFlat(thetaOf(h.kernel)))
        const out = [first]
        for (let r = 1; r <= (h.optimise && p0 > 0 ? h.restarts : 0); r++) {
          const u = toFlat(uniform(foldIn(k, r), [p0]))
          out.push(
            first.map((t, i) => {
              const lo = box.lower[i] ?? -Infinity
              const hi = box.upper[i] ?? Infinity
              return Number.isFinite(lo) && Number.isFinite(hi)
                ? lo + (hi - lo) * (u[i] ?? 0)
                : t + (4 * (u[i] ?? 0.5) - 2)
            }),
          )
        }
        return out
      },
    })
    const posteriorAt = (
      theta: readonly number[],
    ): { lap?: ReturnType<typeof laplaceAlgorithm>; ep?: ReturnType<typeof epAlgorithm> } => {
      const kernel = theta.length === 0 ? h.kernel : withTheta(h.kernel, theta)
      const K = kernel.k(p.xs)
      return h.method === 'laplace'
        ? { lap: laplaceAlgorithm(K, p.t, { likelihood: h.likelihood, tol: h.tol }) }
        : { ep: epAlgorithm(K, p.t, { tol: h.tol }) }
    }
    let cached: { readonly key: string; readonly value: ReturnType<typeof posteriorAt> } | undefined
    const posteriorFor = (theta: readonly number[]): ReturnType<typeof posteriorAt> => {
      const key = theta.join(',')
      if (cached?.key !== key) cached = { key, value: posteriorAt(theta) }
      return cached.value
    }
    const begin = (s: SearchState, key: never): GpcState => {
      const { lap, ep } = posteriorFor(currentOf(s).x)
      return { phase: 'posterior', search: s, laplace: lap?.init(key) ?? null, ep: ep?.init(key) ?? null }
    }
    return {
      name: 'gp-classifier',
      config: { method: h.method, likelihood: h.likelihood, optimise: h.optimise, restarts: h.restarts },
      schema: channels({
        metrics: {
          gradNorm: {
            description: 'the infinity norm of the projected gradient of the approximate likelihood in theta',
          },
          run: { description: 'the optimiser run in progress' },
        },
      }),
      init: (key) => {
        const s = search.init(key)
        return s.inner === null ? begin(s, key as never) : { phase: 'hyper', search: s, laplace: null, ep: null }
      },
      step: (s, ctx) => {
        if (s.phase === 'hyper') {
          const next = search.step(s.search, ctx)
          const now = currentOf(next)
          ctx.log('gp/logMarginalLikelihood', -now.value)
          ctx.log('gp/hyper', natural(h.kernel, now.x))
          return next.finished ? begin(next, ctx.key as never) : { ...s, search: next }
        }
        const { lap, ep } = posteriorFor(currentOf(s.search).x)
        if (lap !== undefined && s.laplace !== null) {
          const next = lap.step(s.laplace, ctx)
          ctx.log('gp/logMarginalLikelihood', next.logMarginalLikelihood)
          ctx.log('gp/latent', next.f)
          return { ...s, laplace: next, phase: next.converged || next.iterations >= h.maxInner ? 'done' : 'posterior' }
        }
        if (ep !== undefined && s.ep !== null) {
          const next = ep.step(s.ep, ctx)
          if (Number.isFinite(next.logMarginalLikelihood))
            ctx.log('gp/logMarginalLikelihood', next.logMarginalLikelihood)
          ctx.log('gp/latent', next.mu)
          return { ...s, ep: next, phase: next.converged || next.sweeps >= h.maxInner ? 'done' : 'posterior' }
        }
        return { ...s, phase: 'done' }
      },
      done: (s) => (s.phase === 'done' ? 'converged' : undefined),
      observe: (s) => ({
        metrics: {
          run: s.search.run,
          ...(s.search.inner === null || s.phase !== 'hyper' ? {} : { gradNorm: gradNormOf(s.search.inner) }),
        },
      }),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as GpcState,
    }
  },
  maxSteps: (h, data) => {
    const n = matrixView(data.x, WHO).n
    return (h.restarts + 1) * h.maxIter + (h.method === 'ep' ? h.maxInner * n : h.maxInner) + 2
  },
  params: (s, data, h) => {
    const p = prepare(data, h)
    const model = currentOf(s.search)
    let posterior: GpcParams['posterior']
    let lml = -model.value
    if (s.laplace !== null) {
      posterior = { f: Array.from(s.laplace.f) }
      if (Number.isFinite(s.laplace.logMarginalLikelihood)) lml = s.laplace.logMarginalLikelihood
    } else if (s.ep !== null) {
      posterior = { tau: Array.from(s.ep.tau), nu: Array.from(s.ep.nu) }
      if (Number.isFinite(s.ep.logMarginalLikelihood)) lml = s.ep.logMarginalLikelihood
    } else {
      const c = converged(h, p, model.x)
      posterior = c.posterior
      lml = c.lml
    }
    return { classes: p.classes, theta: model.x, logMarginalLikelihood: lml, x: p.rows, t: Array.from(p.t), posterior }
  },
  verbs: verbsOf,
  contract: (data, h): Contract<GpcState, GpcPath> => {
    const p = prepare(data, h)
    return defineContract({
      'gp/logMarginalLikelihood': scalar({
        description: 'the approximate log marginal likelihood',
        direction: 'higher',
      }),
      'gp/hyper': tensor({ role: 'param', description: 'the kernel hyperparameters in natural units' }),
      'gp/latent': tensor({
        role: 'state',
        description: 'the latent function at the training points (the mode, or the EP mean)',
      }),
      'model/posterior': derived<GpcState>({
        description: 'the latent mean and variance over the first feature, with the training labels (args: resolution)',
        cost: 'dear',
        compute: (s, args) => latentGrid(h, p, s, (args as { resolution?: number } | undefined)?.resolution ?? 100),
      }),
    })
  },
  logs: { 'gp/logMarginalLikelihood': Log(), 'gp/hyper': Log() },
})

function latentGrid(h: GpClassifierHyper, p: Prepared, s: GpcState, resolution: number): PosteriorGrid {
  const model = currentOf(s.search)
  const solved =
    s.laplace !== null
      ? { f: Array.from(s.laplace.f) }
      : s.ep !== null
        ? { tau: Array.from(s.ep.tau), nu: Array.from(s.ep.nu) }
        : converged(h, p, model.x).posterior
  const verbs = verbsOf(
    {
      classes: p.classes,
      theta: model.x,
      logMarginalLikelihood: -model.value,
      x: p.rows,
      t: Array.from(p.t),
      posterior: solved,
    },
    h,
  )
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
  const { mean, variance } = verbs.latent(inputs)
  return posteriorGrid({
    inputs,
    mean: Array.from(toFlat(mean)),
    variance: Array.from(toFlat(variance)),
    observed: { inputs: p.rows, targets: Array.from(p.t) },
    labels: { output: 'latent function' },
  })
}
