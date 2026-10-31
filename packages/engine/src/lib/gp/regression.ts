/**
 * Gaussian-process regression over any kernel (scikit-learn's `GaussianProcessRegressor`) as an estimator on the
 * iterative base. The fit is the optimisation of the kernel's hyperparameters on the exact log marginal likelihood,
 * one L-BFGS-B iteration per step with random restarts in turn, each step logging the likelihood and the hyperparameters;
 * the fitted value predicts the posterior mean, standard deviation and covariance and draws posterior samples.
 */

import { MdmlError } from '../base/index.ts'
import { type Distribution, normal as normalDistribution } from '../dists/index.ts'
import {
  type HyperparameterSurface,
  hyperparameterSurface,
  type PosteriorGrid,
  posteriorGrid,
} from '../exposition/index.ts'
import { hyperparameters, type Kernel, rbfKernel, scaleKernel, theta as thetaOf, withTheta } from '../kernels/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView, targetVector } from '../learn/index.ts'
import { choleskyJitter, choSolve, solveTriangular } from '../numerics/linalg/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { samplePosterior } from '../stochastic/index.ts'
import {
  add,
  expandDims,
  fromStorage,
  matmul,
  maximum,
  mul,
  primalOf,
  reshape,
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
import { channels, derived, Log, scalar, stateFromJson, stateToJson, tensor } from '../trace/index.ts'
import { currentOf, lbfgsbSearch, type SearchState } from './_search.ts'
import { logMarginalLikelihood, negativeLogMarginalLikelihood } from './likelihood.ts'

/** The hyperparameters of {@link gpRegression}. */
export interface GpHyper {
  /** The covariance function; its hyperparameters are the starting point of the fit. */
  readonly kernel: Kernel
  /** Noise variance added to the diagonal of the kernel matrix (scikit-learn's alpha; default 1e-10). */
  readonly alpha: number
  /** Fit the targets standardised (mean and standard deviation removed), predicting back in their units. */
  readonly normalizeY: boolean
  /** Whether to optimise the kernel's hyperparameters; otherwise the kernel is used as given. */
  readonly optimise: boolean
  /** Extra optimiser runs from starting points drawn uniformly in log space within the bounds (n_restarts_optimizer). */
  readonly restarts: number
  /** The bounds of every positive hyperparameter, in natural units (default scikit-learn's 1e-5 to 1e5). */
  readonly bounds: readonly [number, number]
  /** L-BFGS-B stops when the projected gradient falls below this (scipy's gtol). */
  readonly gradTol: number
  /** The most L-BFGS-B iterations of one run. */
  readonly maxIter: number
}

/** The state of {@link gpRegression}: a multi-start L-BFGS-B search over the hyperparameters. */
export type GpState = SearchState

/** The fitted parameters of {@link gpRegression}: plain JSON. */
export type GpParams = {
  /** The fitted hyperparameters in log space (the order of `theta(kernel)`). */
  readonly theta: number[]
  readonly logMarginalLikelihood: number
  /** The training inputs and targets (in their own units). */
  readonly x: number[][]
  readonly y: number[]
  readonly yMean: number
  readonly yStd: number
}

/** What a fitted Gaussian-process regressor offers. */
export interface GpVerbs {
  /** The posterior mean `[m]`. */
  readonly predict: (x: TensorLike) => Tensor
  /** The posterior mean and standard deviation `[m]` of the function (plus the noise, if the kernel carries a white-noise part). */
  readonly predictWithStd: (x: TensorLike) => { readonly mean: Tensor; readonly std: Tensor }
  /** The posterior covariance `[m, m]`. */
  readonly predictCovariance: (x: TensorLike) => Tensor
  /** The predictive distribution of each row (normal). */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** `n` functions drawn from the posterior at the rows of x, `[n, m]`. */
  readonly sample: (x: TensorLike, n: number, key: Key) => Tensor
  /** The coefficient of determination of the mean on a labelled dataset. */
  readonly score: (data: LearnData) => number
  /** The fitted kernel. */
  readonly kernel: Kernel
  readonly logMarginalLikelihood: number
  readonly nFeaturesIn: number
}

const WHO = 'gpRegression'

interface Prepared {
  readonly n: number
  readonly d: number
  readonly rows: number[][]
  readonly y: Float64Array
  readonly yn: Float64Array
  readonly yMean: number
  readonly yStd: number
  readonly xs: Tensor
}

function prepare(data: LearnData, h: GpHyper): Prepared {
  const m = matrixView(data.x, WHO)
  const y = targetVector(data.y ?? [], WHO, m.n)
  const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  let yMean = 0
  let yStd = 1
  if (h.normalizeY) {
    yMean = y.reduce((a, b) => a + b, 0) / m.n
    const sd = Math.sqrt(y.reduce((a, b) => a + (b - yMean) ** 2, 0) / m.n)
    yStd = sd > 0 ? sd : 1
  }
  const yn = Float64Array.from(y, (v) => (v - yMean) / yStd)
  return { n: m.n, d: m.d, rows, y, yn, yMean, yStd, xs: fromStorage(m.data, [m.n, m.d], 'float64') }
}

const DEFAULT_KERNEL: Kernel = scaleKernel(rbfKernel(), 1)

/** The box of the free hyperparameters in theta (log) coordinates: positive ones within `bounds`, real ones free. */
function thetaBox(kernel: Kernel, bounds: readonly [number, number]): { lower: number[]; upper: number[] } {
  const lower: number[] = []
  const upper: number[] = []
  for (const hp of hyperparameters(kernel)) {
    for (let i = 0; i < hp.size; i++) {
      lower.push(hp.constraint === 'positive' ? Math.log(bounds[0]) : -Infinity)
      upper.push(hp.constraint === 'positive' ? Math.log(bounds[1]) : Infinity)
    }
  }
  return { lower, upper }
}

/** The best theta so far and its negative log marginal likelihood. */
function modelOf(s: GpState): { readonly theta: number[]; readonly value: number } {
  const m = currentOf(s)
  return { theta: m.x, value: m.value }
}

const natural = (kernel: Kernel, theta: readonly number[]): number[] =>
  hyperparameters(theta.length === 0 ? kernel : withTheta(kernel, theta)).flatMap((hp) => [...hp.value])

/** The posterior of a GP at the rows `xs`, in the units of the standardised targets. */
function conditional(
  kernel: Kernel,
  p: Prepared,
  alpha: number,
  xs: Value,
  covariance: boolean,
): { readonly mean: Value; readonly variance: Value; readonly covariance?: Value } {
  const n = p.n
  const noise = mul(
    fromStorage(
      Float64Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0)),
      [n, n],
      'float64',
    ),
    alpha,
  )
  const K = add(kernel.k(p.xs), noise)
  const { l } = choleskyJitter(K, 1e-6)
  const a = choSolve(l, fromStorage(p.yn, [n], 'float64'))
  const Ks = kernel.k(xs, p.xs)
  const mean = reshape(matmul(Ks, expandDims(a, -1)), [-1])
  const V = solveTriangular(l, transpose(Ks), { lower: true })
  const explained = sum(square(V), 0)
  const variance = maximum(sub(kernel.diag(xs), explained), 0)
  if (!covariance) return { mean, variance }
  return { mean, variance, covariance: sub(kernel.k(xs), matmul(transpose(V), V)) }
}

function verbsOf(params: GpParams, h: GpHyper): GpVerbs {
  const kernel = params.theta.length === 0 ? h.kernel : withTheta(h.kernel, params.theta)
  const d = params.x[0]?.length ?? 0
  const train: Prepared = {
    n: params.y.length,
    d,
    rows: params.x,
    y: Float64Array.from(params.y),
    yn: Float64Array.from(params.y, (v) => (v - params.yMean) / params.yStd),
    yMean: params.yMean,
    yStd: params.yStd,
    xs: fromStorage(Float64Array.from(params.x.flat()), [params.y.length, d], 'float64'),
  }
  const inputs = (x: TensorLike): Value => {
    const m = matrixView(x, WHO, { nFeatures: d })
    return fromStorage(m.data, [m.n, m.d], 'float64')
  }
  const meanOf = (x: TensorLike): Tensor => {
    const post = conditional(kernel, train, h.alpha, inputs(x), false)
    return primalOf(add(mul(post.mean, params.yStd), params.yMean))
  }
  const withStd = (x: TensorLike): { mean: Tensor; std: Tensor } => {
    const post = conditional(kernel, train, h.alpha, inputs(x), false)
    return {
      mean: primalOf(add(mul(post.mean, params.yStd), params.yMean)),
      std: primalOf(mul(sqrt(post.variance), params.yStd)),
    }
  }
  return {
    predict: meanOf,
    predictWithStd: withStd,
    predictCovariance: (x) => {
      const post = conditional(kernel, train, h.alpha, inputs(x), true)
      return primalOf(mul(post.covariance as Value, params.yStd ** 2))
    },
    predictDistribution: (x) => {
      const { mean, std } = withStd(x)
      return normalDistribution(mean, std)
    },
    sample: (x, n, key) => {
      const post = conditional(kernel, train, h.alpha, inputs(x), true)
      const draws = samplePosterior(
        key,
        {
          mean: post.mean,
          covariance: post.covariance as Value,
          std: sqrt(post.variance),
          logMarginalLikelihood: post.mean,
        },
        n,
      )
      return primalOf(add(mul(draws, params.yStd), params.yMean))
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
    logMarginalLikelihood: params.logMarginalLikelihood,
    nFeaturesIn: d,
  }
}

/**
 * Gaussian-process regression over any kernel of `mdml-engine/kernels`. The fit maximises the exact log marginal
 * likelihood over the kernel's free hyperparameters with L-BFGS-B, the gradient coming from `gramGradient`, restarting from
 * random points inside the bounds `restarts` times and keeping the best; each step is one L-BFGS-B iteration, logging
 * `gp/logMarginalLikelihood` and `gp/hyper` (the hyperparameters in natural units), so the fit can be watched climbing.
 * The probes `model/posterior` (mean, variance and samples over the first feature), `model/samples` and `solver/surface`
 * (the likelihood over two hyperparameters, with the optimiser's path) are derived from any step.
 *
 * @concept gaussian-process-regression
 * @reference C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning (2006), chapter 2 and 5.
 * @example
 * ```ts
 * import { gpRegression } from 'mdml-engine/gp'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [1], [2], [3], [4]]
 * const fitted = gpRegression.with({ alpha: 1e-4 }).fit({ x, y: [0, 0.8, 0.9, 0.1, -0.8] })
 * toFlat(fitted.predict([[0], [4]])).map((v) => Math.round(v * 100) / 100) // => [0, -0.8]
 * ```
 */
export const gpRegression: IterativeEstimator<GpHyper, GpState, GpParams, GpVerbs> = iterative<
  GpHyper,
  GpState,
  GpParams,
  GpVerbs
>({
  name: 'gp/regression',
  kind: 'regressor',
  defaults: {
    kernel: DEFAULT_KERNEL,
    alpha: 1e-10,
    normalizeY: false,
    optimise: true,
    restarts: 0,
    bounds: [1e-5, 1e5],
    gradTol: 1e-5,
    maxIter: 500,
  },
  hyperSpace: (h) =>
    dict({
      alpha: real(1e-10, 10, { log: true, default: Math.min(Math.max(h.alpha, 1e-10), 10), label: 'noise variance' }),
      restarts: int(0, 10, { default: Math.min(Math.max(h.restarts, 0), 10), label: 'restarts' }),
    }),
  algorithm: (data, h) => {
    const p = prepare(data, h)
    const p0 = thetaOf(h.kernel).shape[0] ?? 0
    const box = thetaBox(h.kernel, h.bounds)
    const search = lbfgsbSearch({
      objective: negativeLogMarginalLikelihood(h.kernel, p.xs, p.yn, { alpha: h.alpha }),
      lower: box.lower,
      upper: box.upper,
      optimise: h.optimise,
      gradTol: h.gradTol,
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
    return {
      name: 'gp-regression',
      config: { alpha: h.alpha, normalizeY: h.normalizeY, restarts: h.restarts, optimise: h.optimise },
      schema: channels({
        metrics: {
          gradNorm: { description: 'the infinity norm of the projected gradient of the likelihood in theta' },
          run: { description: 'the optimiser run in progress (0 is the kernel’s own start)' },
        },
      }),
      init: (k) => search.init(k),
      step: (s, ctx) => {
        const next = search.step(s, ctx)
        if (s.inner !== null) {
          const now = search.current(next)
          ctx.log('gp/logMarginalLikelihood', -now.value)
          ctx.log('gp/hyper', natural(h.kernel, now.x))
        }
        return next
      },
      done: (s) => (s.finished ? 'converged' : undefined),
      observe: (s) => ({
        metrics: {
          run: s.run,
          ...(s.inner === null ? {} : { gradNorm: s.inner.projectedGradNorm }),
        },
      }),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as GpState,
    }
  },
  maxSteps: (h) => (h.restarts + 1) * h.maxIter + 1,
  params: (s, data, h) => {
    const p = prepare(data, h)
    const model = modelOf(s)
    return {
      theta: model.theta,
      logMarginalLikelihood: -model.value,
      x: p.rows,
      y: Array.from(p.y),
      yMean: p.yMean,
      yStd: p.yStd,
    }
  },
  verbs: verbsOf,
  contract: (data, h) => {
    const p = prepare(data, h)
    const gridOf = (
      args: { readonly resolution?: number; readonly samples?: number; readonly seed?: number } | undefined,
      s: GpState,
    ): PosteriorGrid => {
      const model = modelOf(s)
      const verbs = verbsOf(
        {
          theta: model.theta,
          logMarginalLikelihood: -model.value,
          x: p.rows,
          y: Array.from(p.y),
          yMean: p.yMean,
          yStd: p.yStd,
        },
        h,
      )
      const resolution = Math.max(2, Math.min(args?.resolution ?? 100, 400))
      const column = p.rows.map((r) => r[0] ?? 0)
      const lo = Math.min(...column)
      const hi = Math.max(...column)
      const pad = 0.15 * (hi - lo || 1)
      const centre = Array.from({ length: p.d }, (_, j) => p.rows.reduce((a, r) => a + (r[j] ?? 0), 0) / p.n)
      const inputs = Array.from({ length: resolution }, (_, i) => {
        const row = [...centre]
        row[0] = lo - pad + ((hi - lo + 2 * pad) * i) / (resolution - 1)
        return row
      })
      const { mean, std } = verbs.predictWithStd(inputs)
      const count = args?.samples ?? 0
      const draws = count > 0 ? toFlat(verbs.sample(inputs, count, foldIn(keyOf(args?.seed ?? 0), 0))) : undefined
      return posteriorGrid({
        inputs,
        mean: Array.from(toFlat(mean)),
        variance: Array.from(toFlat(std), (v) => v * v),
        ...(draws === undefined
          ? {}
          : {
              samples: Array.from({ length: count }, (_, i) =>
                Array.from(draws.slice(i * resolution, (i + 1) * resolution)),
              ),
            }),
        observed: { inputs: p.rows, targets: Array.from(p.y) },
      })
    }
    return {
      'gp/logMarginalLikelihood': scalar({
        description: 'the log marginal likelihood at the current hyperparameters',
        direction: 'higher',
      }),
      'gp/hyper': tensor({
        role: 'param',
        description: 'the kernel hyperparameters in natural units, in the order of `hyperparameters(kernel)`',
      }),
      'model/posterior': derived<GpState>({
        description:
          'the posterior mean and variance over the first feature, with the training data (args: resolution, samples, seed)',
        cost: 'dear',
        compute: (s, args) => gridOf(args as never, s),
      }),
      'model/samples': derived<GpState>({
        description: 'functions drawn from the posterior over the first feature (args: resolution, samples, seed)',
        cost: 'dear',
        compute: (s, args) => gridOf({ samples: 5, ...(args as object) } as never, s),
      }),
      'solver/surface': derived<GpState>({
        description:
          'the log marginal likelihood over a grid of the first two free hyperparameters (log scale), other hyperparameters at the fit (args: resolution)',
        cost: 'dear',
        compute: (s, args) => surfaceOf(h, p, s, (args as { resolution?: number } | undefined)?.resolution ?? 25),
      }),
    }
  },
  logs: { 'gp/logMarginalLikelihood': Log(), 'gp/hyper': Log() },
})

import { key as keyOf } from '../random/index.ts'

function surfaceOf(h: GpHyper, p: Prepared, s: GpState, resolution: number): HyperparameterSurface {
  const model = modelOf(s)
  const free = hyperparameters(h.kernel).filter((hp) => hp.size > 0)
  if (model.theta.length < 2) throw new MdmlError('value', 'solver/surface: needs at least two free hyperparameters')
  const names = free.flatMap((hp) =>
    Array.from({ length: hp.size }, (_, i) => (hp.size === 1 ? hp.name : `${hp.name}[${i}]`)),
  )
  const around = (i: number): number[] => {
    const c = model.theta[i] ?? 0
    return Array.from({ length: resolution }, (_, k) => c - 3 + (6 * k) / (resolution - 1))
  }
  const xs = around(0)
  const ys = around(1)
  const values = ys.map((b) =>
    xs.map((a) => {
      const t = [...model.theta]
      t[0] = a
      t[1] = b
      const v = logMarginalLikelihood(h.kernel, p.xs, p.yn, { alpha: h.alpha, theta: t }).value
      return Number.isFinite(v) ? v : -1e10
    }),
  )
  return hyperparameterSurface({
    parameters: [names[0] ?? 'theta0', names[1] ?? 'theta1'],
    x: xs.map(Math.exp),
    y: ys.map(Math.exp),
    values,
    logScale: [true, true],
    objective: 'log marginal likelihood',
    higherIsBetter: true,
    path: [[Math.exp(model.theta[0] ?? 0), Math.exp(model.theta[1] ?? 0)]],
  })
}
