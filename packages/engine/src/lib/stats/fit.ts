/**
 * Maximum-likelihood estimation: Newton's method on any log-likelihood with the score and Hessian from automatic
 * differentiation (`newtonMle`, also as a traced algorithm), the observed information whose inverse gives standard
 * errors, and `fitDistribution`, which fits a named distribution family to data in closed form where one exists and
 * numerically otherwise, as scipy.stats' fit. The minimiser is a slot (`mleSolvers`): the damped Newton by default, any
 * generic solver of `smoothSolvers` as another mode.
 */

import { grad, hessian } from '../autodiff/index.ts'
import { MdmlError, meanOf, varianceOf } from '../base/index.ts'
import {
  bernoulli,
  beta,
  cauchy,
  type Distribution,
  exponential,
  gamma,
  geometric,
  gumbel,
  laplace,
  logistic,
  logNormal,
  negativeBinomial,
  normal,
  pareto,
  poisson,
  studentT,
  uniform,
  weibull,
} from '../dists/index.ts'
import { dpotrf, dpotrs } from '../numerics/linalg/index.ts'
import { familySolvers, type SmoothProblem, solverConfigOf, type VectorParams } from '../optim/index.ts'
import { key as makeKey } from '../random/index.ts'
import { dict, real, type SpaceValue } from '../space/index.ts'
import {
  asTensor,
  exp,
  item,
  neg,
  sigmoid,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type ModeSpec,
  type ModeTable,
  type RunOptions,
  runAlgorithm,
} from '../trace/index.ts'
import { invSpd } from './_linear.ts'
import { mat, sample } from './_util.ts'
import { median } from './quantiles.ts'

/** The state of {@link newtonMleAlgorithm}. */
export interface NewtonMleState {
  /** The unconstrained parameters. */
  readonly eta: Float64Array
  readonly logLikelihood: number
  /** The largest absolute score component. */
  readonly gradientNorm: number
  readonly iterations: number
  readonly converged: boolean
  readonly stalled: boolean
}

/** Options of {@link newtonMle}. */
export interface NewtonMleOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Maximum Newton steps (default 200). */
  readonly maxIterations?: number
  /** Stop when every score component is below tol (1 + |loglik|) (default 1e-10). */
  readonly tol?: number
  /**
   * How the negative log-likelihood is minimised: a value of `mleSolvers.space` (default the Levenberg-damped Newton
   * of {@link newtonMleAlgorithm}, which reproduces today's fits; any generic solver of `smoothSolvers` is another).
   * With a generic solver `tol` is not used (the solver's `gradTol` is).
   */
  readonly solver?: SpaceValue
}

/**
 * Maximum likelihood by Newton's method as a traced algorithm: the score and Hessian of the log-likelihood come from
 * reverse- and forward-mode differentiation; a Hessian that is not negative definite is regularised (Levenberg) and
 * every step is halved until the log-likelihood increases.
 *
 * @param logLikelihood - The log-likelihood of an unconstrained parameter vector (a differentiable tensor function).
 * @param eta0 - The starting point.
 * @param options - tol.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { newtonMleAlgorithm } from 'mdml-engine/stats'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { mul, neg, square, sub, sum, type Value } from 'mdml-engine/tensor'
 * const ll = (eta: Value): Value => neg(sum(square(sub(eta, 3))))
 * Array.from(runAlgorithm(newtonMleAlgorithm(ll, [0]), { maxSteps: 20 }).state.eta) // => [3]
 * ```
 */
export function newtonMleAlgorithm(
  logLikelihood: (eta: Value) => Value,
  eta0: readonly number[],
  options: { readonly tol?: number } = {},
): Algorithm<NewtonMleState> {
  const p = eta0.length
  const tol = options.tol ?? 1e-10
  const value = (eta: Float64Array): number => {
    const v = item(logLikelihood(asTensor(eta)) as Tensor)
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  const score = grad(logLikelihood)
  const hess = hessian(logLikelihood)
  const initial = Float64Array.from(eta0)
  return {
    name: 'newtonMle',
    config: { parameters: p, tol },
    schema: channels({
      metrics: {
        logLikelihood: { description: 'log-likelihood', direction: 'higher' },
        gradientNorm: { description: 'largest absolute score component', direction: 'lower' },
      },
    }),
    init: () => {
      const ll = value(initial)
      if (!Number.isFinite(ll)) throw new MdmlError('value', 'newtonMle: the log-likelihood is not finite at the start')
      return {
        eta: initial,
        logLikelihood: ll,
        gradientNorm: Number.POSITIVE_INFINITY,
        iterations: 0,
        converged: false,
        stalled: false,
      }
    },
    step: (s) => {
      const g = Float64Array.from(toFlat(score(asTensor(s.eta)) as Tensor))
      let gn = 0
      for (const v of g) gn = Math.max(gn, Math.abs(v))
      if (gn <= tol * (1 + Math.abs(s.logLikelihood)))
        return { ...s, gradientNorm: gn, converged: true, iterations: s.iterations + 1 }
      const H = toFlat(hess(asTensor(s.eta)))
      // Solve (-H + lambda I) d = g, raising lambda until the matrix is positive definite.
      let lambda = 0
      let factor: Float64Array | undefined
      for (let attempt = 0; attempt < 60; attempt++) {
        const A = Float64Array.from(
          { length: p * p },
          (_, k) => -(H[k] ?? 0) + (Math.floor(k / p) === k % p ? lambda : 0),
        )
        const r = dpotrf(A, p)
        if (r.info === 0) {
          factor = r.factor
          break
        }
        lambda = lambda === 0 ? 1e-8 * (1 + Math.abs(H[0] ?? 1)) : lambda * 10
      }
      const d = factor === undefined ? g : dpotrs(factor, g, p)
      let t = 1
      let next = s.eta
      let ll = s.logLikelihood
      for (let k = 0; k < 60; k++) {
        const cand = Float64Array.from(s.eta, (v, i) => v + t * (d[i] ?? 0))
        const v = value(cand)
        if (v >= s.logLikelihood - 1e-14 * Math.abs(s.logLikelihood)) {
          next = cand
          ll = v
          break
        }
        t /= 2
      }
      let moved = 0
      for (let i = 0; i < p; i++) moved = Math.max(moved, Math.abs((next[i] ?? 0) - (s.eta[i] ?? 0)))
      let scale = 0
      for (const v of s.eta) scale = Math.max(scale, Math.abs(v))
      const stalled = moved <= 1e-15 * (1 + scale)
      return { eta: next, logLikelihood: ll, gradientNorm: gn, iterations: s.iterations + 1, converged: false, stalled }
    },
    done: (s) => (s.converged ? 'converged' : s.stalled ? 'converged' : undefined),
    observe: (s) => ({ metrics: { logLikelihood: s.logLikelihood, gradientNorm: s.gradientNorm } }),
  }
}

/** A maximum-likelihood problem: a log-likelihood of unconstrained parameters and where to start. */
export interface MleProblem {
  readonly logLikelihood: (eta: Value) => Value
  readonly eta0: readonly number[]
}

/**
 * The negative log-likelihood of an {@link MleProblem} as a smooth problem for the solver slot.
 *
 * @param problem - The log-likelihood and the start.
 * @returns The smooth problem: minimise the negative log-likelihood from the start.
 * @example
 * ```ts
 * import { mleSmoothProblem } from 'mdml-engine/stats'
 * import { neg, square, sum, type Value } from 'mdml-engine/tensor'
 * mleSmoothProblem({ logLikelihood: (t: Value): Value => neg(sum(square(t))), eta0: [1, 2] }).x0.length // => 2
 * ```
 */
export function mleSmoothProblem(problem: MleProblem): SmoothProblem {
  return {
    objective: (eta) => neg(problem.logLikelihood(eta as Value)),
    x0: Float64Array.from(problem.eta0),
  }
}

const levenbergNewton: ModeSpec<MleProblem, NewtonMleState, VectorParams> = {
  description: 'Newton with a Levenberg-regularised Hessian and a halving search, the default of maximum likelihood',
  space: dict({ tol: real(1e-14, 1e-4, { log: true, default: 1e-10, label: 'score tolerance' }) }),
  make: (p, k) => newtonMleAlgorithm(p.logLikelihood, p.eta0, { tol: k['tol'] as number }),
  params: (s) => ({ x: Array.from(s.eta) }),
}

/**
 * The solvers of the maximum-likelihood slot: `levenbergNewton` (default; the library's own damped Newton, which
 * reproduces the fits of {@link newtonMle}) beside every generic solver of `smoothSolvers` applied to the negative
 * log-likelihood. All of them reach the same maximiser; they differ in the path.
 *
 * @index table Solvers of the maximum-likelihood slot.
 * @example
 * ```ts
 * import { mleSolvers } from 'mdml-engine/stats'
 * mleSolvers.names[0] // => 'levenbergNewton'
 * mleSolvers.modeOf({ kind: 'bfgs' }).name // => 'bfgs'
 * ```
 */
export const mleSolvers: ModeTable<MleProblem, VectorParams> = familySolvers<MleProblem>(
  { levenbergNewton },
  mleSmoothProblem,
  {
    default: 'levenbergNewton',
    description: 'how the negative log-likelihood is minimised',
  },
)

/** The result of {@link newtonMle}. */
export interface NewtonMleResult {
  readonly eta: Float64Array
  readonly logLikelihood: number
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Maximise a log-likelihood by Newton's method ({@link newtonMleAlgorithm}, traced).
 *
 * @param logLikelihood - The log-likelihood of an unconstrained parameter vector.
 * @param eta0 - The starting point.
 * @param options - maxIterations, tol, tracer, policy, and `solver` (a value of `mleSolvers.space`: the default
 *   Levenberg Newton or any generic solver of `smoothSolvers`).
 * @returns The maximiser and its log-likelihood.
 * @example
 * ```ts
 * import { newtonMle } from 'mdml-engine/stats'
 * import { neg, square, sub, sum, type Value } from 'mdml-engine/tensor'
 * newtonMle((eta: Value): Value => neg(sum(square(sub(eta, 2)))), [0]).logLikelihood // => 0
 * ```
 */
export function newtonMle(
  logLikelihood: (eta: Value) => Value,
  eta0: readonly number[],
  options: NewtonMleOptions = {},
): NewtonMleResult {
  const { maxIterations, tol, solver, ...run } = options
  const config = solver === undefined ? undefined : solverConfigOf(mleSolvers, solver)
  if (config !== undefined && mleSolvers.modeOf(config).name !== 'levenbergNewton') {
    const generic = runAlgorithm(mleSolvers.make({ logLikelihood, eta0 }, config, run.key ?? makeKey(0)), {
      ...run,
      maxSteps: maxIterations ?? 200,
    })
    const eta = Float64Array.from(mleSolvers.params(config, generic.state).x)
    return {
      eta,
      logLikelihood: item(logLikelihood(asTensor(eta)) as Tensor),
      iterations: generic.steps,
      converged: generic.status === 'converged',
    }
  }
  const r = runAlgorithm(newtonMleAlgorithm(logLikelihood, eta0, tol === undefined ? {} : { tol }), {
    ...run,
    maxSteps: maxIterations ?? 200,
  })
  return {
    eta: r.state.eta,
    logLikelihood: r.state.logLikelihood,
    iterations: r.state.iterations,
    converged: r.state.converged || r.state.stalled,
  }
}

/**
 * The observed information -d^2 l / d theta^2 of a log-likelihood at a point (by automatic differentiation), whose
 * inverse is the asymptotic covariance of the maximum-likelihood estimate.
 *
 * @param logLikelihood - The log-likelihood of a parameter vector.
 * @param theta - The point (usually the MLE).
 * @returns The p x p observed information.
 * @example
 * ```ts
 * import { observedInformation } from 'mdml-engine/stats'
 * import { mul, neg, square, sum, type Value } from 'mdml-engine/tensor'
 * observedInformation((t: Value): Value => neg(sum(mul(square(t), 2))), [1]) // => [[4]]
 * ```
 */
export function observedInformation(logLikelihood: (theta: Value) => Value, theta: readonly number[]): Tensor {
  const p = theta.length
  const H = toFlat(hessian(logLikelihood)(asTensor(Float64Array.from(theta))))
  return mat(
    H.map((v) => -v),
    p,
    p,
  )
}

/** A distribution family that {@link fitDistribution} can fit. */
export type FitFamily =
  | 'normal'
  | 'logNormal'
  | 'exponential'
  | 'gamma'
  | 'weibull'
  | 'beta'
  | 'laplace'
  | 'logistic'
  | 'gumbel'
  | 'cauchy'
  | 'studentT'
  | 'uniform'
  | 'pareto'
  | 'poisson'
  | 'bernoulli'
  | 'geometric'
  | 'negativeBinomial'

/** The result of {@link fitDistribution}. */
export interface DistributionFit {
  readonly family: FitFamily
  /** Parameter values, named as the `dists` factory names them. */
  readonly params: Readonly<Record<string, number>>
  /** Standard errors from the observed information (NaN where the likelihood is not smooth, e.g. bounds). */
  readonly standardErrors: Readonly<Record<string, number>>
  readonly covariance: Tensor
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  readonly n: number
  readonly method: 'closed form' | 'newton'
  readonly converged: boolean
  readonly distribution: Distribution
}

interface FamilySpec {
  readonly names: readonly string[]
  /** Which parameters are positive (fitted on the log scale). */
  readonly positive: readonly boolean[]
  readonly make: (params: readonly Value[]) => Distribution
  /** A closed-form MLE, or starting values for Newton's method. */
  readonly closed?: (x: Float64Array) => number[]
  readonly start?: (x: Float64Array) => number[]
  /** The support constraint on the data. */
  readonly check?: (x: Float64Array) => boolean
  /** Whether the MLE sits on a boundary (no smooth standard errors). */
  readonly irregular?: boolean
}

const positiveData = (x: Float64Array): boolean => x.every((v) => v > 0)
const unitData = (x: Float64Array): boolean => x.every((v) => v > 0 && v < 1)
const countData = (x: Float64Array): boolean => x.every((v) => Number.isInteger(v) && v >= 0)
const sd0 = (x: Float64Array): number => Math.sqrt(varianceOf(x))

const FAMILIES: Readonly<Record<FitFamily, FamilySpec>> = {
  normal: {
    names: ['loc', 'scale'],
    positive: [false, true],
    make: ([m, s]) => normal(m as Value, s as Value),
    closed: (x) => [meanOf(x), sd0(x)],
  },
  logNormal: {
    names: ['mu', 'sigma'],
    positive: [false, true],
    make: ([m, s]) => logNormal(m as Value, s as Value),
    closed: (x) => {
      const l = Float64Array.from(x, Math.log)
      return [meanOf(l), sd0(l)]
    },
    check: positiveData,
  },
  exponential: {
    names: ['rate'],
    positive: [true],
    make: ([r]) => exponential(r as Value),
    closed: (x) => [1 / meanOf(x)],
    check: (x) => x.every((v) => v >= 0),
  },
  gamma: {
    names: ['concentration', 'rate'],
    positive: [true, true],
    make: ([a, r]) => gamma(a as Value, r as Value),
    start: (x) => {
      const m = meanOf(x)
      const v = varianceOf(x)
      return [(m * m) / v, m / v]
    },
    check: positiveData,
  },
  weibull: {
    names: ['concentration', 'scale'],
    positive: [true, true],
    make: ([k, s]) => weibull(k as Value, s as Value),
    start: (x) => {
      const l = Float64Array.from(x, Math.log)
      const k = 1.2 / Math.sqrt(varianceOf(l))
      return [k, Math.exp(meanOf(l) + 0.5772 / k)]
    },
    check: positiveData,
  },
  beta: {
    names: ['a', 'b'],
    positive: [true, true],
    make: ([a, b]) => beta(a as Value, b as Value),
    start: (x) => {
      const m = meanOf(x)
      const v = varianceOf(x)
      const c = (m * (1 - m)) / v - 1
      return [Math.max(0.1, m * c), Math.max(0.1, (1 - m) * c)]
    },
    check: unitData,
  },
  laplace: {
    names: ['loc', 'scale'],
    positive: [false, true],
    make: ([m, b]) => laplace(m as Value, b as Value),
    closed: (x) => {
      const m = median(x)
      return [m, meanOf(Float64Array.from(x, (v) => Math.abs(v - m)))]
    },
    irregular: true,
  },
  logistic: {
    names: ['loc', 'scale'],
    positive: [false, true],
    make: ([m, s]) => logistic(m as Value, s as Value),
    start: (x) => [meanOf(x), (Math.sqrt(3) * sd0(x)) / Math.PI],
  },
  gumbel: {
    names: ['loc', 'scale'],
    positive: [false, true],
    make: ([m, s]) => gumbel(m as Value, s as Value),
    start: (x) => {
      const s = (Math.sqrt(6) * sd0(x)) / Math.PI
      return [meanOf(x) - 0.5772 * s, s]
    },
  },
  cauchy: {
    names: ['loc', 'scale'],
    positive: [false, true],
    make: ([m, s]) => cauchy(m as Value, s as Value),
    start: (x) => {
      const s = sortedQuartiles(x)
      return [median(x), Math.max(1e-6, (s[1] - s[0]) / 2)]
    },
  },
  studentT: {
    names: ['df', 'loc', 'scale'],
    positive: [true, false, true],
    make: ([d, m, s]) => studentT(d as Value, m as Value, s as Value),
    start: (x) => {
      const q = sortedQuartiles(x)
      return [5, median(x), Math.max(1e-6, (q[1] - q[0]) / 1.5)]
    },
  },
  uniform: {
    names: ['low', 'high'],
    positive: [false, false],
    make: ([a, b]) => uniform(a as Value, b as Value),
    closed: (x) => [
      x.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY),
      x.reduce((m, v) => Math.max(m, v), Number.NEGATIVE_INFINITY),
    ],
    irregular: true,
  },
  pareto: {
    names: ['b', 'scale'],
    positive: [true, true],
    make: ([b, s]) => pareto(b as Value, s as Value),
    closed: (x) => {
      const xm = x.reduce((m, v) => Math.min(m, v), Number.POSITIVE_INFINITY)
      let s = 0
      for (const v of x) s += Math.log(v / xm)
      return [x.length / s, xm]
    },
    check: positiveData,
    irregular: true,
  },
  poisson: {
    names: ['rate'],
    positive: [true],
    make: ([r]) => poisson(r as Value),
    closed: (x) => [meanOf(x)],
    check: countData,
  },
  bernoulli: {
    names: ['probs'],
    positive: [false],
    make: ([p]) => bernoulli({ probs: p as Value }),
    closed: (x) => [meanOf(x)],
    check: (x) => x.every((v) => v === 0 || v === 1),
  },
  geometric: {
    names: ['probs'],
    positive: [false],
    make: ([p]) => geometric(p as Value),
    closed: (x) => [1 / meanOf(x)],
    check: (x) => x.every((v) => Number.isInteger(v) && v >= 1),
  },
  negativeBinomial: {
    names: ['n', 'probs'],
    positive: [true, true],
    // probs is fitted through its logit, by representing it as 1 / (1 + exp(-eta)) inside the likelihood below.
    make: ([n, p]) => negativeBinomial(n as Value, p as Value),
    start: (x) => {
      const m = meanOf(x)
      const v = Math.max(varianceOf(x), m * 1.01)
      const p = m / v
      return [(m * p) / (1 - p), p]
    },
    check: countData,
  },
}

function sortedQuartiles(x: Float64Array): [number, number] {
  const s = Float64Array.from(x).sort()
  const at = (q: number): number => {
    const h = (s.length - 1) * q
    const lo = Math.floor(h)
    return (s[lo] ?? 0) + (h - lo) * ((s[Math.min(s.length - 1, lo + 1)] ?? 0) - (s[lo] ?? 0))
  }
  return [at(0.25), at(0.75)]
}

/**
 * Fit a distribution family by maximum likelihood (scipy.stats' fit with the location fixed at zero where the family
 * has one): closed forms where they exist (normal, log-normal, exponential, Laplace, uniform, Pareto, Poisson,
 * Bernoulli, geometric), otherwise Newton's method on the log-likelihood with positive parameters on the log scale
 * (and the negative binomial's probability on the logit scale); standard errors from the inverse observed
 * information.
 *
 * @param x - The sample.
 * @param family - The family.
 * @param options - Newton options (maxIterations, tol, tracer).
 * @returns The parameters, standard errors, covariance, log-likelihood, AIC, BIC and the fitted distribution.
 * @example
 * ```ts
 * import { fitDistribution } from 'mdml-engine/stats'
 * fitDistribution([1, 2, 3, 4], 'normal').params // => { loc: 2.5, scale: 1.118034 }
 * fitDistribution([1, 2, 3, 4], 'normal').standardErrors.loc // => 0.559017
 * ```
 */
export function fitDistribution(x: TensorLike, family: FitFamily, options: NewtonMleOptions = {}): DistributionFit {
  const spec = FAMILIES[family]
  if (spec === undefined) throw new MdmlError('value', `fitDistribution: unknown family ${String(family)}`)
  const a = sample('fitDistribution', x, 2)
  if (spec.check !== undefined && !spec.check(a))
    throw new MdmlError('domain', `fitDistribution: the data are outside the support of ${family}`)
  const data = asTensor(a)
  const p = spec.names.length
  const logit = family === 'negativeBinomial'
  const natural = (eta: Value): Value[] =>
    spec.names.map((_, i) => {
      const e = take(eta, [i])
      if (logit && i === 1) return sigmoid(e)
      return spec.positive[i] === true ? exp(e) : e
    })
  const ll = (params: readonly Value[]): Value => sum(spec.make(params).logProb(data))
  let theta: number[]
  let method: DistributionFit['method']
  let converged = true
  if (spec.closed !== undefined) {
    theta = spec.closed(a)
    method = 'closed form'
  } else {
    const start = (spec.start ?? (() => [1, 1]))(a)
    const eta0 = start.map((v, i) => {
      if (logit && i === 1) return Math.log(v / (1 - v))
      return spec.positive[i] === true ? Math.log(v) : v
    })
    const r = newtonMle((eta) => ll(natural(eta)), eta0, options)
    theta = natural(asTensor(r.eta)).map((v) => item(v as Tensor))
    method = 'newton'
    converged = r.converged
  }
  const thetaTensor = (t: Value): Value[] => spec.names.map((_, i) => take(t, [i]))
  const logLikelihood = item(ll(theta.map((v) => asTensor([v]))) as Tensor)
  let cov: Float64Array
  if (spec.irregular === true) cov = new Float64Array(p * p).fill(Number.NaN)
  else {
    const info = toFlat(observedInformation((t) => ll(thetaTensor(t)), theta))
    try {
      cov = invSpd(info, p)
    } catch {
      cov = new Float64Array(p * p).fill(Number.NaN)
    }
  }
  const n = a.length
  const params: Record<string, number> = {}
  const se: Record<string, number> = {}
  spec.names.forEach((name, i) => {
    params[name] = theta[i] ?? Number.NaN
    se[name] = Math.sqrt(cov[i * p + i] ?? Number.NaN)
  })
  return {
    family,
    params,
    standardErrors: se,
    covariance: mat(cov, p, p),
    logLikelihood,
    aic: 2 * p - 2 * logLikelihood,
    bic: Math.log(n) * p - 2 * logLikelihood,
    n,
    method,
    converged,
    distribution: spec.make(theta.map((v) => asTensor(v))),
  }
}
