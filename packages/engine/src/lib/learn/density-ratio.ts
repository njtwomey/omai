/**
 * Density-ratio estimation `w(x) = p_test(x) / p_train(x)` for covariate shift, three ways: a probabilistic domain
 * classifier of any base estimator (`classifierDensityRatio`), Sugiyama's KLIEP (`kliepProblem`, `kliepSolvers`,
 * `kliepAlgorithm`, `kliep`) and the least-squares fits uLSIF and relative uLSIF (`ulsifProblem`, `ulsifSolvers`,
 * `ulsif`) with closed-form leave-pair-out model selection (`ulsifLoo`). KLIEP and uLSIF are objectives over the
 * coefficients of Gaussian kernels at test points, so the minimiser is a slot: uLSIF's closed form is one explicit mode
 * beside every generic smooth solver, KLIEP's own projected gradient is one beside them too.
 */

import { type Json, MdmlError, maxOf } from '../base/index.ts'
import { ddot, dgemv, dnrm2, solve } from '../numerics/linalg/index.ts'
import { loggedSolver, type SmoothProblem, smoothSolvers, type VectorParams } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { importanceEss } from '../sampling/index.ts'
import { type DictSpace, defaults, dict, real, type SpaceValue } from '../space/index.ts'
import { medianHeuristic } from '../stats/tests/kernel.ts'
import {
  add,
  fromStorage,
  logsumexp,
  matmul,
  mean,
  mul,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  channels,
  type ModeSpec,
  type ModeTable,
  modes,
  runAlgorithm,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { innerFitOptions, keyFor, rbfGram, runOptionsFor, squaredDistances } from './_shared.ts'
import { linearDiscriminant } from './discriminant.ts'
import { type Estimator, type FitOptions, requireVerb } from './estimator.ts'
import { fractionOption, fromView, integerOption, type MatrixView, matrixView } from './validation.ts'

/** What every density-ratio estimator returns: the weights at the data and the fitted ratio as a function. */
export interface DensityRatioResult {
  /** Estimated density ratios `w(x_i)` at the training points `[n_train]`. */
  readonly trainWeights: Tensor<'float64'>
  /** Estimated density ratios `w(x'_j)` at the test points `[n_test]`. */
  readonly testWeights: Tensor<'float64'>
  /** The fitted density ratio `w(x)` at arbitrary points `[m, d]`. */
  readonly predictRatio: (x: TensorLike) => Tensor<'float64'>
  /** Kish's effective sample size of `trainWeights`, `(sum w)^2 / sum w^2`. */
  readonly ess: number
}

/** Kish's effective sample size of nonnegative weights, through the sampling domain's log-weight form. */
function essOf(w: ArrayLike<number>): number {
  return importanceEss(Array.from(w, (v) => Math.log(Math.max(v, 0))))
}

function vectorOf(x: Float64Array): Tensor<'float64'> {
  return fromStorage(x, [x.length], 'float64')
}

// ---------------------------------------------------------------------------------------------------------------------
// The classifier
// ---------------------------------------------------------------------------------------------------------------------

/** Options of {@link classifierDensityRatio}. */
export interface ClassifierDensityRatioOptions extends FitOptions {
  /** The domain classifier, any estimator with `predictProba` (default linear discriminant analysis). */
  readonly estimator?: Estimator
  /** Clip the domain probabilities to `[clipProba, 1 - clipProba]` before forming odds (default 1e-3). */
  readonly clipProba?: number
  /** An upper cap on the returned ratios. */
  readonly maxWeight?: number
}

/**
 * The probabilistic-classifier density ratio (Bickel, Brueckner and Scheffer 2009): fit a domain classifier that tells
 * training rows (`s = 0`) from test rows (`s = 1`), then by Bayes' rule
 * `w(x) = (n_train / n_test) P(s = 1 | x) / P(s = 0 | x)`. The classifier is a slot: any estimator of the library with
 * `predictProba` (logistic regression, a forest, naive Bayes, a network), so the ratio is as good as the classifier's
 * probabilities are calibrated.
 *
 * @param xTrain - Training features `[n_train, d]`.
 * @param xTest - Test features `[n_test, d]`.
 * @param options - The `estimator` (default linear discriminant), `clipProba`, `maxWeight` and fit options.
 * @returns The weights at both samples, the ratio as a function, and its effective sample size.
 * @concept covariate-shift
 * @example
 * ```ts
 * import { classifierDensityRatio } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const w = toFlat(classifierDensityRatio([[-2], [-1], [0], [1]], [[0], [1], [2], [3]]).trainWeights)
 * w[3] > w[0] // => true
 * ```
 */
export function classifierDensityRatio(
  xTrain: TensorLike,
  xTest: TensorLike,
  options: ClassifierDensityRatioOptions = {},
): DensityRatioResult {
  const who = 'classifierDensityRatio'
  const tr = matrixView(xTrain, who)
  const te = matrixView(xTest, who, { nFeatures: tr.d })
  const clip = fractionOption(options.clipProba ?? 1e-3, 'clipProba', who, 0, 0.5)
  const estimator = options.estimator ?? linearDiscriminant()
  const all = new Float64Array((tr.n + te.n) * tr.d)
  all.set(tr.data)
  all.set(te.data, tr.n * tr.d)
  const domain = new Float64Array(tr.n + te.n)
  domain.fill(1, tr.n)
  const fitted = estimator.fit(
    { x: fromStorage(all, [tr.n + te.n, tr.d], 'float64'), y: domain },
    innerFitOptions(options, keyFor(options, who), options.parent),
  )
  const proba = requireVerb(fitted, 'predictProba', who)
  const positive = Math.max(0, fitted.classes?.indexOf(1) ?? 1)
  const scale = tr.n / te.n

  const predictRatio = (x: TensorLike): Tensor<'float64'> => {
    const m = matrixView(x, who, { nFeatures: tr.d })
    const p = matrixView(proba(fromView(m)), who)
    const out = Float64Array.from({ length: m.n }, (_, i) => {
      const q = Math.min(Math.max(p.data[i * p.d + positive] ?? 0.5, clip), 1 - clip)
      const r = scale * (q / (1 - q))
      return options.maxWeight === undefined ? r : Math.min(r, options.maxWeight)
    })
    return vectorOf(out)
  }
  const trainWeights = predictRatio(fromView(tr))
  return { trainWeights, testWeights: predictRatio(fromView(te)), predictRatio, ess: essOf(toFlat(trainWeights)) }
}

// ---------------------------------------------------------------------------------------------------------------------
// The kernel model shared by KLIEP and uLSIF
// ---------------------------------------------------------------------------------------------------------------------

/** The Gaussian kernel model `w(x) = sum_l theta_l exp(-|x - c_l|^2 / (2 sigma^2))` at centres taken from the test sample. */
interface KernelModel {
  readonly tr: MatrixView
  readonly te: MatrixView
  readonly centres: MatrixView
  readonly sigma: number
}

/** Up to `count` centres from the test rows, evenly spaced through them. */
function chooseCentres(te: MatrixView, count: number): MatrixView {
  const b = Math.min(count, te.n)
  if (b === te.n) return te
  const out = new Float64Array(b * te.d)
  for (let l = 0; l < b; l++) {
    const idx = Math.floor((l * te.n) / b)
    out.set(te.data.subarray(idx * te.d, (idx + 1) * te.d), l * te.d)
  }
  return { n: b, d: te.d, data: out }
}

/** The median-heuristic width of the pooled sample, from at most 200 evenly spaced rows. */
function pooledWidth(tr: MatrixView, te: MatrixView): number {
  const pick = (m: MatrixView, cap: number): number[][] => {
    const step = Math.max(1, Math.ceil(m.n / cap))
    const rows: number[][] = []
    for (let i = 0; i < m.n; i += step) rows.push(Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
    return rows
  }
  return Math.max(medianHeuristic([...pick(tr, 100), ...pick(te, 100)]), 1e-6)
}

function kernelModel(
  who: string,
  xTrain: TensorLike,
  xTest: TensorLike,
  nCentres: number,
  sigma?: number,
): KernelModel {
  const tr = matrixView(xTrain, who)
  const te = matrixView(xTest, who, { nFeatures: tr.d })
  const width = sigma ?? pooledWidth(tr, te)
  if (!(width > 0)) throw new MdmlError('value', `${who}: sigma must be > 0`)
  return { tr, te, centres: chooseCentres(te, integerOption(nCentres, 'nCentres', who, 1)), sigma: width }
}

/** The design matrix of a sample on the model's centres, `[n, b]` row-major. */
function design(model: KernelModel, x: MatrixView, sigma = model.sigma): Float64Array {
  return rbfGram(x, model.centres, 1 / (2 * sigma * sigma))
}

function ratioFunction(who: string, model: KernelModel, theta: Float64Array, sigma: number) {
  return (x: TensorLike): Tensor<'float64'> => {
    const m = matrixView(x, who, { nFeatures: model.tr.d })
    return vectorOf(dgemv(design(model, m, sigma), m.n, model.centres.n, theta))
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// KLIEP
// ---------------------------------------------------------------------------------------------------------------------

/** Options of {@link kliepProblem}. */
export interface KliepProblemOptions {
  /** Gaussian kernel width `sigma` (default the median heuristic on the pooled sample). */
  readonly sigma?: number
  /** At most this many kernel centres, taken from the test sample (default 100). */
  readonly nCentres?: number
}

/**
 * KLIEP as a smooth problem. The model is `w(x) = sum_l alpha_l K(x, c_l)` and KLIEP (Sugiyama et al. 2008) maximises
 * `mean_j log w(x'_j)` over the test sample subject to `alpha >= 0` and `mean_i w(x_i) = 1` over the training sample.
 * The objective is invariant to scale, so the constraint is a normalisation after the fact and `alpha = exp(theta)`
 * keeps the sign: the smooth problem minimises, over `theta` and by any solver,
 * `log sum_l b_l e^{theta_l} - mean_j log sum_l K(x'_j, c_l) e^{theta_l}` with `b = mean_i K(x_i, c)`; the coefficients
 * are `exp(theta)` rescaled to `b . alpha = 1`. The problem also carries the design matrices for the projected-gradient
 * mode.
 */
export interface KliepProblem extends SmoothProblem {
  readonly model: KernelModel
  /** Log kernel at the test points, `[n_test, b]`. */
  readonly logPhiTest: Float64Array
  /** Kernel at the training points, `[n_train, b]`. */
  readonly phiTrain: Float64Array
  /** Kernel at the test points, `[n_test, b]`. */
  readonly phiTest: Float64Array
  /** The mean kernel over the training sample, `[b]`. */
  readonly kernelMean: Float64Array
}

/**
 * The KLIEP problem for a pair of samples: the objective over `theta = log alpha` (see {@link KliepProblem}), the
 * starting point `theta = 0`, and the kernel matrices.
 *
 * @param xTrain - Training features `[n_train, d]`.
 * @param xTest - Test features `[n_test, d]`.
 * @param options - Kernel width and number of centres.
 * @returns The problem.
 * @concept covariate-shift
 * @example
 * ```ts
 * import { kliepProblem } from 'mdml-engine/learn'
 * kliepProblem([[0], [1], [2]], [[1], [2], [3]], { sigma: 1 }).x0 // => [0, 0, 0]
 * ```
 */
export function kliepProblem(xTrain: TensorLike, xTest: TensorLike, options: KliepProblemOptions = {}): KliepProblem {
  const model = kernelModel('kliep', xTrain, xTest, options.nCentres ?? 100, options.sigma)
  const { tr, te, centres } = model
  const b = centres.n
  const gamma = 1 / (2 * model.sigma * model.sigma)
  const sqTest = squaredDistances(te, centres)
  const sqTrain = squaredDistances(tr, centres)
  const logPhiTest = Float64Array.from(sqTest, (s) => -gamma * s)
  const phiTest = Float64Array.from(logPhiTest, Math.exp)
  const phiTrain = Float64Array.from(sqTrain, (s) => Math.exp(-gamma * s))
  const kernelMean = new Float64Array(b)
  for (let i = 0; i < tr.n; i++)
    for (let l = 0; l < b; l++) kernelMean[l] = (kernelMean[l] ?? 0) + (phiTrain[i * b + l] ?? 0) / tr.n
  // log b_l by a stable log-sum-exp over the training rows.
  const logKernelMean = Float64Array.from({ length: b }, (_, l) => {
    let top = Number.NEGATIVE_INFINITY
    for (let i = 0; i < tr.n; i++) top = Math.max(top, -gamma * (sqTrain[i * b + l] ?? 0))
    let s = 0
    for (let i = 0; i < tr.n; i++) s += Math.exp(-gamma * (sqTrain[i * b + l] ?? 0) - top)
    return top + Math.log(s / tr.n)
  })
  const logTest = fromStorage(logPhiTest, [te.n, b], 'float64')
  const logMean = fromStorage(logKernelMean, [b], 'float64')
  const objective = (theta: Value): Value =>
    sub(logsumexp(add(logMean, theta)), mean(logsumexp(add(logTest, theta), 1)))
  return { objective, x0: new Array<number>(b).fill(0), model, logPhiTest, phiTrain, phiTest, kernelMean }
}

/** Options of {@link kliepAlgorithm} and {@link kliep}. */
export interface KliepOptions extends KliepProblemOptions, FitOptions {
  /** A `kliepSolvers` configuration (default `{ kind: 'projectedGradient' }`, Sugiyama's own). */
  readonly solver?: SpaceValue
  /** Iteration cap of the run (default 300). */
  readonly maxIterations?: number
}

/** Serialisable state of KLIEP's projected-gradient mode. */
export interface KliepState {
  readonly iteration: number
  readonly alpha: Float64Array
  readonly objective: number
  /** The gain of the objective at the last step. */
  readonly gain: number
  readonly change: number
  /** The step length that last succeeded, tried again doubled (0 before the first step). */
  readonly length: number
}

/** The fitted parameters of every KLIEP mode: the vector reached and the scale it is on. */
export type KliepParams = { readonly x: number[]; readonly scale: 'alpha' | 'log' }

/**
 * The Euclidean projection of `y` onto the KLIEP constraints `alpha >= 0` and `b . alpha = 1`: `alpha_l = max(0, y_l -
 * tau b_l)` with `tau` the root of the decreasing piecewise-linear `sum_l b_l max(0, y_l - tau b_l) = 1`, found by bisection.
 */
function projectKliep(y: Float64Array, kernelMean: Float64Array): Float64Array {
  const at = (tau: number): number => {
    let total = 0
    for (let l = 0; l < y.length; l++)
      total += (kernelMean[l] ?? 0) * Math.max(0, (y[l] ?? 0) - tau * (kernelMean[l] ?? 0))
    return total - 1
  }
  // With every coefficient active the root is tau0 and f(tau0) >= 0; at the largest ratio y_l / b_l, f = -1.
  let lo = (ddot(kernelMean, y) - 1) / Math.max(ddot(kernelMean, kernelMean), 1e-300)
  let hi = Number.NEGATIVE_INFINITY
  for (let l = 0; l < y.length; l++) hi = Math.max(hi, (y[l] ?? 0) / Math.max(kernelMean[l] ?? 0, 1e-300))
  for (let it = 0; it < 200 && hi - lo > 1e-15 * Math.max(1, Math.abs(hi)); it++) {
    const mid = (lo + hi) / 2
    if (at(mid) > 0) lo = mid
    else hi = mid
  }
  const tau = (lo + hi) / 2
  return Float64Array.from(y, (v, l) => Math.max(0, v - tau * (kernelMean[l] ?? 0)))
}

/** The scale-free KLIEP objective `mean_j log(phi_j . alpha)` of normalised coefficients. */
function kliepValue(p: KliepProblem, alpha: Float64Array): number {
  const w = dgemv(p.phiTest, p.model.te.n, p.model.centres.n, alpha)
  let s = 0
  for (const v of w) s += Math.log(Math.max(v, 1e-300))
  return s / w.length
}

const projectedGradient: ModeSpec<KliepProblem, KliepState, KliepParams> = {
  description:
    "Sugiyama's projected gradient ascent on the log-likelihood of the test sample, projecting onto the constraints",
  space: dict({
    learningRate: real(1e-4, 10, { log: true, default: 0.1, label: 'length of the first step' }),
    tol: real(1e-14, 1e-2, { log: true, default: 1e-9, label: 'tolerance on the gain of the objective' }),
  }),
  make: (p, k): Algorithm<KliepState> => {
    const lr = (k['learningRate'] as number | undefined) ?? 0.1
    const tol = (k['tol'] as number | undefined) ?? 1e-9
    const b = p.model.centres.n
    return {
      name: 'kliep',
      config: { b, sigma: p.model.sigma, lr, tol },
      schema: channels({
        metrics: {
          'solver/objective': {
            description: 'mean log ratio over the test sample (higher is better)',
            direction: 'higher',
          },
          'solver/change': { description: 'largest change of a coefficient', direction: 'lower' },
          'weights/ess': { description: 'effective sample size of the training weights' },
        },
        tensors: { 'ratio/estimate': { role: 'state', description: 'the ratio at the training points', shape: ['n'] } },
      }),
      init: () => {
        const start = projectKliep(new Float64Array(b).fill(1), p.kernelMean)
        return {
          iteration: 0,
          alpha: start,
          objective: kliepValue(p, start),
          gain: Number.POSITIVE_INFINITY,
          change: Number.POSITIVE_INFINITY,
          length: 0,
        }
      },
      step: (s) => {
        const wTe = dgemv(p.phiTest, p.model.te.n, b, s.alpha)
        const inv = Float64Array.from(wTe, (v) => 1 / (p.model.te.n * Math.max(v, 1e-300)))
        const grad = dgemv(p.phiTest, p.model.te.n, b, inv, true)
        // Try twice the last step that worked, and halve it until the objective does not fall (the projection can
        // undo an ascent step).
        let next = s.alpha
        let value = s.objective
        let length = s.length > 0 ? 2 * s.length : lr / Math.max(dnrm2(grad), 1e-300)
        let accepted = 0
        for (let tries = 0; tries < 40; tries++) {
          const cand = projectKliep(
            Float64Array.from(s.alpha, (a, l) => a + length * (grad[l] ?? 0)),
            p.kernelMean,
          )
          const v = kliepValue(p, cand)
          if (v > s.objective) {
            next = cand
            value = v
            accepted = length
            break
          }
          length /= 2
        }
        let change = 0
        for (let l = 0; l < b; l++) change = Math.max(change, Math.abs((next[l] ?? 0) - (s.alpha[l] ?? 0)))
        return {
          iteration: s.iteration + 1,
          alpha: next,
          objective: value,
          gain: value - s.objective,
          change,
          length: accepted > 0 ? accepted : length,
        }
      },
      done: (s) => (s.iteration > 0 && s.gain <= tol ? 'converged' : undefined),
      observe: (s) => {
        const wTr = dgemv(p.phiTrain, p.model.tr.n, b, s.alpha)
        return {
          metrics: {
            'solver/objective': s.objective,
            'solver/change': Number.isFinite(s.change) ? s.change : 1,
            'weights/ess': essOf(wTr),
          },
          tensors: { 'ratio/estimate': { role: 'state', value: wTr, shape: [wTr.length] } },
        }
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as KliepState,
    }
  },
  params: (s) => ({ x: Array.from(s.alpha), scale: 'alpha' }),
}

/** A generic smooth solver of the `smoothSolvers` table as a mode over a problem that is a `SmoothProblem`. */
function delegate<P extends SmoothProblem, Q extends { readonly x: number[] }>(
  name: string,
  wrap: (x: number[]) => Q,
): ModeSpec<P, unknown, Q> {
  const mode = smoothSolvers.modeOf({ kind: name })
  return {
    description: mode.description ?? name,
    space: mode.space,
    make: (p, k, key: Key) => smoothSolvers.make(p, { ...k, kind: name }, key),
    params: (s) => wrap(smoothSolvers.params({ kind: name }, s).x),
  }
}

/**
 * The solver slot of KLIEP: its own projected gradient (`projectedGradient`, the default, which reproduces the
 * published algorithm) and every smooth solver of `optim` (`gradientDescent`, `conjugateGradient`, `newton`, `bfgs`,
 * `lbfgs`, `coordinateDescent`, `gradient`) on the unconstrained `theta = log alpha` form.
 */
export const kliepSolvers: ModeTable<KliepProblem, KliepParams> = modes<KliepProblem, KliepParams>(
  {
    projectedGradient,
    ...Object.fromEntries(
      smoothSolvers.names.map((name) => [
        name,
        delegate<KliepProblem, KliepParams>(name, (x) => ({ x, scale: 'log' })),
      ]),
    ),
  },
  { label: 'solver', description: 'how the KLIEP objective is maximised', default: 'projectedGradient' },
)

/** The normalised coefficients of a fitted KLIEP parameter vector. */
function kliepAlpha(p: KliepProblem, params: KliepParams): Float64Array {
  if (params.scale === 'alpha') return Float64Array.from(params.x)
  const top = maxOf(params.x)
  const raw = Float64Array.from(params.x, (t) => Math.exp(t - top))
  const norm = ddot(p.kernelMean, raw)
  return raw.map((a) => a / norm)
}

/** A solver configuration with the defaults of the mode it names filled in where it gives only some knobs. */
function solverConfig<Q, P extends Json>(table: ModeTable<Q, P>, config: SpaceValue | undefined): SpaceValue {
  const name = table.modeOf(config).name
  const given = typeof config === 'object' && config !== null && !Array.isArray(config) ? config : {}
  return { ...(defaults(table.space.options[name] as DictSpace) as object), ...given, [table.tag]: name }
}

/** The problem, the algorithm that solves it and the solver configuration of a KLIEP run. */
function kliepSetup(
  xTrain: TensorLike,
  xTest: TensorLike,
  options: KliepOptions,
): { readonly algorithm: Algorithm<unknown>; readonly problem: KliepProblem; readonly solver: SpaceValue } {
  const problem = kliepProblem(xTrain, xTest, options)
  const solver = solverConfig(kliepSolvers, options.solver)
  const key = keyFor(options, 'kliep')
  const algorithm =
    kliepSolvers.modeOf(solver).name === 'projectedGradient'
      ? kliepSolvers.make(problem, solver, key)
      : loggedSolver(problem, solver, key)
  return { algorithm, problem, solver }
}

/**
 * KLIEP as a stepped algorithm, one iteration of the chosen solver per step: the log-likelihood of the test sample,
 * the largest change of a coefficient and the effective sample size of the training weights (projected gradient), or the
 * objective, gradient norm and step length of the generic solver. The solver is a knob (see {@link kliepSolvers}).
 *
 * @param xTrain - Training features `[n_train, d]`.
 * @param xTest - Test features `[n_test, d]`.
 * @param options - Kernel, `solver` configuration and fit options.
 * @returns The algorithm.
 * @concept covariate-shift
 * @example
 * ```ts
 * import { kliepAlgorithm } from 'mdml-engine/learn'
 * kliepAlgorithm([[0], [1], [2]], [[1], [2], [3]], { sigma: 1 }).name // => 'kliep'
 * ```
 */
export function kliepAlgorithm(xTrain: TensorLike, xTest: TensorLike, options: KliepOptions = {}): Algorithm<unknown> {
  return kliepSetup(xTrain, xTest, options).algorithm
}

/** Result of {@link kliep}. */
export interface KliepResult extends DensityRatioResult {
  /** Fitted nonnegative kernel coefficients `alpha` `[b]`, normalised to `mean_train w = 1`. */
  readonly alpha: readonly number[]
  /** The final log-likelihood of the test sample, `mean_j log w(x'_j)`. */
  readonly objective: number
  /** The kernel width used. */
  readonly sigma: number
  /** The iterations taken. */
  readonly iterations: number
  /** Whether the solver stopped on its tolerance. */
  readonly converged: boolean
}

/**
 * Sugiyama's Kullback-Leibler Importance Estimation Procedure (KLIEP; Sugiyama, Suzuki, Nakajima, Kashima, von
 * Buenau and Kawanabe 2008): the density ratio as a nonnegative combination of Gaussian kernels at test points, fitted
 * by maximum likelihood of the test sample under the reweighted training density and normalised so the training weights
 * average one. The `solver` knob picks how (see {@link kliepSolvers}); every solver reaches the same optimum.
 *
 * @param xTrain - Training features `[n_train, d]`.
 * @param xTest - Test features `[n_test, d]`.
 * @param options - Kernel, `solver`, `maxIterations` and fit options.
 * @returns The weights, `predictRatio`, `alpha`, the objective and the convergence.
 * @concept covariate-shift
 * @example
 * ```ts
 * import { kliep } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const w = toFlat(kliep([[-1], [-0.5], [0], [0.5], [1]], [[0], [0.5], [1], [1.5], [2]], { sigma: 1 }).trainWeights)
 * Math.abs(w.reduce((a, b) => a + b, 0) / w.length - 1) < 1e-9 // => true
 * ```
 */
export function kliep(xTrain: TensorLike, xTest: TensorLike, options: KliepOptions = {}): KliepResult {
  const { algorithm, problem, solver } = kliepSetup(xTrain, xTest, options)
  const maxSteps = integerOption(options.maxIterations ?? 300, 'maxIterations', 'kliep', 1)
  const run = runAlgorithm(algorithm, { ...runOptionsFor(options, 'kliep'), maxSteps })
  const alpha = kliepAlpha(problem, kliepSolvers.params(solver, run.state))
  const { model } = problem
  const b = model.centres.n
  const predictRatio = ratioFunction('kliep', model, alpha, model.sigma)
  const wTrain = dgemv(problem.phiTrain, model.tr.n, b, alpha)
  return {
    trainWeights: vectorOf(wTrain),
    testWeights: vectorOf(dgemv(problem.phiTest, model.te.n, b, alpha)),
    predictRatio,
    ess: essOf(wTrain),
    alpha: Array.from(alpha),
    objective: kliepValue(problem, alpha),
    sigma: model.sigma,
    iterations: run.steps,
    converged: run.status === 'converged',
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// uLSIF
// ---------------------------------------------------------------------------------------------------------------------

/** Options of {@link ulsifProblem}. */
export interface UlsifProblemOptions extends KliepProblemOptions {
  /** The ridge penalty `lambda > 0` (default 0.1). */
  readonly lambda?: number
  /** The relative mixing `alpha` in `[0, 1)`: 0 is uLSIF, above 0 the relative ratio `p / (alpha p + (1 - alpha) q)`. */
  readonly alpha?: number
}

/**
 * The uLSIF least-squares problem: minimise over `theta` the squared-loss fit of the ratio,
 * `1/2 theta' H theta - h' theta + lambda / 2 |theta|^2`, with `H = (1 - alpha) mean_i phi(x_i) phi(x_i)' +
 * alpha mean_j phi(x'_j) phi(x'_j)'` and `h = mean_j phi(x'_j)`. Its minimiser is `(H + lambda I)^{-1} h` before the
 * clip at zero, so the closed form is one mode of the solver slot ({@link ulsifSolvers}).
 */
export interface UlsifProblem extends SmoothProblem {
  readonly model: KernelModel
  readonly lambda: number
  readonly alpha: number
  /** `H`, `[b, b]` row-major. */
  readonly H: Float64Array
  /** `h`, `[b]`. */
  readonly h: Float64Array
}

/** `H` and `h` of the least-squares problem on a design pair. */
function momentsOf(
  phiTr: Float64Array,
  nTr: number,
  phiTe: Float64Array,
  nTe: number,
  b: number,
  alpha: number,
): { readonly H: Float64Array; readonly h: Float64Array } {
  const H = new Float64Array(b * b)
  const rank = (phi: Float64Array, n: number, weight: number): void => {
    if (weight === 0) return
    for (let i = 0; i < n; i++) {
      for (let r = 0; r < b; r++) {
        const pr = (weight * (phi[i * b + r] ?? 0)) / n
        if (pr === 0) continue
        for (let c = 0; c < b; c++) H[r * b + c] = (H[r * b + c] ?? 0) + pr * (phi[i * b + c] ?? 0)
      }
    }
  }
  rank(phiTr, nTr, 1 - alpha)
  rank(phiTe, nTe, alpha)
  const h = new Float64Array(b)
  for (let j = 0; j < nTe; j++) for (let l = 0; l < b; l++) h[l] = (h[l] ?? 0) + (phiTe[j * b + l] ?? 0) / nTe
  return { H, h }
}

/**
 * The uLSIF problem for a pair of samples ({@link UlsifProblem}) with a smooth objective over `theta`, starting at zero.
 *
 * @param xTrain - Training features `[n_train, d]`.
 * @param xTest - Test features `[n_test, d]`.
 * @param options - Kernel width, centres, `lambda` and the relative `alpha`.
 * @returns The problem.
 * @concept covariate-shift
 * @example
 * ```ts
 * import { ulsifProblem } from 'mdml-engine/learn'
 * ulsifProblem([[0], [1], [2]], [[1], [2], [3]], { sigma: 1 }).h.length // => 3
 * ```
 */
export function ulsifProblem(xTrain: TensorLike, xTest: TensorLike, options: UlsifProblemOptions = {}): UlsifProblem {
  const who = 'ulsif'
  const alpha = options.alpha ?? 0
  if (!(alpha >= 0 && alpha < 1)) throw new MdmlError('value', `${who}: alpha must lie in [0, 1)`)
  const lambda = options.lambda ?? 0.1
  if (!(lambda > 0)) throw new MdmlError('value', `${who}: lambda must be > 0`)
  const model = kernelModel(who, xTrain, xTest, options.nCentres ?? 100, options.sigma)
  const b = model.centres.n
  const { H, h } = momentsOf(design(model, model.tr), model.tr.n, design(model, model.te), model.te.n, b, alpha)
  const A = fromStorage(
    Float64Array.from(H, (v, i) => (i % (b + 1) === 0 ? v + lambda : v)),
    [b, b],
    'float64',
  )
  const hv = fromStorage(h, [b], 'float64')
  const objective = (theta: Value): Value => sub(mul(0.5, sum(mul(theta, matmul(A, theta)))), sum(mul(hv, theta)))
  return { objective, x0: new Array<number>(b).fill(0), model, lambda, alpha, H, h }
}

const closedForm: ModeSpec<UlsifProblem, { readonly theta: Float64Array; readonly solved: boolean }, VectorParams> = {
  description: 'the closed form: solve (H + lambda I) theta = h once',
  space: dict({}),
  make: (p) => ({
    name: 'ulsif-closed-form',
    config: { b: p.h.length, lambda: p.lambda },
    init: () => ({ theta: new Float64Array(p.h.length), solved: false }),
    step: () => {
      const b = p.h.length
      const A = Float64Array.from(p.H, (v, i) => (i % (b + 1) === 0 ? v + p.lambda : v))
      const theta = toFlat(solve(fromStorage(A, [b, b], 'float64'), fromStorage(p.h, [b], 'float64')))
      return { theta: Float64Array.from(theta), solved: true }
    },
    done: (s) => (s.solved ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as { readonly theta: Float64Array; readonly solved: boolean },
  }),
  params: (s) => ({ x: Array.from(s.theta) }),
}

/**
 * The solver slot of uLSIF: the `closedForm` mode (the default, and uLSIF's published algorithm) and every smooth solver
 * of `optim` on the same quadratic objective, which reach the same coefficients.
 */
export const ulsifSolvers: ModeTable<UlsifProblem, VectorParams> = modes<UlsifProblem, VectorParams>(
  {
    closedForm,
    ...Object.fromEntries(
      smoothSolvers.names.map((name) => [name, delegate<UlsifProblem, VectorParams>(name, (x) => ({ x }))]),
    ),
  },
  { label: 'solver', description: 'how the least-squares objective is minimised', default: 'closedForm' },
)

/**
 * The closed-form leave-pair-out score of the uLSIF family (Kanamori, Hido and Sugiyama 2009, Section 3): with the
 * first `n = min(n_train, n_test)` rows of each sample paired, leave out pair `i`, refit
 * `theta_i = max(0, (H_i + lambda I)^{-1} h_i)` (two Sherman-Morrison updates of one inverse, `O(n b^2)` for all `n`
 * fits) and score `(1 - alpha) (theta_i' phi(x_i))^2 / 2 + alpha (theta_i' phi(x'_i))^2 / 2 - theta_i' phi(x'_i)`; the
 * score is the mean over pairs. Lower is better.
 *
 * @param phiTrain - Training design `[n_train, b]` row-major.
 * @param phiTest - Test design `[n_test, b]` row-major.
 * @param b - The number of basis functions.
 * @param lambda - The ridge penalty.
 * @param alpha - The relative mixing (0 for uLSIF).
 * @returns The mean leave-pair-out squared-loss score.
 * @concept covariate-shift
 * @example
 * ```ts
 * import { ulsifLoo } from 'mdml-engine/learn'
 * Number.isFinite(ulsifLoo(Float64Array.of(1, 0.5, 0.5, 1, 0.2, 0.2), Float64Array.of(0.5, 1, 1, 0.5, 0.3, 0.3), 2, 0.1, 0)) // => true
 * ```
 */
export function ulsifLoo(
  phiTrain: Float64Array,
  phiTest: Float64Array,
  b: number,
  lambda: number,
  alpha: number,
): number {
  const n = Math.min(phiTrain.length, phiTest.length) / b
  if (n < 2) throw new MdmlError('value', 'ulsifLoo: need at least two paired rows')
  const rows = (phi: Float64Array): Float64Array[] =>
    Array.from({ length: n }, (_, i) => phi.subarray(i * b, (i + 1) * b))
  const U = rows(phiTrain)
  const V = rows(phiTest)
  // M0 = ((1 - alpha) S_tr + alpha S_te) / (n - 1) + lambda I over the n pairs.
  const pairs = momentsOf(phiTrain.subarray(0, n * b), n, phiTest.subarray(0, n * b), n, b, alpha)
  const M0 = Float64Array.from(pairs.H, (v, i) => (v * n) / (n - 1) + (i % (b + 1) === 0 ? lambda : 0))
  const B1 = Float64Array.from(
    toFlat(solve(fromStorage(M0, [b, b], 'float64'), fromStorage(identity(b), [b, b], 'float64'))),
  )
  const mv = (x: ArrayLike<number>): Float64Array => dgemv(B1, b, b, x)
  const sumV = new Float64Array(b)
  for (const v of V) for (let l = 0; l < b; l++) sumV[l] = (sumV[l] ?? 0) + (v[l] ?? 0)
  const B1sum = mv(sumV)
  const a1 = (1 - alpha) / (n - 1)
  const a2 = alpha / (n - 1)
  let total = 0
  for (let i = 0; i < n; i++) {
    const u = U[i] as Float64Array
    const v = V[i] as Float64Array
    const B1u = mv(u)
    const B1v = mv(v)
    const d1 = 1 - a1 * ddot(u, B1u)
    // y = h_i = (sum_v - v_i) / (n - 1); B1 y, then B2 = (M0 - a1 u u')^{-1}.
    const B1y = Float64Array.from(B1sum, (s, l) => (s - (B1v[l] ?? 0)) / (n - 1))
    const B2y = Float64Array.from(B1y, (t, l) => t + (a1 * (B1u[l] ?? 0) * ddot(u, B1y)) / d1)
    const B2v = Float64Array.from(B1v, (t, l) => t + (a1 * (B1u[l] ?? 0) * ddot(u, B1v)) / d1)
    const d2 = 1 - a2 * ddot(v, B2v)
    const theta = Float64Array.from(B2y, (t, l) => Math.max(0, t + (a2 * (B2v[l] ?? 0) * ddot(v, B2y)) / d2))
    const fu = ddot(theta, u)
    const fv = ddot(theta, v)
    total += 0.5 * (1 - alpha) * fu * fu + 0.5 * alpha * fv * fv - fv
  }
  return total / n
}

function identity(b: number): Float64Array {
  const out = new Float64Array(b * b)
  for (let i = 0; i < b; i++) out[i * b + i] = 1
  return out
}

/** Options of {@link ulsif}. */
export interface UlsifOptions extends UlsifProblemOptions, FitOptions {
  /** Candidate kernel widths, chosen between by the leave-pair-out score (default the median heuristic times 1/4 to 4). */
  readonly sigmas?: readonly number[]
  /** Candidate ridge penalties, chosen between by the leave-pair-out score (default 0.001 to 10). */
  readonly lambdas?: readonly number[]
  /** A `ulsifSolvers` configuration (default `{ kind: 'closedForm' }`). */
  readonly solver?: SpaceValue
  /** Iteration cap of an iterative solver (default 1000). */
  readonly maxIterations?: number
}

/** Result of {@link ulsif}. */
export interface UlsifResult extends DensityRatioResult {
  /** Nonnegative kernel coefficients `theta` `[b]`. */
  readonly theta: readonly number[]
  /** The kernel width used. */
  readonly sigma: number
  /** The ridge penalty used. */
  readonly lambda: number
  /** The leave-pair-out score of the chosen pair (`NaN` when fewer than two paired rows). */
  readonly looScore: number
  /** The relative mixing `alpha`. */
  readonly alpha: number
}

/**
 * Least-squares importance fitting (uLSIF, Kanamori, Hido and Sugiyama 2009; relative uLSIF with `alpha > 0`, Yamada
 * et al. 2013): the density ratio as a nonnegative combination of Gaussian kernels fitted by squared loss, with kernel
 * width and ridge penalty chosen from the candidate lists by the closed-form leave-pair-out score ({@link ulsifLoo}).
 * The default solver is the closed form; the `solver` knob takes any generic smooth solver to the same coefficients.
 * The relative ratio `p / (alpha p + (1 - alpha) q)` is bounded by `1 / alpha`.
 *
 * @param xTrain - Training features `[n_train, d]`.
 * @param xTest - Test features `[n_test, d]`.
 * @param options - Kernel, candidate lists, `alpha`, `solver` and fit options.
 * @returns The weights, `predictRatio`, `theta`, the chosen `sigma` and `lambda`, and the score.
 * @concept covariate-shift
 * @example
 * ```ts
 * import { ulsif } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const r = ulsif([[-1], [-0.5], [0], [0.5], [1]], [[0], [0.5], [1], [1.5], [2]], { sigma: 1, lambda: 0.1 })
 * const w = toFlat(r.trainWeights)
 * w[4] > w[0] // => true
 * ```
 */
export function ulsif(xTrain: TensorLike, xTest: TensorLike, options: UlsifOptions = {}): UlsifResult {
  const who = 'ulsif'
  const alpha = options.alpha ?? 0
  const base = kernelModel(who, xTrain, xTest, options.nCentres ?? 100, options.sigma)
  const sigmas =
    options.sigmas ?? (options.sigma === undefined ? [0.25, 0.5, 1, 2, 4].map((f) => f * base.sigma) : [base.sigma])
  const lambdas = options.lambdas ?? (options.lambda === undefined ? [1e-3, 1e-2, 1e-1, 1, 10] : [options.lambda])
  if (sigmas.length === 0 || lambdas.length === 0)
    throw new MdmlError('value', `${who}: candidate lists must be non-empty`)
  const b = base.centres.n
  const paired = Math.min(base.tr.n, base.te.n) >= 2

  let best = { sigma: sigmas[0] as number, lambda: lambdas[0] as number, loo: Number.POSITIVE_INFINITY }
  for (const sigma of sigmas) {
    if (!(sigma > 0)) throw new MdmlError('value', `${who}: sigma must be > 0`)
    const phiTr = design(base, base.tr, sigma)
    const phiTe = design(base, base.te, sigma)
    for (const lambda of lambdas) {
      if (!(lambda > 0)) throw new MdmlError('value', `${who}: lambda must be > 0`)
      const loo = paired ? ulsifLoo(phiTr, phiTe, b, lambda, alpha) : 0
      if (loo < best.loo || best.loo === Number.POSITIVE_INFINITY) best = { sigma, lambda, loo }
    }
  }

  const problem = ulsifProblem(xTrain, xTest, {
    nCentres: options.nCentres ?? 100,
    sigma: best.sigma,
    lambda: best.lambda,
    alpha,
  })
  const solver = solverConfig(ulsifSolvers, options.solver)
  const closed = ulsifSolvers.modeOf(solver).name === 'closedForm'
  const key = keyFor(options, who)
  const algorithm = closed ? ulsifSolvers.make(problem, solver, key) : loggedSolver(problem, solver, key)
  const run = runAlgorithm(algorithm, {
    ...runOptionsFor(options, who),
    maxSteps: closed ? 1 : integerOption(options.maxIterations ?? 1000, 'maxIterations', who, 1),
  })
  const theta = Float64Array.from(ulsifSolvers.params(solver, run.state).x, (t) => Math.max(0, t))
  const { model } = problem
  const phiTrain = design(model, model.tr)
  const phiTest = design(model, model.te)
  const wTrain = dgemv(phiTrain, model.tr.n, b, theta)
  return {
    trainWeights: vectorOf(wTrain),
    testWeights: vectorOf(dgemv(phiTest, model.te.n, b, theta)),
    predictRatio: ratioFunction(who, model, theta, model.sigma),
    ess: essOf(wTrain),
    theta: Array.from(theta),
    sigma: best.sigma,
    lambda: best.lambda,
    looScore: paired ? best.loo : Number.NaN,
    alpha,
  }
}
