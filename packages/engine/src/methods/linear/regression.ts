import {
  coefficientTable,
  crossValidationCurve,
  type Exposition,
  regularisationPath,
} from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  featureNamesOf,
  type LearnData,
} from '../../lib/learn/index.ts'
import type { Splitter } from '../../lib/learn/model-selection/index.ts'
import { orthogonalMatchingPursuit as omp } from '../../lib/optim/index.ts'
import { bool, dict, int, real, type Space } from '../../lib/space/index.ts'
import { primalOf, toFlat } from '../../lib/tensor/index.ts'
import { centre, designOf, interceptFor, logGrid, matrixTensor, vectorTensor } from './_design.ts'
import {
  type FittedLinearModel,
  fitLinear,
  type LinearModelHyper,
  type LinearModelParams,
  linearModelDefaults,
  linearVerbs,
} from './model.ts'
import { alphaGrid, curveOf, cvPath, pathOn, pathRecord } from './paths.ts'
import { type LinearSolver, leastSquaresSolve, svdOf, svdRidge } from './solvers.ts'

const GAUSSIAN: LinearModelHyper = linearModelDefaults({})
const sumWeights = (data: LearnData, who: string): number => designOf(data, who).w.reduce((a, b) => a + b, 0)

function gaussianEstimator<H extends object>(
  name: string,
  hyper: H,
  space: (h: H) => Space,
  fit: (data: LearnData, h: H, options: FitOptions) => { params: LinearModelParams; exposition: Exposition[] },
): ExplainedEstimator<H, FittedLinearModel> {
  return defineExplainedEstimator(
    { name, kind: 'regressor', hyperSpace: space, fit, methods: (p) => linearVerbs(p, GAUSSIAN, name) },
    hyper,
  )
}

/** Hyperparameters of {@link linearRegression}. */
export interface LinearRegressionHyper {
  readonly fitIntercept: boolean
  /** The direct solver (default Cholesky of the normal equations, falling back to the SVD when singular). */
  readonly solver: 'auto' | 'cholesky' | 'qr' | 'svd'
}

/**
 * Ordinary (weighted) least squares (sklearn.linear_model.LinearRegression). Exposition: the coefficient table with
 * classical standard errors, t-test p-values and 95% intervals.
 *
 * @param hyper - Intercept (default true) and direct solver.
 * @returns The estimator.
 */
export function linearRegression(
  hyper: Partial<LinearRegressionHyper> = {},
): ExplainedEstimator<LinearRegressionHyper, FittedLinearModel> {
  return gaussianEstimator(
    'linear-regression',
    { fitIntercept: hyper.fitIntercept ?? true, solver: hyper.solver ?? 'auto' },
    (h) =>
      dict({
        fitIntercept: bool({
          label: 'fit intercept',
          description: 'Estimate an intercept (otherwise the fit passes through the origin).',
          default: h.fitIntercept,
        }),
      }),
    (data, h, options) => {
      const f = fitLinear(
        data,
        linearModelDefaults({ fitIntercept: h.fitIntercept, solver: h.solver }),
        options,
        'linearRegression',
      )
      return { params: f.params, exposition: [f.table] }
    },
  )
}

/** Hyperparameters of {@link ridge}. */
export interface RidgeHyper {
  /** The penalty on |beta|^2 against the unscaled sum of squares (scikit-learn's Ridge). */
  readonly alpha: number
  readonly fitIntercept: boolean
  readonly solver: 'auto' | 'cholesky' | 'qr' | 'svd' | 'lbfgs'
  /** Points of the exposition path around alpha (default 30; 0 for none). */
  readonly pathLength: number
}

/**
 * Ridge regression, min |y - x beta - b|^2 + alpha |beta|^2 (sklearn.linear_model.Ridge). Exposition: the coefficient
 * table and the ridge path over alpha 1e-3 to 1e3 times the chosen alpha (from one SVD), with the chosen alpha marked.
 *
 * @param hyper - Penalty (default 1), intercept, solver and exposition path length.
 * @returns The estimator.
 */
export function ridge(hyper: Partial<RidgeHyper> = {}): ExplainedEstimator<RidgeHyper, FittedLinearModel> {
  return gaussianEstimator(
    'ridge',
    {
      alpha: hyper.alpha ?? 1,
      fitIntercept: hyper.fitIntercept ?? true,
      solver: hyper.solver ?? 'auto',
      pathLength: hyper.pathLength ?? 30,
    },
    (h) =>
      dict({
        alpha: real(1e-3, 1e3, { log: true, default: h.alpha, label: 'penalty' }),
        fitIntercept: bool({
          label: 'fit intercept',
          description: 'Estimate an intercept (otherwise the fit passes through the origin).',
          default: h.fitIntercept,
        }),
      }),
    (data, h, options) => {
      const s = sumWeights(data, 'ridge')
      const solver: LinearSolver = h.solver === 'auto' ? 'cholesky' : h.solver
      const f = fitLinear(
        data,
        linearModelDefaults({ alpha: h.alpha / s, fitIntercept: h.fitIntercept, solver, tol: 1e-12 }),
        options,
        'ridge',
      )
      const exposition: Exposition[] = [f.table]
      if (h.pathLength > 0) {
        const alphas = logGrid(h.alpha * 1e3, 1e-6, h.pathLength)
        const c = centre(f.design, h.fitIntercept)
        const coefs = svdRidge(c.x, f.design.x.n, f.design.x.d, c.y, alphas).map((v) => Array.from(v))
        exposition.push(
          regularisationPath({
            penalties: alphas,
            features: featureNamesOf(data, f.design.x.d),
            coefficients: coefs,
            chosen: h.alpha,
          }),
        )
      }
      return { params: f.params, exposition }
    },
  )
}

/** Hyperparameters of {@link elasticNet}. */
export interface ElasticNetHyper {
  /** The penalty (scikit-learn's per-sample scaling). */
  readonly alpha: number
  /** The L1 share (1 for the lasso). */
  readonly l1Ratio: number
  readonly fitIntercept: boolean
  readonly maxIter: number
  readonly tol: number
  /** Points of the exposition path from alpha_max (default 30; 0 for none). */
  readonly pathLength: number
}

function enetFit(who: string) {
  return (data: LearnData, h: ElasticNetHyper, options: FitOptions) => {
    const f = fitLinear(
      data,
      linearModelDefaults({
        alpha: h.alpha,
        l1Ratio: h.l1Ratio,
        fitIntercept: h.fitIntercept,
        solver: 'coordinate-descent',
        maxIter: h.maxIter,
        tol: h.tol,
      }),
      options,
      who,
    )
    const exposition: Exposition[] = [f.table]
    if (h.pathLength > 0) {
      const opts = {
        l1Ratio: h.l1Ratio,
        fitIntercept: h.fitIntercept,
        nAlphas: h.pathLength,
        tol: h.tol,
        maxIter: h.maxIter,
      }
      const path = pathOn(f.design, alphaGrid(f.design, opts), opts)
      exposition.push(pathRecord(path, featureNamesOf(data, f.design.x.d), h.alpha))
    }
    return { params: f.params, exposition }
  }
}

const enetSpace = (h: ElasticNetHyper, withRatio: boolean): Space =>
  dict({
    alpha: real(1e-4, 10, { log: true, default: h.alpha, label: 'penalty' }),
    ...(withRatio ? { l1Ratio: real(0, 1, { default: h.l1Ratio, label: 'L1 share' }) } : {}),
    fitIntercept: bool({
      label: 'fit intercept',
      description: 'Estimate an intercept (otherwise the fit passes through the origin).',
      default: h.fitIntercept,
    }),
  })

const enetDefaults = (hyper: Partial<ElasticNetHyper>, l1Ratio: number): ElasticNetHyper => ({
  alpha: hyper.alpha ?? 1,
  l1Ratio: hyper.l1Ratio ?? l1Ratio,
  fitIntercept: hyper.fitIntercept ?? true,
  maxIter: hyper.maxIter ?? 1000,
  tol: hyper.tol ?? 1e-4,
  pathLength: hyper.pathLength ?? 30,
})

/**
 * The elastic net, min |y - x beta - b|^2 / (2n) + alpha l1Ratio |beta|_1 + alpha (1 - l1Ratio) |beta|^2 / 2, by
 * coordinate descent with duality gaps (sklearn.linear_model.ElasticNet). Exposition: the coefficient table and the
 * regularisation path from alpha_max (coefficients, active sets and duality gaps) with the chosen alpha marked.
 *
 * @param hyper - Penalty (default 1), L1 share (default 0.5), intercept, solver limits and exposition path length.
 * @returns The estimator.
 */
export function elasticNet(
  hyper: Partial<ElasticNetHyper> = {},
): ExplainedEstimator<ElasticNetHyper, FittedLinearModel> {
  return gaussianEstimator('elastic-net', enetDefaults(hyper, 0.5), (h) => enetSpace(h, true), enetFit('elasticNet'))
}

/**
 * The lasso, {@link elasticNet} with l1Ratio = 1 (sklearn.linear_model.Lasso).
 *
 * @param hyper - Penalty (default 1), intercept, solver limits and exposition path length.
 * @returns The estimator.
 */
export function lasso(
  hyper: Partial<Omit<ElasticNetHyper, 'l1Ratio'>> = {},
): ExplainedEstimator<ElasticNetHyper, FittedLinearModel> {
  return gaussianEstimator(
    'lasso',
    enetDefaults({ ...hyper, l1Ratio: 1 }, 1),
    (h) => enetSpace(h, false),
    enetFit('lasso'),
  )
}

/** Hyperparameters of {@link elasticNetCV}. */
export interface ElasticNetCVHyper {
  readonly l1Ratio: number
  readonly nAlphas: number
  readonly eps: number
  /** Explicit penalties (overrides nAlphas and eps). */
  readonly alphas?: readonly number[]
  readonly cv?: Splitter
  readonly fitIntercept: boolean
  readonly maxIter: number
  readonly tol: number
  /** Refit at the one-standard-error penalty instead of the best (default false, as scikit-learn). */
  readonly oneStandardError: boolean
}

/** Parameters of a cross-validated elastic net. */
export type ElasticNetCVParams = LinearModelParams & { readonly alpha: number }

function cvEstimator(name: string, hyper: ElasticNetCVHyper, withRatio: boolean) {
  return defineExplainedEstimator(
    {
      name,
      kind: 'regressor',
      hyperSpace: (h: ElasticNetCVHyper): Space =>
        dict({
          ...(withRatio ? { l1Ratio: real(0.01, 1, { default: h.l1Ratio, label: 'L1 share' }) } : {}),
          nAlphas: int(5, 100, { default: h.nAlphas, label: 'grid size' }),
        }),
      fit: (data: LearnData, h: ElasticNetCVHyper, options: FitOptions) => {
        const opts = {
          l1Ratio: h.l1Ratio,
          nAlphas: h.nAlphas,
          eps: h.eps,
          fitIntercept: h.fitIntercept,
          tol: h.tol,
          maxIter: h.maxIter,
          ...(h.alphas === undefined ? {} : { alphas: h.alphas }),
          ...(h.cv === undefined ? {} : { cv: h.cv }),
        }
        const cv = cvPath(data, opts)
        const alpha = h.oneStandardError ? cv.oneStandardErrorAlpha : cv.bestAlpha
        const f = fitLinear(
          data,
          linearModelDefaults({
            alpha,
            l1Ratio: h.l1Ratio,
            fitIntercept: h.fitIntercept,
            solver: 'coordinate-descent',
            maxIter: h.maxIter,
            tol: h.tol,
          }),
          options,
          name,
        )
        const path = pathOn(f.design, cv.alphas, opts)
        const exposition: Exposition[] = [
          f.table,
          cv.curve,
          pathRecord(path, featureNamesOf(data, f.design.x.d), alpha),
        ]
        return { params: { ...f.params, alpha }, exposition }
      },
      methods: (p: ElasticNetCVParams) => ({ ...linearVerbs(p, GAUSSIAN, name), alpha: p.alpha }),
    },
    hyper,
  )
}

const cvDefaults = (hyper: Partial<ElasticNetCVHyper>, l1Ratio: number): ElasticNetCVHyper => ({
  l1Ratio: hyper.l1Ratio ?? l1Ratio,
  nAlphas: hyper.nAlphas ?? 100,
  eps: hyper.eps ?? 1e-3,
  ...(hyper.alphas === undefined ? {} : { alphas: hyper.alphas }),
  ...(hyper.cv === undefined ? {} : { cv: hyper.cv }),
  fitIntercept: hyper.fitIntercept ?? true,
  maxIter: hyper.maxIter ?? 1000,
  tol: hyper.tol ?? 1e-4,
  oneStandardError: hyper.oneStandardError ?? false,
})

/** A fitted cross-validated elastic net: the chosen penalty is `alpha`. */
export type FittedElasticNetCV = FittedLinearModel & { readonly alpha: number }

/**
 * The elastic net with its penalty chosen by cross-validation along the path (sklearn.linear_model.ElasticNetCV).
 * Exposition: the coefficient table, the cross-validation curve (mean test MSE per alpha with standard errors, the
 * best and one-standard-error alphas) and the full-data regularisation path with the chosen alpha.
 *
 * @param hyper - L1 share (default 0.5), grid, splitter (default 5-fold), intercept and solver limits.
 * @returns The estimator.
 */
export function elasticNetCV(
  hyper: Partial<ElasticNetCVHyper> = {},
): ExplainedEstimator<ElasticNetCVHyper, FittedElasticNetCV> {
  return cvEstimator('elastic-net-cv', cvDefaults(hyper, 0.5), true)
}

/**
 * The lasso with its penalty chosen by cross-validation (sklearn.linear_model.LassoCV); exposition as
 * {@link elasticNetCV}.
 *
 * @param hyper - Grid, splitter (default 5-fold), intercept and solver limits.
 * @returns The estimator.
 */
export function lassoCV(
  hyper: Partial<Omit<ElasticNetCVHyper, 'l1Ratio'>> = {},
): ExplainedEstimator<ElasticNetCVHyper, FittedElasticNetCV> {
  return cvEstimator('lasso-cv', cvDefaults({ ...hyper, l1Ratio: 1 }, 1), false)
}

/** Hyperparameters of {@link ridgeCV}. */
export interface RidgeCVHyper {
  readonly alphas: readonly number[]
  readonly fitIntercept: boolean
}

/**
 * Ridge regression with its penalty chosen by efficient leave-one-out cross-validation (sklearn.linear_model.RidgeCV):
 * from one SVD, the leave-one-out residuals e_i / (1 - h_ii) for every alpha. Exposition: the coefficient table, the
 * leave-one-out curve (mean squared error per alpha, with the standard error of the mean over samples) and the ridge
 * path with the chosen alpha.
 *
 * @param hyper - The penalties (default 0.1, 1, 10) and intercept.
 * @returns The estimator.
 */
export function ridgeCV(hyper: Partial<RidgeCVHyper> = {}): ExplainedEstimator<RidgeCVHyper, FittedElasticNetCV> {
  return defineExplainedEstimator(
    {
      name: 'ridge-cv',
      kind: 'regressor',
      hyperSpace: (h: RidgeCVHyper): Space =>
        dict({
          fitIntercept: bool({
            label: 'fit intercept',
            description: 'Estimate an intercept (otherwise the fit passes through the origin).',
            default: h.fitIntercept,
          }),
        }),
      fit: (data: LearnData, h: RidgeCVHyper, options: FitOptions) => {
        const design = designOf(data, 'ridgeCV')
        const { n, d } = design.x
        const c = centre(design, h.fitIntercept)
        const sw = design.w.reduce((a, b) => a + b, 0)
        const coefs = svdRidge(c.x, n, d, c.y, h.alphas)
        // Leverages h_ii of the centred ridge smoother (plus w_i / sum w for the unpenalised intercept), per alpha.
        const errors = h.alphas.map((alpha, a) => {
          const coef = coefs[a] ?? new Float64Array(d)
          const lev = ridgeLeverage(c.x, n, d, alpha)
          return Array.from({ length: n }, (_, i) => {
            let pred = 0
            for (let j = 0; j < d; j++) pred += (c.x[i * d + j] ?? 0) * (coef[j] ?? 0)
            const r = (c.y[i] ?? 0) - pred
            const hii = (lev[i] ?? 0) + (h.fitIntercept ? (design.w[i] ?? 0) / sw : 0)
            return (r / (1 - hii)) ** 2 / Math.max(design.w[i] ?? 1, 1e-300)
          })
        })
        const { foldScores: _perSample, ...curve } = curveOf('alpha', h.alphas, errors, false)
        const alpha = curve.best ?? h.alphas[0] ?? 1
        const f = fitLinear(
          data,
          linearModelDefaults({ alpha: alpha / sw, fitIntercept: h.fitIntercept, tol: 1e-12 }),
          options,
          'ridgeCV',
        )
        const path = regularisationPath({
          penalties: h.alphas,
          features: featureNamesOf(data, d),
          coefficients: coefs.map((v) => Array.from(v)),
          chosen: alpha,
        })
        return {
          params: { ...f.params, alpha },
          exposition: [f.table, crossValidationCurve(curve), path],
        }
      },
      methods: (p: ElasticNetCVParams) => ({ ...linearVerbs(p, GAUSSIAN, 'ridgeCV'), alpha: p.alpha }),
    },
    { alphas: hyper.alphas ?? [0.1, 1, 10], fitIntercept: hyper.fitIntercept ?? true },
  )
}

/** Diagonal of x (x^T x + alpha I)^{-1} x^T from the SVD of x. */
function ridgeLeverage(x: Float64Array, n: number, d: number, alpha: number): Float64Array {
  const out = new Float64Array(n)
  // h_ii = sum_a u_ia^2 s_a^2 / (s_a^2 + alpha), from one SVD per call (cheap for demo sizes).
  const { u, s } = svdOf(x, n, d)
  const k = s.length
  for (let i = 0; i < n; i++) {
    let h = 0
    for (let a = 0; a < k; a++) {
      const sa = s[a] ?? 0
      h += (u[i * k + a] ?? 0) ** 2 * ((sa * sa) / (sa * sa + alpha))
    }
    out[i] = h
  }
  return out
}

/** Hyperparameters of {@link orthogonalMatchingPursuit}. */
export interface OmpHyper {
  /** Atoms to select (default max(floor(d / 10), 1), as scikit-learn). */
  readonly nNonzeroCoefs?: number
  readonly fitIntercept: boolean
}

/**
 * Orthogonal matching pursuit regression (sklearn.linear_model.OrthogonalMatchingPursuit), from the core's traced
 * pursuit on the centred design. Exposition: the coefficient table and the greedy path: the least-squares
 * coefficients on each prefix of the selected atoms (the "penalty" is the number of atoms).
 *
 * @param hyper - Number of atoms and intercept.
 * @returns The estimator.
 */
export function orthogonalMatchingPursuit(
  hyper: Partial<OmpHyper> = {},
): ExplainedEstimator<OmpHyper, FittedLinearModel> {
  return gaussianEstimator(
    'orthogonal-matching-pursuit',
    {
      ...(hyper.nNonzeroCoefs === undefined ? {} : { nNonzeroCoefs: hyper.nNonzeroCoefs }),
      fitIntercept: hyper.fitIntercept ?? true,
    },
    (h) => dict({ nNonzeroCoefs: int(1, 50, { default: h.nNonzeroCoefs ?? 1, label: 'atoms' }) }),
    (data, h, options) => {
      const design = designOf(data, 'orthogonalMatchingPursuit')
      const { n, d } = design.x
      const c = centre(design, h.fitIntercept)
      const r = omp(matrixTensor(c.x, n, d), vectorTensor(c.y), {
        ...(h.nNonzeroCoefs === undefined ? {} : { nNonzeroCoefs: h.nNonzeroCoefs }),
        ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
        ...(options.policy === undefined ? {} : { policy: options.policy }),
      })
      const coef = Float64Array.from(toFlat(primalOf(r.coef)))
      const f = fitLinear(
        data,
        linearModelDefaults({ fitIntercept: h.fitIntercept }),
        options,
        'orthogonalMatchingPursuit',
      )
      const intercept = h.fitIntercept ? interceptFor(c, coef) : 0
      const prefixes = r.support.map((_, k) => {
        const support = r.support.slice(0, k + 1)
        const sub = new Float64Array(n * support.length)
        for (let i = 0; i < n; i++)
          for (const [a, j] of support.entries()) sub[i * support.length + a] = c.x[i * d + j] ?? 0
        const ls = leastSquaresSolve(
          { x: { n, d: support.length, data: sub }, y: c.y, w: new Float64Array(n).fill(1) },
          0,
          false,
          'svd',
        )
        const full = new Array<number>(d).fill(0)
        for (const [a, j] of support.entries()) full[j] = ls.coef[a] ?? 0
        return full
      })
      const params: LinearModelParams = {
        ...f.params,
        coef: Array.from(coef),
        intercept,
        solver: 'omp',
        reason: 'greedy pursuit',
      }
      const names = featureNamesOf(data, d)
      return {
        params,
        exposition: [
          coefficientTable({
            names: [...names, ...(h.fitIntercept ? ['intercept'] : [])],
            estimates: [...coef, ...(h.fitIntercept ? [intercept] : [])],
          }),
          regularisationPath({
            penalties: r.support.map((_, k) => k + 1),
            features: names,
            coefficients: prefixes,
            active: r.support.map((_, k) => r.support.slice(0, k + 1)),
          }),
        ],
      }
    },
  )
}

/** {@link linearRegression} as a class. */
export class LinearRegression extends ExplainedEstimatorClass<LinearRegressionHyper, FittedLinearModel> {
  /** @param hyper - As {@link linearRegression}. */
  constructor(hyper: Partial<LinearRegressionHyper> = {}) {
    super(linearRegression(hyper))
  }
}

/** {@link ridge} as a class. */
export class Ridge extends ExplainedEstimatorClass<RidgeHyper, FittedLinearModel> {
  /** @param hyper - As {@link ridge}. */
  constructor(hyper: Partial<RidgeHyper> = {}) {
    super(ridge(hyper))
  }
}

/** {@link lasso} as a class. */
export class Lasso extends ExplainedEstimatorClass<ElasticNetHyper, FittedLinearModel> {
  /** @param hyper - As {@link lasso}. */
  constructor(hyper: Partial<Omit<ElasticNetHyper, 'l1Ratio'>> = {}) {
    super(lasso(hyper))
  }
}

/** {@link elasticNet} as a class. */
export class ElasticNet extends ExplainedEstimatorClass<ElasticNetHyper, FittedLinearModel> {
  /** @param hyper - As {@link elasticNet}. */
  constructor(hyper: Partial<ElasticNetHyper> = {}) {
    super(elasticNet(hyper))
  }
}

/** {@link lassoCV} as a class. */
export class LassoCV extends ExplainedEstimatorClass<ElasticNetCVHyper, FittedElasticNetCV> {
  /** @param hyper - As {@link lassoCV}. */
  constructor(hyper: Partial<Omit<ElasticNetCVHyper, 'l1Ratio'>> = {}) {
    super(lassoCV(hyper))
  }
}

/** {@link elasticNetCV} as a class. */
export class ElasticNetCV extends ExplainedEstimatorClass<ElasticNetCVHyper, FittedElasticNetCV> {
  /** @param hyper - As {@link elasticNetCV}. */
  constructor(hyper: Partial<ElasticNetCVHyper> = {}) {
    super(elasticNetCV(hyper))
  }
}

/** {@link ridgeCV} as a class. */
export class RidgeCV extends ExplainedEstimatorClass<RidgeCVHyper, FittedElasticNetCV> {
  /** @param hyper - As {@link ridgeCV}. */
  constructor(hyper: Partial<RidgeCVHyper> = {}) {
    super(ridgeCV(hyper))
  }
}

/** {@link orthogonalMatchingPursuit} as a class. */
export class OrthogonalMatchingPursuit extends ExplainedEstimatorClass<OmpHyper, FittedLinearModel> {
  /** @param hyper - As {@link orthogonalMatchingPursuit}. */
  constructor(hyper: Partial<OmpHyper> = {}) {
    super(orthogonalMatchingPursuit(hyper))
  }
}
