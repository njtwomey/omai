/**
 * Robust linear regression that resists outliers, as in scikit-learn: Huber regression (squared loss for small
 * residuals and linear beyond, with a jointly fitted scale), RANSAC (fits on random minimal samples keeping the one
 * with most inliers), Theil-Sen (the spatial median of fits on small subsets) and quantile regression (the pinball
 * loss, as a linear program).
 */

import { quantileOf, WhiteboxError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { anomalyScores, coefficientTable, type Exposition } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  featureNamesOf,
  type LearnData,
  takeRows,
} from '../../lib/learn/index.ts'
import { lbfgsb, linprog } from '../../lib/optim/index.ts'
import { foldIn, type Key, subsample } from '../../lib/random/index.ts'
import { bool, dict, int, real, type Space } from '../../lib/space/index.ts'
import { ransac } from '../../lib/stats/index.ts'
import {
  abs,
  add,
  div,
  greater,
  matmul,
  mul,
  sliceAxis,
  square,
  sub,
  sum,
  type Tensor,
  toFlat,
  type Value,
  where,
} from '../../lib/tensor/index.ts'
import { type Design, designOf, designPredictor, matrixTensor, vectorTensor } from './_design.ts'
import { type FittedLinearModel, type LinearModelParams, linearModelDefaults, linearVerbs } from './model.ts'
import { leastSquaresSolve } from './solvers.ts'

const GAUSSIAN = linearModelDefaults({})

function paramsOf(coef: ArrayLike<number>, intercept: number, solver: string, iterations: number): LinearModelParams {
  return {
    coef: Array.from(coef),
    intercept,
    scale: 1,
    solver,
    reason: 'robust fit',
    iterations,
    converged: true,
    objective: Number.NaN,
  }
}

function table(data: LearnData, d: number, coef: ArrayLike<number>, intercept: number, fitIntercept: boolean) {
  return coefficientTable({
    names: [...featureNamesOf(data, d), ...(fitIntercept ? ['intercept'] : [])],
    estimates: [...Array.from(coef), ...(fitIntercept ? [intercept] : [])],
  })
}

function robustEstimator<H extends object>(
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

/** Hyperparameters of {@link huberRegressor}. */
export interface HuberHyper {
  /** Residuals beyond epsilon sigma count linearly (default 1.35). */
  readonly epsilon: number
  /** L2 penalty on the coefficients (default 1e-4). */
  readonly alpha: number
  readonly fitIntercept: boolean
  readonly maxIter: number
}

/** Parameters of a fitted Huber regressor. */
export type HuberParams = LinearModelParams & { readonly sigma: number; readonly outliers: number[] }

/**
 * Huber regression (sklearn.linear_model.HuberRegressor): coefficients and scale sigma minimise
 * sum_i (sigma + H_epsilon(r_i / sigma) sigma) + alpha |beta|^2 by L-BFGS-B with sigma > 0, so outliers beyond
 * epsilon sigma pull linearly. Exposition: the coefficient table and anomaly scores |r_i| / sigma against epsilon.
 *
 * @param hyper - Epsilon (default 1.35), penalty (default 1e-4), intercept and iterations.
 * @returns The estimator.
 */
export function huberRegressor(hyper: Partial<HuberHyper> = {}): ExplainedEstimator<HuberHyper, FittedLinearModel> {
  return robustEstimator(
    'huber-regressor',
    {
      epsilon: hyper.epsilon ?? 1.35,
      alpha: hyper.alpha ?? 1e-4,
      fitIntercept: hyper.fitIntercept ?? true,
      maxIter: hyper.maxIter ?? 100,
    },
    (h) =>
      dict({
        epsilon: real(1, 3, {
          label: 'ε (Huber threshold)',
          description: 'Residuals beyond ε scale estimates get absolute rather than squared loss.',
          default: h.epsilon,
        }),
        alpha: real(0, 1, {
          label: 'α (L2 penalty)',
          description: 'The L2 penalty on the coefficients.',
          default: h.alpha,
        }),
      }),
    (data, h, options) => {
      const design = designOf(data, 'huberRegressor')
      const { n, d } = design.x
      const cols = d + (h.fitIntercept ? 1 : 0)
      const aug = new Float64Array(n * cols)
      for (let i = 0; i < n; i++) {
        aug.set(design.x.data.subarray(i * d, (i + 1) * d), i * cols)
        if (h.fitIntercept) aug[i * cols + d] = 1
      }
      const a = matrixTensor(aug, n, cols)
      const y = vectorTensor(design.y)
      const w = vectorTensor(design.w)
      const sw = design.w.reduce((s, v) => s + v, 0)
      const eps = h.epsilon
      const f = (p: Value): Value => {
        const beta = sliceAxis(p, 0, 0, cols)
        const sigma = sliceAxis(p, 0, cols, cols + 1)
        const r = abs(sub(y, matmul(a, beta)))
        const outlier = greater(r, mul(sigma, eps))
        const huber = where(outlier, sub(mul(r, 2 * eps), mul(sigma, eps * eps)), div(square(r), sigma))
        const penalty = sum(square(sliceAxis(p, 0, 0, d)))
        return add(add(mul(sum(sigma), sw), sum(mul(w, huber))), mul(penalty, h.alpha))
      }
      const x0 = new Float64Array(cols + 1)
      x0[cols] = 1
      const bounds: [number | null, number | null][] = Array.from({ length: cols + 1 }, (_, j) =>
        j === cols ? [2.220446049250313e-15, null] : [null, null],
      )
      const r = lbfgsb(f, x0, {
        bounds,
        maxIterations: h.maxIter,
        gradTol: 1e-10,
        ftol: 1e-15,
        ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
        ...(options.policy === undefined ? {} : { policy: options.policy }),
      })
      const p = toFlat(r.x as Tensor)
      const coef = p.slice(0, d)
      const intercept = h.fitIntercept ? (p[d] ?? 0) : 0
      const sigma = p[cols] ?? 1
      const pred = designPredictor(design, coef, intercept)
      const scores = Array.from(pred, (v, i) => Math.abs((design.y[i] ?? 0) - v) / sigma)
      const params: HuberParams = {
        ...paramsOf(coef, intercept, 'lbfgsb', r.steps),
        sigma,
        outliers: scores.flatMap((s, i) => (s > eps ? [i] : [])),
      }
      return {
        params,
        exposition: [
          table(data, d, coef, intercept, h.fitIntercept),
          anomalyScores({ method: 'huber', scores, threshold: eps }),
        ],
      }
    },
  )
}

/** Hyperparameters of {@link ransacRegressor}. */
export interface RansacHyper {
  /** Rows per minimal sample (default d + 1). */
  readonly minSamples?: number
  /** Inlier threshold on |residual| (default the median absolute deviation of y, as scikit-learn). */
  readonly residualThreshold?: number
  readonly maxTrials: number
  readonly fitIntercept: boolean
}

/**
 * RANSAC regression (sklearn.linear_model.RANSACRegressor with least squares inside), from the core's traced keyed
 * RANSAC: least squares on random minimal samples, the model with the most inliers refitted on them. Exposition: the
 * coefficient table and anomaly scores |residual| against the threshold, with the inlier indicator as evidence.
 *
 * @param hyper - Sample size, threshold, trials (default 100) and intercept.
 * @returns The estimator.
 */
export function ransacRegressor(hyper: Partial<RansacHyper> = {}): ExplainedEstimator<RansacHyper, FittedLinearModel> {
  return robustEstimator(
    'ransac-regressor',
    {
      ...(hyper.minSamples === undefined ? {} : { minSamples: hyper.minSamples }),
      ...(hyper.residualThreshold === undefined ? {} : { residualThreshold: hyper.residualThreshold }),
      maxTrials: hyper.maxTrials ?? 100,
      fitIntercept: hyper.fitIntercept ?? true,
    },
    (h) =>
      dict({
        maxTrials: int(10, 1000, {
          label: 'trials',
          description: 'Random minimal subsets tried.',
          default: h.maxTrials,
        }),
        fitIntercept: bool({
          label: 'fit intercept',
          description: 'Estimate an intercept (otherwise the fit passes through the origin).',
          default: h.fitIntercept,
        }),
      }),
    (data, h, options) => {
      const design = designOf(data, 'ransacRegressor')
      const { n, d } = design.x
      const ys = Array.from(design.y)
      const median = quantileOf(ys, 0.5)
      const threshold =
        h.residualThreshold ??
        quantileOf(
          ys.map((v) => Math.abs(v - median)),
          0.5,
        )
      const subset = (rows: readonly number[]): Design => ({
        x: takeRows(design.x, rows),
        y: Float64Array.from(rows, (r) => design.y[r] ?? 0),
        w: Float64Array.from(rows, (r) => design.w[r] ?? 0),
      })
      const model = {
        fit: (rows: readonly number[]) => leastSquaresSolve(subset(rows), 0, h.fitIntercept, 'svd'),
        residual: (m: { coef: Float64Array; intercept: number }, i: number) => {
          let s = m.intercept
          for (let j = 0; j < d; j++) s += (design.x.data[i * d + j] ?? 0) * (m.coef[j] ?? 0)
          return (design.y[i] ?? 0) - s
        },
      }
      const r = ransac(
        Array.from({ length: n }, (_, i) => i),
        model,
        {
          minSamples: h.minSamples ?? Math.min(n, d + (h.fitIntercept ? 1 : 0)),
          residualThreshold: threshold,
          maxTrials: h.maxTrials,
          ...(options.key === undefined ? {} : { key: options.key }),
          ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        },
      )
      const inliers = new Set(r.inliers)
      const scores = Array.from({ length: n }, (_, i) => Math.abs(model.residual(r.model, i)))
      return {
        params: paramsOf(r.model.coef, r.model.intercept, 'ransac', r.trials),
        exposition: [
          table(data, d, r.model.coef, r.model.intercept, h.fitIntercept),
          anomalyScores({
            method: 'ransac',
            scores,
            threshold,
            evidence: { name: 'inlier', values: scores.map((_, i) => (inliers.has(i) ? 1 : 0)) },
          }),
        ],
      }
    },
  )
}

/** Hyperparameters of {@link theilSenRegressor}. */
export interface TheilSenHyper {
  /** Subsets solved at most (default 10000); all subsets when there are fewer. */
  readonly maxSubpopulation: number
  readonly fitIntercept: boolean
  readonly maxIter: number
  readonly tol: number
}

function combinations(n: number, k: number, limit: number): number[][] | undefined {
  let count = 1
  for (let i = 0; i < k; i++) {
    count = (count * (n - i)) / (i + 1)
    if (count > limit) return undefined
  }
  const out: number[][] = []
  const pick = (start: number, acc: number[]): void => {
    if (acc.length === k) {
      out.push([...acc])
      return
    }
    for (let i = start; i < n; i++) {
      acc.push(i)
      pick(i + 1, acc)
      acc.pop()
    }
  }
  pick(0, [])
  return out
}

/** The spatial median of points (rows), by the modified Weiszfeld iteration of Vardi and Zhang (scikit-learn's). */
function spatialMedian(points: readonly Float64Array[], maxIter: number, tol: number): Float64Array {
  const p = points[0]?.length ?? 0
  if (p === 1)
    return Float64Array.of(
      quantileOf(
        points.map((x) => x[0] ?? 0),
        0.5,
      ),
    )
  let y: Float64Array = new Float64Array(p)
  for (const x of points) for (let j = 0; j < p; j++) y[j] = (y[j] ?? 0) + (x[j] ?? 0) / points.length
  for (let it = 0; it < maxIter; it++) {
    let onPoint = 0
    const num = new Float64Array(p)
    const dir = new Float64Array(p)
    let invSum = 0
    for (const x of points) {
      let dist = 0
      for (let j = 0; j < p; j++) dist += ((x[j] ?? 0) - (y[j] ?? 0)) ** 2
      dist = Math.sqrt(dist)
      if (dist < 2.220446049250313e-16) {
        onPoint = 1
        continue
      }
      invSum += 1 / dist
      for (let j = 0; j < p; j++) {
        num[j] = (num[j] ?? 0) + (x[j] ?? 0) / dist
        dir[j] = (dir[j] ?? 0) + ((x[j] ?? 0) - (y[j] ?? 0)) / dist
      }
    }
    let q = 0
    for (const v of dir) q += v * v
    q = Math.sqrt(q)
    let direction: Float64Array
    if (q > 2.220446049250313e-16) direction = Float64Array.from(num, (v) => v / invSum)
    else {
      direction = new Float64Array(p).fill(1)
      q = 1
    }
    const next = Float64Array.from(
      direction,
      (v, j) => Math.max(0, 1 - onPoint / q) * v + Math.min(1, onPoint / q) * (y[j] ?? 0),
    )
    let change = 0
    for (let j = 0; j < p; j++) change += ((next[j] ?? 0) - (y[j] ?? 0)) ** 2
    y = next
    if (change < tol * tol) break
  }
  return y
}

/**
 * Theil-Sen regression (sklearn.linear_model.TheilSenRegressor): exact least-squares fits on every subset of
 * p = d + 1 rows (or `maxSubpopulation` keyed random subsets when there are more), combined by their spatial median.
 * Exposition: the coefficient table.
 *
 * @param hyper - Subset budget (default 10000), intercept, and the Weiszfeld iteration limits.
 * @returns The estimator.
 */
export function theilSenRegressor(
  hyper: Partial<TheilSenHyper> = {},
): ExplainedEstimator<TheilSenHyper, FittedLinearModel> {
  return robustEstimator(
    'theil-sen-regressor',
    {
      maxSubpopulation: hyper.maxSubpopulation ?? 10000,
      fitIntercept: hyper.fitIntercept ?? true,
      maxIter: hyper.maxIter ?? 300,
      tol: hyper.tol ?? 1e-3,
    },
    (h) =>
      dict({
        fitIntercept: bool({
          label: 'fit intercept',
          description: 'Estimate an intercept (otherwise the fit passes through the origin).',
          default: h.fitIntercept,
        }),
      }),
    (data, h, options) => {
      const design = designOf(data, 'theilSenRegressor')
      const { n, d } = design.x
      const k = d + (h.fitIntercept ? 1 : 0)
      if (n < k) throw new WhiteboxError('value', `theilSenRegressor: needs at least ${k} samples`)
      const subsets =
        combinations(n, k, h.maxSubpopulation) ??
        Array.from({ length: h.maxSubpopulation }, (_, s) => Array.from(toFlat(subsample(foldKey(options, s), n, k))))
      const solutions = subsets.map((rows) => {
        // An exact fit through the k points, intercept included as a column.
        const aug = new Float64Array(rows.length * k)
        rows.forEach((r, i) => {
          for (let j = 0; j < d; j++) aug[i * k + j] = design.x.data[r * d + j] ?? 0
          if (h.fitIntercept) aug[i * k + d] = 1
        })
        return leastSquaresSolve(
          {
            x: { n: rows.length, d: k, data: aug },
            y: Float64Array.from(rows, (r) => design.y[r] ?? 0),
            w: new Float64Array(rows.length).fill(1),
          },
          0,
          false,
          'svd',
        ).coef
      })
      const median = spatialMedian(solutions, h.maxIter, h.tol)
      const coef = median.slice(0, d)
      const intercept = h.fitIntercept ? (median[d] ?? 0) : 0
      return {
        params: paramsOf(coef, intercept, 'theil-sen', subsets.length),
        exposition: [table(data, d, coef, intercept, h.fitIntercept)],
      }
    },
  )
}

function foldKey(options: FitOptions, s: number): Key {
  return foldIn(resolveKey(options, 'theilSenRegressor'), s)
}

/** Hyperparameters of {@link quantileRegressor}. */
export interface QuantileHyper {
  /** The quantile (default 0.5, the median). */
  readonly quantile: number
  /** L1 penalty (default 1, as scikit-learn). */
  readonly alpha: number
  readonly fitIntercept: boolean
}

/**
 * Quantile regression (sklearn.linear_model.QuantileRegressor): min sum_i w_i rho_q(y_i - x_i beta - b) / sum_i w_i +
 * alpha |beta|_1 with the pinball loss rho_q, as a linear program solved by the core's linprog. Exposition: the
 * coefficient table.
 *
 * @param hyper - Quantile, L1 penalty and intercept.
 * @returns The estimator.
 */
export function quantileRegressor(
  hyper: Partial<QuantileHyper> = {},
): ExplainedEstimator<QuantileHyper, FittedLinearModel> {
  return robustEstimator(
    'quantile-regressor',
    { quantile: hyper.quantile ?? 0.5, alpha: hyper.alpha ?? 1, fitIntercept: hyper.fitIntercept ?? true },
    (h) =>
      dict({
        quantile: real(0.05, 0.95, {
          label: 'quantile',
          description: 'The conditional quantile fitted (0.5 is the median).',
          default: h.quantile,
        }),
        alpha: real(0, 10, {
          label: 'α (L1 penalty)',
          description: 'The L1 penalty on the coefficients (0 for plain quantile regression).',
          default: h.alpha,
        }),
      }),
    (data, h) => {
      const design = designOf(data, 'quantileRegressor')
      const { n, d } = design.x
      const p = d + (h.fitIntercept ? 1 : 0)
      const sw = design.w.reduce((s, v) => s + v, 0)
      // Variables [beta+ (p), beta- (p), u+ (n), u- (n)] >= 0 with X beta + u+ - u- = y.
      const m = 2 * p + 2 * n
      const c = new Float64Array(m)
      for (let j = 0; j < p; j++) {
        const pen = j < d ? h.alpha * sw : 0
        c[j] = pen
        c[p + j] = pen
      }
      for (let i = 0; i < n; i++) {
        c[2 * p + i] = (design.w[i] ?? 0) * h.quantile
        c[2 * p + n + i] = (design.w[i] ?? 0) * (1 - h.quantile)
      }
      const aEq = new Float64Array(n * m)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) {
          const v = j < d ? (design.x.data[i * d + j] ?? 0) : 1
          aEq[i * m + j] = v
          aEq[i * m + p + j] = -v
        }
        aEq[i * m + 2 * p + i] = 1
        aEq[i * m + 2 * p + n + i] = -1
      }
      const r = linprog({ c, aEq: matrixTensor(aEq, n, m), bEq: vectorTensor(design.y) })
      if (!r.success) throw new WhiteboxError('convergence', `quantileRegressor: ${r.message}`)
      const z = toFlat(r.x)
      const beta = Array.from({ length: p }, (_, j) => (z[j] ?? 0) - (z[p + j] ?? 0))
      const coef = beta.slice(0, d)
      const intercept = h.fitIntercept ? (beta[d] ?? 0) : 0
      return {
        params: paramsOf(coef, intercept, 'linprog', r.iterations),
        exposition: [table(data, d, coef, intercept, h.fitIntercept)],
      }
    },
  )
}

/** {@link huberRegressor} as a class. */
export class HuberRegressor extends ExplainedEstimatorClass<HuberHyper, FittedLinearModel> {
  /** @param hyper - As {@link huberRegressor}. */
  constructor(hyper: Partial<HuberHyper> = {}) {
    super(huberRegressor(hyper))
  }
}

/** {@link ransacRegressor} as a class. */
export class RansacRegressor extends ExplainedEstimatorClass<RansacHyper, FittedLinearModel> {
  /** @param hyper - As {@link ransacRegressor}. */
  constructor(hyper: Partial<RansacHyper> = {}) {
    super(ransacRegressor(hyper))
  }
}

/** {@link theilSenRegressor} as a class. */
export class TheilSenRegressor extends ExplainedEstimatorClass<TheilSenHyper, FittedLinearModel> {
  /** @param hyper - As {@link theilSenRegressor}. */
  constructor(hyper: Partial<TheilSenHyper> = {}) {
    super(theilSenRegressor(hyper))
  }
}

/** {@link quantileRegressor} as a class. */
export class QuantileRegressor extends ExplainedEstimatorClass<QuantileHyper, FittedLinearModel> {
  /** @param hyper - As {@link quantileRegressor}. */
  constructor(hyper: Partial<QuantileHyper> = {}) {
    super(quantileRegressor(hyper))
  }
}
