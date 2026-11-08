/**
 * Regression when the features are measured with error too. Total least squares (orthogonal regression) minimises the
 * squared distances to the fitted hyperplane in both the features and the target, each scaled by its error standard
 * deviation; with one feature and a known ratio of the error variances it is Deming regression. The closed form is the
 * smallest right singular vector of the centred, scaled `[X y]`; ordinary least squares, which assumes error-free
 * features, is the limit of vanishing feature error.
 */

import { OmaiError } from '../base/index.ts'
import { defineEstimator, type Estimator, type Fitted } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { dict, real } from '../space/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { type AffineVerbs, affineVerbs } from './_verbs.ts'
import { centre, designOf } from './design.ts'

/** The hyperparameters of {@link totalLeastSquares}. */
export interface TotalLeastSquaresHyper {
  /** The standard deviation of the error of the features: one number for all or one per feature (default 1). */
  readonly errorX: number | readonly number[]
  /** The standard deviation of the error of the target (default 1). */
  readonly errorY: number
  readonly fitIntercept: boolean
}

/** The fitted parameters of {@link totalLeastSquares}. */
export type TotalLeastSquaresParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The squared distance of the data from the fitted hyperplane, in units of the errors (the smallest squared singular value). */
  readonly orthogonalSumOfSquares: number
  /** The singular values of the scaled, centred `[X y]`. */
  readonly singularValues: number[]
}

/**
 * Total least squares (errors-in-variables) regression: the hyperplane `y = x . beta + b` minimising
 * `sum_i |x_i - x*_i|^2 / sx^2 + (y_i - y*_i)^2 / sy^2` over the hyperplane's points `(x*, y*)`, with `sx` the standard
 * deviation of the feature errors (one per feature, or one for all) and `sy` that of the target. Equal errors give
 * orthogonal regression; with one feature and `errorY^2 / errorX^2` equal to Deming's `delta` it is Deming regression.
 * It is the closed form by the SVD of the centred, scaled `[X y]` (the smallest right singular vector), so the slope is
 * not attenuated by feature noise the way a least-squares slope is.
 *
 * @param hyper - `errorX`, `errorY`, `fitIntercept`.
 * @returns The estimator.
 * @throws OmaiError `value` when the best hyperplane is vertical (no finite coefficients) or an error is not positive.
 * @reference W. E. Deming, Statistical Adjustment of Data (1943); G. H. Golub and C. F. Van Loan, An analysis of the
 * total least squares problem, SIAM J. Numer. Anal. 17 (1980).
 * @example
 * ```ts
 * import { totalLeastSquares } from 'omai-engine/linear'
 * const x = [[0], [1], [2], [3], [4]]
 * const f = totalLeastSquares().fit({ x, y: [0, 2, 4, 6, 8] })
 * Math.round((f.params.coef[0] ?? 0) * 1000) / 1000 // => 2
 * ```
 */
export function totalLeastSquares(
  hyper: Partial<TotalLeastSquaresHyper> = {},
): Estimator<TotalLeastSquaresHyper, Fitted<TotalLeastSquaresParams> & AffineVerbs> {
  return defineEstimator<TotalLeastSquaresHyper, TotalLeastSquaresParams, AffineVerbs>(
    {
      name: 'total-least-squares',
      kind: 'regressor',
      hyperSpace: (h) => dict({ errorY: real(0.01, 100, { log: true, default: h.errorY }) }),
      fit: (data, h) => {
        const design = designOf(data, 'totalLeastSquares')
        const { n, d } = design.x
        const sx = typeof h.errorX === 'number' ? new Array<number>(d).fill(h.errorX) : [...h.errorX]
        if (sx.length !== d)
          throw new OmaiError('shape', `totalLeastSquares: ${sx.length} feature errors for ${d} features`)
        if (!sx.every((v) => v > 0) || !(h.errorY > 0))
          throw new OmaiError('value', 'totalLeastSquares: the error standard deviations must be positive')
        if (n <= d) throw new OmaiError('value', 'totalLeastSquares: needs more rows than features')
        const c = centre(design, h.fitIntercept)
        const m = d + 1
        const z = new Float64Array(n * m)
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < d; j++) z[i * m + j] = (c.x[i * d + j] ?? 0) / (sx[j] ?? 1)
          z[i * m + d] = (c.y[i] ?? 0) / h.errorY
        }
        const r = linalg.svd(fromStorage(z, [n, m], 'float64'))
        const vt = toFlat(r.vh)
        const s = Array.from(toFlat(r.s))
        const last = Math.min(s.length, m) - 1
        // The right singular vector of the smallest singular value: row `last` of V^T (a null direction when n < m).
        const v = Array.from({ length: m }, (_, j) => vt[last * m + j] ?? 0)
        const vy = v[d] ?? 0
        if (Math.abs(vy) < 1e-12)
          throw new OmaiError('value', 'totalLeastSquares: the best hyperplane is vertical in the target')
        // Scaled model y/sy = sum b_j x_j/sx_j with b = -v[:d]/v[d]; undo the scaling.
        const coef = Array.from({ length: d }, (_, j) => ((-(v[j] ?? 0) / vy) * h.errorY) / (sx[j] ?? 1))
        const intercept = h.fitIntercept ? c.yMean - coef.reduce((a, b, j) => a + b * (c.xMean[j] ?? 0), 0) : 0
        return {
          coef,
          intercept,
          orthogonalSumOfSquares: (s[last] ?? 0) ** 2,
          singularValues: s,
        }
      },
      methods: (p) => affineVerbs(p.coef, p.intercept, 'totalLeastSquares'),
    },
    { errorX: 1, errorY: 1, fitIntercept: true, ...hyper },
  )
}

/** The hyperparameters of {@link deming}. */
export interface DemingHyper {
  /** The ratio `delta = Var(error in y) / Var(error in x)` (default 1: orthogonal regression). */
  readonly delta: number
  readonly fitIntercept: boolean
}

/**
 * Deming regression: the line through noisy `x` and `y` when the ratio `delta` of the error variance in `y` to that in
 * `x` is known. It is {@link totalLeastSquares} with `errorX = 1` and `errorY = sqrt(delta)`; `delta = 1` is orthogonal
 * regression, a large `delta` approaches least squares of `y` on `x`, and a small one the inverted regression of `x` on
 * `y`. With several features it is the same fit with the same error in each.
 *
 * @param hyper - `delta`, `fitIntercept`.
 * @returns The estimator.
 * @reference W. E. Deming, Statistical Adjustment of Data (1943).
 * @example
 * ```ts
 * import { deming } from 'omai-engine/linear'
 * const x = [[1.1], [1.9], [3.2], [3.8], [5.1]]
 * const slope = deming({ delta: 1 }).fit({ x, y: [1, 2, 3, 4, 5] }).params.coef[0] ?? 0
 * slope > 0.9 // => true
 * ```
 */
export function deming(
  hyper: Partial<DemingHyper> = {},
): Estimator<TotalLeastSquaresHyper, Fitted<TotalLeastSquaresParams> & AffineVerbs> {
  const delta = hyper.delta ?? 1
  if (!(delta > 0)) throw new OmaiError('value', `deming: delta must be positive, got ${delta}`)
  return totalLeastSquares({ errorX: 1, errorY: Math.sqrt(delta), fitIntercept: hyper.fitIntercept ?? true })
}
