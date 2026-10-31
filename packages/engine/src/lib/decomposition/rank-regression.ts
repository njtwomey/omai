/**
 * Regression through a few latent directions, as closed forms: principal-component regression (least squares on the
 * leading principal components of the features, the discarded directions being the regularisation) and reduced-rank
 * regression (a multi-output least squares whose coefficient matrix is constrained to a given rank, by the SVD of the
 * fitted values; Anderson, Izenman). The weighting of the outputs is a named mode: the identity gives the least-squares
 * reduced-rank fit, the inverse output covariance gives the canonical-correlation form.
 */

import { MdmlError } from '../base/index.ts'
import {
  defineEstimator,
  type Estimator,
  type Fitted,
  type LearnData,
  matrixView,
  requireTargets,
  targetVector,
  withDefaultScore,
} from '../learn/index.ts'
import { dict, int, real } from '../space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { centred, columnMeans, mm, pinvRows, thinSvd, tr } from './_linear.ts'

const toRows = (flat: ArrayLike<number>, n: number, d: number): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => flat[i * d + j] ?? 0))

const matrixTensor = (rows: readonly (readonly number[])[], d: number): Tensor =>
  fromStorage(Float64Array.from(rows.flat()), [rows.length, d], 'float64')

// ---------------------------------------------------------------------------------------------------------------------
// Principal-component regression

/** The hyperparameters of {@link pcr}. */
export interface PcrHyper {
  /** How many leading principal components to regress on. */
  readonly nComponents: number
  readonly fitIntercept: boolean
  /** Whether to standardise the features to unit variance first (default false). */
  readonly scale: boolean
}

/** The fitted parameters of {@link pcr}. */
export type PcrParams = {
  /** The coefficients on the original features. */
  readonly coef: number[]
  readonly intercept: number
  /** The coefficients on the principal-component scores. */
  readonly scoreCoef: number[]
  /** The principal axes used, `[k][d]`, in the units of the (scaled) features. */
  readonly components: number[][]
  /** The variance of each retained component `s_i^2 / (n - 1)`. */
  readonly explainedVariance: number[]
  /** The share of the total feature variance each retained component holds. */
  readonly explainedVarianceRatio: number[]
  readonly mean: number[]
  /** The per-feature scale applied before the decomposition (ones unless `scale`). */
  readonly std: number[]
}

/** What a fitted principal-component regression offers. */
export interface PcrVerbs {
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  /** The scores on the retained principal components. */
  readonly transform: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

/**
 * Principal-component regression: least squares of the target on the scores of the leading `nComponents` principal
 * components of the (centred, optionally standardised) features, with the coefficients mapped back to the original
 * features. With every component it is ordinary least squares; fewer components discard the low-variance directions of
 * the features, the way ridge shrinks them. The same fit as scikit-learn's `make_pipeline(PCA(k), LinearRegression())`.
 *
 * @param hyper - `nComponents` (default 1), `fitIntercept`, `scale`.
 * @returns The estimator.
 * @throws MdmlError `value` for more components than `min(n, d)` or for sample weights (not supported).
 * @example
 * ```ts
 * import { pcr } from 'mdml-engine/decomposition'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0, 0.1], [1, 1.2], [2, 1.9], [3, 3.1], [4, 3.9]]
 * const y = [1, 3, 5, 7, 9]
 * const f = pcr({ nComponents: 1 }).fit({ x, y })
 * Math.round((toFlat(f.predict([[5, 5]]))[0] ?? 0) * 10) / 10 // => 11.1
 * ```
 */
export function pcr(hyper: Partial<PcrHyper> = {}): Estimator<PcrHyper, Fitted<PcrParams> & PcrVerbs> {
  return defineEstimator<PcrHyper, PcrParams, PcrVerbs>(
    {
      name: 'pcr',
      kind: 'regressor',
      hyperSpace: (h) => dict({ nComponents: int(1, 50, { default: h.nComponents }) }),
      fit: (data, h) => {
        if (data.weights !== undefined) throw new MdmlError('value', 'pcr: sample weights are not supported')
        const m = matrixView(data.x, 'pcr')
        const y = targetVector(data.y ?? [], 'pcr', m.n)
        const k = h.nComponents
        if (!Number.isInteger(k) || k < 1 || k > Math.min(m.n, m.d))
          throw new MdmlError('value', `pcr: nComponents must be an integer in [1, ${Math.min(m.n, m.d)}], got ${k}`)
        const mean = h.fitIntercept ? columnMeans(m) : new Array<number>(m.d).fill(0)
        const xc = centred(m, mean)
        const std = new Array<number>(m.d).fill(1)
        if (h.scale)
          for (let j = 0; j < m.d; j++) {
            let s = 0
            for (let i = 0; i < m.n; i++) s += (xc[i * m.d + j] ?? 0) ** 2
            const sd = Math.sqrt(s / m.n)
            std[j] = sd === 0 ? 1 : sd
            for (let i = 0; i < m.n; i++) xc[i * m.d + j] = (xc[i * m.d + j] ?? 0) / (std[j] ?? 1)
          }
        const yMean = h.fitIntercept ? y.reduce((a, v) => a + v, 0) / m.n : 0
        const { u, s, vt } = thinSvd(xc, m.n, m.d)
        const components = vt.slice(0, k)
        // Least squares on orthogonal scores: coefficient_c = u_c . y / s_c.
        const scoreCoef = components.map((_, c) => {
          const sc = s[c] ?? 0
          if (!(sc > 1e-12 * (s[0] ?? 1))) return 0
          let dot = 0
          for (let i = 0; i < m.n; i++) dot += (u[i]?.[c] ?? 0) * ((y[i] ?? 0) - yMean)
          return dot / sc
        })
        const scaled = new Array<number>(m.d).fill(0)
        components.forEach((v, c) => {
          for (let j = 0; j < m.d; j++) scaled[j] = (scaled[j] ?? 0) + (scoreCoef[c] ?? 0) * (v[j] ?? 0)
        })
        const coef = scaled.map((v, j) => v / (std[j] ?? 1))
        const intercept = h.fitIntercept ? yMean - coef.reduce((a, c, j) => a + c * (mean[j] ?? 0), 0) : 0
        const total = s.reduce((a, v) => a + v * v, 0)
        return {
          coef,
          intercept,
          scoreCoef,
          components,
          explainedVariance: s.slice(0, k).map((v) => (v * v) / Math.max(m.n - 1, 1)),
          explainedVarianceRatio: s.slice(0, k).map((v) => (total === 0 ? 0 : (v * v) / total)),
          mean,
          std,
        }
      },
      methods: (p) => {
        const d = p.coef.length
        const base = {
          nFeaturesIn: d,
          predict: (x: TensorLike): Tensor => {
            const m = matrixView(x, 'pcr', { nFeatures: d })
            return fromStorage(
              Float64Array.from({ length: m.n }, (_, i) => {
                let v = p.intercept
                for (let j = 0; j < d; j++) v += (m.data[i * d + j] ?? 0) * (p.coef[j] ?? 0)
                return v
              }),
              [m.n],
              'float64',
            )
          },
          transform: (x: TensorLike): Tensor => {
            const m = matrixView(x, 'pcr', { nFeatures: d })
            const k = p.components.length
            const out = new Float64Array(m.n * k)
            for (let i = 0; i < m.n; i++)
              for (let c = 0; c < k; c++) {
                let v = 0
                for (let j = 0; j < d; j++)
                  v += (((m.data[i * d + j] ?? 0) - (p.mean[j] ?? 0)) / (p.std[j] ?? 1)) * (p.components[c]?.[j] ?? 0)
                out[i * k + c] = v
              }
            return fromStorage(out, [m.n, k], 'float64')
          },
          featureImportances: p.coef.map(Math.abs),
        }
        return { ...base, score: withDefaultScore('regressor', base).score ?? ((): number => Number.NaN) }
      },
    },
    { nComponents: 1, fitIntercept: true, scale: false, ...hyper },
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Reduced-rank regression

/** How the outputs are weighted in the rank constraint. */
export type RrrWeighting = 'identity' | 'covariance'

/** The hyperparameters of {@link reducedRankRegression}. */
export interface RrrHyper {
  /** The rank of the coefficient matrix. */
  readonly rank: number
  /**
   * `identity`: minimise the unweighted squared error subject to the rank (Anderson, Izenman). `covariance`: weight the
   * outputs by their inverse covariance, which makes the latent directions the canonical variates.
   */
  readonly weighting: RrrWeighting
  /** A ridge penalty on the full-rank fit before the rank constraint (default 0). */
  readonly alpha: number
  readonly fitIntercept: boolean
}

/** The fitted parameters of {@link reducedRankRegression}. */
export type RrrParams = {
  /** The rank-constrained coefficients, `[d][k]`. */
  readonly coef: number[][]
  readonly intercept: number[]
  /** The full-rank (ridge) least-squares coefficients, `[d][k]`. */
  readonly fullCoef: number[][]
  readonly rank: number
  /** The singular values of the weighted fitted values: the rank constraint keeps the leading `rank`. */
  readonly singularValues: number[]
  /** The input loadings `A`, `[d][rank]`: `coef = A C^T`, the scores of `x` being `x A`. */
  readonly inputLoadings: number[][]
  /** The output loadings `C`, `[k][rank]`. */
  readonly outputLoadings: number[][]
  readonly mean: number[]
}

/** What a fitted reduced-rank regression offers. */
export interface RrrVerbs {
  readonly nFeaturesIn: number
  /** The predicted outputs, `[n, k]`. */
  readonly predict: (x: TensorLike) => Tensor
  /** The latent scores `(x - mean) A`, `[n, rank]`. */
  readonly transform: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  /** The uniform-average R-squared over the outputs. */
  readonly score: (data: LearnData) => number
}

/** The symmetric square root and inverse square root of the output covariance, by its SVD. */
function covarianceRoots(yc: Float64Array, n: number, k: number): { root: number[][]; inverse: number[][] } {
  const { s, vt } = thinSvd(yc, n, k)
  const top = s[0] ?? 0
  const build = (power: number): number[][] =>
    Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => {
        let v = 0
        for (let c = 0; c < s.length; c++) {
          const sc = s[c] ?? 0
          if (!(sc > 1e-12 * top)) continue
          v += (vt[c]?.[i] ?? 0) * (vt[c]?.[j] ?? 0) * (sc / Math.sqrt(n)) ** power
        }
        return v
      }),
    )
  return { root: build(1), inverse: build(-1) }
}

/**
 * Reduced-rank regression: the multi-output least-squares fit `Y ~ X B` whose coefficient matrix `B` has at most the
 * given rank. It is the full least-squares (or ridge) fit projected onto the leading right singular directions of its
 * fitted values, `B_rr = B_ols V_r V_r^T`, which is the best rank-`r` fit in the squared error (the Eckart-Young bound:
 * the extra error is the sum of the discarded squared singular values). `weighting: 'covariance'` weights the outputs by
 * their inverse covariance, which gives the canonical-correlation form. Rank equal to the number of outputs is
 * ordinary least squares.
 *
 * @param hyper - `rank` (default 1), `weighting`, `alpha`, `fitIntercept`.
 * @returns The estimator; the targets are an `[n, k]` matrix.
 * @reference T. W. Anderson, Estimating linear restrictions on regression coefficients for multivariate normal
 * distributions, Ann. Math. Statist. 22 (1951); A. J. Izenman, Reduced-rank regression for the multivariate linear
 * model, J. Multivariate Anal. 5 (1975).
 * @example
 * ```ts
 * import { reducedRankRegression } from 'mdml-engine/decomposition'
 * const x = [[0, 1], [1, 0], [2, 1], [3, 2], [4, 1], [5, 3]]
 * const y = x.map((r) => [(r[0] ?? 0) + (r[1] ?? 0), 2 * ((r[0] ?? 0) + (r[1] ?? 0))])
 * const f = reducedRankRegression({ rank: 1 }).fit({ x, y })
 * f.params.rank // => 1
 * ```
 */
export function reducedRankRegression(
  hyper: Partial<RrrHyper> = {},
): Estimator<RrrHyper, Fitted<RrrParams> & RrrVerbs> {
  return defineEstimator<RrrHyper, RrrParams, RrrVerbs>(
    {
      name: 'reduced-rank-regression',
      kind: 'regressor',
      hyperSpace: (h) => dict({ rank: int(1, 20, { default: h.rank }), alpha: real(0, 10, { default: h.alpha }) }),
      fit: (data, h) => {
        if (data.weights !== undefined)
          throw new MdmlError('value', 'reducedRankRegression: sample weights are not supported')
        const m = matrixView(data.x, 'reducedRankRegression')
        const yt = asTensor(requireTargets(data, 'reducedRankRegression'))
        const k = yt.shape.length === 1 ? 1 : (yt.shape[1] ?? 1)
        if ((yt.shape[0] ?? 0) !== m.n)
          throw new MdmlError('shape', `reducedRankRegression: ${yt.shape[0]} targets for ${m.n} rows`)
        if (!Number.isInteger(h.rank) || h.rank < 1 || h.rank > Math.min(m.d, k))
          throw new MdmlError(
            'value',
            `reducedRankRegression: the rank must be an integer in [1, ${Math.min(m.d, k)}], got ${h.rank}`,
          )
        if (!(h.alpha >= 0))
          throw new MdmlError('value', `reducedRankRegression: alpha must be non-negative, got ${h.alpha}`)
        const yFlat = toFlat(yt)
        const xMean = h.fitIntercept ? columnMeans(m) : new Array<number>(m.d).fill(0)
        const yMean = Array.from({ length: k }, (_, c) => {
          if (!h.fitIntercept) return 0
          let s = 0
          for (let i = 0; i < m.n; i++) s += yFlat[i * k + c] ?? 0
          return s / m.n
        })
        const xc = centred(m, xMean)
        const yc = Float64Array.from(yFlat, (v, idx) => v - (yMean[idx % k] ?? 0))
        const X = toRows(xc, m.n, m.d)
        const Y = toRows(yc, m.n, k)
        // The ridge least-squares fit: B = (X'X + alpha I)^+ X'Y.
        const xt = tr(X)
        const gram = mm(xt, X)
        for (let j = 0; j < m.d; j++) (gram[j] as number[])[j] = (gram[j]?.[j] ?? 0) + h.alpha
        const full = mm(pinvRows(gram), mm(xt, Y))
        const fitted = mm(X, full)
        // The output weight Gamma^{1/2} (the inverse square root of the output covariance, or the identity) and its inverse.
        const roots = h.weighting === 'covariance' ? covarianceRoots(yc, m.n, k) : undefined
        const weighted = roots === undefined ? fitted : mm(fitted, roots.inverse)
        const { s, vt } = thinSvd(Float64Array.from(weighted.flat()), m.n, k)
        const vr = vt.slice(0, h.rank) // [rank][k]
        // B_rr = B_full Gamma^{1/2} V_r V_r^T Gamma^{-1/2} = left outMap.
        const outMap = roots === undefined ? vr : mm(vr, roots.root) // [rank][k]
        const left = mm(full, roots === undefined ? tr(vr) : mm(roots.inverse, tr(vr))) // [d][rank]
        const coef = mm(left, outMap) // [d][k]
        const intercept = yMean.map((v, c) => v - xMean.reduce((a, xm, j) => a + xm * (coef[j]?.[c] ?? 0), 0))
        return {
          coef,
          intercept,
          fullCoef: full,
          rank: h.rank,
          singularValues: s,
          inputLoadings: left,
          outputLoadings: tr(outMap),
          mean: xMean,
        }
      },
      methods: (p) => {
        const d = p.coef.length
        const k = p.intercept.length
        const rankOf = p.rank
        const predictRows = (x: TensorLike): number[][] => {
          const m = matrixView(x, 'reducedRankRegression', { nFeatures: d })
          return Array.from({ length: m.n }, (_, i) =>
            Array.from({ length: k }, (_, c) => {
              let v = p.intercept[c] ?? 0
              for (let j = 0; j < d; j++) v += (m.data[i * d + j] ?? 0) * (p.coef[j]?.[c] ?? 0)
              return v
            }),
          )
        }
        return {
          nFeaturesIn: d,
          predict: (x) => matrixTensor(predictRows(x), k),
          transform: (x) => {
            const m = matrixView(x, 'reducedRankRegression', { nFeatures: d })
            const out = Array.from({ length: m.n }, (_, i) =>
              Array.from({ length: rankOf }, (_, c) => {
                let v = 0
                for (let j = 0; j < d; j++)
                  v += ((m.data[i * d + j] ?? 0) - (p.mean[j] ?? 0)) * (p.inputLoadings[j]?.[c] ?? 0)
                return v
              }),
            )
            return matrixTensor(out, rankOf)
          },
          featureImportances: p.coef.map((row) => row.reduce((a, v) => a + Math.abs(v), 0)),
          score: (data) => {
            const yt = toFlat(asTensor(requireTargets(data, 'reducedRankRegression score')))
            const pred = predictRows(data.x)
            const n = pred.length
            let total = 0
            for (let c = 0; c < k; c++) {
              let mean = 0
              for (let i = 0; i < n; i++) mean += (yt[i * k + c] ?? 0) / n
              let res = 0
              let tot = 0
              for (let i = 0; i < n; i++) {
                res += ((yt[i * k + c] ?? 0) - (pred[i]?.[c] ?? 0)) ** 2
                tot += ((yt[i * k + c] ?? 0) - mean) ** 2
              }
              total += tot === 0 ? 0 : 1 - res / tot
            }
            return total / k
          },
        }
      },
    },
    { rank: 1, weighting: 'identity', alpha: 0, fitIntercept: true, ...hyper },
  )
}
