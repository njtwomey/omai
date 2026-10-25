/**
 * Cross-decomposition of two blocks of variables, as in scikit-learn's `cross_decomposition`: PLS regression (latent
 * directions of the inputs that best predict the outputs, by NIPALS), PLS canonical, canonical correlation analysis
 * (directions of maximal correlation between the blocks) and PLS-SVD, with fit functions and class wrappers.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { componentsRecord } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  featureNamesOf,
  type LearnData,
  matrixView,
  requireTargets,
} from '../../lib/learn/index.ts'
import { r2Score } from '../../lib/metrics/index.ts'
import { dict, int } from '../../lib/space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { mm, pinvRows, thinSvd, tr } from './_linear.ts'

/** Which PLS: regression (PLS2), canonical (PLS mode A), CCA (mode B), or the SVD of the cross-covariance. */
export type PlsVariant = 'regression' | 'canonical' | 'cca' | 'svd'

/** The parameters of a fitted cross-decomposition. */
export type PlsParams = {
  readonly variant: PlsVariant
  readonly xMean: number[]
  readonly yMean: number[]
  readonly xStd: number[]
  readonly yStd: number[]
  /** weights[f][k], loadings[f][k] and rotations[f][k], as sklearn's `x_weights_` and friends. */
  readonly xWeights: number[][]
  readonly yWeights: number[][]
  readonly xLoadings: number[][]
  readonly yLoadings: number[][]
  readonly xRotations: number[][]
  readonly yRotations: number[][]
  /** coef[t][f]: predictions are (x - xMean) coef^T + yMean (regression variants). */
  readonly coef: number[][]
}

/** The verbs of a cross-decomposition: x scores, and for the PLS variants predictions and R^2. */
export interface PlsVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly predict?: (x: TensorLike) => Tensor
  readonly score?: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/** A fitted cross-decomposition. */
export type FittedPls = ExplainedFitted<PlsParams, PlsVerbs>

const EPS = Number.EPSILON
const dot = (a: readonly number[], b: readonly number[]): number => a.reduce((s, v, i) => s + v * (b[i] ?? 0), 0)
const column = (m: readonly (readonly number[])[], j: number): number[] => m.map((r) => r[j] ?? 0)
const matVec = (m: readonly (readonly number[])[], v: readonly number[]): number[] => m.map((r) => dot(r, v))
const tMatVec = (m: readonly (readonly number[])[], v: readonly number[]): number[] =>
  (m[0] ?? []).map((_, j) => m.reduce((s, r, i) => s + (r[j] ?? 0) * (v[i] ?? 0), 0))

/** sklearn's `_center_scale_xy`: centre, and divide by the ddof-1 standard deviation (zeros become one). */
function standardise(rows: number[][], scale: boolean): { z: number[][]; mean: number[]; std: number[] } {
  const n = rows.length
  const d = rows[0]?.length ?? 0
  const mean = Array.from({ length: d }, (_, j) => rows.reduce((a, r) => a + (r[j] ?? 0), 0) / n)
  const std = Array.from({ length: d }, (_, j) => {
    if (!scale) return 1
    const s = Math.sqrt(rows.reduce((a, r) => a + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / (n - 1))
    return s === 0 ? 1 : s
  })
  return { z: rows.map((r) => r.map((v, j) => (v - (mean[j] ?? 0)) / (std[j] ?? 1))), mean, std }
}

/** The first pair of singular vectors of X^T Y by the NIPALS power method (sklearn's, modes A and B). */
function powerMethod(
  x: number[][],
  y: number[][],
  mode: 'A' | 'B',
  normYWeights: boolean,
  maxIter: number,
  tol: number,
): { xw: number[]; yw: number[] } {
  const q = y[0]?.length ?? 0
  let yScore = Array.from({ length: q }, (_, j) => column(y, j)).find((c) => c.some((v) => Math.abs(v) > EPS))
  if (yScore === undefined) throw new WhiteboxError('value', 'pls: the y residual is constant')
  const xPinv = mode === 'B' ? pinvRows(x) : undefined
  const yPinv = mode === 'B' ? pinvRows(y) : undefined
  let xw: number[] = []
  let yw: number[] = []
  let old: number[] | undefined
  for (let it = 0; it < maxIter; it++) {
    xw =
      xPinv !== undefined
        ? matVec(xPinv, yScore)
        : tMatVec(x, yScore).map((v) => v / dot(yScore as number[], yScore as number[]))
    const xn = Math.sqrt(dot(xw, xw)) + EPS
    xw = xw.map((v) => v / xn)
    const xScore = matVec(x, xw)
    yw = yPinv !== undefined ? matVec(yPinv, xScore) : tMatVec(y, xScore).map((v) => v / dot(xScore, xScore))
    if (normYWeights) {
      const yn = Math.sqrt(dot(yw, yw)) + EPS
      yw = yw.map((v) => v / yn)
    }
    const yss = dot(yw, yw) + EPS
    yScore = matVec(y, yw).map((v) => v / yss)
    const diff =
      old === undefined ? Number.POSITIVE_INFINITY : xw.reduce((s, v, i) => s + (v - (old?.[i] ?? 0)) ** 2, 0)
    if (diff < tol || q === 1) break
    old = xw
  }
  return { xw, yw }
}

/** Hyperparameters of the cross-decompositions. */
export type PlsHyper = {
  readonly nComponents?: number
  /** Standardise X and Y (default true). */
  readonly scale?: boolean
  readonly maxIter?: number
  readonly tol?: number
}

function fitPls(variant: PlsVariant, data: LearnData, h: PlsHyper): PlsParams {
  const xm = matrixView(data.x, 'pls', { minSamples: 2 })
  const yt = asTensor(requireTargets(data, 'pls'))
  const yRows =
    yt.shape.length === 1
      ? toFlat(yt).map((v) => [v])
      : Array.from({ length: yt.shape[0] ?? 0 }, (_, i) =>
          toFlat(yt).slice(i * (yt.shape[1] ?? 1), (i + 1) * (yt.shape[1] ?? 1)),
        )
  const xRows = Array.from({ length: xm.n }, (_, i) => Array.from(xm.data.subarray(i * xm.d, (i + 1) * xm.d)))
  const k = h.nComponents ?? 2
  const p = xm.d
  const q = yRows[0]?.length ?? 1
  const limit = variant === 'regression' ? p : Math.min(xm.n, p, q)
  if (!(Number.isInteger(k) && k >= 1 && k <= limit))
    throw new WhiteboxError('value', `pls: nComponents must be in 1..${limit}`)
  const scale = h.scale ?? true
  const X = standardise(xRows, scale)
  const Y = standardise(yRows, scale)
  const xWeights: number[][] = []
  const yWeights: number[][] = []
  const xLoadings: number[][] = []
  const yLoadings: number[][] = []
  if (variant === 'svd') {
    const c = mm(tr(X.z), Y.z)
    const { u, vt } = thinSvd(Float64Array.from(c.flat()), p, q)
    for (let j = 0; j < k; j++) {
      let xw = column(u, j)
      let yw = vt[j] ?? []
      const big = xw.reduce((b, v, i) => (Math.abs(v) > Math.abs(xw[b] ?? 0) ? i : b), 0)
      const sign = Math.sign(xw[big] ?? 1) || 1
      xw = xw.map((v) => v * sign)
      yw = yw.map((v) => v * sign)
      xWeights.push(xw)
      yWeights.push(yw)
    }
    return {
      variant,
      xMean: X.mean,
      yMean: Y.mean,
      xStd: X.std,
      yStd: Y.std,
      xWeights: tr(xWeights),
      yWeights: tr(yWeights),
      xLoadings: [],
      yLoadings: [],
      xRotations: tr(xWeights),
      yRotations: tr(yWeights),
      coef: [],
    }
  }
  let xk = X.z.map((r) => [...r])
  let yk = Y.z.map((r) => [...r])
  const mode = variant === 'cca' ? 'B' : 'A'
  const normY = variant !== 'regression'
  for (let j = 0; j < k; j++) {
    for (let c = 0; c < q; c++) if (yk.every((r) => Math.abs(r[c] ?? 0) < 10 * EPS)) for (const r of yk) r[c] = 0
    let { xw, yw } = powerMethod(xk, yk, mode, normY, h.maxIter ?? 500, h.tol ?? 1e-6)
    const big = xw.reduce((b, v, i) => (Math.abs(v) > Math.abs(xw[b] ?? 0) ? i : b), 0)
    const sign = Math.sign(xw[big] ?? 1) || 1
    xw = xw.map((v) => v * sign)
    yw = yw.map((v) => v * sign)
    const xs = matVec(xk, xw)
    const yss = normY ? 1 : dot(yw, yw)
    const ys = matVec(yk, yw).map((v) => v / yss)
    const xl = tMatVec(xk, xs).map((v) => v / dot(xs, xs))
    xk = xk.map((r, i) => r.map((v, f) => v - (xs[i] ?? 0) * (xl[f] ?? 0)))
    const yl =
      variant === 'regression'
        ? tMatVec(yk, xs).map((v) => v / dot(xs, xs))
        : tMatVec(yk, ys).map((v) => v / dot(ys, ys))
    const by = variant === 'regression' ? xs : ys
    yk = yk.map((r, i) => r.map((v, t) => v - (by[i] ?? 0) * (yl[t] ?? 0)))
    xWeights.push(xw)
    yWeights.push(yw)
    xLoadings.push(xl)
    yLoadings.push(yl)
  }
  const xW = tr(xWeights)
  const yW = tr(yWeights)
  const xL = tr(xLoadings)
  const yL = tr(yLoadings)
  const xRotations = mm(xW, pinvRows(mm(tr(xL), xW)))
  const yRotations = mm(yW, pinvRows(mm(tr(yL), yW)))
  const coef = tr(mm(xRotations, tr(yL)).map((row) => row.map((v, t) => v * (Y.std[t] ?? 1)))).map((row) =>
    row.map((v, f) => v / (X.std[f] ?? 1)),
  )
  return {
    variant,
    xMean: X.mean,
    yMean: Y.mean,
    xStd: X.std,
    yStd: Y.std,
    xWeights: xW,
    yWeights: yW,
    xLoadings: xL,
    yLoadings: yL,
    xRotations,
    yRotations,
    coef,
  }
}

function plsVerbs(p: PlsParams): PlsVerbs {
  const d = p.xMean.length
  const rows = (x: TensorLike): number[][] => {
    const m = matrixView(x, 'pls', { nFeatures: d })
    return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * d, (i + 1) * d)))
  }
  const toTensor = (r: number[][]): Tensor =>
    fromStorage(Float64Array.from(r.flat()), [r.length, r[0]?.length ?? 0], 'float64')
  const predict = (x: TensorLike): number[][] =>
    rows(x).map((r) =>
      p.coef.map(
        (c, t) =>
          dot(
            c,
            r.map((v, f) => v - (p.xMean[f] ?? 0)),
          ) + (p.yMean[t] ?? 0),
      ),
    )
  const regression = p.coef.length > 0
  return {
    transform: (x) =>
      toTensor(
        mm(
          rows(x).map((r) => r.map((v, f) => (v - (p.xMean[f] ?? 0)) / (p.xStd[f] ?? 1))),
          p.xRotations,
        ),
      ),
    ...(regression
      ? {
          predict: (x: TensorLike) => {
            const y = predict(x)
            return p.yMean.length === 1
              ? fromStorage(Float64Array.from(y.map((r) => r[0] ?? 0)), [y.length], 'float64')
              : toTensor(y)
          },
          score: (data: LearnData) => {
            const y = predict(data.x)
            const truth = toFlat(asTensor(requireTargets(data, 'pls score')))
            return r2Score(truth, y.flat()).estimate.value
          },
        }
      : {}),
    nFeaturesIn: d,
  }
}

function plsEstimator(variant: PlsVariant, name: string, method: string) {
  return (hyper: PlsHyper = {}): Estimator<PlsHyper, FittedPls> =>
    defineExplainedEstimator<PlsHyper, PlsParams, PlsVerbs>(
      {
        name,
        kind: variant === 'regression' ? 'regressor' : 'transformer',
        hyperSpace: (h) => dict({ nComponents: int(1, 20, { default: h.nComponents ?? 2, label: 'components' }) }),
        fit: (data, h) => {
          const params = fitPls(variant, data, h)
          const d = params.xMean.length
          const scores = toFlat(plsVerbs(params).transform(data.x))
          const k = params.xRotations[0]?.length ?? 0
          return {
            params,
            exposition: [
              componentsRecord({
                method,
                features: featureNamesOf(data, d),
                components: tr(params.xWeights),
                scores: Array.from({ length: scores.length / Math.max(1, k) }, (_, i) =>
                  scores.slice(i * k, (i + 1) * k),
                ),
              }),
            ],
          }
        },
        methods: plsVerbs,
      },
      hyper,
    )
}

/**
 * PLS regression (sklearn.cross_decomposition.PLSRegression, PLS2 by NIPALS with regression deflation). Its exposition
 * is a `components` record of the x weights and x scores.
 *
 * @param hyper - Components, scaling, iteration cap and tolerance.
 * @returns The estimator.
 */
export const plsRegression: (hyper?: PlsHyper) => Estimator<PlsHyper, FittedPls> = plsEstimator(
  'regression',
  'pls-regression',
  'plsRegression',
)

/**
 * PLS canonical (sklearn.cross_decomposition.PLSCanonical, mode A with symmetric deflation).
 *
 * @param hyper - Components, scaling, iteration cap and tolerance.
 * @returns The estimator.
 */
export const plsCanonical: (hyper?: PlsHyper) => Estimator<PlsHyper, FittedPls> = plsEstimator(
  'canonical',
  'pls-canonical',
  'plsCanonical',
)

/**
 * Canonical correlation analysis (sklearn.cross_decomposition.CCA, NIPALS mode B).
 *
 * @param hyper - Components, scaling, iteration cap and tolerance.
 * @returns The estimator.
 */
export const cca: (hyper?: PlsHyper) => Estimator<PlsHyper, FittedPls> = plsEstimator('cca', 'cca', 'cca')

/**
 * PLS-SVD (sklearn.cross_decomposition.PLSSVD): the leading singular vectors of the standardised cross-covariance.
 *
 * @param hyper - Components and scaling.
 * @returns The estimator.
 */
export const plsSvd: (hyper?: PlsHyper) => Estimator<PlsHyper, FittedPls> = plsEstimator('svd', 'pls-svd', 'plsSvd')

/** PLS regression as a class. */
export class PLSRegression extends EstimatorClass<PlsHyper, FittedPls> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link plsRegression}.
   */
  constructor(hyper: PlsHyper = {}) {
    super(plsRegression(hyper))
  }
}

/** CCA as a class. */
export class CCA extends EstimatorClass<PlsHyper, FittedPls> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link cca}.
   */
  constructor(hyper: PlsHyper = {}) {
    super(cca(hyper))
  }
}
