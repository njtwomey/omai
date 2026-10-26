/**
 * The design shared by the linear models: features, targets and weights read once, weighted centring, the intercept
 * that goes with centred coefficients, linear prediction, and the log grid of penalties.
 */

import { type LearnData, type MatrixView, matrixView, sampleWeightsOf, targetVector } from '../learn/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'

/** Features, targets and sample weights of a fit. */
export interface Design {
  readonly x: MatrixView
  readonly y: Float64Array
  readonly w: Float64Array
}

/** Read the design of a supervised linear fit. */
export function designOf(data: LearnData, who: string): Design {
  const x = matrixView(data.x, who)
  const y = targetVector(data.y ?? [], who, x.n)
  return { x, y, w: sampleWeightsOf(data.weights, x.n, who) }
}

/** Weighted means of the columns and of the targets (zeros when there is no intercept). */
export interface Centred {
  /** sqrt(w) (x - xMean), row-major. */
  readonly x: Float64Array
  /** sqrt(w) (y - yMean). */
  readonly y: Float64Array
  readonly xMean: Float64Array
  readonly yMean: number
}

/** Centre (when fitting an intercept) and scale rows by sqrt(w), so weighted least squares becomes ordinary. */
export function centre(design: Design, fitIntercept: boolean, y: ArrayLike<number> = design.y): Centred {
  const { x, w } = design
  const sw = w.reduce((a, b) => a + b, 0)
  const xMean = new Float64Array(x.d)
  let yMean = 0
  if (fitIntercept && sw > 0) {
    for (let i = 0; i < x.n; i++) {
      const wi = w[i] ?? 0
      for (let j = 0; j < x.d; j++) xMean[j] = (xMean[j] ?? 0) + (wi * (x.data[i * x.d + j] ?? 0)) / sw
      yMean += (wi * (y[i] ?? 0)) / sw
    }
  }
  const xc = new Float64Array(x.n * x.d)
  const yc = new Float64Array(x.n)
  for (let i = 0; i < x.n; i++) {
    const r = Math.sqrt(w[i] ?? 0)
    for (let j = 0; j < x.d; j++) xc[i * x.d + j] = r * ((x.data[i * x.d + j] ?? 0) - (xMean[j] ?? 0))
    yc[i] = r * ((y[i] ?? 0) - yMean)
  }
  return { x: xc, y: yc, xMean, yMean }
}

/** The intercept that goes with coefficients fitted on centred data. */
export function interceptFor(c: Centred, coef: ArrayLike<number>): number {
  return c.yMean - ddot(c.xMean, coef)
}

/** Linear predictor x . coef + intercept for each row of new data. */
export function linearPredictor(x: TensorLike, coef: readonly number[], intercept: number, who: string): Float64Array {
  const m = matrixView(x, who, { nFeatures: coef.length })
  const out = new Float64Array(m.n)
  for (let i = 0; i < m.n; i++) out[i] = intercept + ddot(m.data.subarray(i * m.d, (i + 1) * m.d), coef)
  return out
}

/** Linear predictor of the design's own rows. */
export function designPredictor(design: Design, coef: ArrayLike<number>, intercept: number): Float64Array {
  const { n, d, data } = design.x
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = intercept + ddot(data.subarray(i * d, (i + 1) * d), coef)
  return out
}

/** A float64 vector tensor. */
export function vectorTensor(v: ArrayLike<number>): Tensor {
  return fromStorage(Float64Array.from(v), [v.length], 'float64')
}

/** A float64 matrix tensor from row-major data. */
export function matrixTensor(data: ArrayLike<number>, rows: number, cols: number): Tensor {
  return fromStorage(Float64Array.from(data), [rows, cols], 'float64')
}

/** Log-spaced penalties from `max` down to `max * ratio` (sklearn's alpha grid). */
export function logGrid(max: number, ratio: number, n: number): number[] {
  if (n === 1) return [max]
  return Array.from({ length: n }, (_, i) => max * ratio ** (i / (n - 1)))
}
