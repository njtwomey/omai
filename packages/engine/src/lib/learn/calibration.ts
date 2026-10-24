import { type Json, WhiteboxError } from '../base/index.ts'
import { interp } from '../numerics/interpolate/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { fminbound, lbfgs } from '../optim/index.ts'
import { foldIn } from '../random/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  logSoftmax,
  mul,
  neg,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { innerFitOptions, keyFor } from './_shared.ts'
import { type Estimator, type FitOptions, type Fitted, type LearnData, requireTargets } from './estimator.ts'
import { type CvOption, checkCv, crossValPredict, splitInputOf } from './model-selection/cross-validation.ts'
import { splitIndices } from './model-selection/splitters.ts'
import { encodeLabels, matrixView, sampleWeightsOf, takeData, targetVector } from './validation.ts'

/** A fitted calibration map from scores to probabilities: its method, its parameters as data, and the map. */
export interface Calibrator {
  readonly method: CalibrationMethod
  readonly params: Json
  /** Calibrated probabilities: `[n]` (positive class) for binary maps, `[n, k]` for multiclass maps. */
  readonly predict: (scores: TensorLike) => Tensor
}

/** The calibration methods. */
export type CalibrationMethod = 'sigmoid' | 'isotonic' | 'beta' | 'histogram' | 'temperature' | 'vector'

/** Options of the binary calibrators: sample weights. */
export interface CalibratorOptions {
  readonly weights?: TensorLike
}

function vector(x: TensorLike, who: string): Float64Array {
  return targetVector(x, who)
}

function binaryTargets(y: TensorLike, n: number, who: string): Float64Array {
  const t = targetVector(y, who, n)
  for (const v of t) if (v !== 0 && v !== 1) throw new WhiteboxError('value', `${who}: labels must be 0 or 1`)
  return t
}

/**
 * The weighted pool-adjacent-violators algorithm (Ayer et al. 1955; sklearn's contiguous isotonic regression): the
 * non-decreasing sequence closest to y in weighted least squares, by merging adjacent violating blocks into their
 * weighted means. Exact, O(n).
 *
 * @param y - The values, in order.
 * @param weights - Their weights (default ones).
 * @returns The fitted non-decreasing values.
 * @concept isotonic-regression
 * @example
 * ```ts
 * import { poolAdjacentViolators } from 'whitebox-engine/learn'
 * poolAdjacentViolators([1, 3, 2, 4]) // => [1, 2.5, 2.5, 4]
 * ```
 */
export function poolAdjacentViolators(y: ArrayLike<number>, weights?: ArrayLike<number>): number[] {
  const n = y.length
  const v = Float64Array.from(y)
  const w = weights === undefined ? new Float64Array(n).fill(1) : Float64Array.from(weights)
  const target = Int32Array.from({ length: n }, (_, i) => i)
  let i = 0
  while (i < n) {
    let k = (target[i] ?? i) + 1
    if (k === n) break
    if ((v[i] ?? 0) < (v[k] ?? 0)) {
      i = k
      continue
    }
    let sumWy = (w[i] ?? 0) * (v[i] ?? 0)
    let sumW = w[i] ?? 0
    for (;;) {
      const prev = v[k] ?? 0
      sumWy += (w[k] ?? 0) * (v[k] ?? 0)
      sumW += w[k] ?? 0
      k = (target[k] ?? k) + 1
      if (k === n || prev < (v[k] ?? 0)) {
        v[i] = sumWy / sumW
        w[i] = sumW
        target[i] = k - 1
        target[k - 1] = i
        if (i > 0) i = target[i - 1] ?? 0
        break
      }
    }
  }
  i = 0
  while (i < n) {
    const k = (target[i] ?? i) + 1
    for (let j = i + 1; j < k; j++) v[j] = v[i] ?? 0
    i = k
  }
  return Array.from(v)
}

/** A fitted isotonic map: thresholds and values, interpolated linearly and clipped at the ends. */
export type IsotonicParams = { readonly x: number[]; readonly y: number[] }

function isotonicFit(scores: Float64Array, targets: Float64Array, weights: Float64Array): IsotonicParams {
  const order = Array.from(scores.keys()).sort(
    (a, b) => (scores[a] ?? 0) - (scores[b] ?? 0) || (targets[a] ?? 0) - (targets[b] ?? 0),
  )
  const xs: number[] = []
  const ys: number[] = []
  const ws: number[] = []
  const eps = 1e-15
  let cx = scores[order[0] ?? 0] ?? 0
  let cy = 0
  let cw = 0
  for (const j of order) {
    const xj = scores[j] ?? 0
    if (xj - cx >= eps) {
      xs.push(cx)
      ws.push(cw)
      ys.push(cy / cw)
      cx = xj
      cw = weights[j] ?? 0
      cy = (targets[j] ?? 0) * (weights[j] ?? 0)
    } else {
      cw += weights[j] ?? 0
      cy += (targets[j] ?? 0) * (weights[j] ?? 0)
    }
  }
  xs.push(cx)
  ws.push(cw)
  ys.push(cy / cw)
  const fitted = poolAdjacentViolators(ys, ws)
  // Keep the end points and the points where the fitted value changes (sklearn's trim_duplicates).
  const keep = fitted.map((v, i) => i === 0 || i === fitted.length - 1 || v !== fitted[i - 1] || v !== fitted[i + 1])
  return { x: xs.filter((_, i) => keep[i]), y: fitted.filter((_, i) => keep[i]) }
}

function isotonicPredict(p: IsotonicParams): (scores: TensorLike) => Tensor {
  const lo = p.x[0] ?? 0
  const hi = p.x[p.x.length - 1] ?? 0
  return (scores) => {
    const s = vector(scores, 'isotonic').map((v) => Math.min(Math.max(v, lo), hi))
    if (p.x.length === 1)
      return fromStorage(
        Float64Array.from(s, () => p.y[0] ?? 0),
        [s.length],
        'float64',
      )
    return interp(s, p.x, p.y) as Tensor
  }
}

/**
 * Isotonic calibration (Zadrozny and Elkan 2002; sklearn IsotonicRegression with clipping): the non-decreasing step
 * map from scores to the observed frequencies that fits best in least squares, interpolated linearly between its
 * thresholds and clipped beyond them.
 *
 * @param scores - Classifier scores `[n]` (decision values or probabilities).
 * @param labels - Binary labels `[n]` (0 or 1).
 * @param options - Sample weights.
 * @returns The calibrator.
 * @example
 * ```ts
 * import { isotonicCalibration } from 'whitebox-engine/learn'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const c = isotonicCalibration([0.1, 0.4, 0.35, 0.8], [0, 1, 0, 1])
 * toFlat(c.predict([0.1, 0.375, 0.9])) // => [0, 0.5, 1]
 * ```
 */
export function isotonicCalibration(
  scores: TensorLike,
  labels: TensorLike,
  options: CalibratorOptions = {},
): Calibrator {
  const s = vector(scores, 'isotonicCalibration')
  const t = binaryTargets(labels, s.length, 'isotonicCalibration')
  const params = isotonicFit(s, t, sampleWeightsOf(options.weights, s.length, 'isotonicCalibration'))
  return { method: 'isotonic', params, predict: isotonicPredict(params) }
}

/** Parameters of a Platt map p = 1 / (1 + exp(a s + b)). */
export type PlattParams = { readonly a: number; readonly b: number }

const sigmoid = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))

/** Newton's method for a weighted logistic fit of targets t on features (with an intercept as the last column). */
function logisticNewton(
  features: readonly Float64Array[],
  t: Float64Array,
  w: Float64Array,
  init: readonly number[],
  who: string,
): number[] {
  const p = features.length
  let beta = [...init]
  for (let iter = 0; iter < 100; iter++) {
    const g = new Float64Array(p)
    const h = new Float64Array(p * p)
    for (let i = 0; i < t.length; i++) {
      let z = 0
      for (let a = 0; a < p; a++) z += (beta[a] ?? 0) * (features[a]?.[i] ?? 0)
      const q = sigmoid(z)
      const r = (w[i] ?? 0) * (q - (t[i] ?? 0))
      const s = (w[i] ?? 0) * q * (1 - q)
      for (let a = 0; a < p; a++) {
        g[a] = (g[a] ?? 0) + r * (features[a]?.[i] ?? 0)
        for (let b = 0; b < p; b++)
          h[a * p + b] = (h[a * p + b] ?? 0) + s * (features[a]?.[i] ?? 0) * (features[b]?.[i] ?? 0)
      }
    }
    for (let a = 0; a < p; a++) h[a * p + a] = (h[a * p + a] ?? 0) + 1e-12
    const step = toFlat(solve(fromStorage(h, [p, p], 'float64'), fromStorage(g, [p], 'float64')))
    beta = beta.map((v, a) => v - (step[a] ?? 0))
    if (
      step.reduce((m, v) => Math.max(m, Math.abs(v)), 0) <
      1e-12 * (1 + beta.reduce((m, v) => Math.max(m, Math.abs(v)), 0))
    )
      return beta
    if (beta.some((v) => !Number.isFinite(v)))
      throw new WhiteboxError('convergence', `${who}: the fit diverged (separable scores?)`)
  }
  return beta
}

/**
 * Platt scaling (Platt 1999; sklearn's sigmoid calibration): p = 1 / (1 + exp(a s + b)) fitted by maximum likelihood
 * to Platt's smoothed targets (n+ + 1) / (n+ + 2) and 1 / (n- + 2), which keep the map away from 0 and 1. Solved by
 * Newton's method (a convex problem; sklearn uses L-BFGS-B, so the two agree to its tolerance).
 *
 * @param scores - Classifier scores `[n]`.
 * @param labels - Binary labels `[n]`.
 * @param options - Sample weights.
 * @returns The calibrator, with `a` and `b` in its parameters.
 * @concept probability-calibration
 * @example
 * ```ts
 * import { plattScaling } from 'whitebox-engine/learn'
 * const c = plattScaling([-2, -1, 1, 2], [0, 0, 1, 1])
 * c.params.a < 0 // => true
 * ```
 */
export function plattScaling(
  scores: TensorLike,
  labels: TensorLike,
  options: CalibratorOptions = {},
): Calibrator & { readonly params: PlattParams } {
  const s = vector(scores, 'plattScaling')
  const y = binaryTargets(labels, s.length, 'plattScaling')
  const w = sampleWeightsOf(options.weights, s.length, 'plattScaling')
  let prior0 = 0
  let prior1 = 0
  y.forEach((v, i) => {
    if (v > 0) prior1 += w[i] ?? 0
    else prior0 += w[i] ?? 0
  })
  const t = y.map((v) => (v > 0 ? (prior1 + 1) / (prior1 + 2) : 1 / (prior0 + 2)))
  // Fit p = sigmoid(-(a s + b)) as a logistic model in (-s, -1).
  const [na = 0, nb = 0] = logisticNewton(
    [s.map((v) => -v), new Float64Array(s.length).fill(-1)],
    t,
    w,
    [0, Math.log((prior0 + 1) / (prior1 + 1))],
    'plattScaling',
  )
  const params = { a: na, b: nb }
  return { method: 'sigmoid', params, predict: predictorFor('sigmoid', params) }
}

/** Parameters of a beta calibration map: logit(p) = a ln s - b ln(1 - s) + c. */
export type BetaParams = { readonly a: number; readonly b: number; readonly c: number }

/**
 * Beta calibration (Kull, Silva Filho and Flach 2017) of probability scores: logit(q) = a ln s - b ln(1 - s) + c, a
 * logistic regression on the two log features with a, b >= 0 (a negative coefficient is fixed at 0 and the rest
 * refitted). It contains Platt scaling of logits (a = b) and the identity (a = b = 1, c = 0).
 *
 * @param probabilities - Scores in (0, 1) `[n]` (clipped to [1e-12, 1 - 1e-12]).
 * @param labels - Binary labels `[n]`.
 * @param options - Sample weights.
 * @returns The calibrator.
 * @example
 * ```ts
 * import { betaCalibration } from 'whitebox-engine/learn'
 * const c = betaCalibration([0.1, 0.3, 0.6, 0.9, 0.2, 0.8], [0, 0, 1, 1, 1, 0])
 * c.params.a >= 0 && c.params.b >= 0 // => true
 * ```
 */
export function betaCalibration(
  probabilities: TensorLike,
  labels: TensorLike,
  options: CalibratorOptions = {},
): Calibrator & { readonly params: BetaParams } {
  const s = vector(probabilities, 'betaCalibration').map((v) => Math.min(Math.max(v, 1e-12), 1 - 1e-12))
  const y = binaryTargets(labels, s.length, 'betaCalibration')
  const w = sampleWeightsOf(options.weights, s.length, 'betaCalibration')
  const f1 = s.map(Math.log)
  const f2 = s.map((v) => -Math.log1p(-v))
  const ones = new Float64Array(s.length).fill(1)
  let [a = 0, b = 0, c = 0] = logisticNewton([f1, f2, ones], y, w, [0, 0, 0], 'betaCalibration')
  if (a < 0 || b < 0) {
    const keepA = a >= b
    const [coef = 0, intercept = 0] = logisticNewton([keepA ? f1 : f2, ones], y, w, [0, 0], 'betaCalibration')
    const fixed = Math.max(coef, 0)
    if (keepA) [a, b, c] = [fixed, 0, coef < 0 ? Math.log(weightedMean(y, w) / (1 - weightedMean(y, w))) : intercept]
    else [a, b, c] = [0, fixed, coef < 0 ? Math.log(weightedMean(y, w) / (1 - weightedMean(y, w))) : intercept]
  }
  const params = { a, b, c }
  return { method: 'beta', params, predict: predictorFor('beta', params) }
}

function weightedMean(y: Float64Array, w: Float64Array): number {
  let s = 0
  let sw = 0
  y.forEach((v, i) => {
    s += v * (w[i] ?? 0)
    sw += w[i] ?? 0
  })
  return s / sw
}

/** Parameters of a histogram-binning map: bin edges and each bin's calibrated probability. */
export type HistogramParams = { readonly edges: number[]; readonly values: number[] }

/**
 * Histogram binning (Zadrozny and Elkan 2001): split the score range into bins (equal width over [0, 1] for
 * probabilities, or equal mass at quantiles) and map each score to its bin's (weighted) positive rate; an empty bin
 * gets the overall positive rate.
 *
 * @param scores - Scores `[n]`.
 * @param labels - Binary labels `[n]`.
 * @param options - nBins (default 10), strategy (`'uniform'` over [0, 1], default, or `'quantile'`) and weights.
 * @returns The calibrator.
 * @example
 * ```ts
 * import { histogramBinning } from 'whitebox-engine/learn'
 * import { toFlat } from 'whitebox-engine/tensor'
 * const c = histogramBinning([0.1, 0.2, 0.7, 0.9], [0, 1, 1, 1], { nBins: 2 })
 * toFlat(c.predict([0.3, 0.8])) // => [0.5, 1]
 * ```
 */
export function histogramBinning(
  scores: TensorLike,
  labels: TensorLike,
  options: CalibratorOptions & { readonly nBins?: number; readonly strategy?: 'uniform' | 'quantile' } = {},
): Calibrator {
  const s = vector(scores, 'histogramBinning')
  const y = binaryTargets(labels, s.length, 'histogramBinning')
  const w = sampleWeightsOf(options.weights, s.length, 'histogramBinning')
  const k = options.nBins ?? 10
  if (!(Number.isInteger(k) && k >= 1)) throw new WhiteboxError('value', 'histogramBinning: nBins must be >= 1')
  const sorted = Float64Array.from(s).sort()
  const edges =
    options.strategy === 'quantile'
      ? Array.from({ length: k + 1 }, (_, i) => {
          const h = ((sorted.length - 1) * i) / k
          const lo = Math.floor(h)
          return (sorted[lo] ?? 0) + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)] ?? 0) - (sorted[lo] ?? 0))
        })
      : Array.from({ length: k + 1 }, (_, i) => i / k)
  const sums = new Float64Array(k)
  const counts = new Float64Array(k)
  s.forEach((v, i) => {
    const b = histogramBin(edges, k, v)
    sums[b] = (sums[b] ?? 0) + (y[i] ?? 0) * (w[i] ?? 0)
    counts[b] = (counts[b] ?? 0) + (w[i] ?? 0)
  })
  const overall = weightedMean(y, w)
  const params = { edges, values: Array.from(sums, (v, b) => ((counts[b] ?? 0) > 0 ? v / (counts[b] ?? 1) : overall)) }
  return { method: 'histogram', params, predict: predictorFor('histogram', params) }
}

/**
 * Logits `[n, k]` from decision values or probabilities, as sklearn's `_convert_to_logits`: rows of probabilities
 * become their logarithms, a binary decision value d becomes (-d, d), other scores pass through.
 *
 * @param scores - Scores `[n]` or `[n, k]`.
 * @returns The logits, row-major.
 * @example
 * ```ts
 * import { logitsOf } from 'whitebox-engine/learn'
 * Array.from(logitsOf([2]).data) // => [-2, 2]
 * ```
 */
export function logitsOf(scores: TensorLike): { readonly n: number; readonly k: number; readonly data: Float64Array } {
  const t = asTensor(scores)
  const m = matrixView(t.shape.length === 1 ? reshape(t, [-1, 1]) : t, 'logitsOf')
  if (m.d === 1) {
    const data = new Float64Array(m.n * 2)
    for (let i = 0; i < m.n; i++) {
      data[2 * i] = -(m.data[i] ?? 0)
      data[2 * i + 1] = m.data[i] ?? 0
    }
    return { n: m.n, k: 2, data }
  }
  const isProba =
    m.data.every((v) => v >= 0 && v <= 1) &&
    Array.from({ length: m.n }, (_, i) => {
      let s = 0
      for (let j = 0; j < m.d; j++) s += m.data[i * m.d + j] ?? 0
      return Math.abs(s - 1) <= 1e-8 + 1e-5
    }).every(Boolean)
  return { n: m.n, k: m.d, data: isProba ? m.data.map((v) => Math.log(v + 1e-12)) : m.data }
}

function softmaxRows(z: Float64Array, n: number, k: number): Float64Array {
  const out = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    let mx = Number.NEGATIVE_INFINITY
    for (let j = 0; j < k; j++) mx = Math.max(mx, z[i * k + j] ?? 0)
    let s = 0
    for (let j = 0; j < k; j++) s += Math.exp((z[i * k + j] ?? 0) - mx)
    for (let j = 0; j < k; j++) out[i * k + j] = Math.exp((z[i * k + j] ?? 0) - mx) / s
  }
  return out
}

function labelCodes(labels: TensorLike, n: number, k: number, who: string): Int32Array {
  const t = targetVector(labels, who, n)
  return Int32Array.from(t, (v) => {
    if (!(Number.isInteger(v) && v >= 0 && v < k))
      throw new WhiteboxError('value', `${who}: labels must be class indices in [0, ${k})`)
    return v
  })
}

/**
 * Temperature scaling (Guo et al. 2017; sklearn's temperature calibration): divide the logits by one temperature T
 * fitted by minimising the log loss (Brent's bounded search over log(1 / T) in [-10, 10]). Probabilities are turned
 * into logits by their logarithm; a binary decision value d becomes the logits (-d, d).
 *
 * @param scores - Logits or probabilities `[n, k]`, or binary decision values `[n]`.
 * @param labels - Class indices `[n]` in 0..k-1.
 * @param options - Sample weights.
 * @returns The calibrator (multiclass: it predicts `[n, k]`), with the inverse temperature `beta` in its parameters.
 * @example
 * ```ts
 * import { temperatureScaling } from 'whitebox-engine/learn'
 * const c = temperatureScaling([[4, 0], [3, 0], [0, 4], [3, 0]], [0, 0, 1, 1])
 * c.params.beta < 1 // => true
 * ```
 */
export function temperatureScaling(
  scores: TensorLike,
  labels: TensorLike,
  options: CalibratorOptions = {},
): Calibrator & { readonly params: { readonly beta: number } } {
  const z = logitsOf(scores)
  const y = labelCodes(labels, z.n, z.k, 'temperatureScaling')
  const w = sampleWeightsOf(options.weights, z.n, 'temperatureScaling')
  const sw = w.reduce((a, b) => a + b, 0)
  const loss = (logBeta: number): number => {
    const beta = Math.exp(logBeta)
    let total = 0
    for (let i = 0; i < z.n; i++) {
      let mx = Number.NEGATIVE_INFINITY
      for (let j = 0; j < z.k; j++) mx = Math.max(mx, beta * (z.data[i * z.k + j] ?? 0))
      let s = 0
      for (let j = 0; j < z.k; j++) s += Math.exp(beta * (z.data[i * z.k + j] ?? 0) - mx)
      total += (w[i] ?? 0) * (mx + Math.log(s) - beta * (z.data[i * z.k + (y[i] ?? 0)] ?? 0))
    }
    return total / sw
  }
  const r = fminbound(loss, { bounds: [-10, 10], tol: 64 * 2.220446049250313e-16, maxIterations: 500 })
  const params = { beta: Math.exp(r.x) }
  return { method: 'temperature', params, predict: predictorFor('temperature', params) }
}

/** Parameters of a vector-scaling map: a scale and a bias per class. */
export type VectorParams = { readonly scale: number[]; readonly bias: number[] }

/**
 * Vector scaling (Guo et al. 2017): softmax(w * z + b) with one scale and one bias per class, fitted by minimising the
 * log loss (L-BFGS on the convex objective); temperature scaling is the special case w = 1 / T, b = 0.
 *
 * @param scores - Logits or probabilities `[n, k]` (or binary decision values `[n]`).
 * @param labels - Class indices `[n]`.
 * @returns The calibrator.
 * @example
 * ```ts
 * import { vectorScaling } from 'whitebox-engine/learn'
 * const c = vectorScaling([[2, 0, 0], [0, 2, 0], [0, 0, 2], [2, 0, 0], [0, 0, 2], [0, 2, 0]], [0, 1, 2, 1, 2, 1])
 * c.params.scale.length // => 3
 * ```
 */
export function vectorScaling(scores: TensorLike, labels: TensorLike): Calibrator & { readonly params: VectorParams } {
  const z = logitsOf(scores)
  const y = labelCodes(labels, z.n, z.k, 'vectorScaling')
  const onehot = new Float64Array(z.n * z.k)
  y.forEach((c, i) => {
    onehot[i * z.k + c] = 1
  })
  const zt = fromStorage(z.data, [z.n, z.k], 'float64')
  const yt = fromStorage(onehot, [z.n, z.k], 'float64')
  const objective = (p: Value): Value => {
    const scale = sliceAxis(p, 0, 0, z.k)
    const bias = sliceAxis(p, 0, z.k, 2 * z.k)
    return mul(neg(sum(mul(yt, logSoftmax(add(mul(zt, scale), bias), -1)))), 1 / z.n)
  }
  const x0 = fromStorage(
    Float64Array.from({ length: 2 * z.k }, (_, i) => (i < z.k ? 1 : 0)),
    [2 * z.k],
    'float64',
  )
  const r = lbfgs(objective, x0, { gradTol: 1e-10, maxIterations: 2000 })
  const p = toFlat(r.x as Tensor)
  const params = { scale: p.slice(0, z.k), bias: p.slice(z.k) }
  return { method: 'vector', params, predict: predictorFor('vector', params) }
}

function clipProbability(q: number): number {
  return Math.min(Math.max(q, 1e-12), 1 - 1e-12)
}

function histogramBin(edges: readonly number[], k: number, t: number): number {
  let b = 0
  for (let i = 1; i < k; i++) if (t >= (edges[i] ?? 0)) b = i
  return b
}

/** The map of a calibration method with given parameters. */
function predictorFor(method: CalibrationMethod, params: Json): (scores: TensorLike) => Tensor {
  const p = params as Record<string, unknown>
  const elementwise =
    (f: (v: number) => number, who: string) =>
    (x: TensorLike): Tensor => {
      const v = vector(x, who)
      return fromStorage(Float64Array.from(v, f), [v.length], 'float64')
    }
  if (method === 'sigmoid') {
    const { a, b } = p as PlattParams
    return elementwise((z) => sigmoid(-(a * z + b)), 'plattScaling')
  }
  if (method === 'isotonic') return isotonicPredict(p as IsotonicParams)
  if (method === 'beta') {
    const { a, b, c } = p as BetaParams
    return elementwise(
      (q) => sigmoid(a * Math.log(clipProbability(q)) - b * Math.log1p(-clipProbability(q)) + c),
      'betaCalibration',
    )
  }
  if (method === 'histogram') {
    const { edges, values } = p as HistogramParams
    return elementwise((t) => values[histogramBin(edges, values.length, t)] ?? 0, 'histogramBinning')
  }
  const multiclass =
    (f: (v: number, j: number) => number) =>
    (x: TensorLike): Tensor => {
      const q = logitsOf(x)
      return fromStorage(
        softmaxRows(
          q.data.map((v, i) => f(v, i % q.k)),
          q.n,
          q.k,
        ),
        [q.n, q.k],
        'float64',
      )
    }
  if (method === 'temperature') {
    const beta = (p as { readonly beta: number }).beta
    return multiclass((v) => beta * v)
  }
  const { scale, bias } = p as VectorParams
  return multiclass((v, j) => {
    if (j >= scale.length) throw new WhiteboxError('shape', `vectorScaling: fitted for ${scale.length} classes`)
    return (scale[j] ?? 1) * v + (bias[j] ?? 0)
  })
}

/**
 * Rebuild a calibrator from its method and parameters (after serialisation).
 *
 * @param method - The method.
 * @param params - Its parameters.
 * @returns The calibrator.
 * @example
 * ```ts
 * import { reviveCalibrator } from 'whitebox-engine/learn'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(reviveCalibrator('sigmoid', { a: -1, b: 0 }).predict([0])) // => [0.5]
 * ```
 */
export function reviveCalibrator(method: CalibrationMethod, params: Json): Calibrator {
  return { method, params, predict: predictorFor(method, params) }
}

const BINARY: Readonly<
  Record<
    'sigmoid' | 'isotonic' | 'beta' | 'histogram',
    (s: TensorLike, y: TensorLike, o: CalibratorOptions) => Calibrator
  >
> = {
  sigmoid: plattScaling,
  isotonic: isotonicCalibration,
  beta: betaCalibration,
  histogram: histogramBinning,
}

/** The scores a classifier exposes for calibration: decision values if it has them, else probabilities. */
function responseOf(
  fitted: Fitted,
  x: TensorLike,
): { readonly values: Float64Array; readonly n: number; readonly k: number; readonly kind: 'decision' | 'proba' } {
  const f = fitted.decisionFunction ?? fitted.predictProba
  if (f === undefined)
    throw new WhiteboxError('value', `calibration: ${fitted.estimator} has neither decisionFunction nor predictProba`)
  const out = f(x)
  const n = out.shape[0] ?? 0
  const k = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
  return {
    values: Float64Array.from(toFlat(out)),
    n,
    k,
    kind: fitted.decisionFunction !== undefined ? 'decision' : 'proba',
  }
}

/** One fitted member of a calibrated classifier: the classifier and one calibrator per class column. */
interface Member {
  readonly fitted: Fitted
  readonly calibrators: readonly Calibrator[]
}

function calibrateMember(
  fitted: Fitted,
  scores: ReturnType<typeof responseOf>,
  y: Int32Array,
  classes: readonly number[],
  method: CalibrationMethod,
  w: Float64Array | undefined,
): Member {
  const k = classes.length
  if (method === 'temperature' || method === 'vector') {
    const input =
      scores.k === 1 && scores.kind === 'proba'
        ? Array.from(scores.values, (p) => [1 - p, p])
        : fromStorage(scores.values, scores.k === 1 ? [scores.n] : [scores.n, scores.k], 'float64')
    const cal =
      method === 'temperature'
        ? temperatureScaling(input, y, w === undefined ? {} : { weights: w })
        : vectorScaling(input, y)
    return { fitted, calibrators: [cal] }
  }
  const fit = BINARY[method]
  const fittedClasses = fitted.classes ?? classes
  const columns = k === 2 ? [scores.k === 1 ? 0 : 1] : fittedClasses.map((_, c) => c)
  const calibrators = columns.map((col) => {
    const cls = k === 2 ? 1 : classes.indexOf(fittedClasses[col] ?? -1)
    const s = Float64Array.from(
      { length: scores.n },
      (_, i) => scores.values[i * scores.k + (scores.k === 1 ? 0 : col)] ?? 0,
    )
    return fit(
      s,
      Float64Array.from(y, (v) => (v === cls ? 1 : 0)),
      w === undefined ? {} : { weights: w },
    )
  })
  return { fitted, calibrators }
}

function memberProba(m: Member, x: TensorLike, classes: readonly number[], method: CalibrationMethod): Float64Array {
  const s = responseOf(m.fitted, x)
  const k = classes.length
  if (method === 'temperature' || method === 'vector') {
    const input =
      s.k === 1 && s.kind === 'proba'
        ? Array.from(s.values, (p) => [1 - p, p])
        : fromStorage(s.values, s.k === 1 ? [s.n] : [s.n, s.k], 'float64')
    return Float64Array.from(toFlat((m.calibrators[0] as Calibrator).predict(input)))
  }
  const out = new Float64Array(s.n * k)
  const fittedClasses = m.fitted.classes ?? classes
  if (k === 2) {
    const col = s.k === 1 ? 0 : 1
    const p = toFlat(
      (m.calibrators[0] as Calibrator).predict(
        Float64Array.from({ length: s.n }, (_, i) => s.values[i * s.k + col] ?? 0),
      ),
    )
    for (let i = 0; i < s.n; i++) {
      out[2 * i + 1] = p[i] ?? 0
      out[2 * i] = 1 - (p[i] ?? 0)
    }
    return out
  }
  m.calibrators.forEach((cal, col) => {
    const cls = classes.indexOf(fittedClasses[col] ?? -1)
    const p = toFlat(cal.predict(Float64Array.from({ length: s.n }, (_, i) => s.values[i * s.k + col] ?? 0)))
    for (let i = 0; i < s.n; i++) out[i * k + cls] = p[i] ?? 0
  })
  for (let i = 0; i < s.n; i++) {
    let total = 0
    for (let c = 0; c < k; c++) total += out[i * k + c] ?? 0
    for (let c = 0; c < k; c++) out[i * k + c] = total === 0 ? 1 / k : (out[i * k + c] ?? 0) / total
  }
  return out
}

/** Hyperparameters of {@link calibratedClassifierCV}. */
export interface CalibratedHyper {
  readonly estimator: Estimator
  readonly method: CalibrationMethod
  /** The splitter or number of folds (default 5, stratified). */
  readonly cv?: CvOption
  /** Keep one calibrated classifier per fold and average them (default true), or calibrate one classifier on out-of-fold scores. */
  readonly ensemble: boolean
}

/** Parameters of a fitted calibrated classifier. */
export type CalibratedParams = {
  readonly classes: number[]
  readonly members: { readonly estimator: Json; readonly calibrators: Json[] }[]
}

/** A fitted calibrated classifier. */
export interface FittedCalibrated extends Fitted<CalibratedParams> {
  readonly classes: readonly number[]
  readonly predictProba: (x: TensorLike) => Tensor
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * Calibrate a classifier with cross-fitting (sklearn.calibration.CalibratedClassifierCV): on each fold, fit the
 * classifier on the training rows and a calibrator (Platt, isotonic, beta, histogram, temperature or vector scaling)
 * on its scores for the held-out rows; predictions average the folds' calibrated probabilities. With `ensemble:
 * false`, one calibrator is fitted on out-of-fold scores and the classifier refitted on all the data. Multiclass
 * one-vs-rest calibrations are renormalised (uniform when they sum to 0).
 *
 * @param hyper - estimator, method (default `'sigmoid'`), cv and ensemble.
 * @returns The estimator.
 * @example
 * ```ts
 * import { calibratedClassifierCV, defineEstimator } from 'whitebox-engine/learn'
 * import { tensor } from 'whitebox-engine/tensor'
 * const scoreByFirst = defineEstimator(
 *   { name: 'first', kind: 'classifier', fit: () => ({}), methods: () => ({ classes: [0, 1], decisionFunction: (x) => tensor(x.shape === undefined ? x.map((r) => r[0]) : Array.from({ length: x.shape[0] }, (_, i) => x.data[i * x.shape[1]])) }) },
 *   {},
 * )
 * const x = [[-2], [-1], [-0.5], [0.5], [1], [2], [-1.5], [1.5]]
 * const f = calibratedClassifierCV({ estimator: scoreByFirst, cv: 2 }).fit({ x, y: [0, 0, 1, 0, 1, 1, 0, 1] })
 * f.predictProba([[3]]).shape // => [1, 2]
 * ```
 */
export function calibratedClassifierCV(
  hyper: Partial<CalibratedHyper> & { readonly estimator: Estimator },
): Estimator<CalibratedHyper, FittedCalibrated> {
  const h: CalibratedHyper = {
    method: hyper.method ?? 'sigmoid',
    ensemble: hyper.ensemble ?? true,
    estimator: hyper.estimator,
    ...(hyper.cv === undefined ? {} : { cv: hyper.cv }),
  }
  const build = (classes: readonly number[], members: readonly Member[]): FittedCalibrated => {
    const k = classes.length
    const predictProba = (x: TensorLike): Tensor => {
      const parts = members.map((m) => memberProba(m, x, classes, h.method))
      const n = (parts[0]?.length ?? 0) / k
      const out = new Float64Array(n * k)
      for (const p of parts)
        p.forEach((v, i) => {
          out[i] = (out[i] ?? 0) + v / parts.length
        })
      return fromStorage(out, [n, k], 'float64')
    }
    return {
      estimator: 'calibrated-classifier',
      kind: 'classifier',
      params: {
        classes: [...classes],
        members: members.map((m) => ({ estimator: m.fitted.params, calibrators: m.calibrators.map((c) => c.params) })),
      },
      classes,
      predictProba,
      predict: (x) => {
        const p = toFlat(predictProba(x))
        const n = p.length / k
        return fromStorage(
          Float64Array.from({ length: n }, (_, i) => {
            let best = 0
            for (let c = 1; c < k; c++) if ((p[i * k + c] ?? 0) > (p[i * k + best] ?? 0)) best = c
            return classes[best] ?? 0
          }),
          [n],
          'float64',
        )
      },
      ...(members[0]?.fitted.nFeaturesIn === undefined ? {} : { nFeaturesIn: members[0].fitted.nFeaturesIn }),
    }
  }
  return {
    name: 'calibrated-classifier',
    kind: 'classifier',
    hyper: h,
    fit: (data: LearnData, options: FitOptions = {}) => {
      const y = requireTargets(data, 'calibratedClassifierCV')
      const { classes, codes } = encodeLabels(y, 'calibratedClassifierCV')
      const w =
        data.weights === undefined ? undefined : sampleWeightsOf(data.weights, codes.length, 'calibratedClassifierCV')
      const key = keyFor(options, 'calibratedClassifierCV')
      const splitter = checkCv(h.cv, { kind: 'classifier' })
      if (h.ensemble) {
        const folds = splitIndices(splitter, splitInputOf(data), foldIn(key, 0))
        const members = folds.map((fold, f) => {
          const fitted = h.estimator.fit(
            takeData(data, fold.train),
            innerFitOptions(options, foldIn(key, 1 + f), options.parent),
          )
          const test = takeData(data, fold.test)
          return calibrateMember(
            fitted,
            responseOf(fitted, test.x),
            Int32Array.from(fold.test, (i) => codes[i] ?? 0),
            classes,
            h.method,
            w === undefined ? undefined : Float64Array.from(fold.test, (i) => w[i] ?? 1),
          )
        })
        return build(classes, members)
      }
      const probe = h.estimator.fit(data, innerFitOptions(options, foldIn(key, 1), options.parent))
      const method = probe.decisionFunction !== undefined ? 'decisionFunction' : 'predictProba'
      const oof = crossValPredict(h.estimator, data, {
        cv: splitter,
        method,
        key: foldIn(key, 2),
        ...(options.parent === undefined ? {} : { parent: options.parent }),
      })
      const n = oof.shape[0] ?? 0
      const kk = oof.shape.length === 2 ? (oof.shape[1] ?? 1) : 1
      const scores = {
        values: Float64Array.from(toFlat(oof)),
        n,
        k: kk,
        kind: method === 'decisionFunction' ? ('decision' as const) : ('proba' as const),
      }
      return build(classes, [calibrateMember(probe, scores, codes, classes, h.method, w)])
    },
    revive: (params) => {
      const p = params as CalibratedParams
      return build(
        p.classes,
        p.members.map((m) => ({
          fitted: h.estimator.revive(m.estimator),
          calibrators: m.calibrators.map((c) => reviveCalibrator(h.method, c)),
        })),
      )
    },
    with: (changes) => {
      const { estimator, ...rest } = changes as Partial<CalibratedHyper> & Readonly<Record<string, unknown>>
      const nested = Object.fromEntries(
        Object.entries(rest)
          .filter(([k]) => k.startsWith('estimator.'))
          .map(([k, v]) => [k.slice('estimator.'.length), v]),
      )
      const own = Object.fromEntries(Object.entries(rest).filter(([k]) => !k.startsWith('estimator.')))
      return calibratedClassifierCV({ ...h, ...own, estimator: (estimator ?? h.estimator).with(nested) })
    },
  }
}

/**
 * Calibrate an already fitted classifier on held-out data (sklearn's CalibratedClassifierCV on a frozen estimator).
 *
 * @param fitted - The fitted classifier.
 * @param data - Calibration data (not used to fit the classifier).
 * @param method - The calibration method (default `'sigmoid'`).
 * @returns Calibrated class probabilities as a function of features.
 * @example
 * ```ts
 * import { calibrateFitted } from 'whitebox-engine/learn'
 * import { tensor } from 'whitebox-engine/tensor'
 * const fitted = { estimator: 'raw', kind: 'classifier', params: {}, classes: [0, 1], decisionFunction: (x) => tensor(x.map((r) => r[0])) }
 * const proba = calibrateFitted(fitted, { x: [[-1], [-0.5], [0.5], [1]], y: [0, 1, 0, 1] }, 'isotonic')
 * proba([[2]]).shape // => [1, 2]
 * ```
 */
export function calibrateFitted(
  fitted: Fitted,
  data: LearnData,
  method: CalibrationMethod = 'sigmoid',
): (x: TensorLike) => Tensor {
  const { classes, codes } = encodeLabels(requireTargets(data, 'calibrateFitted'), 'calibrateFitted')
  const member = calibrateMember(
    fitted,
    responseOf(fitted, data.x),
    codes,
    fitted.classes ?? classes,
    method,
    undefined,
  )
  const all = fitted.classes ?? classes
  return (x) => {
    const p = memberProba(member, x, all, method)
    return fromStorage(p, [p.length / all.length, all.length], 'float64')
  }
}
