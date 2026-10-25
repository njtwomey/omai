/**
 * Proper scoring rules for probabilistic forecasts, which reward a forecaster for reporting its true beliefs: the
 * logarithmic score, the continuous ranked probability score (CRPS) in closed form or from samples, the spherical score
 * and the multivariate energy score.
 */

import { MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import type { Distribution } from '../dists/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import { ndtr, normalPdf } from '../numerics/special/index.ts'
import {
  abs,
  add,
  asTensor,
  broadcastTo,
  div,
  exp,
  expandDims,
  greater,
  less,
  log,
  maximum,
  mean,
  mul,
  neg,
  sigmoid,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  take,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { defineMetric, flat, type MetricFunction, metricResult, sampleWeights, weightedMean } from './metric.ts'

const spec = (name: string, direction: 'higher' | 'lower'): MetricSpec => ({
  name,
  direction,
  range: direction === 'lower' ? [-Infinity, Infinity] : [0, Infinity],
  inputs: 'distributions',
  category: 'probabilistic',
  concept: 'proper-scoring-rules',
})

/** Options of the distributional scores. */
export interface ScoreOptions {
  /** Per-observation weights (default all ones). */
  readonly sampleWeight?: TensorLike
}

const summarise = (name: string, values: readonly number[], options: ScoreOptions): MetricResult =>
  metricResult(name, weightedMean(values, sampleWeights(name, values.length, options.sampleWeight)), values.length)

/**
 * The logarithmic score (negative log predictive density, the ignorance score): -mean log p(y), the strictly proper
 * local scoring rule.
 *
 * @param dist - The predictive distribution (its batch shape broadcasts against y).
 * @param y - Observations.
 * @param options - Sample weights.
 * @returns The metric result (lower is better).
 * @concept proper-scoring-rules
 * @example
 * ```ts
 * import { logScore } from 'mdml-engine/metrics'
 * import { normal } from 'mdml-engine/dists'
 * Math.abs(logScore(normal(0, 1), [0]).estimate.value - 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export const logScore: MetricFunction<[Distribution, TensorLike, ScoreOptions?]> = defineMetric(
  spec('logScore', 'lower'),
  (dist: Distribution, y: TensorLike, options: ScoreOptions = {}): MetricResult =>
    summarise('logScore', toFlat(asTensor(neg(dist.logProb(y)))), options),
)

/** A parameter of a distribution broadcast to the observations. */
const param = (dist: Distribution, key: string, y: Value): Value => {
  const p = dist.params[key]
  if (p === undefined) throw new MdmlError('value', `${dist.name} has no parameter ${key}`)
  return broadcastTo(p, broadcastShape(p.shape, y.shape))
}

const broadcastShape = (a: readonly number[], b: readonly number[]): number[] => {
  const n = Math.max(a.length, b.length)
  return Array.from({ length: n }, (_, i) => {
    const da = a[a.length - n + i] ?? 1
    const db = b[b.length - n + i] ?? 1
    return da === 1 ? db : da
  })
}

/** CRPS in closed form for the families that have one (Gneiting and Raftery, 2007; Jordan et al., 2019). */
function closedFormCrps(dist: Distribution, y: Value): Value | undefined {
  switch (dist.name) {
    case 'normal': {
      const mu = param(dist, 'loc', y)
      const sigma = param(dist, 'scale', y)
      const z = div(sub(y, mu), sigma)
      const inner = add(mul(z, sub(mul(ndtr(z), 2), 1)), sub(mul(normalPdf(z), 2), 1 / Math.sqrt(Math.PI)))
      return mul(sigma, inner)
    }
    case 'laplace': {
      const mu = param(dist, 'loc', y)
      const b = param(dist, 'scale', y)
      const z = abs(div(sub(y, mu), b))
      return mul(b, sub(add(z, exp(neg(z))), 0.75))
    }
    case 'logistic': {
      const mu = param(dist, 'loc', y)
      const s = param(dist, 'scale', y)
      const z = div(sub(y, mu), s)
      return mul(s, sub(sub(z, mul(log(sigmoid(z)), 2)), 1))
    }
    case 'exponential': {
      const rate = param(dist, 'rate', y)
      const f = where(greater(y, 0), sub(1, exp(neg(mul(rate, maximum(y, 0))))), 0)
      return add(sub(abs(y), div(mul(f, 2), rate)), div(0.5, rate))
    }
    case 'uniform': {
      const lo = param(dist, 'low', y)
      const hi = param(dist, 'high', y)
      const width = sub(hi, lo)
      // E|X - y| - E|X - X'| / 2, with E|X - X'| = (b - a) / 3.
      const inside = div(add(square(sub(y, lo)), square(sub(hi, y))), mul(width, 2))
      const middle = div(add(lo, hi), 2)
      const outside = abs(sub(y, middle))
      const inRange = mul(cast01(greater(y, lo)), cast01(less(y, hi)))
      const e = add(mul(inRange, inside), mul(sub(1, inRange), outside))
      return sub(e, div(width, 6))
    }
    default:
      return undefined
  }
}

const cast01 = (b: Value): Value => where(b, 1, 0)

/** The support of a distribution as an integration range. */
function supportRange(dist: Distribution, i: number): [number, number] {
  const s = dist.support
  if (s.kind === 'positive' || s.kind === 'nonNegative') return [0, Number.POSITIVE_INFINITY]
  if (s.kind === 'unitInterval') return [0, 1]
  if (s.kind === 'interval') return [toFlat(asTensor(s.low))[i] ?? 0, toFlat(asTensor(s.high))[i] ?? 0]
  return [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
}

/** The cdf of observation i's distribution at points `[K]` (columns of a broadcast evaluation). */
const cdfColumn =
  (dist: Distribution, n: number, i: number) =>
  (x: Value): Value => {
    if (dist.batchShape.length === 0) return dist.cdf(x)
    const grid = broadcastTo(expandDims(x, 1), [x.shape[0] ?? 0, n])
    return sum(take(dist.cdf(grid), [i], 1), 1)
  }

/** CRPS by quadrature: int (F(x) - 1{x >= y})^2 dx over the support. */
function quadratureCrps(dist: Distribution, y: Float64Array): number[] {
  return Array.from(y, (yi, i) => {
    const cdf = cdfColumn(dist, y.length, i)
    const [lo, hi] = supportRange(dist, i)
    const below =
      yi > lo
        ? (toFlat(
            asTensor(
              quad((x) => square(cdf(x)), lo, Math.min(yi, hi), { epsabs: 1e-12, epsrel: 1e-10, limit: 200 }).value,
            ),
          )[0] ?? 0)
        : 0
    const above =
      yi < hi
        ? (toFlat(
            asTensor(
              quad((x) => square(sub(1, cdf(x))), Math.max(yi, lo), hi, { epsabs: 1e-12, epsrel: 1e-10, limit: 200 })
                .value,
            ),
          )[0] ?? 0)
        : 0
    // Below the support F = 0 and above it F = 1, which add (lo - y) and (y - hi) when y lies outside.
    return below + above + Math.max(lo - yi, 0) + Math.max(yi - hi, 0)
  })
}

/**
 * The continuous ranked probability score, CRPS(F, y) = int (F(x) - 1{x >= y})^2 dx = E|X - y| - E|X - X'| / 2:
 * closed forms for the normal, Laplace, logistic, exponential and uniform families, and adaptive quadrature of the CDF
 * for any other distribution.
 *
 * @param dist - The predictive distribution (batch shape matching y, or scalar).
 * @param y - Observations.
 * @param options - Sample weights.
 * @returns The mean CRPS (lower is better), in the units of y.
 * @example
 * ```ts
 * import { crps } from 'mdml-engine/metrics'
 * import { normal } from 'mdml-engine/dists'
 * Math.abs(crps(normal(0, 1), [0]).estimate.value - (Math.SQRT2 - 1) / Math.sqrt(Math.PI)) < 1e-15 // => true
 * ```
 */
export const crps: MetricFunction<[Distribution, TensorLike, ScoreOptions?]> = defineMetric(
  spec('crps', 'lower'),
  (dist: Distribution, y: TensorLike, options: ScoreOptions = {}): MetricResult => {
    const yv = asTensor(y, 'float64')
    const closed = closedFormCrps(dist, yv)
    const values = closed === undefined ? quadratureCrps(dist, flat(y)) : toFlat(asTensor(closed))
    return summarise('crps', values, options)
  },
)

/**
 * The spherical score p(y) / ||p||_2 (Good, 1971), a strictly proper rule (higher is better): for continuous
 * distributions ||p||_2^2 = int p^2 (in closed form for the normal, by quadrature otherwise); for a categorical
 * distribution p_y / sqrt(sum_k p_k^2).
 *
 * @param dist - The predictive distribution.
 * @param y - Observations (class indices for a categorical).
 * @param options - Sample weights.
 * @returns The mean spherical score.
 * @example
 * ```ts
 * import { sphericalScore } from 'mdml-engine/metrics'
 * import { categorical } from 'mdml-engine/dists'
 * sphericalScore(categorical({ probs: [0, 1] }), [1]).estimate.value // => 1
 * ```
 */
export const sphericalScore: MetricFunction<[Distribution, TensorLike, ScoreOptions?]> = defineMetric(
  spec('sphericalScore', 'higher'),
  (dist: Distribution, y: TensorLike, options: ScoreOptions = {}): MetricResult => {
    const yv = asTensor(y, 'float64')
    const density = toFlat(asTensor(dist.prob(yv)))
    let norms: number[]
    if (dist.name === 'categorical') {
      const probs = dist.params['probs'] as Value
      const sq = toFlat(asTensor(sqrt(sum(square(probs), -1))))
      norms = density.map((_, i) => sq[sq.length === 1 ? 0 : i] ?? 1)
    } else if (dist.name === 'normal') {
      const sigma = toFlat(asTensor(param(dist, 'scale', yv)))
      norms = sigma.map((s) => Math.sqrt(1 / (2 * s * Math.sqrt(Math.PI))))
    } else {
      const n = density.length
      norms = density.map((_, i) => {
        const [lo, hi] = supportRange(dist, i)
        const pdf = (x: Value): Value => exp(mul(logColumn(dist, n, i)(x), 2))
        return Math.sqrt(
          toFlat(asTensor(quad(pdf, lo, hi, { epsabs: 1e-13, epsrel: 1e-11, limit: 200 }).value))[0] ?? 1,
        )
      })
    }
    return summarise(
      'sphericalScore',
      density.map((p, i) => p / (norms[i] ?? 1)),
      options,
    )
  },
)

const logColumn =
  (dist: Distribution, n: number, i: number) =>
  (x: Value): Value => {
    if (dist.batchShape.length === 0) return dist.logProb(x)
    const grid = broadcastTo(expandDims(x, 1), [x.shape[0] ?? 0, n])
    return sum(take(dist.logProb(grid), [i], 1), 1)
  }

/**
 * The energy score (Gneiting and Raftery, 2007), the multivariate generalisation of the CRPS from an ensemble:
 * ES = mean_i ||X_i - y|| - (1 / 2M^2) sum_{i,j} ||X_i - X_j||, for M samples X_i of the forecast.
 *
 * @param samples - Forecast samples `[M, D]` (or `[N, M, D]` for N forecasts).
 * @param y - Observations `[D]` (or `[N, D]`).
 * @param options - Sample weights over forecasts.
 * @returns The mean energy score (lower is better).
 * @example
 * ```ts
 * import { energyScore } from 'mdml-engine/metrics'
 * energyScore([[0, 0], [0, 0]], [3, 4]).estimate.value // => 5
 * ```
 */
export const energyScore: MetricFunction<[TensorLike, TensorLike, ScoreOptions?]> = defineMetric(
  { ...spec('energyScore', 'lower'), inputs: 'values' },
  (samples: TensorLike, y: TensorLike, options: ScoreOptions = {}): MetricResult => {
    const s = asTensor(samples, 'float64')
    const batched = s.shape.length === 3
    const x = batched ? s : expandDims(s, 0)
    const obs = batched ? asTensor(y, 'float64') : expandDims(asTensor(y, 'float64'), 0)
    const m = x.shape[1] ?? 1
    const toObs = sqrt(sum(square(sub(x, expandDims(obs, 1))), -1))
    const pairs = sqrt(sum(square(sub(expandDims(x, 2), expandDims(x, 1))), -1))
    const es = sub(mean(toObs, -1), div(sum(pairs, [-2, -1]), 2 * m * m))
    return summarise('energyScore', toFlat(asTensor(es)), options)
  },
)
