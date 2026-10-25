/**
 * Feature scalers, scikit-learn style: standardisation to zero mean and unit variance, min-max and max-abs scaling, the
 * robust scaler (median and percentile range), row normalisation, the quantile transformer to a uniform or normal
 * distribution, and the Yeo-Johnson and Box-Cox power transforms, each with an inverse.
 */

import { MdmlError } from '../../base/index.ts'
import { interp } from '../../numerics/interpolate/index.ts'
import { ndtr, ndtri } from '../../numerics/special/index.ts'
import { subsample } from '../../random/index.ts'
import { bool, choice, dict, int, type Space } from '../../space/index.ts'
import { boxcox, invBoxcox, invYeojohnson, quantile, yeojohnson } from '../../stats/index.ts'
import { fromStorage, linspace, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { handleZeros, inputNames, keyFor, mapColumns, mapEntries, present } from '../_shared.ts'
import { defineEstimator, type Estimator, type Fitted } from '../estimator.ts'
import { columnOf, fromView, type MatrixView, matrixView, sampleWeightsOf, takeRows } from '../validation.ts'

const EPS = 2.220446049250313e-16

/** The verbs every scaler offers. */
export interface ScalerVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/** A fitted scaler: parameters per feature plus `transform` and `inverseTransform`. */
export type FittedScaler<P extends Readonly<Record<string, number[] | number | boolean | number[][]>>> = Fitted<P> &
  ScalerVerbs

function scalerVerbs(
  name: string,
  d: number,
  forward: (m: MatrixView) => MatrixView,
  backward: (m: MatrixView) => MatrixView,
): ScalerVerbs {
  const check = (x: TensorLike): MatrixView => matrixView(x, name, { allowNaN: true, nFeatures: d })
  return {
    transform: (x) => fromView(forward(check(x))),
    inverseTransform: (x) => fromView(backward(check(x))),
    nFeaturesIn: d,
    featureNamesOut: (input) => inputNames(d, input, name),
  }
}

/** Hyperparameters of {@link standardScaler}. */
export interface StandardScalerHyper {
  readonly withMean: boolean
  readonly withStd: boolean
}

/** Parameters of a fitted standard scaler (NaN-aware, weighted). */
export type StandardScalerParams = {
  readonly mean: number[]
  readonly var: number[]
  readonly scale: number[]
  readonly nSamplesSeen: number[]
}

/** True when a variance is zero up to rounding (sklearn's `_is_constant_feature`). */
function isConstant(variance: number, mean: number, n: number): boolean {
  return variance <= n * EPS * variance + (n * mean * EPS) ** 2
}

/**
 * Standardise features to zero mean and unit variance (sklearn.preprocessing.StandardScaler): z = (x - mean) / scale
 * with the population standard deviation, NaN ignored in fit and kept in transform, constant features left unscaled,
 * and sample weights supported.
 *
 * @param hyper - `withMean` and `withStd` (both default true).
 * @returns The estimator.
 * @concept standardisation
 * @example
 * ```ts
 * import { standardScaler } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * const s = standardScaler().fit({ x: [[0, 1], [2, 1]] })
 * s.params.mean // => [1, 1]
 * toArray(s.transform([[2, 3]])) // => [[1, 2]]
 * toArray(s.inverseTransform([[1, 2]])) // => [[2, 3]]
 * ```
 */
export function standardScaler(
  hyper: Partial<StandardScalerHyper> = {},
): Estimator<StandardScalerHyper, FittedScaler<StandardScalerParams>> {
  return defineEstimator(
    {
      name: 'standard-scaler',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ withMean: bool(), withStd: bool() }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'standardScaler', { allowNaN: true })
        const w = sampleWeightsOf(data.weights, m.n, 'standardScaler')
        const mean: number[] = []
        const variance: number[] = []
        const scale: number[] = []
        const seen: number[] = []
        for (let j = 0; j < m.d; j++) {
          let sw = 0
          let s = 0
          for (let i = 0; i < m.n; i++) {
            const v = m.data[i * m.d + j] ?? Number.NaN
            if (Number.isNaN(v)) continue
            sw += w[i] ?? 0
            s += (w[i] ?? 0) * v
          }
          const mu = sw > 0 ? s / sw : Number.NaN
          let ss = 0
          let corr = 0
          for (let i = 0; i < m.n; i++) {
            const v = m.data[i * m.d + j] ?? Number.NaN
            if (Number.isNaN(v)) continue
            ss += (w[i] ?? 0) * (v - mu) ** 2
            corr += (w[i] ?? 0) * (v - mu)
          }
          const v = sw > 0 ? (ss - (corr * corr) / sw) / sw : Number.NaN
          mean.push(mu)
          variance.push(v)
          seen.push(sw)
          scale.push(isConstant(v, mu, sw) ? 1 : handleZeros(Math.sqrt(v)))
        }
        return { mean, var: variance, scale: h.withStd ? scale : scale.map(() => 1), nSamplesSeen: seen }
      },
      methods: (p, h) =>
        scalerVerbs(
          'standardScaler',
          p.mean.length,
          (m) => mapEntries(m, (v, j) => (h.withMean ? v - (p.mean[j] ?? 0) : v) / (p.scale[j] ?? 1)),
          (m) => mapEntries(m, (v, j) => v * (p.scale[j] ?? 1) + (h.withMean ? (p.mean[j] ?? 0) : 0)),
        ),
    },
    { withMean: hyper.withMean ?? true, withStd: hyper.withStd ?? true },
  )
}

/** Hyperparameters of {@link minMaxScaler}. */
export interface MinMaxScalerHyper {
  readonly featureRange: readonly [number, number]
  readonly clip: boolean
}

/** Parameters of a fitted min-max scaler. */
export type MinMaxScalerParams = {
  readonly dataMin: number[]
  readonly dataMax: number[]
  readonly scale: number[]
  readonly min: number[]
}

function columnRange(col: Float64Array): [number, number] {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of col) {
    if (Number.isNaN(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return lo > hi ? [Number.NaN, Number.NaN] : [lo, hi]
}

/**
 * Scale each feature to a range (sklearn.preprocessing.MinMaxScaler): x * scale + min with scale = (hi - lo) / (max -
 * min); constant features map to `lo`; NaN is ignored in fit and kept; `clip` clamps transformed values to the range.
 *
 * @param hyper - `featureRange` (default [0, 1]) and `clip` (default false).
 * @returns The estimator.
 * @example
 * ```ts
 * import { minMaxScaler } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * const s = minMaxScaler().fit({ x: [[0], [4]] })
 * toArray(s.transform([[1], [8]])) // => [[0.25], [2]]
 * toArray(minMaxScaler({ clip: true }).fit({ x: [[0], [4]] }).transform([[8]])) // => [[1]]
 * ```
 */
export function minMaxScaler(
  hyper: Partial<MinMaxScalerHyper> = {},
): Estimator<MinMaxScalerHyper, FittedScaler<MinMaxScalerParams>> {
  const range = hyper.featureRange ?? [0, 1]
  if (!(range[0] < range[1])) throw new MdmlError('value', 'minMaxScaler: featureRange must be increasing')
  return defineEstimator(
    {
      name: 'min-max-scaler',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ clip: bool() }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'minMaxScaler', { allowNaN: true })
        const [lo, hi] = h.featureRange
        const p = { dataMin: [] as number[], dataMax: [] as number[], scale: [] as number[], min: [] as number[] }
        for (let j = 0; j < m.d; j++) {
          const [a, b] = columnRange(columnOf(m, j))
          const s = (hi - lo) / handleZeros(b - a)
          p.dataMin.push(a)
          p.dataMax.push(b)
          p.scale.push(s)
          p.min.push(lo - a * s)
        }
        return p
      },
      methods: (p, h) =>
        scalerVerbs(
          'minMaxScaler',
          p.scale.length,
          (m) =>
            mapEntries(m, (v, j) => {
              const t = v * (p.scale[j] ?? 1) + (p.min[j] ?? 0)
              return h.clip ? Math.min(Math.max(t, h.featureRange[0]), h.featureRange[1]) : t
            }),
          (m) => mapEntries(m, (v, j) => (v - (p.min[j] ?? 0)) / (p.scale[j] ?? 1)),
        ),
    },
    { featureRange: range, clip: hyper.clip ?? false },
  )
}

/** Parameters of a fitted max-abs scaler. */
export type MaxAbsScalerParams = { readonly maxAbs: number[]; readonly scale: number[] }

/**
 * Scale each feature by its largest absolute value (sklearn.preprocessing.MaxAbsScaler), keeping zeros (and sparsity)
 * in place.
 *
 * @returns The estimator.
 * @example
 * ```ts
 * import { maxAbsScaler } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(maxAbsScaler().fit({ x: [[-4], [2]] }).transform([[1]])) // => [[0.25]]
 * ```
 */
export function maxAbsScaler(): Estimator<Record<string, never>, FittedScaler<MaxAbsScalerParams>> {
  return defineEstimator(
    {
      name: 'max-abs-scaler',
      kind: 'transformer',
      fit: (data) => {
        const m = matrixView(data.x, 'maxAbsScaler', { allowNaN: true })
        const maxAbs: number[] = []
        for (let j = 0; j < m.d; j++) {
          const [a, b] = columnRange(columnOf(m, j))
          maxAbs.push(Math.max(Math.abs(a), Math.abs(b)))
        }
        return { maxAbs, scale: maxAbs.map((v) => (Number.isNaN(v) ? 1 : handleZeros(v))) }
      },
      methods: (p) =>
        scalerVerbs(
          'maxAbsScaler',
          p.scale.length,
          (m) => mapEntries(m, (v, j) => v / (p.scale[j] ?? 1)),
          (m) => mapEntries(m, (v, j) => v * (p.scale[j] ?? 1)),
        ),
    },
    {},
  )
}

/** Hyperparameters of {@link robustScaler}. */
export interface RobustScalerHyper {
  readonly withCentering: boolean
  readonly withScaling: boolean
  /** The percentiles whose difference is the scale (default [25, 75], the interquartile range). */
  readonly quantileRange: readonly [number, number]
  /** Divide by the normal distribution's range for those percentiles, so normal data gets unit variance. */
  readonly unitVariance: boolean
}

/** Parameters of a fitted robust scaler. */
export type RobustScalerParams = { readonly center: number[]; readonly scale: number[] }

/**
 * Centre by the median and scale by a percentile range (sklearn.preprocessing.RobustScaler), so outliers barely move
 * the parameters. NaN is ignored in fit and kept.
 *
 * @param hyper - Centring, scaling, the percentile range (default [25, 75]) and unit variance (default false).
 * @returns The estimator.
 * @example
 * ```ts
 * import { robustScaler } from 'mdml-engine/learn/preprocessing'
 * const s = robustScaler().fit({ x: [[1], [2], [3], [4], [100]] })
 * s.params // => { center: [3], scale: [2] }
 * ```
 */
export function robustScaler(
  hyper: Partial<RobustScalerHyper> = {},
): Estimator<RobustScalerHyper, FittedScaler<RobustScalerParams>> {
  const q = hyper.quantileRange ?? [25, 75]
  if (!(q[0] >= 0 && q[0] < q[1] && q[1] <= 100)) {
    throw new MdmlError('value', 'robustScaler: quantileRange must satisfy 0 <= lo < hi <= 100')
  }
  return defineEstimator(
    {
      name: 'robust-scaler',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ withCentering: bool(), withScaling: bool(), unitVariance: bool() }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'robustScaler', { allowNaN: true })
        const center: number[] = []
        const scale: number[] = []
        const adjust = h.unitVariance
          ? (toFlat(ndtri(h.quantileRange[1] / 100))[0] ?? 0) - (toFlat(ndtri(h.quantileRange[0] / 100))[0] ?? 0)
          : 1
        for (let j = 0; j < m.d; j++) {
          const v = present(columnOf(m, j))
          center.push(h.withCentering ? (v.length === 0 ? Number.NaN : quantile(v, 0.5)) : 0)
          if (h.withScaling) {
            const r = v.length === 0 ? 0 : quantile(v, h.quantileRange[1] / 100) - quantile(v, h.quantileRange[0] / 100)
            scale.push(handleZeros(r) / adjust)
          } else scale.push(1)
        }
        return { center, scale }
      },
      methods: (p) =>
        scalerVerbs(
          'robustScaler',
          p.scale.length,
          (m) => mapEntries(m, (v, j) => (v - (p.center[j] ?? 0)) / (p.scale[j] ?? 1)),
          (m) => mapEntries(m, (v, j) => v * (p.scale[j] ?? 1) + (p.center[j] ?? 0)),
        ),
    },
    {
      withCentering: hyper.withCentering ?? true,
      withScaling: hyper.withScaling ?? true,
      quantileRange: q,
      unitVariance: hyper.unitVariance ?? false,
    },
  )
}

/** The norm {@link normalizer} divides each row by. */
export type RowNorm = 'l1' | 'l2' | 'max'

/**
 * Scale each sample (row) to unit norm (sklearn.preprocessing.Normalizer); rows of zero norm are left as they are. It
 * learns nothing, so `fit` only records the number of features; the inverse is not defined (the norms are lost), so
 * `inverseTransform` is the identity on unit rows only and is omitted.
 *
 * @param hyper - The norm (default `'l2'`).
 * @returns The estimator.
 * @example
 * ```ts
 * import { normalizer } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(normalizer().fit({ x: [[3, 4]] }).transform([[3, 4], [0, 0]])) // => [[0.6, 0.8], [0, 0]]
 * toArray(normalizer({ norm: 'l1' }).fit({ x: [[1, 3]] }).transform([[1, 3]])) // => [[0.25, 0.75]]
 * ```
 */
export function normalizer(hyper: { readonly norm?: RowNorm } = {}): Estimator<
  { readonly norm: RowNorm },
  Fitted<{ readonly nFeatures: number }> & {
    readonly transform: (x: TensorLike) => Tensor
    readonly nFeaturesIn: number
    readonly featureNamesOut: (input?: readonly string[]) => string[]
  }
> {
  return defineEstimator(
    {
      name: 'normalizer',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ norm: choice(['l1', 'l2', 'max']) }),
      fit: (data) => ({ nFeatures: matrixView(data.x, 'normalizer').d }),
      methods: (p, h) => ({
        transform: (x) => {
          const m = matrixView(x, 'normalizer', { nFeatures: p.nFeatures })
          const out = new Float64Array(m.data.length)
          for (let i = 0; i < m.n; i++) {
            let norm = 0
            for (let j = 0; j < m.d; j++) {
              const v = m.data[i * m.d + j] ?? 0
              if (h.norm === 'l1') norm += Math.abs(v)
              else if (h.norm === 'l2') norm += v * v
              else norm = Math.max(norm, Math.abs(v))
            }
            const s = handleZeros(h.norm === 'l2' ? Math.sqrt(norm) : norm)
            for (let j = 0; j < m.d; j++) out[i * m.d + j] = (m.data[i * m.d + j] ?? 0) / s
          }
          return fromStorage(out, [m.n, m.d], 'float64')
        },
        nFeaturesIn: p.nFeatures,
        featureNamesOut: (input) => inputNames(p.nFeatures, input, 'normalizer'),
      }),
    },
    { norm: hyper.norm ?? 'l2' },
  )
}

/** Hyperparameters of {@link quantileTransformer}. */
export interface QuantileTransformerHyper {
  /** The number of quantiles (landmarks) of the empirical CDF (default 1000, capped at the number of samples). */
  readonly nQuantiles: number
  /** Map to a uniform or a standard normal distribution (default `'uniform'`). */
  readonly outputDistribution: 'uniform' | 'normal'
  /** The most samples used to estimate quantiles (default 10000; a random subsample beyond that). */
  readonly subsample: number
}

/** Parameters of a fitted quantile transformer: per feature, its quantiles at `references`. */
export type QuantileTransformerParams = { readonly quantiles: number[][]; readonly references: number[] }

const BOUNDS = 1e-7

function quantileColumn(
  col: Float64Array,
  q: readonly number[],
  refs: readonly number[],
  normal: boolean,
  inverse: boolean,
): Float64Array {
  const out = Float64Array.from(col)
  const qt = Float64Array.from(q)
  const rt = Float64Array.from(refs)
  const [loX, hiX, loY, hiY] = inverse
    ? [0, 1, q[0] ?? 0, q[q.length - 1] ?? 0]
    : [q[0] ?? 0, q[q.length - 1] ?? 0, 0, 1]
  if (inverse && normal) {
    const c = toFlat(ndtr(fromStorage(Float64Array.from(out), [out.length], 'float64')))
    for (let i = 0; i < out.length; i++) out[i] = c[i] ?? 0
  }
  const lower = Array.from(out, (v) => (normal ? v - BOUNDS < loX : v === loX))
  const upper = Array.from(out, (v) => (normal ? v + BOUNDS > hiX : v === hiX))
  const idx: number[] = []
  for (let i = 0; i < out.length; i++) if (!Number.isNaN(out[i] ?? Number.NaN)) idx.push(i)
  const finite = fromStorage(
    Float64Array.from(idx, (i) => out[i] ?? 0),
    [idx.length],
    'float64',
  )
  let mapped: number[]
  if (inverse) mapped = toFlat(interp(finite, rt, qt))
  else {
    const up = toFlat(interp(finite, qt, rt))
    const negX = fromStorage(
      Float64Array.from(idx, (i) => -(out[i] ?? 0)),
      [idx.length],
      'float64',
    )
    const down = toFlat(
      interp(
        negX,
        Float64Array.from(qt)
          .reverse()
          .map((v) => -v),
        Float64Array.from(rt)
          .reverse()
          .map((v) => -v),
      ),
    )
    mapped = up.map((u, k) => 0.5 * (u - (down[k] ?? 0)))
  }
  idx.forEach((i, k) => {
    out[i] = mapped[k] ?? 0
  })
  for (let i = 0; i < out.length; i++) {
    if (upper[i] === true) out[i] = hiY
    if (lower[i] === true) out[i] = loY
  }
  if (!inverse && normal) {
    const z = toFlat(ndtri(fromStorage(Float64Array.from(out), [out.length], 'float64')))
    const lo = toFlat(ndtri(BOUNDS - EPS))[0] ?? 0
    const hi = toFlat(ndtri(1 - (BOUNDS - EPS)))[0] ?? 0
    for (let i = 0; i < out.length; i++) {
      const v = z[i] ?? 0
      out[i] = Number.isNaN(v) ? v : Math.min(Math.max(v, lo), hi)
    }
  }
  return out
}

/**
 * Map each feature through its empirical CDF to a uniform or normal distribution (sklearn QuantileTransformer): the
 * quantiles at `nQuantiles` evenly spaced levels are the landmarks; values in between interpolate linearly (averaging
 * the rising and falling interpolations, so ties map to their mid-rank); values outside clip to the ends. Monotone and
 * invertible on the data's range.
 *
 * @param hyper - The number of quantiles, the output distribution and the subsample size.
 * @param options - The key for subsampling (only used beyond `subsample` rows).
 * @returns The estimator.
 * @example
 * ```ts
 * import { quantileTransformer } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * const q = quantileTransformer({ nQuantiles: 5 }).fit({ x: [[1], [2], [3], [4], [5]] })
 * toArray(q.transform([[2], [3.5], [9]])) // => [[0.25], [0.625], [1]]
 * toArray(q.inverseTransform([[0.625]])) // => [[3.5]]
 * ```
 */
export function quantileTransformer(
  hyper: Partial<QuantileTransformerHyper> = {},
): Estimator<QuantileTransformerHyper, FittedScaler<QuantileTransformerParams>> {
  return defineEstimator(
    {
      name: 'quantile-transformer',
      kind: 'transformer',
      hyperSpace: (): Space =>
        dict({ nQuantiles: int(10, 2000, { log: true }), outputDistribution: choice(['uniform', 'normal']) }),
      fit: (data, h, options) => {
        let m = matrixView(data.x, 'quantileTransformer', { allowNaN: true })
        if (h.nQuantiles < 2) throw new MdmlError('value', 'quantileTransformer: nQuantiles must be at least 2')
        if (m.n > h.subsample) {
          m = takeRows(m, toFlat(subsample(keyFor(options, 'quantileTransformer'), m.n, h.subsample)))
        }
        const nq = Math.min(h.nQuantiles, m.n)
        const references = toFlat(linspace(0, 1, nq))
        const quantiles: number[][] = []
        for (let j = 0; j < m.d; j++) {
          const v = present(columnOf(m, j))
          const qs = v.length === 0 ? references.map(() => Number.NaN) : toFlat(quantile(v, references))
          for (let k = 1; k < qs.length; k++) qs[k] = Math.max(qs[k] ?? 0, qs[k - 1] ?? 0)
          quantiles.push(qs)
        }
        return { quantiles, references }
      },
      methods: (p, h) => {
        const normal = h.outputDistribution === 'normal'
        const apply = (inverse: boolean) => (m: MatrixView) =>
          mapColumns(m, (col, j) => quantileColumn(col, p.quantiles[j] ?? [], p.references, normal, inverse))
        return scalerVerbs('quantileTransformer', p.quantiles.length, apply(false), apply(true))
      },
    },
    {
      nQuantiles: hyper.nQuantiles ?? 1000,
      outputDistribution: hyper.outputDistribution ?? 'uniform',
      subsample: hyper.subsample ?? 10000,
    },
  )
}

/** Hyperparameters of {@link powerTransformer}. */
export interface PowerTransformerHyper {
  /** Yeo-Johnson (any sign, the default) or Box-Cox (strictly positive data). */
  readonly method: 'yeo-johnson' | 'box-cox'
  /** Standardise the transformed features (default true). */
  readonly standardize: boolean
}

/** Parameters of a fitted power transformer: one lambda per feature, and the standardisation after it. */
export type PowerTransformerParams = { readonly lambdas: number[]; readonly mean: number[]; readonly scale: number[] }

// The Yeo-Johnson transform of one value (sklearn's _yeo_johnson_transform, in scipy's expm1 / log1p form).
function yeoJohnsonValue(x: number, lambda: number): number {
  if (x >= 0) return Math.abs(lambda) < EPS ? Math.log1p(x) : Math.expm1(lambda * Math.log1p(x)) / lambda
  return Math.abs(lambda - 2) < EPS ? -Math.log1p(-x) : -Math.expm1((2 - lambda) * Math.log1p(-x)) / (2 - lambda)
}

function powerColumn(
  col: Float64Array,
  lambda: number,
  method: PowerTransformerHyper['method'],
  inverse: boolean,
): Float64Array {
  const out = Float64Array.from(col)
  const idx: number[] = []
  for (let i = 0; i < col.length; i++) if (!Number.isNaN(col[i] ?? Number.NaN)) idx.push(i)
  if (idx.length === 0) return out
  const v = Float64Array.from(idx, (i) => col[i] ?? 0)
  let r: number[]
  if (method === 'box-cox') r = toFlat(inverse ? invBoxcox(v, lambda) : boxcox(v, { lambda }).transformed)
  else r = inverse ? toFlat(invYeojohnson(v, lambda)) : Array.from(v, (t) => yeoJohnsonValue(t, lambda))
  idx.forEach((i, k) => {
    out[i] = r[k] ?? 0
  })
  return out
}

/**
 * Make features more Gaussian with a per-feature power transform (sklearn PowerTransformer): Yeo-Johnson or Box-Cox
 * with lambda by maximum likelihood (scipy's Brent search), then standardised. NaN is ignored in fit and kept.
 *
 * @param hyper - The method (default `'yeo-johnson'`) and whether to standardise (default true).
 * @returns The estimator.
 * @throws MdmlError with code `domain` when Box-Cox meets non-positive data.
 * @example
 * ```ts
 * import { powerTransformer } from 'mdml-engine/learn/preprocessing'
 * import { toArray } from 'mdml-engine/tensor'
 * const p = powerTransformer().fit({ x: [[1], [2], [3], [10]] })
 * const back = toArray(p.inverseTransform(p.transform([[3]])))
 * Math.abs(back[0][0] - 3) < 1e-9 // => true
 * ```
 */
export function powerTransformer(
  hyper: Partial<PowerTransformerHyper> = {},
): Estimator<PowerTransformerHyper, FittedScaler<PowerTransformerParams>> {
  return defineEstimator(
    {
      name: 'power-transformer',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ method: choice(['yeo-johnson', 'box-cox']), standardize: bool() }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'powerTransformer', { allowNaN: true })
        const lambdas: number[] = []
        const mean: number[] = []
        const scale: number[] = []
        for (let j = 0; j < m.d; j++) {
          const v = present(columnOf(m, j))
          if (h.method === 'box-cox' && v.some((t) => t <= 0)) {
            throw new MdmlError('domain', 'powerTransformer: Box-Cox needs strictly positive data')
          }
          const fitted = h.method === 'box-cox' ? boxcox(v) : yeojohnson(v)
          lambdas.push(fitted.lambda)
          const t = toFlat(fitted.transformed)
          const mu = t.reduce((a, b) => a + b, 0) / t.length
          const variance = t.reduce((a, b) => a + (b - mu) ** 2, 0) / t.length
          mean.push(h.standardize ? mu : 0)
          scale.push(h.standardize ? (isConstant(variance, mu, t.length) ? 1 : handleZeros(Math.sqrt(variance))) : 1)
        }
        return { lambdas, mean, scale }
      },
      methods: (p, h) =>
        scalerVerbs(
          'powerTransformer',
          p.lambdas.length,
          (m) =>
            mapColumns(m, (col, j) =>
              powerColumn(col, p.lambdas[j] ?? 1, h.method, false).map(
                (v) => (v - (p.mean[j] ?? 0)) / (p.scale[j] ?? 1),
              ),
            ),
          (m) =>
            mapColumns(m, (col, j) =>
              powerColumn(
                col.map((v) => v * (p.scale[j] ?? 1) + (p.mean[j] ?? 0)),
                p.lambdas[j] ?? 1,
                h.method,
                true,
              ),
            ),
        ),
    },
    { method: hyper.method ?? 'yeo-johnson', standardize: hyper.standardize ?? true },
  )
}
