/**
 * Modifiers that change the features of a dataset while keeping its truth exact: missing values under Rubin's
 * mechanisms (MCAR, MAR, MNAR), outlier contamination, independent nuisance features, invertible elementwise transforms
 * (with the log-Jacobian added to the density) and linear mixing to a known correlation structure. Each comes as a
 * function and as a recipe modifier.
 *
 * @index function Missingness, outlier and feature-noise modifiers.
 */

import { MdmlError, sigmoidOf } from '../../base/index.ts'
import { foldIn, type Key, normal, permutation, uniform } from '../../random/index.ts'
import { choice, dict, int, real } from '../../space/index.ts'
import {
  abs,
  add,
  arcsinh,
  cosh,
  div,
  exp,
  fromStorage,
  full,
  log,
  matmul,
  mul,
  neg,
  pow,
  sigmoid,
  sign,
  sinh,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  tensor,
  type Value,
} from '../../tensor/index.ts'
import type { Generated } from '../generator.ts'
import type { Modifier } from '../recipes.ts'
import type { TabularData } from '../schema/tabular.ts'
import type { Truth } from '../truth.ts'
import {
  calibrateIntercept,
  columnStats,
  columnsOf,
  composedFunctions,
  mapColumns,
  requireTabular,
  selectColumns,
  withoutProgram,
} from './_shared.ts'

/** The missingness mechanisms of Rubin's taxonomy. */
export type MissingMechanism = 'MCAR' | 'MAR' | 'MNAR'

/** Options of {@link withMissing}. */
export interface MissingOptions {
  /** `MCAR` (default): independent of everything; `MAR`: driven by observed columns; `MNAR`: driven by the value. */
  readonly mechanism?: MissingMechanism
  /** The expected fraction of entries removed in each affected column (default 0.2). */
  readonly rate?: number
  /** Columns that lose values (default all for MCAR and MNAR, all but the drivers for MAR). */
  readonly columns?: readonly number[]
  /** MAR only: the fully observed columns that drive missingness (default the first column not in `columns`). */
  readonly drivers?: readonly number[]
  /** The logistic slope on the standardised driver (MAR) or own value (MNAR) (default 2). */
  readonly strength?: number
  readonly key: Key
}

/**
 * Remove feature values (set them to NaN) under a known missingness mechanism. MCAR removes each entry with the same
 * probability; MAR removes entries of a column with probability `logistic(a + strength z)` where `z` is the mean
 * standardised value of the always-observed driver columns; MNAR uses the entry's own standardised value (high values
 * go missing more often), the self-masking case. The intercept `a` is calibrated so each column loses `rate` of its
 * entries on average. The truth keeps the complete data, the mask and the missingness probabilities.
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - Mechanism, rate, columns, drivers, strength and key.
 * @returns The data with NaN gaps; `truth.latents.complete`, `truth.latents.missing` (bool `[n, d]`) and
 * `truth.latents.missingProbability`.
 * @throws MdmlError with code `value` for MAR without an observed driver column.
 * @example
 * ```ts
 * import { asGenerated, withMissing, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = withMissing(asGenerated(xy([[1, 2], [3, 4]], [0, 1])), { rate: 1, key: key(0) })
 * Array.from(d.data.x.data).every(Number.isNaN) // => true
 * ```
 */
export function withMissing<D extends TabularData>(input: Generated<D>, options: MissingOptions): Generated<D> {
  const { data, truth } = requireTabular(input, 'withMissing') as Generated<D>
  const x = data.x
  const [n = 0, d = 1] = x.shape
  const mechanism = options.mechanism ?? 'MCAR'
  const rate = options.rate ?? 0.2
  const strength = options.strength ?? 2
  let cols: number[]
  let drivers: number[] = []
  if (mechanism === 'MAR') {
    const all = Array.from({ length: d }, (_, j) => j)
    if (options.drivers !== undefined) drivers = columnsOf(options.drivers, d, 'withMissing')
    else {
      const masked = new Set(options.columns ?? [])
      const free = all.filter((j) => !masked.has(j))
      drivers = free.slice(0, 1)
    }
    if (drivers.length === 0) throw new MdmlError('value', 'withMissing: MAR needs an always-observed driver column')
    const dset = new Set(drivers)
    cols = columnsOf(options.columns ?? all.filter((j) => !dset.has(j)), d, 'withMissing')
    if (cols.some((j) => dset.has(j))) throw new MdmlError('value', 'withMissing: a driver column cannot lose values')
  } else {
    cols = columnsOf(options.columns, d, 'withMissing')
  }
  const { mean, std } = columnStats(x)
  const z = (i: number, j: number) => ((x.data[i * d + j] ?? 0) - (mean[j] ?? 0)) / ((std[j] ?? 0) || 1)
  const prob = new Float64Array(n * d)
  for (const j of cols) {
    if (mechanism === 'MCAR') {
      for (let i = 0; i < n; i++) prob[i * d + j] = rate
      continue
    }
    const score = Float64Array.from({ length: n }, (_, i) =>
      mechanism === 'MAR' ? drivers.reduce((s, k) => s + z(i, k), 0) / drivers.length : z(i, j),
    )
    const a = rate <= 0 ? -Infinity : rate >= 1 ? Infinity : calibrateIntercept(score, strength, rate)
    for (let i = 0; i < n; i++) prob[i * d + j] = sigmoidOf(a + strength * (score[i] ?? 0))
  }
  const u = uniform(options.key, [n, d]).data
  const values = Float64Array.from(x.data as ArrayLike<number>)
  const mask = new Uint8Array(n * d)
  for (let i = 0; i < n * d; i++) {
    if ((u[i] ?? 1) < (prob[i] ?? 0)) {
      values[i] = Number.NaN
      mask[i] = 1
    }
  }
  return {
    data: { ...data, x: fromStorage(values, [n, d], 'float64') },
    truth: {
      ...withoutProgram(truth),
      latents: {
        ...(truth.latents ?? {}),
        complete: x,
        missing: fromStorage(mask, [n, d], 'bool'),
        missingProbability: fromStorage(prob, [n, d], 'float64'),
      },
      extra: { ...(truth.extra ?? {}), missingness: { mechanism, rate, columns: cols, drivers, strength } },
    },
  }
}

/** Options of {@link withOutliers}. */
export interface OutlierOptions {
  /** The fraction of rows contaminated (default 0.05). */
  readonly fraction?: number
  /** Contaminate the features (default), the regression targets, or both. */
  readonly target?: 'features' | 'targets' | 'both'
  /** `shift` (default): move by `scale` standard deviations in a random direction; `uniform`: redraw uniformly
   * within `scale` standard deviations of the mean. */
  readonly kind?: 'shift' | 'uniform'
  /** Size of the contamination in standard deviations (default 5). */
  readonly scale?: number
  readonly key: Key
}

/**
 * Contaminate a fraction of rows with outliers in the features and/or the regression targets (the gross-error model
 * behind robust regression and outlier-detection benchmarks). The truth marks the outliers and keeps the clean values;
 * the regression function and posterior still describe the inliers.
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - Fraction, what to contaminate, kind, scale and key.
 * @returns The contaminated data; `truth.latents.outliers` (bool `[n]`), `truth.latents.cleanX` and `cleanY`.
 * @throws MdmlError with code `value` when targets are contaminated for non-regression data.
 * @example
 * ```ts
 * import { asGenerated, withOutliers, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = withOutliers(asGenerated(xy([[0], [1], [2], [3]], [0, 1, 2, 3])), { fraction: 0.5, key: key(0) })
 * Array.from(d.truth.latents?.['outliers'].data).filter((v) => v === 1).length // => 2
 * ```
 */
export function withOutliers<D extends TabularData>(input: Generated<D>, options: OutlierOptions): Generated<D> {
  const { data, truth } = requireTabular(input, 'withOutliers') as Generated<D>
  const where = options.target ?? 'features'
  if (where !== 'features' && data.kind !== 'regression')
    throw new MdmlError('value', 'withOutliers: only regression targets can be contaminated')
  const scale = options.scale ?? 5
  const x = data.x
  const [n = 0, d = 1] = x.shape
  const m = Math.round((options.fraction ?? 0.05) * n)
  const rows = Array.from(permutation(foldIn(options.key, 0), n).data.slice(0, m))
  const isOut = new Uint8Array(n)
  for (const i of rows) isOut[i] = 1
  let newX: Tensor = x
  if (where !== 'targets') {
    const { mean, std } = columnStats(x)
    const values = Float64Array.from(x.data as ArrayLike<number>)
    const dir = normal(foldIn(options.key, 1), [m, d]).data
    const u = uniform(foldIn(options.key, 2), [m, d], { low: -1, high: 1 }).data
    for (const [r, i] of rows.entries()) {
      let norm = 0
      for (let j = 0; j < d; j++) norm += (dir[r * d + j] ?? 0) ** 2
      norm = Math.sqrt(norm) || 1
      for (let j = 0; j < d; j++) {
        const s = (std[j] ?? 0) || 1
        values[i * d + j] =
          options.kind === 'uniform'
            ? (mean[j] ?? 0) + scale * s * (u[r * d + j] ?? 0)
            : (values[i * d + j] ?? 0) + (scale * s * (dir[r * d + j] ?? 0)) / norm
      }
    }
    newX = fromStorage(values, [n, d], 'float64')
  }
  let newY: Tensor = data.y
  if (where !== 'features') {
    const y = data.y
    const ys = Float64Array.from(y.data as ArrayLike<number>)
    const k = ys.length / Math.max(1, n)
    const { std } = columnStats(y.shape.length === 1 ? y.backend.wrap(y.dtype, [n, 1], y.data) : y)
    const sign = uniform(foldIn(options.key, 3), [m]).data
    for (const [r, i] of rows.entries()) {
      for (let c = 0; c < k; c++)
        ys[i * k + c] = (ys[i * k + c] ?? 0) + ((sign[r] ?? 0) < 0.5 ? -1 : 1) * scale * ((std[c] ?? 0) || 1)
    }
    newY = fromStorage(ys, y.shape, 'float64')
  }
  return {
    data: { ...data, x: newX, y: newY },
    truth: {
      ...withoutProgram(truth),
      latents: { ...(truth.latents ?? {}), outliers: fromStorage(isOut, [n], 'bool'), cleanX: x, cleanY: data.y },
      extra: { ...(truth.extra ?? {}), outliers: { fraction: m / Math.max(1, n), target: where, scale } },
    },
  }
}

/** Options of {@link withNuisanceFeatures}. */
export interface NuisanceOptions {
  /** How many features to append (default 5). */
  readonly count?: number
  /** `gaussian` (default, standard normal), `uniform` (on [0, 1]) or `permuted` (row-shuffled copies of existing
   * features, matching their marginals: "shadow" features). */
  readonly kind?: 'gaussian' | 'uniform' | 'permuted'
  readonly key: Key
}

/**
 * Append features independent of everything else, for feature-selection and robustness studies. The truth is
 * extended consistently: the regression function and posterior ignore the new columns, the relevant features are
 * unchanged, and the log density gains the nuisance density (for Gaussian and uniform nuisance).
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - Count, kind and key.
 * @returns The widened data, with feature names `nuisance0, ...`.
 * @example
 * ```ts
 * import { asGenerated, withNuisanceFeatures, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * withNuisanceFeatures(asGenerated(xy([[1], [2]], [1, 2])), { count: 2, key: key(0) }).data.featureNames // => ['x0', 'nuisance0', 'nuisance1']
 * ```
 */
export function withNuisanceFeatures<D extends TabularData>(
  input: Generated<D>,
  options: NuisanceOptions,
): Generated<D> {
  const { data, truth } = requireTabular(input, 'withNuisanceFeatures') as Generated<D>
  const x = data.x
  const [n = 0, d = 1] = x.shape
  const count = options.count ?? 5
  const kind = options.kind ?? 'gaussian'
  const extra =
    kind === 'gaussian'
      ? normal(options.key, [n, count]).data
      : kind === 'uniform'
        ? uniform(options.key, [n, count]).data
        : (() => {
            const out = new Float64Array(n * count)
            for (let c = 0; c < count; c++) {
              const p = permutation(foldIn(options.key, c), n).data
              const j = c % d
              for (let i = 0; i < n; i++) out[i * count + c] = x.data[(p[i] ?? 0) * d + j] ?? 0
            }
            return out
          })()
  const wide = new Float64Array(n * (d + count))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) wide[i * (d + count) + j] = x.data[i * d + j] ?? 0
    for (let c = 0; c < count; c++) wide[i * (d + count) + d + c] = extra[i * count + c] ?? 0
  }
  const original = (v: Value): Value => sliceAxis(v, 1, 0, d)
  const nuisanceLogDensity = (v: Value): Value => {
    const block = sliceAxis(v, 1, d, d + count)
    if (kind === 'gaussian') return sum(mul(-0.5, add(mul(block, block), Math.log(2 * Math.PI))), 1)
    return full([v.shape[0] ?? 0], 0)
  }
  const logProb = truth.logProb
  const newTruth: Truth = {
    ...withoutProgram(truth),
    ...composedFunctions(truth, original),
    ...(logProb === undefined || kind === 'permuted'
      ? {}
      : { logProb: (v: Value, y?: Value) => add(logProb(original(v), y), nuisanceLogDensity(v)) }),
    relevantFeatures: truth.relevantFeatures ?? Array.from({ length: d }, (_, j) => j),
    extra: {
      ...(truth.extra ?? {}),
      nuisance: { count, kind, columns: Array.from({ length: count }, (_, c) => d + c) },
    },
  }
  if (kind === 'permuted') {
    const { logProb: _drop, ...rest } = newTruth
    return {
      data: widen(data, wide, n, d, count),
      truth: rest,
    }
  }
  return { data: widen(data, wide, n, d, count), truth: newTruth }
}

function widen<D extends TabularData>(data: D, wide: Float64Array, n: number, d: number, count: number): D {
  return {
    ...data,
    x: fromStorage(wide, [n, d + count], 'float64'),
    featureNames: [...data.featureNames, ...Array.from({ length: count }, (_, c) => `nuisance${c}`)],
  }
}

/**
 * An invertible elementwise feature transform with the log absolute derivative of its inverse, so a density pushed
 * through it stays exact: `log p'(x') = log p(inverse(x')) + logAbsDetInverse(x')`.
 */
export interface FeatureTransform {
  readonly name: string
  readonly forward: (x: Value) => Value
  readonly inverse: (y: Value) => Value
  /** Elementwise `log |d inverse(y) / dy|`. */
  readonly logAbsDetInverse: (y: Value) => Value
}

/** The built-in transforms of {@link withTransform}. */
export type TransformName = 'exp' | 'log' | 'sigmoid' | 'logit' | 'sinh' | 'arcsinh' | 'cube'

/**
 * A built-in invertible elementwise transform: `exp`, `log` (for positive features), `sigmoid`, `logit` (for features
 * in (0, 1)), `sinh`, `arcsinh` or `cube`.
 *
 * @param name - The transform.
 * @returns The transform with its inverse and log-Jacobian.
 * @example
 * ```ts
 * import { featureTransform } from 'mdml-engine/data'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(featureTransform('exp').inverse(tensor([1]))) // => [0]
 * ```
 */
export function featureTransform(name: TransformName): FeatureTransform {
  const cbrt = (v: Value): Value => mul(sign(v), pow(abs(v), 1 / 3))
  switch (name) {
    case 'exp':
      return { name, forward: exp, inverse: log, logAbsDetInverse: (y) => neg(log(y)) }
    case 'log':
      return { name, forward: log, inverse: exp, logAbsDetInverse: (y) => y }
    case 'sigmoid':
      // inverse logit(y), derivative 1 / (y (1 - y)).
      return {
        name,
        forward: sigmoid,
        inverse: (y) => log(div(y, sub(1, y))),
        logAbsDetInverse: (y) => neg(add(log(y), log(sub(1, y)))),
      }
    case 'logit':
      return {
        name,
        forward: (x) => log(div(x, sub(1, x))),
        inverse: sigmoid,
        logAbsDetInverse: (y) => add(log(sigmoid(y)), log(sub(1, sigmoid(y)))),
      }
    case 'sinh':
      return { name, forward: sinh, inverse: arcsinh, logAbsDetInverse: (y) => mul(-0.5, log(add(1, mul(y, y)))) }
    case 'arcsinh':
      return { name, forward: arcsinh, inverse: sinh, logAbsDetInverse: (y) => log(cosh(y)) }
    case 'cube':
      return {
        name,
        forward: (x) => mul(mul(x, x), x),
        inverse: cbrt,
        logAbsDetInverse: (y) => {
          const r = cbrt(y)
          return neg(log(mul(3, mul(r, r))))
        },
      }
  }
}

/** Push a truth through a change of the feature columns `x' = forward(x)` with inverse and log-Jacobian. */
function reparameterised(
  truth: Truth,
  inverseX: (x: Value) => Value,
  logDetInverse: (x: Value) => Value,
  original: Tensor,
  record: Readonly<Record<string, unknown>>,
): Truth {
  const logProb = truth.logProb
  return {
    ...withoutProgram(truth),
    ...composedFunctions(truth, inverseX),
    ...(logProb === undefined
      ? {}
      : { logProb: (v: Value, y?: Value) => add(logProb(inverseX(v), y), logDetInverse(v)) }),
    latents: { ...(truth.latents ?? {}), untransformed: original },
    extra: { ...(truth.extra ?? {}), ...record },
  }
}

/** Options of {@link withTransform}. */
export interface TransformOptions {
  /** A built-in transform name or a custom transform. */
  readonly transform: TransformName | FeatureTransform
  /** The columns to transform (default all). */
  readonly columns?: readonly number[]
}

/**
 * Transform feature columns by an invertible map, keeping the truth consistent: the regression function, posterior
 * and quantiles are composed with the inverse, and the log density gains the log-Jacobian of the inverse, so
 * `truth.logProb` remains the exact density of the transformed data.
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - The transform and the columns.
 * @returns The transformed data; `truth.latents.untransformed` keeps the original features.
 * @example
 * ```ts
 * import { asGenerated, withTransform, xy } from 'mdml-engine/data'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * const g = withTransform(asGenerated(xy([[0], [1]], [0, 2]), { regressionFunction: (x) => x }), { transform: 'exp' })
 * toFlat(g.truth.regressionFunction?.(tensor([[1]])) ?? tensor([])) // => [0]
 * ```
 */
export function withTransform<D extends TabularData>(input: Generated<D>, options: TransformOptions): Generated<D> {
  const { data, truth } = requireTabular(input, 'withTransform') as Generated<D>
  const t = typeof options.transform === 'string' ? featureTransform(options.transform) : options.transform
  const d = data.x.shape[1] ?? 1
  const cols = columnsOf(options.columns, d, 'withTransform')
  const x2 = mapColumns(data.x, cols, t.forward) as Tensor
  const inverseX = (v: Value) => mapColumns(v, cols, t.inverse)
  const logDet = (v: Value) => sum(t.logAbsDetInverse(selectColumns(v, cols)), 1)
  return {
    data: { ...data, x: x2 },
    truth: reparameterised(truth, inverseX, logDet, data.x, { transform: { name: t.name, columns: cols } }),
  }
}

/** Options of {@link withCorrelatedFeatures}. */
export interface CorrelationOptions {
  /** The correlation parameter: `rho^|i-j|` for `ar1`, `rho` between every pair for `equicorrelated`. */
  readonly rho: number
  /** `ar1` (default, Toeplitz) or `equicorrelated`. */
  readonly structure?: 'ar1' | 'equicorrelated'
  /** The columns to correlate (default all). */
  readonly columns?: readonly number[]
}

/** The mixing matrix `M` (with `M M^T` the target correlation), its inverse and `log|det M^{-1}|`. */
function correlationFactor(
  d: number,
  rho: number,
  structure: 'ar1' | 'equicorrelated',
): { m: number[][]; inv: number[][]; logDetInverse: number } {
  if (structure === 'ar1') {
    if (!(Math.abs(rho) < 1)) throw new MdmlError('value', 'withCorrelatedFeatures: ar1 needs |rho| < 1')
    const s = Math.sqrt(1 - rho * rho)
    const m = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => (j > i ? 0 : j === 0 ? rho ** i : s * rho ** (i - j))),
    )
    const inv = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => {
        if (i === 0) return j === 0 ? 1 : 0
        if (j === i) return 1 / s
        if (j === i - 1) return -rho / s
        return 0
      }),
    )
    return { m, inv, logDetInverse: -(d - 1) * Math.log(s) }
  }
  if (!(rho > -1 / Math.max(1, d - 1) && rho < 1))
    throw new MdmlError('value', `withCorrelatedFeatures: equicorrelation needs -1/(d-1) < rho < 1`)
  const a = Math.sqrt(1 - rho)
  const b = (Math.sqrt(1 - rho + d * rho) - a) / d
  const c = (1 / Math.sqrt(1 - rho + d * rho) - 1 / a) / d
  const m = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? a : 0) + b))
  const inv = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 / a : 0) + c))
  return { m, inv, logDetInverse: -0.5 * ((d - 1) * Math.log(1 - rho) + Math.log(1 + (d - 1) * rho)) }
}

/**
 * Correlate features by a linear mixing `x' = M x` (per row) whose `M M^T` is a Toeplitz (`ar1`, `rho^|i-j|`) or
 * equicorrelation matrix, so independent unit-variance features gain exactly that correlation. The truth is pushed
 * through the inverse map (with its constant log-Jacobian), so it stays exact.
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - Correlation parameter, structure and columns.
 * @returns The mixed data; `truth.extra.correlation` holds the mixing matrix.
 * @throws MdmlError with code `value` for a parameter outside the valid range.
 * @example
 * ```ts
 * import { asGenerated, withCorrelatedFeatures, xy } from 'mdml-engine/data'
 * const g = withCorrelatedFeatures(asGenerated(xy([[1, 0]], [0])), { rho: 0.5 })
 * Array.from(g.data.x.data) // => [1, 0.5]
 * ```
 */
export function withCorrelatedFeatures<D extends TabularData>(
  input: Generated<D>,
  options: CorrelationOptions,
): Generated<D> {
  const { data, truth } = requireTabular(input, 'withCorrelatedFeatures') as Generated<D>
  const d = data.x.shape[1] ?? 1
  const cols = columnsOf(options.columns, d, 'withCorrelatedFeatures')
  const structure = options.structure ?? 'ar1'
  const { m, inv, logDetInverse } = correlationFactor(cols.length, options.rho, structure)
  const mt = tensor(m.map((row, i) => row.map((_, j) => m[j]?.[i] ?? 0)))
  const invT = tensor(inv.map((row, i) => row.map((_, j) => inv[j]?.[i] ?? 0)))
  const x2 = mapColumns(data.x, cols, (block) => matmul(block, mt)) as Tensor
  const inverseX = (v: Value) => mapColumns(v, cols, (block) => matmul(block, invT))
  const logDet = (v: Value) => full([v.shape[0] ?? 0], logDetInverse)
  return {
    data: { ...data, x: x2 },
    truth: reparameterised(truth, inverseX, logDet, data.x, {
      correlation: { rho: options.rho, structure, columns: cols, mixing: m },
    }),
  }
}

/**
 * {@link withMissing} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { missingModifier } from 'mdml-engine/data'
 * missingModifier.name // => 'missing'
 * ```
 */
export const missingModifier: Modifier = {
  name: 'missing',
  description: 'Remove feature values under an MCAR, MAR or MNAR mechanism.',
  knobs: dict({
    mechanism: choice(['MCAR', 'MAR', 'MNAR'], { default: 'MCAR' }),
    rate: real(0, 1, { default: 0.2, label: 'missing rate' }),
    strength: real(0, 10, { default: 2, label: 'mechanism strength' }),
  }),
  apply: (key, input, knobs) =>
    withMissing(input as Generated<TabularData>, {
      mechanism: knobs['mechanism'] as MissingMechanism,
      rate: knobs['rate'] as number,
      strength: knobs['strength'] as number,
      key,
    }),
}

/**
 * {@link withOutliers} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { outliersModifier } from 'mdml-engine/data'
 * outliersModifier.name // => 'outliers'
 * ```
 */
export const outliersModifier: Modifier = {
  name: 'outliers',
  description: 'Contaminate a fraction of rows with gross errors.',
  knobs: dict({
    fraction: real(0, 0.5, { default: 0.05 }),
    target: choice(['features', 'targets', 'both'], { default: 'features' }),
    kind: choice(['shift', 'uniform'], { default: 'shift' }),
    scale: real(0, 100, { default: 5 }),
  }),
  apply: (key, input, knobs) =>
    withOutliers(input as Generated<TabularData>, {
      fraction: knobs['fraction'] as number,
      target: knobs['target'] as 'features' | 'targets' | 'both',
      kind: knobs['kind'] as 'shift' | 'uniform',
      scale: knobs['scale'] as number,
      key,
    }),
}

/**
 * {@link withNuisanceFeatures} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { nuisanceFeaturesModifier } from 'mdml-engine/data'
 * nuisanceFeaturesModifier.name // => 'nuisanceFeatures'
 * ```
 */
export const nuisanceFeaturesModifier: Modifier = {
  name: 'nuisanceFeatures',
  description: 'Append features independent of the target.',
  knobs: dict({
    count: int(0, 1000, { default: 5 }),
    kind: choice(['gaussian', 'uniform', 'permuted'], { default: 'gaussian' }),
  }),
  apply: (key, input, knobs) =>
    withNuisanceFeatures(input as Generated<TabularData>, {
      count: knobs['count'] as number,
      kind: knobs['kind'] as 'gaussian' | 'uniform' | 'permuted',
      key,
    }),
}

/**
 * {@link withTransform} (built-in transforms) as a recipe modifier.
 *
 * @example
 * ```ts
 * import { transformModifier } from 'mdml-engine/data'
 * transformModifier.name // => 'transform'
 * ```
 */
export const transformModifier: Modifier = {
  name: 'transform',
  description: 'Transform every feature by an invertible elementwise map.',
  knobs: dict({ transform: choice(['exp', 'sigmoid', 'sinh', 'arcsinh', 'cube'], { default: 'exp' }) }),
  apply: (_key, input, knobs) =>
    withTransform(input as Generated<TabularData>, { transform: knobs['transform'] as TransformName }),
}

/**
 * {@link withCorrelatedFeatures} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { correlatedFeaturesModifier } from 'mdml-engine/data'
 * correlatedFeaturesModifier.name // => 'correlatedFeatures'
 * ```
 */
export const correlatedFeaturesModifier: Modifier = {
  name: 'correlatedFeatures',
  description: 'Mix features to a Toeplitz or equicorrelation structure.',
  knobs: dict({
    rho: real(-0.99, 0.99, { default: 0.5, label: 'correlation' }),
    structure: choice(['ar1', 'equicorrelated'], { default: 'ar1' }),
  }),
  apply: (_key, input, knobs) =>
    withCorrelatedFeatures(input as Generated<TabularData>, {
      rho: knobs['rho'] as number,
      structure: knobs['structure'] as 'ar1' | 'equicorrelated',
    }),
}
