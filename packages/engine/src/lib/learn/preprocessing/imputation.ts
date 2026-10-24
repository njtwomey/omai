import { type Json, WhiteboxError } from '../../base/index.ts'
import { solve } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, permutation, uniform } from '../../random/index.ts'
import { choice, dict, int, type Space } from '../../space/index.ts'
import { quantile } from '../../stats/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { inputNames, keyFor, present } from '../_shared.ts'
import { defineEstimator, type Estimator, type Fitted, requireVerb } from '../estimator.ts'
import {
  columnOf,
  fromView,
  hstackViews,
  type MatrixView,
  matrixView,
  takeColumns,
  takeRows,
  targetVector,
} from '../validation.ts'

/** The verbs of a fitted imputer. */
export interface ImputerVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

function isMissing(v: number, marker: number): boolean {
  return Number.isNaN(marker) ? Number.isNaN(v) : v === marker
}

/** Hyperparameters of {@link missingIndicator}. */
export interface MissingIndicatorHyper {
  /** Indicate only the features with missing values in fit (default), or all features. */
  readonly features: 'missing-only' | 'all'
  /** The value that marks a missing entry (default NaN). */
  readonly missingValues: number
  /** Raise when transform finds missing values in a feature that had none in fit (default true). */
  readonly errorOnNew: boolean
}

/** Parameters of a fitted missing indicator. */
export type MissingIndicatorParams = { readonly features: number[]; readonly nFeatures: number }

function missingColumns(m: MatrixView, marker: number): number[] {
  const out: number[] = []
  for (let j = 0; j < m.d; j++) {
    for (let i = 0; i < m.n; i++) {
      if (isMissing(m.data[i * m.d + j] ?? 0, marker)) {
        out.push(j)
        break
      }
    }
  }
  return out
}

function indicatorView(m: MatrixView, features: readonly number[], marker: number): MatrixView {
  const out = new Float64Array(m.n * features.length)
  for (let i = 0; i < m.n; i++) {
    features.forEach((j, c) => {
      out[i * features.length + c] = isMissing(m.data[i * m.d + j] ?? 0, marker) ? 1 : 0
    })
  }
  return { n: m.n, d: features.length, data: out }
}

/**
 * Binary indicators of missing entries (sklearn.impute.MissingIndicator), for the features that had missing values in
 * fit (or all).
 *
 * @param hyper - features, missingValues and errorOnNew.
 * @returns The estimator.
 * @throws WhiteboxError with code `value` in transform for missing values in a new feature when `errorOnNew`.
 * @example
 * ```ts
 * import { missingIndicator } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const m = missingIndicator().fit({ x: [[1, NaN], [2, 3]] })
 * toArray(m.transform([[NaN, NaN], [1, 2]])) // => throws value
 * toArray(m.transform([[1, NaN], [1, 2]])) // => [[1], [0]]
 * ```
 */
export function missingIndicator(
  hyper: Partial<MissingIndicatorHyper> = {},
): Estimator<MissingIndicatorHyper, Fitted<MissingIndicatorParams> & ImputerVerbs> {
  return defineEstimator(
    {
      name: 'missing-indicator',
      kind: 'transformer',
      fit: (data, h) => {
        const m = matrixView(data.x, 'missingIndicator', { allowNaN: true })
        const features =
          h.features === 'all' ? Array.from({ length: m.d }, (_, j) => j) : missingColumns(m, h.missingValues)
        return { features, nFeatures: m.d }
      },
      methods: (p, h) => ({
        nFeaturesIn: p.nFeatures,
        transform: (x) => {
          const m = matrixView(x, 'missingIndicator', { allowNaN: true, nFeatures: p.nFeatures })
          if (h.errorOnNew && h.features === 'missing-only') {
            const known = new Set(p.features)
            const fresh = missingColumns(m, h.missingValues).filter((j) => !known.has(j))
            if (fresh.length > 0) {
              throw new WhiteboxError(
                'value',
                `missingIndicator: features ${fresh.join(', ')} have missing values only in transform`,
              )
            }
          }
          return fromView(indicatorView(m, p.features, h.missingValues))
        },
        featureNamesOut: (input) => {
          const names = inputNames(p.nFeatures, input, 'missingIndicator')
          return p.features.map((j) => `missingindicator_${names[j]}`)
        },
      }),
    },
    {
      features: hyper.features ?? 'missing-only',
      missingValues: hyper.missingValues ?? Number.NaN,
      errorOnNew: hyper.errorOnNew ?? true,
    },
  )
}

/** Hyperparameters of {@link simpleImputer}. */
export interface SimpleImputerHyper {
  readonly strategy: 'mean' | 'median' | 'most_frequent' | 'constant'
  /** The value for `'constant'` (default 0). */
  readonly fillValue: number
  /** The value that marks a missing entry (default NaN). */
  readonly missingValues: number
  /** Append missing indicators of the features with missing values in fit (default false). */
  readonly addIndicator: boolean
  /** Keep features with no observed values (filled with 0, or `fillValue`) instead of dropping them (default false). */
  readonly keepEmptyFeatures: boolean
}

/** Parameters of a fitted simple imputer. */
export type SimpleImputerParams = {
  /** The fill value of each feature (null for an empty feature that is dropped). */
  readonly statistics: (number | null)[]
  /** Features with missing values in fit (indicated when `addIndicator`). */
  readonly indicated: number[]
}

function statistic(values: Float64Array, strategy: SimpleImputerHyper['strategy'], fill: number): number {
  if (strategy === 'constant') return fill
  if (values.length === 0) return Number.NaN
  if (strategy === 'mean') return values.reduce((a, b) => a + b, 0) / values.length
  if (strategy === 'median') return quantile(values, 0.5)
  const counts = new Map<number, number>()
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1)
  let best = Number.NaN
  let bestCount = 0
  for (const [v, c] of counts) if (c > bestCount || (c === bestCount && v < best)) [best, bestCount] = [v, c]
  return best
}

/**
 * Fill missing values with a per-feature statistic (sklearn.impute.SimpleImputer): the mean, median, most frequent
 * value (smallest on ties) or a constant. Features never observed in fit are dropped (or kept and filled with 0);
 * `addIndicator` appends indicators, and with them `inverseTransform` restores the missing entries.
 *
 * @param hyper - strategy (default `'mean'`), fillValue, missingValues, addIndicator and keepEmptyFeatures.
 * @returns The estimator.
 * @concept imputation
 * @example
 * ```ts
 * import { simpleImputer } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const s = simpleImputer().fit({ x: [[1, 2], [NaN, 4], [3, NaN]] })
 * toArray(s.transform([[NaN, NaN]])) // => [[2, 3]]
 * toArray(simpleImputer({ strategy: 'most_frequent' }).fit({ x: [[1], [2], [2]] }).transform([[NaN]])) // => [[2]]
 * ```
 */
export function simpleImputer(
  hyper: Partial<SimpleImputerHyper> = {},
): Estimator<
  SimpleImputerHyper,
  Fitted<SimpleImputerParams> & ImputerVerbs & { readonly inverseTransform: (x: TensorLike) => Tensor }
> {
  return defineEstimator(
    {
      name: 'simple-imputer',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ strategy: choice(['mean', 'median', 'most_frequent']) }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'simpleImputer', { allowNaN: true })
        const statistics: (number | null)[] = []
        for (let j = 0; j < m.d; j++) {
          const col = columnOf(m, j)
          const observed = Number.isNaN(h.missingValues) ? present(col) : col.filter((v) => v !== h.missingValues)
          const s = statistic(observed, h.strategy, h.fillValue)
          statistics.push(Number.isNaN(s) ? (h.keepEmptyFeatures ? 0 : null) : s)
        }
        return { statistics, indicated: h.addIndicator ? missingColumns(m, h.missingValues) : [] }
      },
      methods: (p, h) => {
        const d = p.statistics.length
        const kept = p.statistics.flatMap((s, j) => (s === null ? [] : [j]))
        return {
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'simpleImputer', { allowNaN: true, nFeatures: d })
            const filled = new Float64Array(m.data.length)
            for (let i = 0; i < m.data.length; i++) {
              const v = m.data[i] ?? 0
              filled[i] = isMissing(v, h.missingValues) ? (p.statistics[i % d] ?? Number.NaN) : v
            }
            const out = takeColumns({ n: m.n, d, data: filled }, kept)
            return fromView(h.addIndicator ? hstackViews([out, indicatorView(m, p.indicated, h.missingValues)]) : out)
          },
          inverseTransform: (x) => {
            if (!h.addIndicator) throw new WhiteboxError('value', 'simpleImputer: inverseTransform needs addIndicator')
            const width = kept.length + p.indicated.length
            const m = matrixView(x, 'simpleImputer', { nFeatures: width })
            const out = new Float64Array(m.n * d).fill(Number.NaN)
            for (let i = 0; i < m.n; i++) {
              kept.forEach((j, c) => {
                out[i * d + j] = m.data[i * width + c] ?? 0
              })
              p.indicated.forEach((j, c) => {
                if ((m.data[i * width + kept.length + c] ?? 0) === 1) out[i * d + j] = h.missingValues
              })
            }
            return fromStorage(out, [m.n, d], 'float64')
          },
          featureNamesOut: (input) => {
            const names = inputNames(d, input, 'simpleImputer')
            return [
              ...kept.map((j) => names[j] ?? ''),
              ...(h.addIndicator ? p.indicated.map((j) => `missingindicator_${names[j]}`) : []),
            ]
          },
        }
      },
    },
    {
      strategy: hyper.strategy ?? 'mean',
      fillValue: hyper.fillValue ?? 0,
      missingValues: hyper.missingValues ?? Number.NaN,
      addIndicator: hyper.addIndicator ?? false,
      keepEmptyFeatures: hyper.keepEmptyFeatures ?? false,
    },
  )
}

/** Hyperparameters of {@link knnImputer}. */
export interface KnnImputerHyper {
  /** Neighbours averaged (default 5). */
  readonly nNeighbors: number
  /** Equal weights or inverse-distance weights. */
  readonly weights: 'uniform' | 'distance'
  readonly missingValues: number
  readonly addIndicator: boolean
  readonly keepEmptyFeatures: boolean
}

/** Parameters of a fitted k-NN imputer: the training rows (missing values as null). */
export type KnnImputerParams = { readonly rows: (number | null)[][]; readonly indicated: number[] }

/** sklearn's nan_euclidean distance: sqrt(d / present * sum over coordinates present in both). */
function nanEuclidean(a: ArrayLike<number>, b: ArrayLike<number>, d: number): number {
  let s = 0
  let k = 0
  for (let j = 0; j < d; j++) {
    const u = a[j] ?? Number.NaN
    const v = b[j] ?? Number.NaN
    if (Number.isNaN(u) || Number.isNaN(v)) continue
    s += (u - v) ** 2
    k++
  }
  return k === 0 ? Number.NaN : Math.sqrt((d / k) * s)
}

/**
 * Fill each missing entry from the k nearest training rows that observe that feature (sklearn.impute.KNNImputer):
 * distances are NaN-Euclidean over the coordinates both rows observe, rescaled by the fraction observed; the value is
 * the (inverse-distance weighted) mean of the donors; with no usable distance, the feature's mean.
 *
 * @param hyper - nNeighbors (default 5), weights, missingValues, addIndicator and keepEmptyFeatures.
 * @returns The estimator.
 * @example
 * ```ts
 * import { knnImputer } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const k = knnImputer({ nNeighbors: 2 }).fit({ x: [[0, 0], [1, 10], [2, 20], [9, 90]] })
 * toArray(k.transform([[1.4, NaN]])) // => [[1.4, 15]]
 * ```
 */
export function knnImputer(
  hyper: Partial<KnnImputerHyper> = {},
): Estimator<KnnImputerHyper, Fitted<KnnImputerParams> & ImputerVerbs> {
  return defineEstimator(
    {
      name: 'knn-imputer',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ nNeighbors: int(1, 30), weights: choice(['uniform', 'distance']) }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'knnImputer', { allowNaN: true })
        const rows = Array.from({ length: m.n }, (_, i) =>
          Array.from(m.data.subarray(i * m.d, (i + 1) * m.d), (v) => (isMissing(v, h.missingValues) ? null : v)),
        )
        return { rows, indicated: h.addIndicator ? missingColumns(m, h.missingValues) : [] }
      },
      methods: (p, h) => {
        const d = p.rows[0]?.length ?? 0
        const fit = p.rows.map((r) => Float64Array.from(r, (v) => (v === null ? Number.NaN : v)))
        const valid = Array.from({ length: d }, (_, j) => fit.some((r) => !Number.isNaN(r[j] ?? Number.NaN)))
        const means = Array.from({ length: d }, (_, j) => {
          const v = present(Float64Array.from(fit, (r) => r[j] ?? Number.NaN))
          return v.length === 0 ? 0 : v.reduce((a, b) => a + b, 0) / v.length
        })
        const kept = valid.flatMap((ok, j) => (ok || h.keepEmptyFeatures ? [j] : []))
        return {
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'knnImputer', { allowNaN: true, nFeatures: d })
            const raw = Float64Array.from(m.data, (v) => (isMissing(v, h.missingValues) ? Number.NaN : v))
            const out = Float64Array.from(raw)
            for (let i = 0; i < m.n; i++) {
              const row = raw.subarray(i * d, (i + 1) * d)
              if (!row.some(Number.isNaN)) continue
              const dist = fit.map((f) => nanEuclidean(row, f, d))
              for (let j = 0; j < d; j++) {
                if (!Number.isNaN(row[j] ?? 0)) continue
                if (!valid[j]) {
                  out[i * d + j] = 0
                  continue
                }
                const donors = fit.flatMap((f, t) => (Number.isNaN(f[j] ?? Number.NaN) ? [] : [t]))
                if (donors.every((t) => Number.isNaN(dist[t] ?? Number.NaN))) {
                  out[i * d + j] = means[j] ?? 0
                  continue
                }
                const k = Math.min(h.nNeighbors, donors.length)
                const nearest = donors
                  .map((t) => ({ t, dd: dist[t] ?? Number.NaN }))
                  .sort((a, b) => (Number.isNaN(a.dd) ? 1 : Number.isNaN(b.dd) ? -1 : a.dd - b.dd || a.t - b.t))
                  .slice(0, k)
                const anyZero = nearest.some((e) => e.dd === 0)
                let sw = 0
                let s = 0
                for (const e of nearest) {
                  const w = Number.isNaN(e.dd)
                    ? 0
                    : h.weights === 'uniform'
                      ? 1
                      : anyZero
                        ? e.dd === 0
                          ? 1
                          : 0
                        : 1 / e.dd
                  sw += w
                  s += w * (fit[e.t]?.[j] ?? 0)
                }
                out[i * d + j] = s / sw
              }
            }
            const imputed = takeColumns({ n: m.n, d, data: out }, kept)
            return fromView(
              h.addIndicator ? hstackViews([imputed, indicatorView(m, p.indicated, h.missingValues)]) : imputed,
            )
          },
          featureNamesOut: (input) => {
            const names = inputNames(d, input, 'knnImputer')
            return [
              ...kept.map((j) => names[j] ?? ''),
              ...(h.addIndicator ? p.indicated.map((j) => `missingindicator_${names[j]}`) : []),
            ]
          },
        }
      },
    },
    {
      nNeighbors: hyper.nNeighbors ?? 5,
      weights: hyper.weights ?? 'uniform',
      missingValues: hyper.missingValues ?? Number.NaN,
      addIndicator: hyper.addIndicator ?? false,
      keepEmptyFeatures: hyper.keepEmptyFeatures ?? false,
    },
  )
}

// The regressor iterativeImputer uses by default: ridge regression with intercept, in closed form.
function ridgeImputationModel(alpha = 1): Estimator {
  return defineEstimator(
    {
      name: 'imputation-ridge',
      kind: 'regressor',
      fit: (data) => {
        const m = matrixView(data.x, 'imputationRidge')
        const y = targetVector(data.y ?? [], 'imputationRidge', m.n)
        const xm = Array.from({ length: m.d }, (_, j) => columnOf(m, j).reduce((a, b) => a + b, 0) / m.n)
        const ym = y.reduce((a, b) => a + b, 0) / m.n
        const g = new Float64Array(m.d * m.d)
        const b = new Float64Array(m.d)
        for (let i = 0; i < m.n; i++) {
          for (let j = 0; j < m.d; j++) {
            const xj = (m.data[i * m.d + j] ?? 0) - (xm[j] ?? 0)
            b[j] = (b[j] ?? 0) + xj * ((y[i] ?? 0) - ym)
            for (let k = 0; k < m.d; k++)
              g[j * m.d + k] = (g[j * m.d + k] ?? 0) + xj * ((m.data[i * m.d + k] ?? 0) - (xm[k] ?? 0))
          }
        }
        for (let j = 0; j < m.d; j++) g[j * m.d + j] = (g[j * m.d + j] ?? 0) + alpha
        const coef = toFlat(solve(fromStorage(g, [m.d, m.d], 'float64'), fromStorage(b, [m.d], 'float64')))
        return { coef, intercept: ym - coef.reduce((s, c, j) => s + c * (xm[j] ?? 0), 0) }
      },
      methods: (p: { readonly coef: number[]; readonly intercept: number }) => ({
        predict: (x: TensorLike): Tensor => {
          const m = matrixView(x, 'imputationRidge')
          return fromStorage(
            Float64Array.from({ length: m.n }, (_, i) =>
              p.coef.reduce((s, c, j) => s + c * (m.data[i * m.d + j] ?? 0), p.intercept),
            ),
            [m.n],
            'float64',
          )
        },
      }),
    },
    { alpha },
  )
}

/** Hyperparameters of {@link iterativeImputer}. */
export interface IterativeImputerHyper {
  /** The regressor fitted per feature (default ridge regression, alpha 1). */
  readonly estimator: Estimator
  /** Rounds over the features (default 10). */
  readonly maxIter: number
  /** Stop when the largest row sum of absolute changes falls below tol times the largest observed magnitude. */
  readonly tol: number
  readonly initialStrategy: 'mean' | 'median' | 'most_frequent' | 'constant'
  /** The order features are imputed in each round. */
  readonly imputationOrder: 'ascending' | 'descending' | 'roman' | 'arabic' | 'random'
  /** Skip features with no missing values in fit (default false: they are still modelled). */
  readonly skipComplete: boolean
  /** Use only this many other features as predictors, drawn by absolute correlation (default all). */
  readonly nNearestFeatures?: number
  /** Clip imputed values to these bounds. */
  readonly minValue: number
  readonly maxValue: number
  readonly missingValues: number
  readonly addIndicator: boolean
}

/** One step of a fitted iterative imputation: a feature, its predictors and the fitted regressor's parameters. */
export type ImputationStep = { readonly feature: number; readonly predictors: number[]; readonly params: Json }

/** Parameters of a fitted iterative imputer. */
export type IterativeImputerParams = {
  readonly initial: SimpleImputerParams
  readonly steps: ImputationStep[]
  readonly nIter: number
  readonly kept: number[]
  readonly indicated: number[]
  readonly nFeatures: number
}

/**
 * Multivariate imputation by chained equations (sklearn.impute.IterativeImputer): start from a simple imputation, then
 * for a number of rounds regress each feature with missing values on the others (fitted on the rows where it is
 * observed) and replace its missing entries by the predictions, until the imputations stop changing. `transform`
 * replays the fitted sequence of regressors on new data.
 *
 * @param hyper - estimator, maxIter (default 10), tol (default 1e-3), initialStrategy, imputationOrder and limits.
 * @returns The estimator.
 * @example
 * ```ts
 * import { iterativeImputer } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const x = [[1, 2], [2, 4], [3, 6], [4, 8], [5, NaN]]
 * const f = iterativeImputer().fit({ x })
 * // The ridge regressor shrinks the slope a little: the fill is near, but below, 12.
 * const filled = toArray(f.transform([[6, NaN]]))[0][1]
 * filled > 10 && filled < 12 // => true
 * ```
 */
export function iterativeImputer(
  hyper: Partial<IterativeImputerHyper> = {},
): Estimator<IterativeImputerHyper, Fitted<IterativeImputerParams> & ImputerVerbs> {
  return defineEstimator(
    {
      name: 'iterative-imputer',
      kind: 'transformer',
      hyperSpace: (): Space =>
        dict({ maxIter: int(1, 50), imputationOrder: choice(['ascending', 'descending', 'roman', 'arabic']) }),
      fit: (data, h, options) => {
        const raw = matrixView(data.x, 'iterativeImputer', { allowNaN: true })
        const initialEst = simpleImputer({ strategy: h.initialStrategy, missingValues: h.missingValues })
        const initial = initialEst.fit({ x: fromView(raw) })
        const kept = initial.params.statistics.flatMap((s, j) => (s === null ? [] : [j]))
        const indicated = h.addIndicator ? missingColumns(raw, h.missingValues) : []
        const x = takeColumns(raw, kept)
        const mask = Uint8Array.from(x.data, (v) => (isMissing(v, h.missingValues) ? 1 : 0))
        let filled = matrixView(initial.transform(fromView(raw)), 'iterativeImputer')
        const steps: ImputationStep[] = []
        const key = keyFor(options, 'iterativeImputer')
        const d = x.d
        const frac = Array.from({ length: d }, (_, j) => {
          let c = 0
          for (let i = 0; i < x.n; i++) c += mask[i * d + j] ?? 0
          return c / x.n
        })
        const order = (round: number): number[] => {
          const idx = Array.from({ length: d }, (_, j) => j).filter((j) => !h.skipComplete || (frac[j] ?? 0) > 0)
          const nSkip = d - idx.length
          const asc = Array.from({ length: d }, (_, j) => j)
            .sort((a, b) => (frac[a] ?? 0) - (frac[b] ?? 0) || a - b)
            .slice(nSkip)
          if (h.imputationOrder === 'roman') return idx
          if (h.imputationOrder === 'arabic') return idx.reverse()
          if (h.imputationOrder === 'ascending') return asc
          if (h.imputationOrder === 'descending') return asc.reverse()
          return toFlat(permutation(foldIn(key, round), idx.length)).map((r) => idx[r] ?? 0)
        }
        let nIter = 0
        let observedMax = 0
        for (let i = 0; i < x.data.length; i++)
          if (mask[i] === 0) observedMax = Math.max(observedMax, Math.abs(x.data[i] ?? 0))
        const tol = h.tol * observedMax
        const weights =
          h.nNearestFeatures === undefined || h.nNearestFeatures >= d - 1 ? undefined : absCorrelations(filled)
        const anyMissing = mask.some((v) => v === 1)
        if (anyMissing && d > 1 && h.maxIter > 0 && !mask.every((v) => v === 1)) {
          for (let round = 1; round <= h.maxIter; round++) {
            nIter = round
            const previous = Float64Array.from(filled.data)
            for (const j of order(round)) {
              const others = Array.from({ length: d }, (_, k) => k).filter((k) => k !== j)
              const predictors =
                weights === undefined || h.nNearestFeatures === undefined
                  ? others
                  : drawPredictors(weights, j, h.nNearestFeatures, foldIn(foldIn(key, 1000 + round), j))
              const observedRows: number[] = []
              const missingRows: number[] = []
              for (let i = 0; i < x.n; i++) (mask[i * d + j] === 1 ? missingRows : observedRows).push(i)
              const xp = takeColumns(filled, predictors)
              const target = takeColumns(takeRows(filled, observedRows), [j])
              const fitted = h.estimator.fit({ x: fromView(takeRows(xp, observedRows)), y: toFlat(fromView(target)) })
              steps.push({ feature: j, predictors, params: fitted.params })
              if (missingRows.length === 0) continue
              const pred = toFlat(
                requireVerb(fitted, 'predict', 'iterativeImputer')(fromView(takeRows(xp, missingRows))),
              )
              const next = Float64Array.from(filled.data)
              missingRows.forEach((i, r) => {
                next[i * d + j] = Math.min(Math.max(pred[r] ?? 0, h.minValue), h.maxValue)
              })
              filled = { ...filled, data: next }
            }
            let change = 0
            for (let i = 0; i < x.n; i++) {
              let rowSum = 0
              for (let j = 0; j < d; j++) rowSum += Math.abs((filled.data[i * d + j] ?? 0) - (previous[i * d + j] ?? 0))
              change = Math.max(change, rowSum)
            }
            if (change < tol) break
          }
        }
        return { initial: initial.params, steps, nIter, kept, indicated, nFeatures: raw.d }
      },
      methods: (p, h) => {
        const initial = simpleImputer({ strategy: h.initialStrategy, missingValues: h.missingValues }).revive(p.initial)
        const models = p.steps.map((s) => ({ ...s, fitted: h.estimator.revive(s.params) }))
        return {
          nFeaturesIn: p.nFeatures,
          transform: (x) => {
            const raw = matrixView(x, 'iterativeImputer', { allowNaN: true, nFeatures: p.nFeatures })
            const masked = takeColumns(raw, p.kept)
            let filled = matrixView(initial.transform(fromView(raw)), 'iterativeImputer')
            const d = filled.d
            for (const s of models) {
              const rows: number[] = []
              for (let i = 0; i < filled.n; i++)
                if (isMissing(masked.data[i * d + s.feature] ?? 0, h.missingValues)) rows.push(i)
              if (rows.length === 0) continue
              const pred = toFlat(
                requireVerb(
                  s.fitted,
                  'predict',
                  'iterativeImputer',
                )(fromView(takeRows(takeColumns(filled, s.predictors), rows))),
              )
              const next = Float64Array.from(filled.data)
              rows.forEach((i, r) => {
                next[i * d + s.feature] = Math.min(Math.max(pred[r] ?? 0, h.minValue), h.maxValue)
              })
              filled = { ...filled, data: next }
            }
            return fromView(
              h.addIndicator ? hstackViews([filled, indicatorView(raw, p.indicated, h.missingValues)]) : filled,
            )
          },
          featureNamesOut: (input) => {
            const names = inputNames(p.nFeatures, input, 'iterativeImputer')
            return [...p.kept.map((j) => names[j] ?? ''), ...p.indicated.map((j) => `missingindicator_${names[j]}`)]
          },
        }
      },
    },
    {
      estimator: hyper.estimator ?? ridgeImputationModel(),
      maxIter: hyper.maxIter ?? 10,
      tol: hyper.tol ?? 1e-3,
      initialStrategy: hyper.initialStrategy ?? 'mean',
      imputationOrder: hyper.imputationOrder ?? 'ascending',
      skipComplete: hyper.skipComplete ?? false,
      ...(hyper.nNearestFeatures === undefined ? {} : { nNearestFeatures: hyper.nNearestFeatures }),
      minValue: hyper.minValue ?? Number.NEGATIVE_INFINITY,
      maxValue: hyper.maxValue ?? Number.POSITIVE_INFINITY,
      missingValues: hyper.missingValues ?? Number.NaN,
      addIndicator: hyper.addIndicator ?? false,
    },
  )
}

// sklearn's _get_abs_corr_mat: |corrcoef| of the columns, NaN and small values raised to 1e-6, zero diagonal, columns
// normalised to sum to one (column j is the draw distribution of feature j's predictors).
function absCorrelations(m: MatrixView): number[][] {
  const cols = Array.from({ length: m.d }, (_, j) => columnOf(m, j))
  const centred = cols.map((c) => {
    const mu = c.reduce((a, b) => a + b, 0) / c.length
    return c.map((v) => v - mu)
  })
  const corr = centred.map((a, i) =>
    centred.map((b, j) => {
      if (i === j) return 0
      let ab = 0
      let aa = 0
      let bb = 0
      for (let t = 0; t < a.length; t++) {
        ab += (a[t] ?? 0) * (b[t] ?? 0)
        aa += (a[t] ?? 0) ** 2
        bb += (b[t] ?? 0) ** 2
      }
      const r = Math.abs(ab / Math.sqrt(aa * bb))
      return Number.isNaN(r) ? 1e-6 : Math.max(r, 1e-6)
    }),
  )
  return corr.map((_, j) => {
    const total = corr.reduce((s, row) => s + (row[j] ?? 0), 0)
    return corr.map((row) => (row[j] ?? 0) / total)
  })
}

// Draw k predictors of feature j without replacement, each draw proportional to the remaining weights.
function drawPredictors(weights: readonly (readonly number[])[], j: number, k: number, key: Key): number[] {
  const w = [...(weights[j] ?? [])]
  const u = toFlat(uniform(key, [k]))
  const out: number[] = []
  for (let r = 0; r < k; r++) {
    const total = w.reduce((a, b) => a + b, 0)
    let target = (u[r] ?? 0) * total
    let pick = w.findIndex((v) => v > 0)
    for (let c = 0; c < w.length; c++) {
      if ((w[c] ?? 0) <= 0) continue
      pick = c
      target -= w[c] ?? 0
      if (target < 0) break
    }
    out.push(pick)
    w[pick] = 0
  }
  return out
}
