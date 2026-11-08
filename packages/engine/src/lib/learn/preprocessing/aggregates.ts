/**
 * Aggregate features over groups and windows, the building blocks of tabular and time-series feature engineering: per-row
 * group statistics (the mean, spread or count of the group a row belongs to, optionally leaving the row itself out), rolling
 * window statistics of a series (pandas' `rolling` with trailing windows, `min_periods` and `center`), and lagged copies.
 * Entries without enough history are NaN, as in pandas. These are pure functions of the table; the estimators that learn
 * from a target (target encoding) are in `encoders.ts`.
 */

import { maxOf, minOf, OmaiError } from '../../base/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { matrixView } from '../validation.ts'

/** A statistic of a set of values. */
export type Aggregate = 'mean' | 'sum' | 'min' | 'max' | 'std' | 'median' | 'count'

function aggregateOf(kind: Aggregate, v: readonly number[]): number {
  if (kind === 'count') return v.length
  if (v.length === 0) return Number.NaN
  if (kind === 'sum' || kind === 'mean') {
    let s = 0
    for (const x of v) s += x
    return kind === 'sum' ? s : s / v.length
  }
  if (kind === 'min') return minOf(v)
  if (kind === 'max') return maxOf(v)
  if (kind === 'median') {
    const s = [...v].sort((a, b) => a - b)
    const mid = Math.floor(s.length / 2)
    return s.length % 2 === 1 ? (s[mid] ?? Number.NaN) : ((s[mid - 1] ?? 0) + (s[mid] ?? 0)) / 2
  }
  if (v.length < 2) return Number.NaN
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  return Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1))
}

/** Options of {@link groupAggregates}. */
export interface GroupAggregateOptions {
  /** Leave the row itself out of its group's statistic (a leave-one-out statistic; default false). */
  readonly excludeSelf?: boolean
}

/**
 * Per-row group statistics, like pandas `groupby(...).transform`: for each row, the statistic of the values of the rows that
 * share its group, for each column of `values` and each requested statistic (columns ordered value-major: all statistics of
 * column 0, then column 1, ...). The standard deviation uses n - 1 and is NaN for fewer than two values.
 *
 * @param groups - The group of each row (any numbers).
 * @param values - The values `[n]` or `[n, d]`.
 * @param aggregates - The statistics.
 * @param options - Whether to leave the row out of its own group.
 * @returns The features `[n, d * aggregates.length]`.
 * @concept feature-engineering
 * @example
 * ```ts
 * import { groupAggregates } from 'omai-engine/learn/preprocessing'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(groupAggregates([0, 0, 1], [[1], [3], [5]], ['mean', 'count'])) // => [[2, 2], [2, 2], [5, 1]]
 * ```
 */
export function groupAggregates(
  groups: ArrayLike<number>,
  values: TensorLike,
  aggregates: readonly Aggregate[],
  options: GroupAggregateOptions = {},
): Tensor {
  const t =
    Array.isArray(values) && !Array.isArray((values as unknown[])[0]) ? (values as number[]).map((v) => [v]) : values
  const m = matrixView(t as TensorLike, 'groupAggregates', { allowNaN: true })
  if (groups.length !== m.n) throw new OmaiError('shape', 'groupAggregates: one group per row')
  const members = new Map<number, number[]>()
  for (let i = 0; i < m.n; i++) {
    const g = groups[i] ?? 0
    members.set(g, [...(members.get(g) ?? []), i])
  }
  const k = aggregates.length
  const out = new Float64Array(m.n * m.d * k)
  for (let i = 0; i < m.n; i++) {
    const rows = (members.get(groups[i] ?? 0) ?? []).filter((r) => !(options.excludeSelf === true && r === i))
    for (let j = 0; j < m.d; j++) {
      const v = rows.map((r) => m.data[r * m.d + j] ?? 0)
      aggregates.forEach((a, c) => {
        out[i * m.d * k + j * k + c] = aggregateOf(a, v)
      })
    }
  }
  return fromStorage(out, [m.n, m.d * k], 'float64')
}

/** Options of {@link rollingFeatures}. */
export interface RollingOptions {
  /** The fewest values in a window for the statistic to be defined (default the window length, pandas' default). */
  readonly minPeriods?: number
  /** Centre the window on the row instead of ending at it (default false). */
  readonly center?: boolean
}

/**
 * Rolling-window statistics of a series, like pandas `Series.rolling(window, min_periods, center)`: the window of the last
 * `window` values including the current one (or centred on it), the statistic NaN when it holds fewer than `minPeriods`
 * values. For a matrix the columns are separate series; the output columns are ordered series-major, then statistic.
 *
 * @param series - The series `[n]` or several `[n, d]`.
 * @param window - The window length.
 * @param aggregates - The statistics.
 * @param options - minPeriods and center.
 * @returns The features `[n, d * aggregates.length]`.
 * @concept feature-engineering
 * @example
 * ```ts
 * import { rollingFeatures } from 'omai-engine/learn/preprocessing'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(rollingFeatures([1, 2, 4, 8], 2, ['mean'])) // => [[NaN], [1.5], [3], [6]]
 * ```
 */
export function rollingFeatures(
  series: TensorLike,
  window: number,
  aggregates: readonly Aggregate[],
  options: RollingOptions = {},
): Tensor {
  if (!(Number.isInteger(window) && window >= 1))
    throw new OmaiError('value', 'rollingFeatures: window must be a positive integer')
  const t =
    Array.isArray(series) && !Array.isArray((series as unknown[])[0]) ? (series as number[]).map((v) => [v]) : series
  const m = matrixView(t as TensorLike, 'rollingFeatures', { allowNaN: true })
  const minPeriods = options.minPeriods ?? window
  const k = aggregates.length
  const out = new Float64Array(m.n * m.d * k)
  for (let i = 0; i < m.n; i++) {
    const lo = options.center === true ? i - Math.floor(window / 2) : i - window + 1
    const hi = lo + window - 1
    for (let j = 0; j < m.d; j++) {
      const v: number[] = []
      for (let r = Math.max(0, lo); r <= Math.min(m.n - 1, hi); r++) {
        const x = m.data[r * m.d + j] ?? Number.NaN
        if (!Number.isNaN(x)) v.push(x)
      }
      aggregates.forEach((a, c) => {
        out[i * m.d * k + j * k + c] = v.length >= minPeriods ? aggregateOf(a, v) : Number.NaN
      })
    }
  }
  return fromStorage(out, [m.n, m.d * k], 'float64')
}

/**
 * Lagged copies of a series, like pandas `shift`: column l holds the value `lags[l]` rows earlier (NaN at the start); a
 * negative lag looks ahead.
 *
 * @param series - The series `[n]`.
 * @param lags - The lags.
 * @returns The features `[n, lags.length]`.
 * @concept feature-engineering
 * @example
 * ```ts
 * import { lagFeatures } from 'omai-engine/learn/preprocessing'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(lagFeatures([1, 2, 3], [1, 2])) // => [[NaN, NaN], [1, NaN], [2, 1]]
 * ```
 */
export function lagFeatures(series: ArrayLike<number>, lags: readonly number[]): Tensor {
  const n = series.length
  const out = new Float64Array(n * lags.length).fill(Number.NaN)
  for (let i = 0; i < n; i++)
    lags.forEach((l, c) => {
      const r = i - l
      if (r >= 0 && r < n) out[i * lags.length + c] = series[r] ?? Number.NaN
    })
  return fromStorage(out, [n, lags.length], 'float64')
}
