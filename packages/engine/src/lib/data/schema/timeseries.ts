import { WhiteboxError } from '../../base/index.ts'
import { arange, fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { checkRows, concrete, concreteFloat, defaultNames, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/** A labelled moment in a series (an annotation, an intervention, a known changepoint). */
export interface TimeSeriesEvent {
  readonly time: number
  readonly label: string
}

/** One multichannel series: one row per time step. */
export interface TimeSeriesData {
  readonly kind: 'timeseries'
  /** Sample times `[n]` (increasing; uneven sampling is allowed). */
  readonly times: Tensor
  /** Values `[n, channels]`; `NaN` marks a missing observation. */
  readonly values: Tensor
  /** A segment or regime label per step `[n]` (int32), when known. */
  readonly segments?: Tensor<'int32'>
  /** Samples per unit time, for evenly sampled series. */
  readonly samplingRate?: number
  /** The seasonal period in steps, when the series has one. */
  readonly period?: number
  readonly events: readonly TimeSeriesEvent[]
  readonly channelNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Options of {@link timeSeries}. */
export interface TimeSeriesOptions {
  /** Sample times (default `0, 1/rate, 2/rate, ...`). */
  readonly times?: TensorLike
  readonly samplingRate?: number
  readonly period?: number
  readonly segments?: TensorLike
  readonly events?: readonly TimeSeriesEvent[]
  readonly channelNames?: readonly string[]
  readonly meta?: Partial<DatasetMeta>
}

/**
 * A time series from values (`[n]` for one channel or `[n, channels]`), with evenly spaced times from the sampling
 * rate unless times are given.
 *
 * @param values - The observations.
 * @param options - Times or sampling rate, period, segment labels, events, channel names and metadata.
 * @returns The series.
 * @throws WhiteboxError with code `value` when the times decrease.
 * @example
 * ```ts
 * import { timeSeries } from 'whitebox-engine/data'
 * const s = timeSeries([1, 2, 4], { samplingRate: 2 })
 * Array.from(s.times.data) // => [0, 0.5, 1]
 * s.values.shape // => [3, 1]
 * ```
 */
export function timeSeries(values: TensorLike, options: TimeSeriesOptions = {}): TimeSeriesData {
  let v = concreteFloat(values, 'timeSeries')
  if (v.shape.length === 1) v = v.backend.wrap(v.dtype, [v.shape[0] ?? 0, 1], v.data)
  const n = v.shape[0] ?? 0
  const rate = options.samplingRate ?? 1
  const times =
    options.times === undefined
      ? fromStorage(
          Float64Array.from(arange(n).data, (i) => i / rate),
          [n],
          'float64',
        )
      : concreteFloat(options.times, 'timeSeries')
  for (let i = 1; i < n; i++) {
    if ((times.data[i] ?? 0) < (times.data[i - 1] ?? 0)) throw new WhiteboxError('value', 'timeSeries: times decrease')
  }
  const segments = options.segments === undefined ? undefined : concrete(options.segments, 'int32', 'timeSeries')
  checkRows('timeSeries', n, { times, segments })
  return {
    kind: 'timeseries',
    times,
    values: v,
    ...(segments === undefined ? {} : { segments }),
    ...(options.times === undefined || options.samplingRate !== undefined ? { samplingRate: rate } : {}),
    ...(options.period === undefined ? {} : { period: options.period }),
    events: options.events ?? [],
    channelNames: options.channelNames ?? (v.shape[1] === 1 ? ['value'] : defaultNames('c', v.shape[1] ?? 0)),
    meta: metaFor('timeseries', options.meta),
  }
}
