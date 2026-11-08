/**
 * The schema for point-process data: event times, event locations or both, with optional marks, observed over a window.
 * The `events` constructor builds and checks one.
 */

import { OmaiError } from '../../base/index.ts'
import type { Tensor, TensorLike } from '../../tensor/index.ts'
import { cardinality, checkRows, concrete, concreteFloat, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/**
 * Point-process data: event times (temporal), event locations (spatial) or both, with optional marks, observed over a
 * window. Several independent realisations are told apart by `sequences`.
 */
export interface EventData {
  readonly kind: 'events'
  /** Event times `[n]`, sorted within each sequence. */
  readonly times?: Tensor
  /** Event locations `[n, d]`. */
  readonly locations?: Tensor
  /** A mark (event type) per event `[n]` (int32). */
  readonly marks?: Tensor<'int32'>
  /** The realisation each event belongs to `[n]` (int32). */
  readonly sequences?: Tensor<'int32'>
  /** The observation window in time `[start, end]`. */
  readonly window?: readonly [number, number]
  /** The observed region in space, one `[low, high]` per dimension. */
  readonly region?: readonly (readonly [number, number])[]
  readonly markNames?: readonly string[]
  readonly nSequences: number
  readonly meta: DatasetMeta
}

/** Options of {@link events}. */
export interface EventOptions {
  readonly times?: TensorLike
  readonly locations?: TensorLike
  readonly marks?: TensorLike
  readonly sequences?: TensorLike
  /** Default: from the first to the last event. */
  readonly window?: readonly [number, number]
  readonly region?: readonly (readonly [number, number])[]
  readonly markNames?: readonly string[]
  readonly meta?: Partial<DatasetMeta>
}

/**
 * Point-process data from event times and/or locations.
 *
 * @param options - Times, locations, marks, sequence ids, window, region and metadata.
 * @returns The dataset.
 * @throws OmaiError with code `value` when there are neither times nor locations, or an event lies outside the
 * window.
 * @example
 * ```ts
 * import { events } from 'omai-engine/data'
 * events({ times: [0.5, 1.2, 3], window: [0, 4] }).window // => [0, 4]
 * ```
 */
export function events(options: EventOptions): EventData {
  const times = options.times === undefined ? undefined : concreteFloat(options.times, 'events')
  const locations = options.locations === undefined ? undefined : concreteFloat(options.locations, 'events')
  if (times === undefined && locations === undefined) throw new OmaiError('value', 'events: give times or locations')
  const marks = options.marks === undefined ? undefined : concrete(options.marks, 'int32', 'events')
  const sequences = options.sequences === undefined ? undefined : concrete(options.sequences, 'int32', 'events')
  const n = (times ?? locations)?.shape[0] ?? 0
  checkRows('events', n, { times, locations, marks, sequences })
  let window = options.window
  if (times !== undefined) {
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (const t of times.data) {
      lo = Math.min(lo, t)
      hi = Math.max(hi, t)
    }
    if (window === undefined) window = n === 0 ? [0, 0] : [lo, hi]
    else if (n > 0 && (lo < window[0] || hi > window[1]))
      throw new OmaiError('value', `events: an event lies outside the window [${window[0]}, ${window[1]}]`)
  }
  return {
    kind: 'events',
    ...(times === undefined ? {} : { times }),
    ...(locations === undefined ? {} : { locations }),
    ...(marks === undefined ? {} : { marks }),
    ...(sequences === undefined ? {} : { sequences }),
    ...(window === undefined ? {} : { window }),
    ...(options.region === undefined ? {} : { region: options.region }),
    ...(options.markNames === undefined ? {} : { markNames: options.markNames }),
    nSequences: sequences === undefined ? 1 : Math.max(1, cardinality(sequences.data)),
    meta: metaFor('events', options.meta),
  }
}
