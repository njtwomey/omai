/**
 * The shape survival analysis starts from: right-censored observations (a time and whether the event was seen or the
 * subject was lost), validated once and read as typed arrays, with the table of distinct event times (subjects at
 * risk, events, censorings) that the Kaplan-Meier and Nelson-Aalen estimators, the log-rank tests and the Cox
 * partial likelihood all run over.
 */

import { MdmlError } from '../base/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'

/** Right-censored survival data: `event[i]` is 1 when the event was seen at `time[i]` and 0 when the subject was censored. */
export interface SurvivalSample {
  readonly n: number
  readonly time: Float64Array
  readonly event: Uint8Array
}

/**
 * Validate and read survival times and event indicators.
 *
 * @param time - The times, finite and non-negative.
 * @param event - Event indicators (1 or true: seen, 0 or false: censored); omitted means every event was seen.
 * @param name - The caller, for error messages.
 * @returns The sample.
 * @throws MdmlError with code `value` for a length mismatch, a negative or non-finite time, or an indicator that is not 0 or 1.
 * @example
 * ```ts
 * import { survivalSample } from 'mdml-engine/survival'
 * Array.from(survivalSample([2, 3, 5], [1, 0, 1]).event) // => [1, 0, 1]
 * ```
 */
export function survivalSample(time: TensorLike, event?: TensorLike, name = 'survivalSample'): SurvivalSample {
  const t = Float64Array.from(toFlat(asTensor(time)))
  const n = t.length
  if (n === 0) throw new MdmlError('value', `${name}: needs at least one observation`)
  for (let i = 0; i < n; i++) {
    const v = t[i] ?? Number.NaN
    if (!Number.isFinite(v) || v < 0)
      throw new MdmlError('value', `${name}: time ${i} must be finite and >= 0, got ${v}`)
  }
  const e = new Uint8Array(n)
  if (event === undefined) e.fill(1)
  else {
    const raw = toFlat(asTensor(event))
    if (raw.length !== n) throw new MdmlError('shape', `${name}: ${raw.length} event indicators for ${n} times`)
    for (let i = 0; i < n; i++) {
      const v = raw[i]
      if (v !== 0 && v !== 1) throw new MdmlError('value', `${name}: event ${i} must be 0 or 1, got ${String(v)}`)
      e[i] = v
    }
  }
  return { n, time: t, event: e }
}

/** The subjects ordered by time, and the table of distinct event times. */
export interface EventTable {
  /** Distinct times at which at least one event was seen, ascending. */
  readonly time: Float64Array
  /** Subjects with time at least this one (at risk just before it). */
  readonly atRisk: Float64Array
  /** Events at this time. */
  readonly events: Float64Array
  /** Censorings in `[time, next event time)`: those leaving the risk set before the next event. */
  readonly censored: Float64Array
  /** Censorings before the first event time. */
  readonly censoredBefore: number
}

/**
 * The table of distinct event times: how many were at risk, how many events, how many censored until the next one.
 *
 * @param sample - The sample.
 * @returns The table.
 * @example
 * ```ts
 * import { eventTable, survivalSample } from 'mdml-engine/survival'
 * const t = eventTable(survivalSample([1, 2, 2, 3, 4], [1, 1, 0, 1, 0]))
 * Array.from(t.atRisk) // => [5, 4, 2]
 * Array.from(t.events) // => [1, 1, 1]
 * Array.from(t.censored) // => [0, 1, 1]
 * ```
 */
export function eventTable(sample: SurvivalSample): EventTable {
  const order = Array.from({ length: sample.n }, (_, i) => i).sort(
    (a, b) => (sample.time[a] ?? 0) - (sample.time[b] ?? 0),
  )
  const time: number[] = []
  const atRisk: number[] = []
  const events: number[] = []
  const censored: number[] = []
  let censoredBefore = 0
  let i = 0
  while (i < sample.n) {
    const t = sample.time[order[i] ?? 0] ?? 0
    let d = 0
    let c = 0
    let j = i
    while (j < sample.n && (sample.time[order[j] ?? 0] ?? 0) === t) {
      if (sample.event[order[j] ?? 0] === 1) d++
      else c++
      j++
    }
    if (d > 0) {
      time.push(t)
      atRisk.push(sample.n - i)
      events.push(d)
      censored.push(c)
    } else if (time.length === 0) censoredBefore += c
    else censored[censored.length - 1] = (censored[censored.length - 1] ?? 0) + c
    i = j
  }
  return {
    time: Float64Array.from(time),
    atRisk: Float64Array.from(atRisk),
    events: Float64Array.from(events),
    censored: Float64Array.from(censored),
    censoredBefore,
  }
}
