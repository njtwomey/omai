/**
 * What the prose of 'A signal in the clutter' says: from the readings alone (before anything runs: how many there are and how
 * far apart), and from a result (which readings the model calls clutter, how the bell curve compares with the exact curve, how
 * many humps the exact curve has and how much probability each holds). Pure functions, light enough for the page's bundle.
 */

import type { Case } from './compute.ts'
import { SIGNAL } from './data.ts'

/** A reading is called clutter when its chance of being clutter is above this. */
export const CLUTTER_ABOVE = 0.5

/** A hump of the exact curve: where its top is and how much of the probability lies under it. */
export interface Hump {
  readonly at: number
  readonly share: number
}

/**
 * The humps of a curve: its local maxima that are at least a tenth as high as the highest, with the probability each holds
 * (the curve is cut at the lowest point between two humps).
 *
 * @param curve - The values and the density at each.
 * @returns The humps from left to right; shares add to one.
 */
export function humpsOf(curve: { readonly x: readonly number[]; readonly density: readonly number[] }): Hump[] {
  const d = curve.density
  const top = Math.max(...d)
  const peaks = d
    .map((v, i) => ({ i, v }))
    .filter(({ i, v }) => v >= 0.1 * top && v >= (d[i - 1] ?? 0) && v >= (d[i + 1] ?? 0))
  // keep the highest of peaks that are close together (a flat top gives several equal ones)
  const apart: { i: number; v: number }[] = []
  for (const p of peaks) {
    const last = apart.at(-1)
    if (last !== undefined && (curve.x[p.i] as number) - (curve.x[last.i] as number) < 1.5) {
      if (p.v > last.v) apart[apart.length - 1] = p
    } else apart.push(p)
  }
  const total = d.reduce((a, b) => a + b, 0)
  const cuts = apart.slice(1).map((p, k) => {
    const from = (apart[k] as { i: number }).i
    let low = from
    for (let j = from; j <= p.i; j++) if ((d[j] as number) < (d[low] as number)) low = j
    return low
  })
  return apart.map((p, k) => {
    const lo = k === 0 ? 0 : (cuts[k - 1] as number)
    const hi = k === apart.length - 1 ? d.length : (cuts[k] as number)
    return { at: curve.x[p.i] as number, share: d.slice(lo, hi).reduce((a, b) => a + b, 0) / total }
  })
}

/** The numbers the prose quotes before anything has run. */
export interface DataReading {
  readonly count: number
  readonly lowest: number
  readonly highest: number
}

/**
 * The numbers about the readings alone.
 *
 * @param readings - The readings.
 * @returns The count and the range.
 */
export const readData = (readings: readonly number[]): DataReading => ({
  count: readings.length,
  lowest: Math.min(...readings),
  highest: Math.max(...readings),
})

/** The numbers the prose quotes about one solved case. */
export interface CaseReading {
  /** How many readings the model calls clutter (a chance above {@link CLUTTER_ABOVE}), and how many it keeps. */
  readonly clutter: number
  readonly kept: number
  /** The mean of the readings it keeps. */
  readonly keptMean: number
  /** The humps of the exact curve. */
  readonly humps: readonly Hump[]
  /** How far the bell curve's mean is from the exact mean, and the ratio of the bell's spread to the exact spread. */
  readonly meanGap: number
  readonly spreadRatio: number
  /** How far the bell curve's mean is from the true value of the first twelve readings. */
  readonly fromSignal: number
  /** Whether the bell curve's mean lies where the exact curve is low: under half as high as its highest point. */
  readonly inValley: boolean
  /** The largest hump of the exact curve and the others. */
  readonly main: Hump | undefined
  readonly others: readonly Hump[]
  /** The share of the exact probability that lies within one bell spread of the bell's mean. */
  readonly underBell: number
}

/**
 * The numbers of a case the prose quotes.
 *
 * @param c - The case.
 * @returns The reading.
 */
export function readCase(c: Case): CaseReading {
  const kept = c.readings.filter((_, i) => (c.clutter[i] as number) <= CLUTTER_ABOVE)
  const humps = humpsOf(c.exact.curve)
  const main = [...humps].sort((a, b) => b.share - a.share)[0]
  // the exact density at the grid value nearest to the bell curve's mean, against its highest point
  const nearest = c.exact.curve.x.reduce(
    (best, x, i) => (Math.abs(x - c.bell.mean) < Math.abs((c.exact.curve.x[best] as number) - c.bell.mean) ? i : best),
    0,
  )
  const atBell = (c.exact.curve.density[nearest] as number) / Math.max(...c.exact.curve.density)
  const within = c.exact.curve.x.reduce(
    (s, x, i) => (Math.abs(x - c.bell.mean) <= c.bell.sd ? s + (c.exact.curve.density[i] as number) : s),
    0,
  )
  const total = c.exact.curve.density.reduce((a, b) => a + b, 0)
  return {
    clutter: c.readings.length - kept.length,
    kept: kept.length,
    keptMean: kept.length === 0 ? Number.NaN : kept.reduce((a, b) => a + b, 0) / kept.length,
    humps,
    meanGap: Math.abs(c.bell.mean - c.exact.mean),
    spreadRatio: c.bell.sd / c.exact.sd,
    fromSignal: Math.abs(c.bell.mean - SIGNAL),
    inValley: atBell < 0.5,
    main,
    others: humps.filter((h) => h !== main),
    underBell: within / total,
  }
}
