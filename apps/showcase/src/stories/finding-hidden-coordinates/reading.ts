/**
 * What the prose of 'Finding hidden coordinates' says after the run: how close the recovered coordinate is to the true one, how
 * the spreads the method states compare with the distances it actually missed by, and how well the learned functions follow
 * the measurements. Pure functions of the result, light enough for the page's bundle.
 */

import type { Result } from './compute.ts'

/** The z value of a 90 percent interval of a bell curve. */
export const Z90 = 1.645

/** A run found the coordinate when the recovered one correlates with the true one at least this much. */
export const FOUND = 0.9

/** The numbers the prose quotes after the run. */
export interface Reading {
  readonly points: number
  /** Whether the run found the coordinate (see {@link FOUND}). */
  readonly found: boolean
  readonly correlation: number
  /** The root mean square distance of the recovered coordinates from the true ones. */
  readonly miss: number
  /** The average spread the method states for a coordinate. */
  readonly spread: number
  /** How the stated spread compares with the miss: narrower, about the same or wider. */
  readonly spreadIs: 'narrower than' | 'about the same as' | 'wider than'
  /** How many true coordinates lie inside the 90 percent range of the coordinate found for them. */
  readonly inside: number
  /** The distance of the learned functions from the measurements at the points, and the noise that was put in. */
  readonly residual: number
  readonly noise: number
  /** Whether the learned functions come within one and a half times the noise of the measurements. */
  readonly curvesFit: boolean
  /** The smallest and the largest true coordinate. */
  readonly low: number
  readonly high: number
  readonly sweeps: number
  readonly settled: boolean
}

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const n = result.truth.length
  const miss = Math.sqrt(result.truth.reduce((s, t, i) => s + (t - (result.recovered[i] ?? 0)) ** 2, 0) / n)
  const spread = result.spread.reduce((a, b) => a + b, 0) / n
  const ratio = spread / miss
  const inside = result.truth.filter(
    (t, i) => Math.abs(t - (result.recovered[i] ?? 0)) <= Z90 * (result.spread[i] ?? 0),
  ).length
  return {
    points: n,
    found: result.correlation >= FOUND,
    correlation: result.correlation,
    miss,
    spread,
    spreadIs: ratio < 0.8 ? 'narrower than' : ratio > 1.25 ? 'wider than' : 'about the same as',
    inside,
    residual: result.residual,
    noise: result.noise,
    curvesFit: result.residual <= 1.5 * result.noise,
    low: Math.min(...result.truth),
    high: Math.max(...result.truth),
    sweeps: result.sweeps,
    settled: result.settled,
  }
}
