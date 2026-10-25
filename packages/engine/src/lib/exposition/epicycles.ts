/**
 * The exposition record for a closed curve drawn by its Fourier series: the target curve, the kept coefficients in the
 * order their rotating arrows are chained, the partial sum they trace on a fine grid, and the spectrum with the kept
 * and dropped coefficients, so a view can play the chain of arrows round one cycle.
 */

import { MdmlError } from '../base/index.ts'
import { sameLength } from './_check.ts'

/** A curve: its points' x and y. */
export interface PlaneCurve {
  readonly x: readonly number[]
  readonly y: readonly number[]
}

/**
 * A closed curve drawn by rotating arrows: arrow i has length |c_i| and turns `k[i]` times per cycle, the arrows are
 * chained tip to tail from the centroid, and the last tip traces the partial sum.
 */
export interface Epicycles {
  readonly kind: 'epicycles'
  readonly method: string
  /** What the curve is (a shape's name, a character in a font). */
  readonly source: string
  /** The target curve, one loop. */
  readonly target: PlaneCurve
  /** The centroid c_0. */
  readonly centroid: readonly [number, number]
  /** The kept coefficients in chain order: real and imaginary parts and frequencies. */
  readonly chain: { readonly re: readonly number[]; readonly im: readonly number[]; readonly k: readonly number[] }
  /** The partial sum on a fine grid of t in [0, 1], its last point equal to its first. */
  readonly trace: PlaneCurve
  /** The coefficients above a floor, for the spectrum: frequency, magnitude and whether kept. */
  readonly spectrum: {
    readonly k: readonly number[]
    readonly magnitude: readonly number[]
    readonly kept: readonly boolean[]
  }
  /** The share of the curve's energy about its centroid that the kept terms carry (Parseval). */
  readonly energyKept: number
  /** Frames per cycle when played. */
  readonly steps: number
  /** The magnification of a view that follows the pen (none when absent). */
  readonly zoom?: number
}

/**
 * Epicycles, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when paired arrays differ in length; `value` for fewer than 2 frames per cycle.
 * @example
 * ```ts
 * import { epicycles } from 'mdml-engine/exposition'
 * const r = epicycles({
 *   method: 'fourier series',
 *   source: 'circle',
 *   target: { x: [1, 0, -1, 0], y: [0, 1, 0, -1] },
 *   centroid: [0, 0],
 *   chain: { re: [1], im: [0], k: [1] },
 *   trace: { x: [1, 0, -1, 0, 1], y: [0, 1, 0, -1, 0] },
 *   spectrum: { k: [1], magnitude: [1], kept: [true] },
 *   energyKept: 1,
 *   steps: 4,
 * })
 * r.chain.k // => [1]
 * ```
 */
export function epicycles(record: Omit<Epicycles, 'kind'>): Epicycles {
  sameLength('epicycles', 'target.y', record.target.y.length, record.target.x.length)
  sameLength('epicycles', 'trace.y', record.trace.y.length, record.trace.x.length)
  sameLength('epicycles', 'chain.im', record.chain.im.length, record.chain.re.length)
  sameLength('epicycles', 'chain.k', record.chain.k.length, record.chain.re.length)
  sameLength('epicycles', 'spectrum.magnitude', record.spectrum.magnitude.length, record.spectrum.k.length)
  sameLength('epicycles', 'spectrum.kept', record.spectrum.kept.length, record.spectrum.k.length)
  if (!(Number.isInteger(record.steps) && record.steps >= 2))
    throw new MdmlError('value', 'epicycles: at least 2 frames per cycle')
  return { kind: 'epicycles', ...record }
}
