/**
 * Drawing with rotating circles: a closed outline's complex Fourier series, truncated to the largest terms (by count
 * or by share of energy) and chained as rotating arrows, recorded for a view that plays one cycle with the target, the
 * traced partial sum, a zoom on the pen and the spectrum.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { type Epicycles, type Explained, epicycles } from '../../lib/exposition/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import {
  type ChainOrder,
  type CurveSpectrum,
  chainOrder,
  curveSpectrum,
  keptTerms,
  partialSum,
} from '../../lib/signal/index.ts'

/** Options of {@link fourierDrawing}. */
export interface FourierDrawingOptions {
  /** The terms kept: a number of terms (default 60), or a share of the energy in (0, 1]. */
  readonly keep?: { readonly count: number } | { readonly energy: number }
  /** The order the arrows are chained in (default largest first). */
  readonly order?: ChainOrder
  /** The key for a random order. */
  readonly key?: Key
  /** Frames per cycle when played (default 360). */
  readonly steps?: number
  /** The magnification of the view that follows the pen (default none). */
  readonly zoom?: number
  /** What the outline is, for the record (default 'curve'). */
  readonly source?: string
}

/** The outcome of {@link fourierDrawing}: the spectrum, the kept bins in chain order, and the record. */
export interface FourierDrawing {
  readonly spectrum: CurveSpectrum
  readonly chain: readonly number[]
  readonly energyKept: number
}

// Coefficients more than five decades below the largest are left off the spectrum.
const FLOOR = 1e-5

/**
 * Redraw a closed outline with rotating circles: its Fourier series truncated to the largest terms, chained in the
 * chosen order, with the traced partial sum and the spectrum.
 *
 * @param points - The outline: points round one loop, equally spaced (resampled by arc length).
 * @param options - The terms kept, the chain order and key, the frames per cycle, the zoom and the source's name.
 * @returns The spectrum, the chain and the energy kept, with an `epicycles` record.
 * @throws MdmlError with code `value` for fewer than 4 points.
 */
export function fourierDrawing(
  points: readonly (readonly [number, number])[],
  options: FourierDrawingOptions = {},
): Explained<FourierDrawing> {
  if (points.length < 4) throw new MdmlError('value', 'fourierDrawing: need at least 4 points')
  const s = curveSpectrum(points)
  const kept = keptTerms(s, options.keep ?? { count: 60 })
  const chain = chainOrder(s, kept, options.order ?? 'magnitude', options.key ?? makeKey(0))
  const top = Math.max(1, ...kept.map((b) => Math.abs(s.k[b] ?? 0)))
  const trace = partialSum(s, kept, Math.max(4 * points.length, 2 * top + 1))
  const energyKept = s.energy > 0 ? kept.reduce((a, b) => a + (s.magnitude[b] ?? 0) ** 2, 0) / s.energy : 1
  const floor = FLOOR * (s.magnitude[s.byMagnitude[0] ?? 0] ?? 1)
  const keptSet = new Set(kept)
  const shown = s.byMagnitude.filter((b) => keptSet.has(b) || (s.magnitude[b] ?? 0) >= floor)
  const record: Epicycles = epicycles({
    method: 'fourier series',
    source: options.source ?? 'curve',
    target: { x: points.map((p) => p[0]), y: points.map((p) => p[1]) },
    centroid: [s.re[0] ?? 0, s.im[0] ?? 0],
    chain: {
      re: chain.map((b) => s.re[b] ?? 0),
      im: chain.map((b) => s.im[b] ?? 0),
      k: chain.map((b) => s.k[b] ?? 0),
    },
    trace,
    spectrum: {
      k: shown.map((b) => s.k[b] ?? 0),
      magnitude: shown.map((b) => s.magnitude[b] ?? 0),
      kept: shown.map((b) => keptSet.has(b)),
    },
    energyKept,
    steps: options.steps ?? 360,
    ...(options.zoom === undefined || options.zoom <= 1 ? {} : { zoom: options.zoom }),
  })
  return { spectrum: s, chain, energyKept, exposition: [record] }
}
