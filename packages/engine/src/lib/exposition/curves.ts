import { WhiteboxError } from '../base/index.ts'
import { sameLength } from './_check.ts'

/** One series of a {@link CurveSet}: values along x, an optional band, and which axis and group it belongs to. */
export interface CurveSeries {
  readonly name: string
  /** The x values (default the record's `x`). */
  readonly x?: readonly number[]
  readonly y: readonly number[]
  /** A band of y ± spread (a standard deviation, a rating deviation). */
  readonly spread?: readonly number[]
  /** Or the band's explicit edges. */
  readonly lower?: readonly number[]
  readonly upper?: readonly number[]
  /** The y axis it is read against: the left (default) or the right, for a second quantity on another scale. */
  readonly axis?: 'left' | 'right'
  /**
   * Series of one group share a colour and differ in dash, the first solid (an optimiser's training and validation
   * loss); without groups every series has its own colour.
   */
  readonly group?: string
}

/**
 * Several curves on shared axes: an optimiser race, a study of schemes (initialisations, normalisations), rating
 * trajectories with their deviation bands, coefficients against the samples seen, any training metrics with a
 * second quantity on a right-hand axis. The multi-series generalisation of a learning curve.
 */
export interface CurveSet {
  readonly kind: 'curves'
  /** The x values every series shares unless it gives its own. */
  readonly x?: readonly number[]
  readonly series: readonly CurveSeries[]
  /** What x is (epoch, games played, samples seen). */
  readonly xLabel: string
  /** What the left and right axes read (the right one only when a series uses it). */
  readonly yLabel?: string
  readonly y2Label?: string
  /** Log axes (default linear). */
  readonly xLog?: boolean
  readonly yLog?: boolean
  /** Positions along x to mark with a labelled line (a learning-rate drop, the chosen epoch). */
  readonly marks?: readonly { readonly x: number; readonly label?: string }[]
}

/**
 * A curve set, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a series' values or band do not match its x values; `value` when a series has
 *   no x values of its own and the record has none.
 * @example
 * ```ts
 * import { curveSet } from 'whitebox-engine/exposition'
 * const r = curveSet({ x: [1, 2, 3], xLabel: 'epoch', series: [{ name: 'sgd', y: [1, 0.6, 0.4] }, { name: 'adam', y: [0.8, 0.3, 0.2] }] })
 * r.series.length // => 2
 * ```
 */
export function curveSet(record: Omit<CurveSet, 'kind'>): CurveSet {
  for (const s of record.series) {
    const x = s.x ?? record.x
    if (x === undefined) throw new WhiteboxError('value', `curves: the series ${s.name} has no x values`)
    sameLength('curves', `${s.name}.y`, s.y.length, x.length)
    for (const field of ['spread', 'lower', 'upper'] as const) {
      const column = s[field]
      if (column !== undefined) sameLength('curves', `${s.name}.${field}`, column.length, x.length)
    }
  }
  return { kind: 'curves', ...record }
}
