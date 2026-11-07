/**
 * Exposition records for trustworthy predictions: calibration of probabilistic forecasts as reliability curves with
 * expected and maximum calibration errors, and conformal prediction with its coverage guarantee checked on held-out
 * rows, including intervals for regression and runs on a stream. `setSizes` summarises a conformal record's set sizes.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { sameLength, seal } from './_check.ts'

/** One reliability curve: per non-empty bin the mean forecast, the observed frequency and the count, with its errors. */
export interface ReliabilityCurve {
  /** What the forecasts are (`uncalibrated`, `Platt`, `isotonic`, a group's name). */
  readonly name: string
  /** Curves of one group share a colour (per-group reliability). */
  readonly group?: string
  readonly meanPredicted: readonly number[]
  readonly observed: readonly number[]
  readonly counts: readonly number[]
  /** Expected and maximum calibration error. */
  readonly ece: number
  readonly mce: number
  readonly brier?: number
  /** The calibration map from raw score to probability, on a grid (for calibrators). */
  readonly map?: { readonly x: readonly number[]; readonly y: readonly number[] }
}

/**
 * Calibration of probabilistic forecasts: reliability curves (observed frequency against mean forecast per bin) with
 * their expected and maximum calibration errors, for several calibrators or groups, and the maps of the calibrators.
 */
export interface CalibrationRecord {
  readonly kind: 'calibration'
  readonly nBins: number
  readonly strategy: 'uniform' | 'quantile'
  readonly curves: readonly ReliabilityCurve[]
  /** The rows the curves are measured on (`test`, `calibration`). */
  readonly split?: string
}

/**
 * A calibration record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a curve's columns (or its map's) differ in length.
 * @example
 * ```ts
 * import { calibrationRecord } from 'mdml-engine/exposition'
 * const r = calibrationRecord({ nBins: 2, strategy: 'uniform', curves: [{ name: 'model', meanPredicted: [0.2, 0.8], observed: [0.1, 0.9], counts: [5, 5], ece: 0.1, mce: 0.1 }] })
 * r.curves[0]?.ece // => 0.1
 * ```
 */
export function calibrationRecord(record: Omit<CalibrationRecord, 'kind'>): CalibrationRecord {
  for (const c of record.curves) {
    sameLength('calibration', `${c.name}.observed`, c.observed.length, c.meanPredicted.length)
    sameLength('calibration', `${c.name}.counts`, c.counts.length, c.meanPredicted.length)
    if (c.map !== undefined) sameLength('calibration', `${c.name}.map.y`, c.map.y.length, c.map.x.length)
  }
  return seal({ kind: 'calibration', ...record })
}

/** A conformal predictor run on a stream: at each time, whether the truth was covered, the running coverage, and more. */
export interface ConformalStream {
  readonly t: readonly number[]
  /** The running coverage up to each time. */
  readonly coverage: readonly number[]
  /** A comparison's running coverage (a split conformal predictor that does not adapt). */
  readonly baseline?: readonly number[]
  /** The working miscoverage level at each time (adaptive conformal inference). */
  readonly alphas?: readonly number[]
  /** The interval's width at each time. */
  readonly width?: readonly number[]
}

/**
 * Conformal prediction with its guarantee checked: the nominal coverage `1 - alpha`, the empirical coverage on held-out
 * rows, the calibration scores and their quantile; for regression, intervals along an input (or time) with the truth;
 * for classification, the prediction sets with the distribution of their sizes; optionally the coverage as data
 * arrive.
 */
export interface ConformalRecord {
  readonly kind: 'conformal'
  readonly task: 'regression' | 'classification'
  readonly method: string
  readonly alpha: number
  /** The empirical coverage on held-out rows. */
  readonly coverage: number
  /** The calibration rows' conformity scores and their conformal quantile. */
  readonly scores?: readonly number[]
  readonly quantile?: number
  /** Regression: the input each row is drawn at (sorted), the truth, the point prediction and the interval. */
  readonly x?: readonly number[]
  readonly xLabel?: string
  readonly y?: readonly number[]
  readonly prediction?: readonly number[]
  readonly lower?: readonly number[]
  readonly upper?: readonly number[]
  /** Classification: the class names, each row's set (class indices) and true class. */
  readonly classes?: readonly string[]
  readonly sets?: readonly (readonly number[])[]
  readonly truth?: readonly number[]
  readonly stream?: ConformalStream
}

/**
 * A conformal record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the per-row columns differ in length; `value` for a level outside (0, 1).
 * @example
 * ```ts
 * import { conformalRecord } from 'mdml-engine/exposition'
 * conformalRecord({ task: 'regression', method: 'split', alpha: 0.1, coverage: 0.9, x: [0, 1], y: [1, 2], lower: [0, 1], upper: [2, 3] }).coverage // => 0.9
 * ```
 */
export function conformalRecord(record: Omit<ConformalRecord, 'kind'>): ConformalRecord {
  if (!(record.alpha > 0 && record.alpha < 1)) throw new MdmlError('value', 'conformal: alpha must be in (0, 1)')
  const n = record.x?.length ?? record.sets?.length ?? 0
  for (const field of ['y', 'prediction', 'lower', 'upper'] as const) {
    const v = record[field]
    if (v !== undefined) sameLength('conformal', field, v.length, n)
  }
  if (record.truth !== undefined) sameLength('conformal', 'truth', record.truth.length, n)
  if (record.stream !== undefined) {
    const s = record.stream
    sameLength('conformal', 'stream.coverage', s.coverage.length, s.t.length)
    for (const field of ['baseline', 'alphas', 'width'] as const) {
      const v = s[field]
      if (v !== undefined) sameLength('conformal', `stream.${field}`, v.length, s.t.length)
    }
  }
  return seal({ kind: 'conformal', ...record })
}

/**
 * The distribution of a conformal record's set sizes: how many rows have a set of each size (0 up to the number of
 * classes), and the coverage among them.
 *
 * @param record - A classification record.
 * @returns Counts and coverage by set size.
 * @example
 * ```ts
 * import { conformalRecord, setSizes } from 'mdml-engine/exposition'
 * const r = conformalRecord({ task: 'classification', method: 'lac', alpha: 0.1, coverage: 1, classes: ['a', 'b'], sets: [[0], [0, 1]], truth: [0, 1] })
 * setSizes(r).counts // => [0, 1, 1]
 * ```
 */
export function setSizes(record: ConformalRecord): { readonly counts: number[]; readonly coverage: number[] } {
  const k = record.classes?.length ?? Math.max(0, maxOf((record.sets ?? []).map((s) => s.length)))
  const counts = new Array<number>(k + 1).fill(0)
  const covered = new Array<number>(k + 1).fill(0)
  for (const [i, s] of (record.sets ?? []).entries()) {
    counts[s.length] = (counts[s.length] ?? 0) + 1
    if (s.includes(record.truth?.[i] ?? -1)) covered[s.length] = (covered[s.length] ?? 0) + 1
  }
  return { counts, coverage: counts.map((c, j) => (c === 0 ? Number.NaN : (covered[j] ?? 0) / c)) }
}
