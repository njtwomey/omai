/**
 * The global-effect record of explainability: how a model's output depends on one feature (partial dependence with
 * its ICE curves, or accumulated local effects) or on a pair (a surface), with the data's rug so a reader sees where
 * the curve is supported. Built by `methods/explain` (`dependencePlot`, `aleEffects`) and drawn by render's
 * `DependenceView`.
 */
import { WhiteboxError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'

/** One feature's effect: the averaged curve over a grid, and optionally every row's curve (ICE). */
export interface DependenceCurve {
  /** The feature, an index into the record's `features`. */
  readonly feature: number
  /** The grid of feature values (PDP grid points, or ALE bin edges). */
  readonly grid: readonly number[]
  /** The averaged effect at each grid value (the PDP, or the centred ALE). */
  readonly average: readonly number[]
  /** ICE curves, individual[row][gridIndex] (partial dependence only). */
  readonly individual?: readonly (readonly number[])[]
  /** Rows between consecutive grid values (ALE bins), one fewer than the grid. */
  readonly counts?: readonly number[]
  /** The feature's values in the data, for a rug along the axis. */
  readonly rug?: readonly number[]
  /** Whether the grid holds a categorical feature's levels. */
  readonly categorical?: boolean
}

/** A pair of features' joint effect over a grid (two-way PDP, or second-order ALE). */
export interface DependenceSurface {
  /** The two features, indices into `features`. */
  readonly features: readonly [number, number]
  readonly gridX: readonly number[]
  readonly gridY: readonly number[]
  /** values[i][j] at (gridX[i], gridY[j]). */
  readonly values: readonly (readonly number[])[]
  /** Rows per cell (second-order ALE), counts[i][j] for the cells between grid values. */
  readonly counts?: readonly (readonly number[])[]
}

/**
 * How a model's output depends on its features (Friedman's partial dependence with Goldstein's ICE curves, Apley and
 * Zhu's accumulated local effects): per feature the averaged curve and, for partial dependence, every row's curve;
 * per pair a surface. `transform` says what the curves hold: the raw output, curves centred at their first grid value
 * (c-ICE) or their derivatives (d-ICE).
 */
export interface Dependence {
  readonly kind: 'dependence'
  /** `partial dependence` or `ale`. */
  readonly method: 'partial dependence' | 'ale'
  /** Every feature's name (curves and surfaces index into them). */
  readonly features: readonly string[]
  /** What is explained (`prediction`, `P(malignant)`). */
  readonly output: string
  readonly curves: readonly DependenceCurve[]
  readonly surfaces?: readonly DependenceSurface[]
  /** What the curves hold (default `none`: the output itself). */
  readonly transform?: 'none' | 'centred' | 'derivative'
  /** Which rows the ICE curves are of (dataset indices). */
  readonly indices?: readonly number[]
}

/**
 * A dependence record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `value` when a curve or surface names a feature that is not in `features`; `shape` when a
 *   curve's average, ICE curves or counts do not match its grid, or a surface's values its grids.
 * @example
 * ```ts
 * import { dependence } from 'whitebox-engine/exposition'
 * const r = dependence({ method: 'partial dependence', features: ['age', 'bmi'], output: 'prediction', curves: [{ feature: 1, grid: [0, 1], average: [2, 3], individual: [[1, 2], [3, 4]] }] })
 * r.curves[0].individual.length // => 2
 * ```
 */
export function dependence(record: Omit<Dependence, 'kind'>): Dependence {
  const k = 'dependence'
  const d = record.features.length
  const feature = (f: number): void => {
    if (!(Number.isInteger(f) && f >= 0 && f < d)) throw new WhiteboxError('value', `${k}: feature ${f} is not named`)
  }
  for (const [i, c] of record.curves.entries()) {
    feature(c.feature)
    const g = c.grid.length
    sameLength(k, `curves[${i}].average`, c.average.length, g)
    if (c.individual !== undefined) rowsOf(k, `curves[${i}].individual`, c.individual, g)
    if (c.counts !== undefined) sameLength(k, `curves[${i}].counts`, c.counts.length, Math.max(0, g - 1))
    if (record.indices !== undefined && c.individual !== undefined)
      sameLength(k, `curves[${i}].individual`, c.individual.length, record.indices.length)
  }
  for (const [i, s] of (record.surfaces ?? []).entries()) {
    feature(s.features[0])
    feature(s.features[1])
    sameLength(k, `surfaces[${i}].values`, s.values.length, s.gridX.length)
    rowsOf(k, `surfaces[${i}].values`, s.values, s.gridY.length)
    if (s.counts !== undefined) {
      sameLength(k, `surfaces[${i}].counts`, s.counts.length, Math.max(0, s.gridX.length - 1))
      rowsOf(k, `surfaces[${i}].counts`, s.counts, Math.max(0, s.gridY.length - 1))
    }
  }
  return { kind: k, ...record }
}
