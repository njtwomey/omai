import { sameLength } from './_check.ts'

/** One density over a record's grid, with the part it plays (styled consistently: the prior dashed, the posterior filled). */
export interface DensityCurve {
  readonly name: string
  /** density[i]: the density at x[i]. */
  readonly density: readonly number[]
  readonly role?: 'prior' | 'posterior' | 'likelihood' | 'other'
}

/** The posterior after a prefix of the data: how many observations it has seen, its density and its interval. */
export interface DensityStep {
  readonly observed: number
  readonly density: readonly number[]
  readonly interval?: readonly [number, number]
  /** A point estimate to mark at this step (the posterior mean). */
  readonly estimate?: number
}

/**
 * Densities over one quantity, compared on a shared grid: a prior against its posterior (conjugate models), a
 * likelihood, several posteriors. Optionally the posterior after every prefix of the data, so a view can play the
 * update as the observations arrive, with a credible interval and marked values (estimates, the truth).
 */
export interface DensityCurves {
  readonly kind: 'densities'
  /** The quantity the densities are over (p, λ, μ). */
  readonly parameter: string
  /** The grid. */
  readonly x: readonly number[]
  readonly curves: readonly DensityCurve[]
  /** A credible interval of the posterior, at a level. */
  readonly interval?: { readonly lower: number; readonly upper: number; readonly level?: number }
  /** Values to mark with a vertical line (the posterior mean, the maximum-likelihood estimate, the true value). */
  readonly marks?: readonly { readonly name: string; readonly value: number }[]
  /** The posterior after each prefix of the data, in order (the last one the posterior of all of it). */
  readonly steps?: readonly DensityStep[]
}

/**
 * Density curves, with their shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a density (of a curve or a step) does not match the grid.
 * @example
 * ```ts
 * import { densityCurves } from 'whitebox-engine/exposition'
 * const r = densityCurves({
 *   parameter: 'p',
 *   x: [0, 0.5, 1],
 *   curves: [{ name: 'prior', density: [1, 1, 1], role: 'prior' }, { name: 'posterior', density: [0, 1.5, 0], role: 'posterior' }],
 * })
 * r.curves.length // => 2
 * ```
 */
export function densityCurves(record: Omit<DensityCurves, 'kind'>): DensityCurves {
  const n = record.x.length
  for (const c of record.curves) sameLength('densities', `curves.${c.name}`, c.density.length, n)
  for (const [t, s] of (record.steps ?? []).entries())
    sameLength('densities', `steps[${t}].density`, s.density.length, n)
  return { kind: 'densities', ...record }
}
