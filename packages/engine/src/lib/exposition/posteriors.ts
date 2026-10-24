import { rowsOf, sameLength } from './_check.ts'

/** A predictive distribution on a grid of inputs (GPs, Bayesian regression), with samples. */
export interface PosteriorGrid {
  readonly kind: 'posteriorGrid'
  /** The grid inputs (one per row; a single column for 1-D inputs). */
  readonly inputs: readonly (readonly number[])[]
  readonly mean: readonly number[]
  readonly variance: readonly number[]
  /** samples[s][i]: sample s at input i. */
  readonly samples?: readonly (readonly number[])[]
  /** The training inputs and targets the posterior conditions on. */
  readonly observed?: { readonly inputs: readonly (readonly number[])[]; readonly targets: readonly number[] }
}

/**
 * A posterior grid, with its shapes checked.
 *
 * @param grid - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when means, variances or samples do not match the grid.
 * @example
 * ```ts
 * import { posteriorGrid } from 'whitebox-engine/exposition'
 * posteriorGrid({ inputs: [[0], [1]], mean: [0, 1], variance: [1, 0.5] }).mean // => [0, 1]
 * ```
 */
export function posteriorGrid(grid: Omit<PosteriorGrid, 'kind'>): PosteriorGrid {
  const n = grid.inputs.length
  sameLength('posteriorGrid', 'mean', grid.mean.length, n)
  sameLength('posteriorGrid', 'variance', grid.variance.length, n)
  if (grid.samples !== undefined) rowsOf('posteriorGrid', 'samples', grid.samples, n)
  if (grid.observed !== undefined)
    sameLength('posteriorGrid', 'observed.targets', grid.observed.targets.length, grid.observed.inputs.length)
  return { kind: 'posteriorGrid', ...grid }
}

/** An objective over a grid of two hyperparameters (a marginal likelihood, a validation score), with the optimiser's path. */
export interface HyperparameterSurface {
  readonly kind: 'hyperparameterSurface'
  readonly parameters: readonly [string, string]
  readonly x: readonly number[]
  readonly y: readonly number[]
  /** values[i][j]: the objective at (x[j], y[i]). */
  readonly values: readonly (readonly number[])[]
  /** The points the optimiser visited, in order. */
  readonly path?: readonly (readonly [number, number])[]
  readonly logScale?: readonly [boolean, boolean]
}

/**
 * A hyperparameter surface, with its shapes checked.
 *
 * @param surface - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when the values do not match the grid.
 * @example
 * ```ts
 * import { hyperparameterSurface } from 'whitebox-engine/exposition'
 * hyperparameterSurface({ parameters: ['length', 'noise'], x: [1, 2], y: [0.1], values: [[-3, -2]] }).values // => [[-3, -2]]
 * ```
 */
export function hyperparameterSurface(surface: Omit<HyperparameterSurface, 'kind'>): HyperparameterSurface {
  sameLength('hyperparameterSurface', 'values', surface.values.length, surface.y.length)
  rowsOf('hyperparameterSurface', 'values', surface.values, surface.x.length)
  return { kind: 'hyperparameterSurface', ...surface }
}
