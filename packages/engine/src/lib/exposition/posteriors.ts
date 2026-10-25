/**
 * Exposition records for Bayesian and tuned models: a predictive distribution on a grid of inputs with its mean, bands
 * and samples (Gaussian processes, Bayesian regression), and an objective over a grid of two hyperparameters, such as a
 * marginal likelihood, with the optimiser's path across it.
 */

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
  /** The names of the input and the output, for axis labels ('games' and 'rating', 'p' and 'density'). */
  readonly labels?: { readonly input?: string; readonly output?: string }
  /**
   * Named densities over a one-dimensional grid of a parameter (a conjugate model's prior, likelihood and posterior),
   * drawn as curves; `mean` then holds the posterior's.
   */
  readonly densities?: readonly {
    readonly name: string
    readonly role: 'prior' | 'likelihood' | 'posterior'
    readonly values: readonly number[]
  }[]
  /**
   * Several trajectories over the same inputs, each with its variance (players' ratings over games), drawn together
   * with their deviation bands; `mean` and `variance` then hold the first's.
   */
  readonly series?: readonly {
    readonly name: string
    readonly mean: readonly number[]
    readonly variance: readonly number[]
  }[]
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
  for (const [i, d] of (grid.densities ?? []).entries())
    sameLength('posteriorGrid', `densities[${i}]`, d.values.length, n)
  for (const [i, s] of (grid.series ?? []).entries()) {
    sameLength('posteriorGrid', `series[${i}].mean`, s.mean.length, n)
    sameLength('posteriorGrid', `series[${i}].variance`, s.variance.length, n)
  }
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
  /** What the values are ('log marginal likelihood', 'cross-validated accuracy'). */
  readonly objective?: string
  /** Whether higher values are better (default true: likelihoods and scores). */
  readonly higherIsBetter?: boolean
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
