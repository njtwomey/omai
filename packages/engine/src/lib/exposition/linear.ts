/**
 * Exposition records for linear models: regularisation paths of coefficients along a penalty grid (lasso, ridge,
 * elastic net, GAM smoothness), coefficient tables with standard errors and intervals in the manner of a statsmodels
 * summary, and cross-validated scores along one parameter for choosing it.
 */

import { rowsOf, sameLength, seal } from './_check.ts'

/** Coefficients along a penalty grid (lasso, ridge, elastic net, GAM smoothness sweeps). */
export interface RegularisationPath {
  readonly kind: 'regularisationPath'
  /** The penalties, in the order the path was computed (usually decreasing). */
  readonly penalties: readonly number[]
  readonly features: readonly string[]
  /** coefficients[i][j]: feature j at penalty i. */
  readonly coefficients: readonly (readonly number[])[]
  /** Indices of the non-zero coefficients at each penalty. */
  readonly active?: readonly (readonly number[])[]
  /** The duality gap (or another convergence measure) at each penalty. */
  readonly gaps?: readonly number[]
  /** The penalty finally chosen, if any. */
  readonly chosen?: number
}

/**
 * A regularisation path, with its shapes checked.
 *
 * @param path - The path without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the coefficients do not match the penalties and features.
 * @example
 * ```ts
 * import { regularisationPath } from 'mdml-engine/exposition'
 * regularisationPath({ penalties: [1, 0.1], features: ['a', 'b'], coefficients: [[0, 0], [0.4, 0]] }).kind // => 'regularisationPath'
 * ```
 */
export function regularisationPath(path: Omit<RegularisationPath, 'kind'>): RegularisationPath {
  sameLength('regularisationPath', 'coefficients', path.coefficients.length, path.penalties.length)
  rowsOf('regularisationPath', 'coefficients', path.coefficients, path.features.length)
  if (path.active !== undefined) sameLength('regularisationPath', 'active', path.active.length, path.penalties.length)
  if (path.gaps !== undefined) sameLength('regularisationPath', 'gaps', path.gaps.length, path.penalties.length)
  return seal({ kind: 'regularisationPath', ...path })
}

/** Estimated coefficients with their uncertainty (a statsmodels-style summary table). */
export interface CoefficientTable {
  readonly kind: 'coefficientTable'
  readonly names: readonly string[]
  readonly estimates: readonly number[]
  readonly standardErrors?: readonly number[]
  readonly intervals?: readonly (readonly [number, number])[]
  readonly pValues?: readonly number[]
  /** The interval level, for example 0.95. */
  readonly level?: number
  /**
   * The estimates as they evolved (an online learner's coefficients after each batch of samples): values[t][j] is
   * names[j]'s estimate at `at[t]` of the resource.
   */
  readonly trajectories?: {
    readonly resource: string
    readonly at: readonly number[]
    readonly values: readonly (readonly number[])[]
  }
}

/**
 * A coefficient table, with its shapes checked.
 *
 * @param table - The table without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a column does not match the names.
 * @example
 * ```ts
 * import { coefficientTable } from 'mdml-engine/exposition'
 * coefficientTable({ names: ['x'], estimates: [2], standardErrors: [0.1] }).estimates // => [2]
 * ```
 */
export function coefficientTable(table: Omit<CoefficientTable, 'kind'>): CoefficientTable {
  const n = table.names.length
  sameLength('coefficientTable', 'estimates', table.estimates.length, n)
  for (const field of ['standardErrors', 'intervals', 'pValues'] as const) {
    const column = table[field]
    if (column !== undefined) sameLength('coefficientTable', field, column.length, n)
  }
  if (table.trajectories !== undefined) {
    sameLength(
      'coefficientTable',
      'trajectories.values',
      table.trajectories.values.length,
      table.trajectories.at.length,
    )
    rowsOf('coefficientTable', 'trajectories.values', table.trajectories.values, n)
  }
  return seal({ kind: 'coefficientTable', ...table })
}

/** Cross-validated scores along one parameter, with their spread (for choosing it, e.g. by the one-SE rule). */
export interface CrossValidationCurve {
  readonly kind: 'crossValidationCurve'
  readonly parameter: string
  readonly values: readonly number[]
  readonly meanScores: readonly number[]
  readonly standardErrors: readonly number[]
  /** foldScores[i][f]: the score at value i on fold f. */
  readonly foldScores?: readonly (readonly number[])[]
  /** Whether higher scores are better. */
  readonly higherIsBetter: boolean
  readonly best?: number
  readonly oneStandardError?: number
  /** What the scores are ('mean squared error', 'accuracy', 'held-out log likelihood'). */
  readonly scoreName?: string
  /** Whether the parameter is a penalty strength, drawn decreasing to the right as on its regularisation path. */
  readonly penalty?: boolean
}

/**
 * A cross-validation curve, with its shapes checked.
 *
 * @param curve - The curve without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the scores do not match the values.
 * @example
 * ```ts
 * import { crossValidationCurve } from 'mdml-engine/exposition'
 * crossValidationCurve({ parameter: 'alpha', values: [1, 2], meanScores: [0.8, 0.7], standardErrors: [0.1, 0.1], higherIsBetter: true }).parameter // => 'alpha'
 * ```
 */
export function crossValidationCurve(curve: Omit<CrossValidationCurve, 'kind'>): CrossValidationCurve {
  const n = curve.values.length
  sameLength('crossValidationCurve', 'meanScores', curve.meanScores.length, n)
  sameLength('crossValidationCurve', 'standardErrors', curve.standardErrors.length, n)
  if (curve.foldScores !== undefined) sameLength('crossValidationCurve', 'foldScores', curve.foldScores.length, n)
  return seal({ kind: 'crossValidationCurve', ...curve })
}
