/**
 * Exposition records for model selection: every trial of a hyperparameter search (grid, random, successive halving,
 * Bayesian, TPE) with its parameters, scores and fate, and learning curves of training and validation scores against
 * training-set size, epochs or another resource.
 */

import type { Json } from '../base/index.ts'
import { sameLength, seal } from './_check.ts'

/** Every trial of a search (grid, random, halving, Bayesian, TPE): its parameters, scores and fate. */
export interface SearchTrials {
  readonly kind: 'searchTrials'
  readonly method: string
  readonly parameters: readonly string[]
  readonly trials: readonly {
    readonly params: { readonly [name: string]: Json }
    /** The objective (or mean validation score). */
    readonly value: number
    /** Per fold or per resource step, when available. */
    readonly intermediate?: readonly number[]
    readonly state: 'complete' | 'pruned' | 'failed'
  }[]
  readonly higherIsBetter: boolean
  readonly best?: number
  /** What the values are ('accuracy', 'neg_mean_squared_error'). */
  readonly scoreName?: string
}

/**
 * Search trials (no shape constraints beyond the type; `best` must index a trial).
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @example
 * ```ts
 * import { searchTrials } from 'omai-engine/exposition'
 * searchTrials({ method: 'grid', parameters: ['alpha'], trials: [{ params: { alpha: 1 }, value: 0.8, state: 'complete' }], higherIsBetter: true, best: 0 }).best // => 0
 * ```
 */
export function searchTrials(record: Omit<SearchTrials, 'kind'>): SearchTrials {
  if (record.best !== undefined && !(record.best >= 0 && record.best < record.trials.length))
    sameLength('searchTrials', 'best (index)', record.best, record.trials.length - 1)
  return seal({ kind: 'searchTrials', ...record })
}

/**
 * Training and validation scores against training-set size (or epochs, or another resource); `validation` may be empty
 * for a single series.
 */
export interface LearningCurve {
  readonly kind: 'learningCurve'
  readonly sizes: readonly number[]
  readonly train: readonly number[]
  readonly validation: readonly number[]
  readonly trainSpread?: readonly number[]
  readonly validationSpread?: readonly number[]
  readonly resource?: string
  /** Series names when the two columns are not training and validation scores (e.g. each trial's score and the best so far). */
  readonly labels?: { readonly train?: string; readonly validation?: string }
  /** What the scores are ('accuracy', 'cross-validated R²'). */
  readonly scoreName?: string
  /** Draw the validation column on its own y axis (two measures on different scales: log loss and accuracy). */
  readonly secondaryAxis?: boolean
}

/**
 * A learning curve, with its shapes checked.
 *
 * @param curve - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when the scores do not match the sizes.
 * @example
 * ```ts
 * import { learningCurve } from 'omai-engine/exposition'
 * learningCurve({ sizes: [10, 20], train: [1, 0.95], validation: [0.6, 0.8] }).sizes // => [10, 20]
 * ```
 */
export function learningCurve(curve: Omit<LearningCurve, 'kind'>): LearningCurve {
  const n = curve.sizes.length
  sameLength('learningCurve', 'train', curve.train.length, n)
  // An empty validation column is a single series.
  if (curve.validation.length > 0) sameLength('learningCurve', 'validation', curve.validation.length, n)
  for (const field of ['trainSpread', 'validationSpread'] as const) {
    const column = curve[field]
    if (column !== undefined) sameLength('learningCurve', field, column.length, n)
  }
  return seal({ kind: 'learningCurve', ...curve })
}
