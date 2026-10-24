import type { Json } from '../base/index.ts'
import { sameLength } from './_check.ts'

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
}

/**
 * Search trials (no shape constraints beyond the type; `best` must index a trial).
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @example
 * ```ts
 * import { searchTrials } from 'whitebox-engine/exposition'
 * searchTrials({ method: 'grid', parameters: ['alpha'], trials: [{ params: { alpha: 1 }, value: 0.8, state: 'complete' }], higherIsBetter: true, best: 0 }).best // => 0
 * ```
 */
export function searchTrials(record: Omit<SearchTrials, 'kind'>): SearchTrials {
  if (record.best !== undefined && !(record.best >= 0 && record.best < record.trials.length))
    sameLength('searchTrials', 'best (index)', record.best, record.trials.length - 1)
  return { kind: 'searchTrials', ...record }
}

/** Training and validation scores against training-set size (or epochs, or another resource). */
export interface LearningCurve {
  readonly kind: 'learningCurve'
  readonly sizes: readonly number[]
  readonly train: readonly number[]
  /** The second column; empty (or equal to `train`) for a single series. */
  readonly validation: readonly number[]
  readonly trainSpread?: readonly number[]
  readonly validationSpread?: readonly number[]
  /** What the sizes are (default 'training size'): 'epoch', 'games', 'observations'. */
  readonly resource?: string
  /** Series names when the two columns are not training and validation scores (e.g. each trial's score and the best so far). */
  readonly labels?: { readonly train?: string; readonly validation?: string }
  /** What the scores are, for the y axis (default 'score'): 'loss', 'accuracy'. */
  readonly score?: string
  /**
   * Read the second column against its own right-hand axis (default false), for two quantities on different scales
   * (a running log loss and a running accuracy); each axis is then named after its column's label.
   */
  readonly secondaryAxis?: boolean
}

/**
 * A learning curve, with its shapes checked.
 *
 * @param curve - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when the scores do not match the sizes (an empty `validation` is a single series).
 * @example
 * ```ts
 * import { learningCurve } from 'whitebox-engine/exposition'
 * learningCurve({ sizes: [10, 20], train: [1, 0.95], validation: [0.6, 0.8] }).sizes // => [10, 20]
 * learningCurve({ sizes: [1, 2], train: [0.9, 0.4], validation: [], score: 'loss' }).validation // => []
 * ```
 */
export function learningCurve(curve: Omit<LearningCurve, 'kind'>): LearningCurve {
  const n = curve.sizes.length
  sameLength('learningCurve', 'train', curve.train.length, n)
  if (curve.validation.length > 0) sameLength('learningCurve', 'validation', curve.validation.length, n)
  if (curve.trainSpread !== undefined) sameLength('learningCurve', 'trainSpread', curve.trainSpread.length, n)
  if (curve.validationSpread !== undefined)
    sameLength('learningCurve', 'validationSpread', curve.validationSpread.length, curve.validation.length)
  return { kind: 'learningCurve', ...curve }
}
