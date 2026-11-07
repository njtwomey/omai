/**
 * The confusion matrix as an exposition record: how many rows of each true class a classifier predicted as each class
 * on a named split, optionally at recorded steps of training so a view can play it. `confusionCounts` tallies the
 * counts from predictions and labels.
 */

import { MdmlError } from '../base/index.ts'
import { rowsOf, sameLength, seal } from './_check.ts'

/** The confusion counts at one recorded step of training (an epoch). */
export interface ConfusionStep {
  readonly iteration: number
  readonly counts: readonly (readonly number[])[]
}

/**
 * A classifier's confusion matrix: how many rows of each true class it predicted as each class, on the rows named by
 * `split`; optionally the counts at recorded steps of training, so a view can play the diagonal filling in.
 */
export interface ConfusionMatrix {
  readonly kind: 'confusionMatrix'
  readonly classes: readonly string[]
  /** counts[i][j]: rows of true class i predicted as class j. */
  readonly counts: readonly (readonly number[])[]
  /** The rows counted ('validation', 'training', 'test'). */
  readonly split?: string
  readonly steps?: readonly ConfusionStep[]
  /** What a step is (default 'step'). */
  readonly stepName?: string
}

/**
 * A confusion matrix, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the counts (or a step's) are not one row and column per class.
 * @example
 * ```ts
 * import { confusionMatrix } from 'mdml-engine/exposition'
 * confusionMatrix({ classes: ['cat', 'dog'], counts: [[5, 1], [2, 7]] }).counts[1] // => [2, 7]
 * ```
 */
export function confusionMatrix(record: Omit<ConfusionMatrix, 'kind'>): ConfusionMatrix {
  const k = record.classes.length
  const check = (field: string, counts: readonly (readonly number[])[]): void => {
    sameLength('confusionMatrix', field, counts.length, k)
    rowsOf('confusionMatrix', field, counts, k)
  }
  check('counts', record.counts)
  for (const [t, s] of (record.steps ?? []).entries()) check(`steps[${t}].counts`, s.counts)
  return seal({ kind: 'confusionMatrix', ...record })
}

/**
 * The confusion counts of predictions against the truth, both class indices.
 *
 * @param truth - The true class index of every row.
 * @param predicted - The predicted class index of every row.
 * @param classes - The number of classes.
 * @returns counts[i][j]: rows of true class i predicted as j.
 * @throws MdmlError `shape` when the two do not have the same length; `value` for an index outside the classes.
 * @example
 * ```ts
 * import { confusionCounts } from 'mdml-engine/exposition'
 * confusionCounts([0, 0, 1, 1], [0, 1, 1, 1], 2) // => [[1, 1], [0, 2]]
 * ```
 */
export function confusionCounts(truth: ArrayLike<number>, predicted: ArrayLike<number>, classes: number): number[][] {
  sameLength('confusionCounts', 'predicted', predicted.length, truth.length)
  const counts = Array.from({ length: classes }, () => new Array<number>(classes).fill(0))
  for (let i = 0; i < truth.length; i++) {
    const t = truth[i] ?? -1
    const p = predicted[i] ?? -1
    if (!(Number.isInteger(t) && Number.isInteger(p) && t >= 0 && p >= 0 && t < classes && p < classes))
      throw new MdmlError('value', `confusionCounts: row ${i} has a class outside 0..${classes - 1}`)
    ;(counts[t] as number[])[p] = (counts[t]?.[p] ?? 0) + 1
  }
  return counts
}
