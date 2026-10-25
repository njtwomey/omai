/**
 * The Pareto-front record (`paretoFront`): candidate classifiers as points of accuracy against a fairness violation,
 * in families (how they were made: a grid of multipliers, a moved threshold, hyperparameters), each family's Pareto
 * front, and the point a selection rule chose; views draw the points, the fronts as steps and the choice.
 */

import { WhiteboxError } from '../base/index.ts'
import { sameLength } from './_check.ts'

/** One family of candidates: their violations, accuracies, labels and the indices of its front. */
export interface FrontFamily {
  readonly name: string
  readonly violation: readonly number[]
  readonly accuracy: readonly number[]
  /** A label per candidate (its multiplier, threshold or hyperparameters). */
  readonly labels?: readonly string[]
  /** Indices of the family's Pareto front, by increasing violation. */
  readonly front: readonly number[]
}

/**
 * Candidate classifiers by accuracy and fairness violation, in families, with each family's Pareto front, reference
 * marks (an exact post-processor, the unconstrained model) and the candidate a selection rule chose.
 */
export interface ParetoFront {
  readonly kind: 'paretoFront'
  readonly label?: string
  /** The violation's name (`demographic parity difference`). */
  readonly violationName: string
  readonly families: readonly FrontFamily[]
  readonly marks?: readonly { readonly violation: number; readonly accuracy: number; readonly label: string }[]
  /** The chosen candidate and why (`most accurate with violation at most 0.05`). */
  readonly chosen?: { readonly family: number; readonly index: number; readonly rule: string }
}

/**
 * A Pareto-front record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a family's columns differ in length; `value` when a front index or the chosen
 *   candidate is not a candidate.
 * @example
 * ```ts
 * import { paretoFront } from 'whitebox-engine/exposition'
 * const r = paretoFront({ violationName: 'gap', families: [{ name: 'grid', violation: [0.1, 0.2], accuracy: [0.7, 0.8], front: [0, 1] }] })
 * r.families[0]?.front // => [0, 1]
 * ```
 */
export function paretoFront(record: Omit<ParetoFront, 'kind'>): ParetoFront {
  for (const [f, fam] of record.families.entries()) {
    const n = fam.violation.length
    sameLength('paretoFront', `families[${f}].accuracy`, fam.accuracy.length, n)
    if (fam.labels !== undefined) sameLength('paretoFront', `families[${f}].labels`, fam.labels.length, n)
    for (const i of fam.front)
      if (!(Number.isInteger(i) && i >= 0 && i < n))
        throw new WhiteboxError('value', `paretoFront: families[${f}].front holds ${i}, not a candidate`)
  }
  const c = record.chosen
  if (c !== undefined) {
    const fam = record.families[c.family]
    if (fam === undefined || !(c.index >= 0 && c.index < fam.violation.length))
      throw new WhiteboxError('value', 'paretoFront: the chosen candidate is not a candidate')
  }
  return { kind: 'paretoFront', ...record }
}
