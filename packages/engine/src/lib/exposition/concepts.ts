/**
 * The record of concept-based explanations: TCAV scores of one class for several concepts, each over runs against
 * random sets beside the scores of random directions, with the t test's p-value and the probes' accuracies.
 */

import { MdmlError } from '../base/index.ts'

/** One concept's TCAV test. */
export interface ConceptScore {
  readonly name: string
  /** TCAV scores, one per run against a random set. */
  readonly scores: readonly number[]
  /** Scores of random-vs-random directions, the null. */
  readonly randomScores: readonly number[]
  readonly mean: number
  readonly pValue: number
  /** The concept probes' mean accuracy. */
  readonly accuracy?: number
}

/**
 * Concept sensitivity of a model (TCAV): for a class and a layer, each concept's TCAV scores across runs, the random
 * baseline, and whether the difference is significant.
 */
export interface ConceptScores {
  readonly kind: 'concepts'
  readonly method: string
  /** The class explained. */
  readonly target: string
  /** The layer the concepts live in. */
  readonly layer: string
  readonly concepts: readonly ConceptScore[]
  /** The significance level the view marks (default 0.05). */
  readonly alpha?: number
}

/**
 * A concept-scores record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when a concept has no scores.
 * @example
 * ```ts
 * import { conceptScores } from 'mdml-engine/exposition'
 * const r = conceptScores({ method: 'TCAV', target: 'zebra', layer: 'mixed4c', concepts: [{ name: 'striped', scores: [0.9, 0.8], randomScores: [0.5, 0.4], mean: 0.85, pValue: 0.01 }] })
 * r.concepts[0].name // => 'striped'
 * ```
 */
export function conceptScores(record: Omit<ConceptScores, 'kind'>): ConceptScores {
  for (const c of record.concepts) {
    if (c.scores.length === 0) throw new MdmlError('shape', `concepts: ${c.name} has no scores`)
  }
  return { kind: 'concepts', ...record }
}
