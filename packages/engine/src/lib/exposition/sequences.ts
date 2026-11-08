/**
 * The exposition record of a hidden-state model reading a sequence (hidden Markov models, switching models): each
 * state's posterior at each time, the decoded path and, when known, the true states.
 */

import { OmaiError } from '../base/index.ts'
import { rowsOf, sameLength, seal } from './_check.ts'

/** A hidden-state model's reading of a sequence: each state's posterior at each time, the decoded path, the truth. */
export interface StateSequence {
  readonly kind: 'stateSequence'
  /** The states' names. */
  readonly states: readonly string[]
  /** posteriors[t][k]: P(z_t = k | the whole sequence), from forward-backward. */
  readonly posteriors: readonly (readonly number[])[]
  /** The most probable path (Viterbi), a state per time. */
  readonly decoded?: readonly number[]
  /** The true states, when the sequence was simulated. */
  readonly truth?: readonly number[]
  /** The observations, when one number per time (a die's faces, a measurement), with their name. */
  readonly observations?: { readonly name: string; readonly values: readonly number[] }
}

/**
 * A state sequence, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when a posterior row, the paths or the observations do not match.
 * @example
 * ```ts
 * import { stateSequence } from 'omai-engine/exposition'
 * stateSequence({ states: ['fair', 'loaded'], posteriors: [[0.9, 0.1], [0.2, 0.8]], decoded: [0, 1] }).decoded // => [0, 1]
 * ```
 */
export function stateSequence(record: Omit<StateSequence, 'kind'>): StateSequence {
  const t = record.posteriors.length
  rowsOf('stateSequence', 'posteriors', record.posteriors, record.states.length)
  for (const field of ['decoded', 'truth'] as const) {
    const path = record[field]
    if (path === undefined) continue
    sameLength('stateSequence', field, path.length, t)
    if (path.some((s) => !Number.isInteger(s) || s < 0 || s >= record.states.length))
      throw new OmaiError('value', `stateSequence: ${field} must hold state indices`)
  }
  if (record.observations !== undefined)
    sameLength('stateSequence', 'observations', record.observations.values.length, t)
  return seal({ kind: 'stateSequence', ...record })
}
