import { WhiteboxError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'

/**
 * A hidden-state model decoded along a sequence (hidden Markov models, switching and changepoint models): the
 * posterior probability of every state at every step, the most likely path, the true states when the sequence was
 * simulated (the dishonest casino's dice), and the observations themselves.
 */
export interface StateSequence {
  readonly kind: 'stateSequence'
  readonly method: string
  /** The state names, in index order. */
  readonly states: readonly string[]
  /** posteriors[t][k]: P(z_t = k | x), each row summing to one (forward-backward's smoothed posteriors). */
  readonly posteriors: readonly (readonly number[])[]
  /** The most likely path (Viterbi): a state index per step. */
  readonly decoded?: readonly number[]
  /** The true state at every step, when known. */
  readonly truth?: readonly number[]
  /** The observation at every step (a symbol index or a value), with its name and the symbols' names. */
  readonly observations?: {
    readonly values: readonly number[]
    readonly name?: string
    readonly symbols?: readonly string[]
  }
  /** What a step is (default 'time'): 'roll', 'day', 'position'. */
  readonly stepName?: string
}

/**
 * A state sequence, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when a posterior row does not have one entry per state or a path does not have one
 *   entry per step; `value` when a path names a state that does not exist.
 * @example
 * ```ts
 * import { stateSequence } from 'whitebox-engine/exposition'
 * const r = stateSequence({ method: 'hmm', states: ['fair', 'loaded'], posteriors: [[0.9, 0.1], [0.2, 0.8]], decoded: [0, 1] })
 * r.posteriors.length // => 2
 * ```
 */
export function stateSequence(record: Omit<StateSequence, 'kind'>): StateSequence {
  const t = record.posteriors.length
  const k = record.states.length
  rowsOf('stateSequence', 'posteriors', record.posteriors, k)
  for (const field of ['decoded', 'truth'] as const) {
    const path = record[field]
    if (path === undefined) continue
    sameLength('stateSequence', field, path.length, t)
    if (path.some((s) => !(Number.isInteger(s) && s >= 0 && s < k)))
      throw new WhiteboxError('value', `stateSequence: ${field} names a state outside 0..${k - 1}`)
  }
  if (record.observations !== undefined)
    sameLength('stateSequence', 'observations.values', record.observations.values.length, t)
  return { kind: 'stateSequence', ...record }
}
