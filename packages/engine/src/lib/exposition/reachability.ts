/**
 * The exposition record of a reachability plot (`reachability`): the points of a density-based clustering in the
 * order OPTICS visits them with each one's reachability distance, so clusters are valleys and a threshold cuts them out.
 */

import { MdmlError } from '../base/index.ts'
import { seal } from './_check.ts'

/** A reachability plot: distances in cluster order, optionally with the clusters found and a cut. */
export interface Reachability {
  readonly kind: 'reachability'
  /** The row index visited at each position (default the identity). */
  readonly order?: readonly number[]
  /** The reachability distance at each position; `null` where it is undefined (the first point of a component). */
  readonly reach: readonly (number | null)[]
  /** A cluster per position (-1 for noise), when known; else the cut at `threshold` finds them. */
  readonly labels?: readonly number[]
  /** The distance at which clusters are cut out (drawn as a line). */
  readonly threshold?: number
}

/**
 * A reachability plot, with the lengths of its arrays checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the order or the labels do not have one entry per position; `value` when a distance
 *   is negative or not a number.
 * @example
 * ```ts
 * import { reachabilityRecord } from 'mdml-engine/exposition'
 * reachabilityRecord({ reach: [null, 0.1, 0.1, 0.9, 0.1], threshold: 0.5 }).reach.length // => 5
 * ```
 */
export function reachabilityRecord(record: Omit<Reachability, 'kind'>): Reachability {
  const n = record.reach.length
  if (record.order !== undefined && record.order.length !== n)
    throw new MdmlError('shape', `reachability: order has ${record.order.length} entries, expected ${n}`)
  if (record.labels !== undefined && record.labels.length !== n)
    throw new MdmlError('shape', `reachability: labels has ${record.labels.length} entries, expected ${n}`)
  for (const [i, r] of record.reach.entries())
    if (r !== null && !(r >= 0)) throw new MdmlError('value', `reachability: reach[${i}] is not a distance`)
  return seal({ kind: 'reachability', ...record })
}
