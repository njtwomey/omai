/**
 * The exposition record of a probability simplex (`simplex`): the triangle of three strategies (or the unit square of
 * two populations of two strategies each) with the paths of the state over it, a field of directions, marked points
 * and the fixed points with their stability. Replicator dynamics and learning in games are read this way.
 */

import { OmaiError } from '../base/index.ts'
import { seal } from './_check.ts'
import type { FixedPointClass } from './fields.ts'

/** A named path of states over the simplex. */
export interface SimplexPath {
  readonly name: string
  /** The states in time order: three probabilities summing to one (`triangle`), or two probabilities (`square`). */
  readonly points: readonly (readonly number[])[]
}

/** A fixed point of the dynamics: where, its stability and an optional name. */
export interface SimplexFixedPoint {
  readonly at: readonly number[]
  readonly stability: FixedPointClass
  readonly label?: string
}

/** A field of directions on the simplex: at each state, the change of the state (three numbers summing to zero, or two). */
export interface SimplexField {
  readonly at: readonly (readonly number[])[]
  readonly by: readonly (readonly number[])[]
}

/**
 * The simplex of three strategies (`triangle`; states are barycentric: three probabilities summing to one) or the unit
 * square of two populations (`square`; a state is the probability of the first strategy in each population).
 */
export interface Simplex {
  readonly kind: 'simplex'
  readonly shape: 'triangle' | 'square'
  /** Three vertex names (`triangle`), or the names of the two axes (`square`). */
  readonly labels: readonly string[]
  readonly paths?: readonly SimplexPath[]
  readonly field?: SimplexField
  readonly fixedPoints?: readonly SimplexFixedPoint[]
  /** A caption (`replicator dynamics of rock, paper, scissors`). */
  readonly caption?: string
}

/**
 * A simplex record, with every state checked to have the shape's number of coordinates, to be non-negative and, on a
 * triangle, to sum to one.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws OmaiError `shape` when the labels, a state or a direction has the wrong length; `value` when a state is not
 *   a probability vector.
 * @example
 * ```ts
 * import { simplexRecord } from 'omai-engine/exposition'
 * const r = simplexRecord({
 *   shape: 'triangle',
 *   labels: ['rock', 'paper', 'scissors'],
 *   paths: [{ name: 'run', points: [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]] }],
 * })
 * r.paths?.[0]?.points.length // => 2
 * ```
 */
export function simplexRecord(record: Omit<Simplex, 'kind'>): Simplex {
  const d = record.shape === 'triangle' ? 3 : 2
  if (record.labels.length !== d)
    throw new OmaiError('shape', `simplex: labels has ${record.labels.length} entries, expected ${d}`)
  const state = (name: string, p: readonly number[]): void => {
    if (p.length !== d) throw new OmaiError('shape', `simplex: ${name} has ${p.length} entries, expected ${d}`)
    if (!p.every((v) => v >= -1e-9 && v <= 1 + 1e-9))
      throw new OmaiError('value', `simplex: ${name} is not a probability`)
    if (d === 3 && Math.abs((p[0] ?? 0) + (p[1] ?? 0) + (p[2] ?? 0) - 1) > 1e-6)
      throw new OmaiError('value', `simplex: ${name} does not sum to one`)
  }
  for (const [k, path] of (record.paths ?? []).entries())
    for (const [i, p] of path.points.entries()) state(`paths[${k}].points[${i}]`, p)
  for (const [k, f] of (record.fixedPoints ?? []).entries()) state(`fixedPoints[${k}]`, f.at)
  if (record.field !== undefined) {
    if (record.field.at.length !== record.field.by.length)
      throw new OmaiError(
        'shape',
        `simplex: field has ${record.field.by.length} directions for ${record.field.at.length} states`,
      )
    for (const [i, p] of record.field.at.entries()) state(`field.at[${i}]`, p)
    for (const [i, b] of record.field.by.entries())
      if (b.length !== d) throw new OmaiError('shape', `simplex: field.by[${i}] has ${b.length} entries, expected ${d}`)
  }
  return seal({ kind: 'simplex', ...record })
}
