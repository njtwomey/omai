import { rowsOf, sameLength } from './_check.ts'
import type { InputRows } from './rows.ts'

/** The neighbours found for each query point (k-NN, radius neighbours, LOF). */
export interface NeighbourRecord {
  readonly kind: 'neighbours'
  /** indices[q]: the training rows nearest query q, nearest first. */
  readonly indices: readonly (readonly number[])[]
  readonly distances: readonly (readonly number[])[]
  readonly metric: string
  /** The training rows the indices point into (two coordinates each, with any classes), when the data is small. */
  readonly rows?: InputRows
  /** The query rows (same coordinates), when they are not the training rows themselves (query q is row q). */
  readonly queries?: readonly (readonly number[])[]
}

/**
 * A neighbour record, with its shapes checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when indices and distances disagree.
 * @example
 * ```ts
 * import { neighbourRecord } from 'whitebox-engine/exposition'
 * neighbourRecord({ indices: [[2, 0]], distances: [[0.1, 0.4]], metric: 'euclidean' }).indices // => [[2, 0]]
 * ```
 */
export function neighbourRecord(record: Omit<NeighbourRecord, 'kind'>): NeighbourRecord {
  sameLength('neighbours', 'distances', record.distances.length, record.indices.length)
  for (const [q, row] of record.indices.entries())
    rowsOf('neighbours', `distances[${q}]`, [record.distances[q] ?? []], row.length)
  return { kind: 'neighbours', ...record }
}

/** How each feature contributes to each class's score (naive Bayes, discriminant analysis). */
export interface LikelihoodTable {
  readonly kind: 'likelihoodTable'
  readonly classes: readonly string[]
  readonly features: readonly string[]
  readonly logPriors: readonly number[]
  /** contributions[c][f]: the log-likelihood contribution of feature f to class c (for one query, or averaged). */
  readonly contributions: readonly (readonly number[])[]
  /** What the contributions are for: one query row (its index when known) or the average over rows. */
  readonly scope?:
    | { readonly kind: 'query'; readonly row?: number }
    | { readonly kind: 'average'; readonly rows: number }
}

/**
 * A likelihood table, with its shapes checked.
 *
 * @param table - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when the table does not match classes and features.
 * @example
 * ```ts
 * import { likelihoodTable } from 'whitebox-engine/exposition'
 * likelihoodTable({ classes: ['a', 'b'], features: ['x'], logPriors: [-0.7, -0.7], contributions: [[-1], [-2]] }).classes // => ['a', 'b']
 * ```
 */
export function likelihoodTable(table: Omit<LikelihoodTable, 'kind'>): LikelihoodTable {
  sameLength('likelihoodTable', 'logPriors', table.logPriors.length, table.classes.length)
  sameLength('likelihoodTable', 'contributions', table.contributions.length, table.classes.length)
  rowsOf('likelihoodTable', 'contributions', table.contributions, table.features.length)
  return { kind: 'likelihoodTable', ...table }
}
