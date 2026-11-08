/**
 * Schemas for recommendation and ranking data: user-item interactions (ratings, clicks, purchases) one row per
 * interaction, and learning-to-rank data of query-document feature rows with graded relevance grouped by query.
 */

import type { Tensor, TensorLike } from '../../tensor/index.ts'
import { cardinality, checkRows, concrete, concreteFloat, defaultNames, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/**
 * User-item interactions (ratings, clicks, purchases), one row per interaction. Optional fields carry the time, the
 * session, the display position and the logging policy's propensity of showing the item, for off-policy evaluation.
 */
export interface InteractionData {
  readonly kind: 'interactions'
  /** User indices `[n]` (int32). */
  readonly users: Tensor<'int32'>
  /** Item indices `[n]` (int32). */
  readonly items: Tensor<'int32'>
  /** Ratings, clicks (0 or 1) or counts `[n]`. */
  readonly values: Tensor
  readonly times?: Tensor
  readonly sessions?: Tensor<'int32'>
  /** The rank at which the item was shown (0 is the top). */
  readonly positions?: Tensor<'int32'>
  /** The probability with which the logging policy showed the item. */
  readonly propensities?: Tensor
  readonly nUsers: number
  readonly nItems: number
  readonly userNames?: readonly string[]
  readonly itemNames?: readonly string[]
  readonly meta: DatasetMeta
}

/** Learning-to-rank data: query-document feature rows with graded relevance, grouped by query. */
export interface QueryData {
  readonly kind: 'queries'
  /** Features of each query-document pair `[n, d]`. */
  readonly x: Tensor
  /** Graded relevance `[n]`. */
  readonly y: Tensor
  /** The query each row belongs to `[n]` (int32). */
  readonly queries: Tensor<'int32'>
  readonly featureNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Arrays for {@link interactions}. */
export interface InteractionArrays {
  readonly users: TensorLike
  readonly items: TensorLike
  readonly values: TensorLike
  readonly times?: TensorLike
  readonly sessions?: TensorLike
  readonly positions?: TensorLike
  readonly propensities?: TensorLike
  /** The number of users (default: one more than the largest index). */
  readonly nUsers?: number
  readonly nItems?: number
  readonly userNames?: readonly string[]
  readonly itemNames?: readonly string[]
}

/**
 * An interaction dataset from arrays, checking that every per-row field has one entry per interaction.
 *
 * @param arrays - Users, items, values and the optional fields.
 * @param meta - Name, description and provenance.
 * @returns The dataset.
 * @throws OmaiError with code `shape` when the fields disagree in length.
 * @example
 * ```ts
 * import { interactions } from 'omai-engine/data'
 * interactions({ users: [0, 1, 1], items: [2, 0, 1], values: [5, 3, 4] }).nItems // => 3
 * ```
 */
export function interactions(arrays: InteractionArrays, meta: Partial<DatasetMeta> = {}): InteractionData {
  const users = concrete(arrays.users, 'int32', 'interactions')
  const items = concrete(arrays.items, 'int32', 'interactions')
  const values = concreteFloat(arrays.values, 'interactions')
  const out: InteractionData = {
    kind: 'interactions',
    users,
    items,
    values,
    ...(arrays.times === undefined ? {} : { times: concreteFloat(arrays.times, 'interactions') }),
    ...(arrays.sessions === undefined ? {} : { sessions: concrete(arrays.sessions, 'int32', 'interactions') }),
    ...(arrays.positions === undefined ? {} : { positions: concrete(arrays.positions, 'int32', 'interactions') }),
    ...(arrays.propensities === undefined ? {} : { propensities: concreteFloat(arrays.propensities, 'interactions') }),
    nUsers: arrays.nUsers ?? cardinality(users.data),
    nItems: arrays.nItems ?? cardinality(items.data),
    ...(arrays.userNames === undefined ? {} : { userNames: arrays.userNames }),
    ...(arrays.itemNames === undefined ? {} : { itemNames: arrays.itemNames }),
    meta: metaFor('interactions', meta),
  }
  checkRows('interactions', users.shape[0] ?? 0, {
    items,
    values,
    times: out.times,
    sessions: out.sessions,
    positions: out.positions,
    propensities: out.propensities,
  })
  return out
}

/**
 * A learning-to-rank dataset from features, relevance grades and query ids.
 *
 * @param x - Features `[n, d]`.
 * @param y - Relevance `[n]`.
 * @param queries - Query ids `[n]`.
 * @param meta - Feature names and provenance.
 * @returns The dataset.
 * @example
 * ```ts
 * import { queryData } from 'omai-engine/data'
 * queryData([[1], [2], [3]], [2, 0, 1], [0, 0, 1]).queries.shape // => [3]
 * ```
 */
export function queryData(
  x: TensorLike,
  y: TensorLike,
  queries: TensorLike,
  meta: Partial<DatasetMeta> & { readonly featureNames?: readonly string[] } = {},
): QueryData {
  const xs = concreteFloat(x, 'queryData')
  const ys = concreteFloat(y, 'queryData')
  const qs = concrete(queries, 'int32', 'queryData')
  checkRows('queryData', xs.shape[0] ?? 0, { y: ys, queries: qs })
  const { featureNames, ...rest } = meta
  return {
    kind: 'queries',
    x: xs,
    y: ys,
    queries: qs,
    featureNames: featureNames ?? defaultNames('x', xs.shape[1] ?? 0),
    meta: metaFor('queries', rest),
  }
}
