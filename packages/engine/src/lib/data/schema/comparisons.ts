import { WhiteboxError } from '../../base/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { cardinality, checkRows, concrete, concreteFloat, defaultNames, metaFor } from './_coerce.ts'
import type { DatasetMeta } from './common.ts'

/**
 * Outcomes of games, matches or judged comparisons between players (or items), for rating models (Bradley-Terry,
 * Elo, Glicko, TrueSkill, Plackett-Luce). Each row is one contest among `k` participants; `ranks` gives each
 * participant's finishing place (0 is first), and equal ranks are ties. Pairwise data has `k = 2`.
 */
export interface ComparisonData {
  readonly kind: 'comparisons'
  /** Player indices `[n, k]` (int32). */
  readonly participants: Tensor<'int32'>
  /** Finishing places `[n, k]` (int32), 0 first; ties share a place. */
  readonly ranks: Tensor<'int32'>
  /** When each contest happened `[n]`, for dynamic ratings. */
  readonly times?: Tensor
  readonly playerNames: readonly string[]
  readonly meta: DatasetMeta
}

/**
 * Comparison data from participants and their finishing places.
 *
 * @param participants - Player indices `[n, k]`.
 * @param ranks - Finishing places `[n, k]` (0 first; ties share a place).
 * @param options - Times, player names and metadata.
 * @returns The dataset.
 * @throws WhiteboxError with code `shape` when participants and ranks differ in shape.
 * @example
 * ```ts
 * import { comparisons } from 'whitebox-engine/data'
 * comparisons([[0, 1], [1, 2]], [[0, 1], [0, 0]]).playerNames // => ['p0', 'p1', 'p2']
 * ```
 */
export function comparisons(
  participants: TensorLike,
  ranks: TensorLike,
  options: {
    readonly times?: TensorLike
    readonly playerNames?: readonly string[]
    readonly meta?: Partial<DatasetMeta>
  } = {},
): ComparisonData {
  const p = concrete(participants, 'int32', 'comparisons')
  const r = concrete(ranks, 'int32', 'comparisons')
  if (p.shape.length !== 2 || p.shape.join() !== r.shape.join())
    throw new WhiteboxError('shape', `comparisons: participants ${p.shape} and ranks ${r.shape} must both be [n, k]`)
  const times = options.times === undefined ? undefined : concreteFloat(options.times, 'comparisons')
  checkRows('comparisons', p.shape[0] ?? 0, { times })
  return {
    kind: 'comparisons',
    participants: p,
    ranks: r,
    ...(times === undefined ? {} : { times }),
    playerNames: options.playerNames ?? defaultNames('p', cardinality(p.data)),
    meta: metaFor('comparisons', options.meta),
  }
}

/**
 * Pairwise comparison data from winners and losers (with optional ties), the common case of {@link comparisons}.
 *
 * @param winners - The winner of each game `[n]`.
 * @param losers - The loser of each game `[n]`.
 * @param options - Which games were draws, times, player names and metadata.
 * @returns The dataset, with participants `[winner, loser]`.
 * @example
 * ```ts
 * import { pairwise } from 'whitebox-engine/data'
 * Array.from(pairwise([0, 2], [1, 0], { ties: [false, true] }).ranks.data) // => [0, 1, 0, 0]
 * ```
 */
export function pairwise(
  winners: TensorLike,
  losers: TensorLike,
  options: {
    readonly ties?: readonly boolean[]
    readonly times?: TensorLike
    readonly playerNames?: readonly string[]
    readonly meta?: Partial<DatasetMeta>
  } = {},
): ComparisonData {
  const w = concrete(winners, 'int32', 'pairwise')
  const l = concrete(losers, 'int32', 'pairwise')
  const n = w.shape[0] ?? 0
  checkRows('pairwise', n, { losers: l })
  const p = new Int32Array(2 * n)
  const r = new Int32Array(2 * n)
  for (let i = 0; i < n; i++) {
    p[2 * i] = w.data[i] ?? 0
    p[2 * i + 1] = l.data[i] ?? 0
    r[2 * i + 1] = options.ties?.[i] === true ? 0 : 1
  }
  return comparisons(fromStorage(p, [n, 2], 'int32'), fromStorage(r, [n, 2], 'int32'), options)
}
