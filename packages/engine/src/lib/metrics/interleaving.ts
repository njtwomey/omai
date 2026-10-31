/**
 * Interleaved comparison of two rankers on one user (Joachims 2003; Radlinski, Kurup and Joachims 2008): team-draft
 * interleaving merges two ranked lists into one by alternating picks in random order, remembers which ranker supplied
 * each position, and credits each click to its ranker; the ranker with more clicks wins the impression, and the share of
 * impressions won estimates the preference far more sensitively than an A/B test of the two rankers on separate users.
 */

import { MdmlError } from '../base/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'

/** The ranker that supplied a position. */
export type Team = 'a' | 'b'

/** An interleaved list and the ranker behind each position. */
export interface Interleaving {
  readonly items: readonly number[]
  readonly teams: readonly Team[]
}

/**
 * Team-draft interleaving: in each round a coin flip decides which ranker picks first; each ranker contributes its
 * highest-ranked item not yet shown, the pick recorded as that ranker's. Lists shorter than `length` end when both run out.
 *
 * @param a - The first ranker's list, best first.
 * @param b - The second ranker's list.
 * @param key - The key for the coin flips.
 * @param length - The length of the merged list (default all distinct items).
 * @returns The merged list with the team of each position.
 * @concept interleaving
 * @example
 * ```ts
 * import { teamDraftInterleave } from 'mdml-engine/metrics'
 * import { key } from 'mdml-engine/random'
 * const r = teamDraftInterleave([1, 2, 3], [3, 4, 5], key(0))
 * [r.items.length, new Set(r.items).size] // => [5, 5]
 * ```
 */
export function teamDraftInterleave(
  a: readonly number[],
  b: readonly number[],
  key: Key,
  length: number = new Set([...a, ...b]).size,
): Interleaving {
  const items: number[] = []
  const teams: Team[] = []
  const used = new Set<number>()
  const next = (list: readonly number[]): number | undefined => list.find((x) => !used.has(x))
  for (let round = 0; items.length < length; round++) {
    const first: Team = (uniform(foldIn(key, round), [1]).data[0] ?? 0) < 0.5 ? 'a' : 'b'
    const order: Team[] = first === 'a' ? ['a', 'b'] : ['b', 'a']
    let added = false
    for (const team of order) {
      if (items.length >= length) break
      const pick = next(team === 'a' ? a : b)
      if (pick === undefined) continue
      items.push(pick)
      teams.push(team)
      used.add(pick)
      added = true
    }
    if (!added) break
  }
  return { items, teams }
}

/**
 * The outcome of one impression: the clicks credited to each ranker and the winner (a tie when equal).
 *
 * @param teams - The team of each position of the interleaved list.
 * @param clicks - Whether each position was clicked (1) or not (0).
 * @returns The clicks per team and the winner.
 * @throws MdmlError with code `shape` for different lengths.
 * @concept interleaving
 * @example
 * ```ts
 * import { interleavingCredit } from 'mdml-engine/metrics'
 * interleavingCredit(['a', 'b', 'a', 'b'], [1, 0, 0, 1]) // => { a: 1, b: 1, winner: 'tie' }
 * ```
 */
export function interleavingCredit(
  teams: readonly Team[],
  clicks: readonly number[],
): { readonly a: number; readonly b: number; readonly winner: Team | 'tie' } {
  if (teams.length !== clicks.length)
    throw new MdmlError('shape', 'interleavingCredit: teams and clicks differ in length')
  let a = 0
  let b = 0
  for (const [i, t] of teams.entries()) {
    if ((clicks[i] ?? 0) > 0) {
      if (t === 'a') a++
      else b++
    }
  }
  return { a, b, winner: a > b ? 'a' : b > a ? 'b' : 'tie' }
}

/**
 * The preference for ranker A over many impressions: wins over wins plus losses, ties left out, with a binomial
 * standard error (0.5 means no preference).
 *
 * @param winners - The winner of each impression.
 * @returns The share of decided impressions A won, its standard error and the counts.
 * @concept interleaving
 * @example
 * ```ts
 * import { interleavingPreference } from 'mdml-engine/metrics'
 * interleavingPreference(['a', 'a', 'b', 'tie']).preference // => 0.6666666666666666
 * ```
 */
export function interleavingPreference(winners: readonly (Team | 'tie')[]): {
  readonly preference: number
  readonly standardError: number
  readonly wins: number
  readonly losses: number
  readonly ties: number
} {
  const wins = winners.filter((w) => w === 'a').length
  const losses = winners.filter((w) => w === 'b').length
  const decided = wins + losses
  const p = decided === 0 ? 0.5 : wins / decided
  return {
    preference: p,
    standardError: decided === 0 ? Number.NaN : Math.sqrt((p * (1 - p)) / decided),
    wins,
    losses,
    ties: winners.length - decided,
  }
}
