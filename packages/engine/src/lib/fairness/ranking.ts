/**
 * Fairness in rankings: the exposure each group receives from the positions it is ranked in (Singh and Joachims, 2018)
 * and its ratio to the group's relevance (disparate treatment), FA*IR's ranked group fairness test and top-k
 * algorithm (Zehlike et al., 2017), and deterministic re-ranking to minimum shares at every prefix (Geyik, Ambler and
 * Kenthapadi, 2019).
 */

import { MdmlError } from '../base/index.ts'

/** How attention falls off with position (1-based): logarithmic (DCG's 1 / log2(1 + rank)) or geometric. */
export type PositionBias = { readonly kind: 'log' } | { readonly kind: 'geometric'; readonly p: number }

/** The attention of a 1-based position. */
export function positionWeight(rank: number, bias: PositionBias = { kind: 'log' }): number {
  return bias.kind === 'log' ? 1 / Math.log2(1 + rank) : bias.p * (1 - bias.p) ** (rank - 1)
}

/** Each group's exposure in a ranking. */
export interface Exposure {
  readonly groups: readonly string[]
  /** Each group's mean exposure per member. */
  readonly exposure: readonly number[]
  /** Each group's mean relevance per member (when relevances are given). */
  readonly relevance?: readonly number[]
  /** The smallest over the largest mean exposure (1 is demographic parity of exposure). */
  readonly exposureRatio: number
  /**
   * Singh and Joachims's disparate treatment ratio: the smallest over the largest exposure-to-relevance ratio of the
   * groups (1 when exposure is proportional to relevance).
   */
  readonly treatmentRatio?: number
}

/**
 * The exposure of each group in a ranking (Singh and Joachims, 2018): the attention its members receive from their
 * positions, averaged per member, the ratio between the least and most exposed groups (demographic parity of
 * exposure) and, given relevances, the ratio between the groups' exposure per unit of relevance (disparate treatment).
 *
 * @param groups - The group of the item at each position, top first.
 * @param options - Each position's item relevance, the position bias and how many positions are seen (default all).
 * @returns The groups' exposure, relevance and ratios.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { rankingExposure } from 'mdml-engine/fairness'
 * const r = rankingExposure(['a', 'a', 'b', 'b'])
 * r.exposureRatio.toFixed(3) // => '0.571'
 * ```
 */
export function rankingExposure(
  groups: readonly (string | number)[],
  options: { readonly relevance?: readonly number[]; readonly bias?: PositionBias; readonly k?: number } = {},
): Exposure {
  const names = [...new Set(groups.map(String))].sort()
  const k = options.k ?? groups.length
  const exp = names.map(() => 0)
  const rel = names.map(() => 0)
  const count = names.map(() => 0)
  for (const [i, g] of groups.entries()) {
    const j = names.indexOf(String(g))
    count[j] = (count[j] ?? 0) + 1
    if (i < k) exp[j] = (exp[j] ?? 0) + positionWeight(i + 1, options.bias)
    rel[j] = (rel[j] ?? 0) + (options.relevance?.[i] ?? 0)
  }
  const exposure = exp.map((e, j) => e / Math.max(1, count[j] ?? 0))
  const ratio = (v: readonly number[]): number => (Math.max(...v) === 0 ? 1 : Math.min(...v) / Math.max(...v))
  if (options.relevance === undefined) return { groups: names, exposure, exposureRatio: ratio(exposure) }
  if (options.relevance.length !== groups.length)
    throw new MdmlError('shape', 'rankingExposure: one relevance per item')
  const relevance = rel.map((r, j) => r / Math.max(1, count[j] ?? 0))
  return {
    groups: names,
    exposure,
    relevance,
    exposureRatio: ratio(exposure),
    treatmentRatio: ratio(exposure.map((e, j) => e / Math.max(relevance[j] ?? 0, 1e-300))),
  }
}

/**
 * FA*IR's table of minimum protected candidates (Zehlike et al., 2017): at each prefix length i, the smallest number of
 * protected candidates a ranking may have without failing a binomial test at level alpha against a protected share p,
 * `F^-1(alpha; i, p)` (unadjusted for the multiple tests, as the paper's basic test).
 *
 * @param k - The ranking's length.
 * @param p - The minimum protected share.
 * @param alpha - The significance (default 0.1).
 * @returns mtable[i - 1] for prefixes 1 to k.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { fairMinimumTable } from 'mdml-engine/fairness'
 * fairMinimumTable(10, 0.5) // => [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
 * ```
 */
export function fairMinimumTable(k: number, p: number, alpha = 0.1): number[] {
  if (!(p > 0 && p < 1)) throw new MdmlError('value', 'fairMinimumTable: p is in (0, 1)')
  return Array.from({ length: k }, (_, i) => {
    const n = i + 1
    // The smallest m with P(X <= m) >= alpha (scipy's binom.ppf), the CDF summed by the pmf's recurrence.
    let pmf = (1 - p) ** n
    let cdf = pmf
    let m = 0
    while (m < n && cdf < alpha) {
      pmf *= ((n - m) / (m + 1)) * (p / (1 - p))
      cdf += pmf
      m++
    }
    return m
  })
}

/** The result of {@link fairTopK}. */
export interface FairRanking {
  /** The selected candidates' indices, best first. */
  readonly ranking: readonly number[]
  /** The minimum protected count at each prefix, and the count reached. */
  readonly minimums: readonly number[]
  readonly protectedCounts: readonly number[]
  /** Whether the ranking by score alone would already pass. */
  readonly scoreRankingPasses: boolean
}

/**
 * FA*IR top-k (Zehlike et al., 2017): the best k candidates by score, subject to ranked group fairness, that every
 * prefix holds at least FA*IR's minimum number of protected candidates (taking the best protected candidate whenever the
 * minimum demands, the best overall otherwise).
 *
 * @param scores - Each candidate's score.
 * @param isProtected - Whether each candidate is in the protected group.
 * @param k - How many to rank.
 * @param options - The minimum protected share p (default 0.5) and alpha (default 0.1).
 * @returns The ranking, the minimums and counts, and whether the score ranking passed.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { fairTopK } from 'mdml-engine/fairness'
 * const scores = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
 * const prot = [false, false, false, false, false, true, true, true, true, true]
 * fairTopK(scores, prot, 6).ranking // => [0, 1, 2, 5, 3, 4]
 * ```
 */
export function fairTopK(
  scores: readonly number[],
  isProtected: readonly boolean[],
  k: number,
  options: { readonly p?: number; readonly alpha?: number } = {},
): FairRanking {
  if (scores.length !== isProtected.length) throw new MdmlError('shape', 'fairTopK: one protected flag per score')
  const minimums = fairMinimumTable(k, options.p ?? 0.5, options.alpha ?? 0.1)
  const order = scores.map((_, i) => i).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0) || a - b)
  const prot = order.filter((i) => isProtected[i] === true)
  const rest = order.filter((i) => isProtected[i] !== true)
  const ranking: number[] = []
  const counts: number[] = []
  let have = 0
  let pi = 0
  let ri = 0
  for (let i = 0; i < Math.min(k, scores.length); i++) {
    const need = (minimums[i] ?? 0) > have
    const takeProtected =
      pi < prot.length && (need || ri >= rest.length || (scores[prot[pi] ?? 0] ?? 0) >= (scores[rest[ri] ?? 0] ?? 0))
    if (takeProtected) {
      ranking.push(prot[pi] ?? 0)
      pi++
      have++
    } else {
      ranking.push(rest[ri] ?? 0)
      ri++
    }
    counts.push(have)
  }
  let plain = 0
  const passes = order.slice(0, k).every((i, j) => {
    if (isProtected[i] === true) plain++
    return plain >= (minimums[j] ?? 0)
  })
  return { ranking, minimums, protectedCounts: counts, scoreRankingPasses: passes }
}

/**
 * Deterministic re-ranking to minimum shares (Geyik, Ambler and Kenthapadi, 2019's DetGreedy): at every prefix of
 * length i each group holds at least floor(share x i) places; at each position, a group below its floor takes it with
 * its best remaining candidate, otherwise the best remaining candidate of the groups still below their ceiling (ceil(share
 * x i)) does.
 *
 * @param scores - Each candidate's score.
 * @param groups - Each candidate's group.
 * @param shares - Each group's desired share (summing to at most 1).
 * @param k - How many to rank (default all).
 * @returns The re-ranked candidates' indices, best first.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { detGreedyRerank } from 'mdml-engine/fairness'
 * detGreedyRerank([9, 8, 7, 6, 5, 4], ['a', 'a', 'a', 'a', 'b', 'b'], { a: 0.5, b: 0.5 }) // => [0, 4, 1, 5, 2, 3]
 * ```
 */
export function detGreedyRerank(
  scores: readonly number[],
  groups: readonly (string | number)[],
  shares: Readonly<Record<string, number>>,
  k: number = scores.length,
): number[] {
  if (scores.length !== groups.length) throw new MdmlError('shape', 'detGreedyRerank: one group per score')
  const names = [...new Set(groups.map(String))]
  const queues = new Map(
    names.map((g) => [
      g,
      groups
        .flatMap((v, i) => (String(v) === g ? [i] : []))
        .sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0) || a - b),
    ]),
  )
  const counts = new Map(names.map((g) => [g, 0]))
  const out: number[] = []
  const best = (candidates: readonly string[]): string | undefined => {
    let pick: string | undefined
    for (const g of candidates) {
      const head = queues.get(g)?.[0]
      if (head === undefined) continue
      const cur = pick === undefined ? undefined : queues.get(pick)?.[0]
      if (cur === undefined || (scores[head] ?? 0) > (scores[cur] ?? 0)) pick = g
    }
    return pick
  }
  for (let i = 1; i <= Math.min(k, scores.length); i++) {
    const below = names.filter((g) => (counts.get(g) ?? 0) < Math.floor((shares[g] ?? 0) * i))
    const underCeil = names.filter((g) => (counts.get(g) ?? 0) < Math.ceil((shares[g] ?? 0) * i))
    const g = best(below) ?? best(underCeil) ?? best(names)
    if (g === undefined) break
    out.push(queues.get(g)?.shift() ?? 0)
    counts.set(g, (counts.get(g) ?? 0) + 1)
  }
  return out
}
