/**
 * Social choice: aggregating ranked ballots into a collective choice. Preference profiles, scoring rules (plurality,
 * Borda), pairwise majorities with Condorcet winners and losers, Copeland and the Smith set, instant-runoff voting, and
 * a search for the classic paradoxes (majority cycles, Condorcet winners losing), with the Tennessee capital and
 * Condorcet paradox profiles as examples.
 */

import { MdmlError, maxOf, minOf } from '../base/index.ts'

/**
 * A preference profile: the candidates and a list of ballots, each a strict ranking of candidate indices (best
 * first, possibly truncated), cast by `counts[k]` voters.
 *
 * @concept social-choice
 */
export interface Profile {
  readonly candidates: readonly string[]
  readonly ballots: readonly (readonly number[])[]
  readonly counts: readonly number[]
}

/**
 * A preference profile from ballots of candidate names or indices.
 *
 * @param candidates - The candidates' names.
 * @param ballots - Rankings, best first (names or indices; truncated rankings allowed).
 * @param counts - How many voters cast each ballot (default 1 each).
 * @returns The profile.
 * @throws MdmlError with code `value` for unknown or repeated candidates, or `shape` for mismatched counts.
 * @concept social-choice
 * @example
 * ```ts
 * import { profile } from 'mdml-engine/games'
 * profile(['a', 'b'], [['b', 'a'], [0, 1]]).ballots // => [[1, 0], [0, 1]]
 * ```
 */
export function profile(
  candidates: readonly string[],
  ballots: readonly (readonly (number | string)[])[],
  counts?: readonly number[],
): Profile {
  const index = (c: number | string): number => {
    const i = typeof c === 'number' ? c : candidates.indexOf(c)
    if (!Number.isInteger(i) || i < 0 || i >= candidates.length)
      throw new MdmlError('value', `profile: unknown candidate '${c}'`)
    return i
  }
  const parsed = ballots.map((b) => {
    const r = b.map(index)
    if (new Set(r).size !== r.length) throw new MdmlError('value', 'profile: a ballot ranks a candidate twice')
    return r
  })
  const n = counts ?? parsed.map(() => 1)
  if (n.length !== parsed.length) throw new MdmlError('shape', 'profile: one count per ballot')
  return { candidates: [...candidates], ballots: parsed, counts: [...n] }
}

/** The result of a scoring rule: every candidate's score and the winners (all with the top score). */
export interface ScoreResult {
  readonly scores: readonly number[]
  readonly winners: readonly number[]
}

function winnersOf(scores: readonly number[]): number[] {
  const top = maxOf(scores)
  return scores.flatMap((s, i) => (s === top ? [i] : []))
}

/**
 * Plurality: every ballot gives one point to its first choice.
 *
 * @param p - The profile.
 * @returns First-choice tallies and the winners.
 * @concept social-choice
 * @example
 * ```ts
 * import { plurality, tennesseeProfile } from 'mdml-engine/games'
 * plurality(tennesseeProfile()).winners // => [0]
 * ```
 */
export function plurality(p: Profile): ScoreResult {
  const scores = new Array<number>(p.candidates.length).fill(0)
  for (const [k, b] of p.ballots.entries()) {
    const first = b[0]
    if (first !== undefined) scores[first] = (scores[first] ?? 0) + (p.counts[k] ?? 0)
  }
  return { scores, winners: winnersOf(scores) }
}

/**
 * The Borda count: with m candidates, a ballot gives m - 1 points to its first choice, m - 2 to the second, and so on
 * (unranked candidates get 0).
 *
 * @param p - The profile.
 * @returns Borda scores and the winners.
 * @concept social-choice
 * @example
 * ```ts
 * import { borda, tennesseeProfile } from 'mdml-engine/games'
 * borda(tennesseeProfile()).winners // => [1]
 * ```
 */
export function borda(p: Profile): ScoreResult {
  const m = p.candidates.length
  const scores = new Array<number>(m).fill(0)
  for (const [k, b] of p.ballots.entries())
    for (const [r, c] of b.entries()) scores[c] = (scores[c] ?? 0) + (p.counts[k] ?? 0) * (m - 1 - r)
  return { scores, winners: winnersOf(scores) }
}

/**
 * The pairwise majority matrix: `M[a][b]` voters rank a above b (a ranked candidate is above an unranked one).
 *
 * @param p - The profile.
 * @returns The m x m matrix of pairwise support.
 * @concept condorcet
 * @example
 * ```ts
 * import { pairwiseMatrix, profile } from 'mdml-engine/games'
 * pairwiseMatrix(profile(['a', 'b'], [['a', 'b'], ['b', 'a'], ['a']])) // => [[0, 2], [1, 0]]
 * ```
 */
export function pairwiseMatrix(p: Profile): number[][] {
  const m = p.candidates.length
  const mat = Array.from({ length: m }, () => new Array<number>(m).fill(0))
  for (const [k, b] of p.ballots.entries()) {
    const w = p.counts[k] ?? 0
    const rank = new Array<number>(m).fill(Infinity)
    for (const [r, c] of b.entries()) rank[c] = r
    for (let a = 0; a < m; a++)
      for (let c = 0; c < m; c++)
        if (a !== c && (rank[a] ?? Infinity) < (rank[c] ?? Infinity)) (mat[a] ?? [])[c] = (mat[a]?.[c] ?? 0) + w
  }
  return mat
}

/**
 * The Condorcet winner: the candidate who beats every other in a pairwise majority, if any.
 *
 * @param p - The profile.
 * @returns The candidate, or `null` (for example under a majority cycle).
 * @concept condorcet
 * @example
 * ```ts
 * import { condorcetParadoxProfile, condorcetWinner, tennesseeProfile } from 'mdml-engine/games'
 * [condorcetWinner(tennesseeProfile()), condorcetWinner(condorcetParadoxProfile())] // => [1, null]
 * ```
 */
export function condorcetWinner(p: Profile): number | null {
  const mat = pairwiseMatrix(p)
  for (let a = 0; a < mat.length; a++)
    if (mat.every((_, b) => b === a || (mat[a]?.[b] ?? 0) > (mat[b]?.[a] ?? 0))) return a
  return null
}

/**
 * The Condorcet loser: the candidate who loses to every other in a pairwise majority, if any.
 *
 * @param p - The profile.
 * @returns The candidate, or `null`.
 * @concept condorcet
 * @example
 * ```ts
 * import { condorcetLoser, tennesseeProfile } from 'mdml-engine/games'
 * condorcetLoser(tennesseeProfile()) // => 0
 * ```
 */
export function condorcetLoser(p: Profile): number | null {
  const mat = pairwiseMatrix(p)
  for (let a = 0; a < mat.length; a++)
    if (mat.every((_, b) => b === a || (mat[a]?.[b] ?? 0) < (mat[b]?.[a] ?? 0))) return a
  return null
}

/**
 * Copeland's rule: a candidate scores 1 for every pairwise majority win and 1/2 for every tie.
 *
 * @param p - The profile.
 * @returns Copeland scores and the winners.
 * @concept condorcet
 * @example
 * ```ts
 * import { copeland, tennesseeProfile } from 'mdml-engine/games'
 * copeland(tennesseeProfile()).scores // => [0, 3, 2, 1]
 * ```
 */
export function copeland(p: Profile): ScoreResult {
  const mat = pairwiseMatrix(p)
  const scores = mat.map((row, a) => {
    let s = 0
    for (let b = 0; b < row.length; b++) {
      if (a === b) continue
      const d = (row[b] ?? 0) - (mat[b]?.[a] ?? 0)
      s += d > 0 ? 1 : d === 0 ? 0.5 : 0
    }
    return s
  })
  return { scores, winners: winnersOf(scores) }
}

/**
 * The Smith set: the smallest non-empty set of candidates who each beat every candidate outside it in a pairwise
 * majority. It is the Condorcet winner alone when there is one; a larger Smith set is a majority cycle.
 *
 * @param p - The profile.
 * @returns The Smith set, ascending.
 * @concept condorcet
 * @example
 * ```ts
 * import { condorcetParadoxProfile, smithSet } from 'mdml-engine/games'
 * smithSet(condorcetParadoxProfile()) // => [0, 1, 2]
 * ```
 */
export function smithSet(p: Profile): number[] {
  const mat = pairwiseMatrix(p)
  const m = mat.length
  const order = copeland(p)
    .scores.map((s, i) => ({ s, i }))
    .sort((x, y) => y.s - x.s || x.i - y.i)
    .map((x) => x.i)
  for (let k = 1; k <= m; k++) {
    const inside = order.slice(0, k)
    const outside = order.slice(k)
    if (inside.every((a) => outside.every((b) => (mat[a]?.[b] ?? 0) > (mat[b]?.[a] ?? 0))))
      return [...inside].sort((x, y) => x - y)
  }
  return order.sort((x, y) => x - y)
}

/** One round of instant runoff: the tallies of the remaining candidates and who was eliminated. */
export interface RunoffRound {
  readonly tallies: readonly number[]
  readonly eliminated: number | null
}

/**
 * Instant-runoff voting (the alternative vote): repeatedly count every ballot for its highest remaining choice;
 * a candidate with a majority of the ballots still counting wins, otherwise the one with the fewest votes is
 * eliminated (ties eliminate the later candidate).
 *
 * @param p - The profile.
 * @returns The winner and every round's tallies (eliminated candidates tally NaN).
 * @concept social-choice
 * @example
 * ```ts
 * import { instantRunoff, tennesseeProfile } from 'mdml-engine/games'
 * instantRunoff(tennesseeProfile()).winner // => 3
 * ```
 */
export function instantRunoff(p: Profile): { winner: number; rounds: RunoffRound[] } {
  const m = p.candidates.length
  if (m === 0) throw new MdmlError('value', 'instantRunoff: no candidates')
  const alive = new Array<boolean>(m).fill(true)
  const rounds: RunoffRound[] = []
  for (let r = 0; r < m; r++) {
    const tallies = alive.map((a) => (a ? 0 : Number.NaN))
    let counting = 0
    for (const [k, b] of p.ballots.entries()) {
      const c = b.find((x) => alive[x] === true)
      if (c === undefined) continue
      tallies[c] = (tallies[c] ?? 0) + (p.counts[k] ?? 0)
      counting += p.counts[k] ?? 0
    }
    const live = alive.flatMap((a, i) => (a ? [i] : []))
    const best = live.reduce((a, i) => ((tallies[i] ?? 0) > (tallies[a] ?? 0) ? i : a), live[0] ?? 0)
    if (live.length === 1 || (tallies[best] ?? 0) * 2 > counting) {
      rounds.push({ tallies, eliminated: null })
      return { winner: best, rounds }
    }
    const low = minOf(live.map((i) => tallies[i] ?? 0))
    const out = live.filter((i) => tallies[i] === low).reduce((a, i) => Math.max(a, i), -1)
    alive[out] = false
    rounds.push({ tallies, eliminated: out })
  }
  return { winner: alive.indexOf(true), rounds }
}

/**
 * The profile with one candidate removed (from the list and every ballot; later indices shift down).
 *
 * @param p - The profile.
 * @param candidate - The candidate to drop.
 * @returns The smaller profile.
 * @example
 * ```ts
 * import { tennesseeProfile, withoutCandidate } from 'mdml-engine/games'
 * withoutCandidate(tennesseeProfile(), 0).candidates // => ['Nashville', 'Chattanooga', 'Knoxville']
 * ```
 */
export function withoutCandidate(p: Profile, candidate: number): Profile {
  return {
    candidates: p.candidates.filter((_, i) => i !== candidate),
    ballots: p.ballots.map((b) => b.filter((c) => c !== candidate).map((c) => (c > candidate ? c - 1 : c))),
    counts: p.counts,
  }
}

/** A rule's winner on a profile. */
type Rule = (p: Profile) => number

const RULES: Readonly<Record<string, Rule>> = {
  plurality: (p) => plurality(p).winners[0] ?? -1,
  borda: (p) => borda(p).winners[0] ?? -1,
  copeland: (p) => copeland(p).winners[0] ?? -1,
  instantRunoff: (p) => instantRunoff(p).winner,
}

/** The paradoxes {@link socialChoiceParadoxes} finds in a profile. */
export interface ParadoxReport {
  /** Each rule's winner (the first, under ties). */
  readonly winners: Readonly<Record<string, number>>
  readonly condorcetWinner: number | null
  readonly condorcetLoser: number | null
  /** Condorcet's paradox: the majority relation has a cycle (the Smith set has several candidates). */
  readonly majorityCycle: boolean
  /** Rules that do not elect the Condorcet winner. */
  readonly condorcetWinnerLoses: readonly string[]
  /** Rules that elect the Condorcet loser. */
  readonly condorcetLoserWins: readonly string[]
  /** Spoilers: removing a losing candidate changes a rule's winner (independence of irrelevant alternatives fails). */
  readonly spoilers: readonly {
    readonly rule: string
    readonly removed: number
    readonly before: number
    readonly after: number
  }[]
}

/**
 * Look for the classic social-choice paradoxes in a profile under plurality, Borda, Copeland and instant runoff:
 * Condorcet's paradox (a majority cycle), Condorcet winners who lose and Condorcet losers who win, and spoiler
 * candidates whose withdrawal changes the winner (Arrow's independence of irrelevant alternatives).
 *
 * @param p - The profile.
 * @returns What was found.
 * @concept social-choice
 * @example
 * ```ts
 * import { socialChoiceParadoxes, tennesseeProfile } from 'mdml-engine/games'
 * socialChoiceParadoxes(tennesseeProfile()).condorcetWinnerLoses // => ['plurality', 'instantRunoff']
 * ```
 */
export function socialChoiceParadoxes(p: Profile): ParadoxReport {
  const winners: Record<string, number> = {}
  for (const [name, rule] of Object.entries(RULES)) winners[name] = rule(p)
  const cw = condorcetWinner(p)
  const cl = condorcetLoser(p)
  const spoilers: { rule: string; removed: number; before: number; after: number }[] = []
  for (const [name, rule] of Object.entries(RULES)) {
    const before = winners[name] ?? -1
    for (let c = 0; c < p.candidates.length; c++) {
      if (c === before || p.candidates.length < 3) continue
      const after = rule(withoutCandidate(p, c))
      const mapped = after >= c ? after + 1 : after
      if (mapped !== before) spoilers.push({ rule: name, removed: c, before, after: mapped })
    }
  }
  return {
    winners,
    condorcetWinner: cw,
    condorcetLoser: cl,
    majorityCycle: smithSet(p).length > 1,
    condorcetWinnerLoses: cw === null ? [] : Object.keys(RULES).filter((r) => winners[r] !== cw),
    condorcetLoserWins: cl === null ? [] : Object.keys(RULES).filter((r) => winners[r] === cl),
    spoilers,
  }
}

/**
 * The textbook example of where Tennessee's capital would be if its voters each favoured the nearest city: Memphis
 * 42%, Nashville 26%, Chattanooga 15%, Knoxville 17%. Plurality picks Memphis (the Condorcet loser), instant runoff
 * Knoxville, and Borda, Copeland and the Condorcet criterion Nashville.
 *
 * @returns The profile (voters in percent).
 * @concept social-choice
 * @example
 * ```ts
 * import { tennesseeProfile } from 'mdml-engine/games'
 * tennesseeProfile().counts // => [42, 26, 15, 17]
 * ```
 */
export function tennesseeProfile(): Profile {
  return profile(
    ['Memphis', 'Nashville', 'Chattanooga', 'Knoxville'],
    [
      ['Memphis', 'Nashville', 'Chattanooga', 'Knoxville'],
      ['Nashville', 'Chattanooga', 'Knoxville', 'Memphis'],
      ['Chattanooga', 'Knoxville', 'Nashville', 'Memphis'],
      ['Knoxville', 'Chattanooga', 'Nashville', 'Memphis'],
    ],
    [42, 26, 15, 17],
  )
}

/**
 * Condorcet's paradox: three voters ranking a > b > c, b > c > a and c > a > b, so that every candidate loses a
 * pairwise majority to another and the majority preference is a cycle.
 *
 * @returns The profile.
 * @concept condorcet
 * @example
 * ```ts
 * import { condorcetParadoxProfile, socialChoiceParadoxes } from 'mdml-engine/games'
 * socialChoiceParadoxes(condorcetParadoxProfile()).majorityCycle // => true
 * ```
 */
export function condorcetParadoxProfile(): Profile {
  return profile(
    ['a', 'b', 'c'],
    [
      ['a', 'b', 'c'],
      ['b', 'c', 'a'],
      ['c', 'a', 'b'],
    ],
  )
}
