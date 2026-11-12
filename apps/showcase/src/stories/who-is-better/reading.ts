/**
 * What the prose of 'Who is better?' says: the win, draw and loss counts read off the games (before anything runs), and, from
 * the result, the ladder of ratings, which intervals overlap, how the spread of the ratings fell as games came in, and what
 * the first games did to the ratings of their players. Pure functions, light enough for the page's bundle.
 */

import type { Rating, Result } from './compute.ts'
import { type Games, PLAYERS, TRUE_SKILLS } from './data.ts'

/** The z value of a 90 percent interval of a bell curve. */
export const Z90 = 1.645

/** How many of the first games the page follows one at a time. */
export const EARLY = 12

/** One player's record. */
export interface Tally {
  readonly name: string
  readonly wins: number
  readonly draws: number
  readonly losses: number
}

/**
 * The win, draw and loss counts of every player.
 *
 * @param games - The games.
 * @returns One record per player, in the order of the players.
 */
export function tally(games: Games): Tally[] {
  return PLAYERS.map((name, p) => {
    const mine = games.a.flatMap((a, g) => {
      const b = games.b[g] as number
      const r = games.result[g] as number
      if (a === p) return [r === 2 ? 'w' : r === 1 ? 'd' : 'l']
      if (b === p) return [r === 0 ? 'w' : r === 1 ? 'd' : 'l']
      return []
    })
    return {
      name,
      wins: mine.filter((m) => m === 'w').length,
      draws: mine.filter((m) => m === 'd').length,
      losses: mine.filter((m) => m === 'l').length,
    }
  })
}

/** A player in the ladder: the rating, its 90 percent interval and the true skill the games were made from. */
export interface LadderRow {
  readonly index: number
  readonly name: string
  readonly mean: number
  readonly sd: number
  readonly lo: number
  readonly hi: number
  readonly truth: number
}

/** What a game did to the ratings of its two players. */
export interface Move {
  /** The game, counted from 1. */
  readonly game: number
  readonly winner: string
  readonly loser: string
  /** The loser's rating minus the winner's, before the game: positive for a win over a higher-rated player. */
  readonly gap: number
  /** How much the winner's rating rose, and how much the loser's fell (a positive number). */
  readonly rise: number
  readonly fall: number
}

/** The numbers the prose quotes after the run. */
export interface Reading {
  /** The players from the highest rating to the lowest. */
  readonly ladder: readonly LadderRow[]
  /** Whether the ladder is in the order of the true skills. */
  readonly inTrueOrder: boolean
  /** How many of the true skills lie inside their player's 90 percent interval. */
  readonly covered: number
  /** The pairs of neighbours on the ladder whose 90 percent intervals overlap. */
  readonly overlapping: readonly (readonly [string, string])[]
  /** The average spread of the skills after `k` games, for the numbers of games the run rated. */
  readonly spreadAfter: (k: number) => number | undefined
  /** How many games there were. */
  readonly games: number
  /** What each of the first games with a winner did to its two players. */
  readonly moves: readonly Move[]
  /** The win over the highest-rated opponent and the win over the lowest-rated opponent, among {@link Reading.moves}. */
  readonly upset: Move | undefined
  readonly expected: Move | undefined
  /** Whether the upset raised its winner's rating by more than the expected win raised its winner's. */
  readonly upsetMovedMore: boolean
}

const intervalOf = (r: Rating): readonly [number, number] => [r.mean - Z90 * r.sd, r.mean + Z90 * r.sd]

/**
 * The numbers of a result the prose quotes.
 *
 * @param result - The result.
 * @returns The reading.
 */
export function readResult(result: Result): Reading {
  const ladder = result.ratings
    .map((r, index): LadderRow => {
      const [lo, hi] = intervalOf(r)
      return {
        index,
        name: PLAYERS[index] as string,
        mean: r.mean,
        sd: r.sd,
        lo,
        hi,
        truth: TRUE_SKILLS[index] as number,
      }
    })
    .sort((a, b) => b.mean - a.mean)
  const overlapping = ladder
    .slice(1)
    .map((row, i): readonly [string, string] | undefined =>
      (ladder[i] as LadderRow).lo < row.hi ? [(ladder[i] as LadderRow).name, row.name] : undefined,
    )
    .filter((p): p is readonly [string, string] => p !== undefined)
  const spreadAfter = (k: number): number | undefined => {
    const at = result.history.find((h) => h.after === k)
    return at === undefined ? undefined : at.ratings.reduce((s, r) => s + r.sd, 0) / at.ratings.length
  }
  const moves: Move[] = []
  for (let g = 1; g <= EARLY; g++) {
    const before = result.history.find((h) => h.after === g - 1)
    const after = result.history.find((h) => h.after === g)
    const a = result.games.a[g - 1] as number
    const b = result.games.b[g - 1] as number
    const outcome = result.games.result[g - 1] as number
    if (before === undefined || after === undefined || outcome === 1) continue
    const [w, l] = outcome === 2 ? [a, b] : [b, a]
    moves.push({
      game: g,
      winner: PLAYERS[w] as string,
      loser: PLAYERS[l] as string,
      gap: (before.ratings[l] as Rating).mean - (before.ratings[w] as Rating).mean,
      rise: (after.ratings[w] as Rating).mean - (before.ratings[w] as Rating).mean,
      fall: (before.ratings[l] as Rating).mean - (after.ratings[l] as Rating).mean,
    })
  }
  const byGap = [...moves].sort((x, y) => y.gap - x.gap)
  const upset = byGap[0]
  const expected = byGap.at(-1)
  return {
    ladder,
    inTrueOrder: ladder.every((row, i) => row.truth === [...TRUE_SKILLS].sort((x, y) => y - x)[i]),
    covered: ladder.filter((row) => row.truth >= row.lo && row.truth <= row.hi).length,
    overlapping,
    spreadAfter,
    games: result.games.a.length,
    moves,
    upset,
    expected,
    upsetMovedMore: upset !== undefined && expected !== undefined && upset.rise > expected.rise,
  }
}
