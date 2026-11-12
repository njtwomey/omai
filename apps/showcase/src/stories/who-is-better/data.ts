/**
 * The data of 'Who is better?': a small tournament of five players that is made up here, so that the true skills are known
 * and the page can say how close the ratings come to them. Each pair meets four times; the order of the games is shuffled
 * within each round; each game is won, drawn or lost by the first player as the difference of the two skills, plus noise,
 * falls below, inside or above a draw margin. This is the same performance model the rating model assumes, so the model
 * is right about how the games were made and only the skills are unknown. The simulation is a pure function of the seed.
 */

import { foldIn, key, normal, uniform } from 'omai-engine/random'
import {
  add,
  arange,
  argsort,
  div,
  equal,
  flatten,
  greater,
  less,
  mod,
  mul,
  nonzero,
  ones,
  reshape,
  sub,
  type Tensor,
  take,
  tensor,
  toFlat,
  triu,
  where,
} from 'omai-engine/tensor'

/** The players, from the best to the worst by their true skill. */
export const PLAYERS = ['Ana', 'Ben', 'Cleo', 'Dev', 'Eli'] as const

/** The skills the games are made from (the page shows them after play). */
export const TRUE_SKILLS = [1.5, 0.7, 0.1, -0.6, -1.4] as const

/** A game is drawn when the performance gap is inside this margin. */
export const DRAW_MARGIN = 0.7

/** How many times each pair meets. */
export const ROUNDS = 4

/** The games of the tournament, in the order they were played. */
export interface Games {
  /** The first player of each game, as an index into {@link PLAYERS}. */
  readonly a: readonly number[]
  /** The second player of each game. */
  readonly b: readonly number[]
  /** 0 when the first player lost, 1 when the game was drawn and 2 when the first player won. */
  readonly result: readonly number[]
}

/** The seed the tournament is made from: the same games every time, so the page can show them before anything runs. */
export const TOURNAMENT_SEED = 1

/**
 * The games of the tournament for a seed, in order: round after round, every pair meeting once in a round, in a shuffled
 * order, with the two players taking turns to be listed first.
 *
 * @param seed - Decides the order of the games within a round and the results.
 * @param rounds - How many times each pair meets.
 * @returns The games.
 */
export function playGames(seed: number, rounds: number = ROUNDS): Games {
  const k = key(seed)
  const n = PLAYERS.length
  const skills = tensor([...TRUE_SKILLS])
  // every pair once, as the two columns of the upper triangle
  const [first, second] = nonzero(triu(ones([n, n]), 1)) as [Tensor, Tensor]
  const perRound = toFlat(first).length
  const games = rounds * perRound
  // in odd rounds the two players of a pair swap places
  const swap = reshape(equal(mod(arange(rounds), 2), 1), [rounds, 1])
  const a = flatten(where(swap, second, first))
  const b = flatten(where(swap, first, second))
  // shuffle within each round: sort by the round number plus a random number between 0 and 1
  const round = div(sub(arange(games), mod(arange(games), perRound)), perRound)
  const order = argsort(add(round, uniform(foldIn(k, 0), [games])), 0)
  const home = take(a, order)
  const away = take(b, order)
  // the performance gap: the difference of the skills plus noise with a standard deviation of the square root of two
  const gap = add(sub(take(skills, home), take(skills, away)), mul(normal(foldIn(k, 1), [games]), Math.SQRT2))
  // 0 below minus the margin, 1 inside it, 2 above it
  const result = where(less(gap, -DRAW_MARGIN), 0, where(greater(gap, DRAW_MARGIN), 2, 1))
  return { a: toFlat(home), b: toFlat(away), result: toFlat(result) }
}

/**
 * The same games in another order, drawn from a seed: the order in which the games reach the model.
 *
 * @param games - The games.
 * @param seed - Decides the order.
 * @returns The games, shuffled.
 */
export function shuffled(games: Games, seed: number): Games {
  const order = argsort(uniform(foldIn(key(seed), 7), [games.a.length]), 0)
  const pick = (v: readonly number[]): number[] => toFlat(take(tensor([...v]), order))
  return { a: pick(games.a), b: pick(games.b), result: pick(games.result) }
}

/** The tournament everyone sees: every pair meets four times. */
export const tournament = (): Games => playGames(TOURNAMENT_SEED)
