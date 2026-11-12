/**
 * The computation of 'Who is better?': a tournament of won, drawn and lost games is turned into a rating for each player by
 * expectation propagation. Each player's skill is a bell curve (a mean and a spread); each game is a factor that joins the
 * skills of its two players. EP sends each skill's bell curve to the games it plays in, each game sends back what its result
 * says about each player, and this repeats until the messages stop changing. The model is run on the whole tournament and
 * again on the first game, the first two games and so on, so the page can show how each game moved the ratings. The result is
 * plain data and a pure function of the seed (which decides the order and the results of the games), so it runs in the story's
 * worker. The regions are the code shown.
 */

// region imports
import { epRoute } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { session } from 'omai-engine/trace'
// endregion

import type { ModelRun } from 'omai-engine/exposition'
import { arange, sliceAxis, toFlat } from 'omai-engine/tensor'
import type { Say } from '../../kit/protocol.ts'
import { replayOfModel } from '../../kit/replay.ts'
import { DRAW_MARGIN, EARLY, type Games, PLAYERS, playGames, ROUNDS, shuffled, tournament } from './data.ts'

/** A rating: the mean and the standard deviation of a player's skill. */
export interface Rating {
  readonly mean: number
  readonly sd: number
}

/** What the figures and the prose read. All of it is plain numbers. */
export interface Result {
  /** The games, in the order they were played. */
  readonly games: Games
  /** The ratings after all the games, one per player. */
  readonly ratings: readonly Rating[]
  /** The ratings after the first `after[i]` games, one list of ratings per entry; the first entry is 0 games (the prior). */
  readonly history: readonly { readonly after: number; readonly ratings: readonly Rating[] }[]
  /** How many sweeps of messages the run on all the games took, and whether the messages settled. */
  readonly sweeps: number
  readonly settled: boolean
  /** The messages of the run on all the games, as the library records them, for a replay (undefined when it was not kept). */
  readonly record: ModelRun | undefined
}

/** How much to do: a test uses fewer rounds and keeps no record. */
export interface Size {
  readonly rounds: number
  readonly record: boolean
}

/** The size of the real run: every pair meets four times, forty games. */
export const DEFAULT_SIZE: Size = { rounds: ROUNDS, record: true }

/**
 * The ratings after the first `count` games by expectation propagation: write the model, pass messages until they settle and
 * read each skill's mean and standard deviation.
 *
 * @param games - The games played so far.
 * @param keepRecord - Whether to log every message, for the replay.
 * @returns The ratings, the sweeps taken, whether the messages settled and the record of the run.
 */
export function rate(
  games: Games,
  keepRecord = false,
): { ratings: Rating[]; sweeps: number; settled: boolean; record: ModelRun | undefined } {
  // region model
  // a game is lost, drawn or won as the gap in performance falls below, inside or above the draw margin
  const cuts = [-DRAW_MARGIN / Math.SQRT2, DRAW_MARGIN / Math.SQRT2]
  const tournament = defineModel('tournament', (m) => {
    const player = m.range('player', { size: PLAYERS.length })
    const game = m.range('game')
    // who played whom, and the result for the first player: 0 lost, 1 drawn, 2 won
    const a = m.data('a', { dims: [game], support: 'index', indexes: player })
    const b = m.data('b', { dims: [game], support: 'index', indexes: player })
    const result = m.data('result', { dims: [game], support: 'count' })
    // every player's skill starts as a bell curve centred on zero
    const skill = m.latent('skill', d.normal(0, 1), { dims: [player] })
    // each player's performance is the skill plus noise, so the gap has a spread of the square root of two
    const gap = ops.div(skill.at(a).sub(skill.at(b)), Math.SQRT2)
    m.observe('result', d.orderedProbit(gap, cuts), result)
  })
  // endregion
  // region infer
  const bound = tournament.bind({ a: games.a, b: games.b, result: games.result })
  // ask the library how expectation propagation runs this model
  const route = epRoute(bound)
  if (route.via !== 'ep') throw new Error(`no plain message-passing route: ${route.verdict.reason}`)
  // one step is one sweep: every game sends a message to its two players, who send theirs back
  const run = session(route.method, { steps: 300, mode: 'eager', logs: 'all', config: { tol: 1e-8 } })
  // the bell curve each skill settled on
  const beliefs = route.method.posterior?.(run.state, run.config)
  // endregion
  if (beliefs === undefined) throw new Error('the method returned no beliefs')
  const mean = toFlat(beliefs.mean('skill'))
  const sd = toFlat(beliefs.stddev('skill'))
  return {
    ratings: PLAYERS.map((_, i) => ({ mean: mean[i] as number, sd: sd[i] as number })),
    sweeps: run.t,
    settled: run.status === 'converged',
    record: keepRecord ? replayOfModel(bound, run.run()) : undefined,
  }
}

/** The first `count` games. */
const firstGames = (games: Games, count: number): Games => ({
  a: toFlat(sliceAxis(games.a, 0, 0, count)),
  b: toFlat(sliceAxis(games.b, 0, 0, count)),
  result: toFlat(sliceAxis(games.result, 0, 0, count)),
})

/** The numbers of games after which the players are rated: each of the first `EARLY`, then every few. */
export function checkpoints(total: number): number[] {
  const early = toFlat(arange(1, Math.min(EARLY, total) + 1))
  const later = toFlat(arange(EARLY, total + 1)).filter((k) => k > EARLY && (k % 5 === 0 || k === total))
  return [...early, ...later]
}

/**
 * The inference of the story, from a seed: take the tournament's games in an order drawn from the seed, rate the players
 * after the first games and after the last.
 *
 * @param seed - Decides the order in which the games arrive.
 * @param say - Says what is happening, for the play button.
 * @param size - How much to do.
 * @returns The result.
 */
export function fit(seed: number, say: Say = () => {}, size: Size = DEFAULT_SIZE): Result {
  say('Setting out the games')
  const games = shuffled(size.rounds === ROUNDS ? tournament() : playGames(1, size.rounds), seed)
  say('Passing messages between the players and the games')
  const final = rate(games, size.record)
  const total = games.a.length
  // Rating after every game would take too long, so the first dozen games are rated one at a time and the later ones in steps.
  const stops = checkpoints(total)
  // before any game every skill is the prior bell curve, centred on zero with a standard deviation of one
  const prior = PLAYERS.map((): Rating => ({ mean: 0, sd: 1 }))
  const history = [
    { after: 0, ratings: prior },
    ...stops.map((after) => {
      say(`Rating the players after ${after} of ${total} games`)
      return { after, ratings: after === total ? final.ratings : rate(firstGames(games, after)).ratings }
    }),
  ]
  return {
    games,
    ratings: final.ratings,
    history,
    sweeps: final.sweeps,
    settled: final.settled,
    record: final.record,
  }
}

/**
 * The run of the story.
 *
 * @param seed - Decides the games.
 * @param say - Says what is happening.
 * @returns The result.
 */
export const run = (seed: number, say?: Say): Result => fit(seed, say, DEFAULT_SIZE)
