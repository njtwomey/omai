/**
 * A TrueSkill-style rating model written with the operator set: each player has a skill `s_i ~ N(0, 1)`, a game between `a`
 * and `b` is won by `a` with probability `Phi((s_a - s_b) / (sqrt(2) beta))` (each player's performance is the skill plus
 * `N(0, beta²)` noise and the better performance wins, the Thurstone model of TrueSkill with the draws left out), and the
 * players of each game are index data gathered with `at`. Three players keep the posterior at three coordinates, so the truth is
 * a grid; the factor-graph version of this model is what expectation propagation was made for (the plan's TrueSkill row),
 * and the library's own `pgm/ratings` TrueSkill is the hand-written reference it is to reproduce.
 */

import { ndtr } from '../../numerics/special/index.ts'
import { uniform as drawUniform, foldIn, type Key } from '../../random/index.ts'
import { add, arange, item, less, mod, mul, reshape, take, tensor, toFlat, where } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, type ModelDefinition, ops } from '../model/index.ts'
import { type BenchmarkData, type BenchmarkModel, defineBenchmark } from './benchmark.ts'
import { importancePlan, laplacePlan, nutsPlan, sviPlan, temperedPlan } from './plans.ts'
import { gridTruth } from './truth.ts'

const BETA = 1
const SKILLS = [1, 0, -1]

function generateGames(key: Key, size = 24): BenchmarkData {
  const players = SKILLS.length
  // Every pair meets in turn: {0, 1}, {0, 2}, {1, 2}, {0, 1}, ...
  const pairs = [
    [0, 1],
    [0, 2],
    [1, 2],
  ] as const
  // Which of the two is listed first is a coin flip, so that the first player is not always the stronger one.
  const flip = drawUniform(foldIn(key, 1), [size])
  // the pair of game g is row g mod 3 of the table, its first player the column the coin picks
  const row = mul(mod(arange(size), pairs.length), 2)
  const heads = less(flip, 0.5)
  const table = reshape(tensor(pairs), [-1])
  const a = toFlat(take(table, add(row, where(heads, 0, 1))))
  const b = toFlat(take(table, add(row, where(heads, 1, 0))))
  const u = toFlat(drawUniform(foldIn(key, 0), [size]))
  const win = a.map((ai, g) => {
    const diff = ((SKILLS[ai] ?? 0) - (SKILLS[b[g] ?? 0] ?? 0)) / (Math.SQRT2 * BETA)
    return (u[g] ?? 1) < item(ndtr(diff)) ? 1 : 0
  })
  return { data: { a, b, win }, options: { sizes: { player: players } }, generating: { skills: SKILLS, beta: BETA } }
}

/**
 * Three players rated from the outcomes of pairwise games under the TrueSkill (Thurstone) performance model: `s_i ~ N(0,
 * 1)`, `P(a beats b) = Phi((s_a - s_b) / sqrt(2))`, 24 games between every pair in turn (skills `(1, 0, -1)`). The
 * players of a game are index data (`at` gathers their skills), the win probability is the `probit` operator, so the model is
 * the one a message-passing engine takes: a Gaussian prior on each skill and a probit factor on each difference. The truth
 * is a three-dimensional grid; EP (a later release) is expected to be the closest approximation, as in Minka's and
 * Herbrich's work, and the NUTS, tempered SMC and Laplace results are the references it will be compared with.
 *
 * @concept posterior-validation
 * @index dataset Three players rated from pairwise games, TrueSkill-style (probit): the EP target; truth is a 3-D grid.
 * @example
 * ```ts
 * import { trueskillThreePlayers } from 'omai-engine/probprog/models'
 * trueskillThreePlayers.bound().sizes // => { player: 3, game: 24 }
 * ```
 */
export const trueskillThreePlayers: BenchmarkModel = defineBenchmark({
  name: 'trueskill-three-players',
  title: 'TrueSkill, three players',
  description:
    'Three players with N(0, 1) skills and 24 pairwise games won with probability Phi((s_a - s_b) / (sqrt 2 beta)): the TrueSkill performance model with a grid truth, the target of expectation propagation.',
  tags: ['ranking', 'trueskill', 'probit', 'factor-graph', 'ep', 'quadrature', 'three-dimensional'],
  references: [
    'Herbrich, Minka and Graepel (2007), TrueSkill: a Bayesian skill rating system, NIPS 19',
    'Minka (2001), A family of algorithms for approximate Bayesian inference, chapter 3 (EP)',
    'the `pgm/ratings` TrueSkill of omai-engine (the hand-written reference)',
  ],
  definition: defineModel('trueskill-three-players', (m) => {
    const player = m.range('player')
    const game = m.range('game')
    const a = m.data('a', {
      dims: [game],
      support: 'index',
      role: 'index',
      indexes: player,
      doc: 'the first player of each game',
    })
    const b = m.data('b', {
      dims: [game],
      support: 'index',
      role: 'index',
      indexes: player,
      doc: 'the second player of each game',
    })
    const win = m.data('win', { dims: [game], support: 'binary', role: 'response', doc: '1 when the first player won' })
    const skill = m.latent('skill', d.normal(0, 1), { dims: [player], doc: 'the skill of each player' })
    const diff = ops.div(skill.at(a).sub(skill.at(b)), Math.SQRT2 * BETA)
    m.observe('win', d.bernoulli({ probs: ops.probit(diff) }), win)
  }),
  generate: generateGames,
  truth: (data, quality) =>
    gridTruth(
      trueskillThreePlayers.program(data),
      quality === 'fast' ? { points: 17, maxRefinements: 0 } : { points: 31, maxRefinements: 1 },
    ),
  plans: (scale) => [
    nutsPlan(scale),
    temperedPlan(scale),
    importancePlan(scale, {
      expect: 'report',
      note: 'weights from the prior: a concentrated posterior leaves few effective draws',
    }),
    laplacePlan(scale, { note: 'near Gaussian: close to the grid, a little narrow' }),
    sviPlan(scale, {
      note: 'a mean-field normal ignores the negative correlation between the skills of two players who met',
    }),
  ],
})

/** The games of a small tournament, in order, with the skills that produced them. */
export interface TrueskillTournament {
  /** The players' names (`A`, `B`, ...). */
  readonly players: readonly string[]
  /** The skills the games were drawn from, one per player. */
  readonly skills: readonly number[]
  /** Each game as the index of the first and of the second player. */
  readonly pairs: readonly (readonly [number, number])[]
  /** 1 where the first player of the game won. */
  readonly wins: readonly number[]
}

/** What {@link trueskillTournament} takes besides the key. */
export interface TrueskillOptions {
  /** The skills of the players (default `[1, 0, -1]`: three players). */
  readonly skills?: readonly number[]
  /** The number of games (default 12), played between the pairs in turn. */
  readonly games?: number
}

/**
 * The games of a small tournament drawn from known skills: pairs meet in turn, in a random order within each round, and
 * the first of a pair wins with probability `Phi((s_a - s_b) / sqrt 2)` (the TrueSkill performance model with noise 1
 * and no draws). The same key gives the same games, and the first `n` games of a longer tournament are those of the
 * shorter one, so a page can add games one at a time.
 *
 * @param key - The random key.
 * @param options - The skills and the number of games.
 * @returns The tournament.
 * @concept posterior-validation
 * @index dataset A small TrueSkill tournament drawn from known skills, games in a fixed order.
 * @example
 * ```ts
 * import { trueskillTournament } from 'omai-engine/probprog/models'
 * import { key } from 'omai-engine/random'
 * const t = trueskillTournament(key(1), { games: 6 })
 * [t.players, t.pairs.length, t.wins.length] // => [['A', 'B', 'C'], 6, 6]
 * ```
 */
export function trueskillTournament(key: Key, options: TrueskillOptions = {}): TrueskillTournament {
  const skills = options.skills ?? SKILLS
  const games = options.games ?? 12
  const n = skills.length
  const meetings: [number, number][] = []
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) meetings.push([a, b])
  const flip = toFlat(drawUniform(foldIn(key, 1), [games]))
  const u = toFlat(drawUniform(foldIn(key, 0), [games]))
  // banned-ok: non-numeric: one pair of players per game
  const pairs = Array.from({ length: games }, (_, g): [number, number] => {
    const [a, b] = meetings[g % meetings.length] ?? [0, 1]
    return (flip[g] ?? 0) < 0.5 ? [a, b] : [b, a]
  })
  const wins = pairs.map(([a, b], g) => {
    const diff = ((skills[a] ?? 0) - (skills[b] ?? 0)) / (Math.SQRT2 * BETA)
    return (u[g] ?? 1) < item(ndtr(diff)) ? 1 : 0
  })
  return { players: skills.map((_, i) => String.fromCharCode(65 + i)), skills, pairs, wins }
}

/**
 * The tournament as a model with one node per player and one observation per game (no plates), so that its factor graph
 * has a factor for each game and message passing can be watched game by game: `skill_A ~ N(0, 1)`, and game `g` between
 * `a` and `b` is `win_g ~ Bernoulli(Phi((skill_a - skill_b) / sqrt 2))` with the result as the data `won_g`.
 *
 * @param tournament - The players and the games to include (take the first `games` of a longer tournament to add games).
 * @param games - How many of the tournament's games the model has (default all).
 * @returns The model and the data to bind to it.
 * @concept model-spec
 * @example
 * ```ts
 * import { trueskillModel, trueskillTournament } from 'omai-engine/probprog/models'
 * import { key } from 'omai-engine/random'
 * const { definition, data } = trueskillModel(trueskillTournament(key(1), { games: 4 }))
 * Object.keys(data) // => ['won_1', 'won_2', 'won_3', 'won_4']
 * definition.bind(data).sizes // => {}
 * ```
 */
export function trueskillModel(
  tournament: TrueskillTournament,
  games: number = tournament.pairs.length,
): { readonly definition: ModelDefinition; readonly data: Readonly<Record<string, number>> } {
  const pairs = tournament.pairs.slice(0, games)
  const definition = defineModel(`trueskill-${tournament.players.length}-players-${pairs.length}-games`, (m) => {
    const skill = tournament.players.map((p) => m.latent(`skill_${p}`, d.normal(0, 1), { doc: `the skill of ${p}` }))
    for (const [g, [a, b]] of pairs.entries()) {
      const won = m.data(`won_${g + 1}`, {
        support: 'binary',
        role: 'response',
        doc: `1 when ${tournament.players[a]} beat ${tournament.players[b]}`,
      })
      const sa = skill[a]
      const sb = skill[b]
      if (sa === undefined || sb === undefined) continue
      m.observe(`win_${g + 1}`, d.bernoulli({ probs: ops.probit(ops.div(sa.sub(sb), Math.SQRT2 * BETA)) }), won)
    }
  })
  return { definition, data: Object.fromEntries(pairs.map((_, g) => [`won_${g + 1}`, tournament.wins[g] ?? 0])) }
}
