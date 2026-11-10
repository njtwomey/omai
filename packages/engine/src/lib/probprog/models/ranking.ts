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
import { item, toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
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
  const flip = toFlat(drawUniform(foldIn(key, 1), [size]))
  const a = Array.from({ length: size }, (_, g) => pairs[g % pairs.length]?.[(flip[g] ?? 0) < 0.5 ? 0 : 1] ?? 0)
  const b = Array.from({ length: size }, (_, g) => pairs[g % pairs.length]?.[(flip[g] ?? 0) < 0.5 ? 1 : 0] ?? 1)
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
