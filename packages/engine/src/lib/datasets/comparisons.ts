/**
 * Comparison data from players with latent skills, for rating and ranking models: Elo-scale pairwise games (optionally
 * with drifting skills), round-robin tournaments with Bradley-Terry-Davidson outcomes and ties, and multi-player races
 * under the TrueSkill (Thurstone) performance model, each with the true skills.
 */

import { type ComparisonData, comparisons, type DatasetGenerator, type Generated } from '../data/index.ts'
import { foldIn, type Key, normal, permutation, randint, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage } from '../tensor/index.ts'
import { stamp, vector } from './_util.ts'

const ELO = 400 / Math.LN10

/** Win, draw and loss probabilities under the Davidson extension of Bradley-Terry (Elo scale). */
function davidson(si: number, sj: number, nu: number): { win: number; draw: number; loss: number } {
  const a = Math.exp(si / ELO / 2)
  const b = Math.exp(sj / ELO / 2)
  const pi = a * a
  const pj = b * b
  const d = nu * a * b
  const z = pi + pj + d
  return { win: pi / z, draw: d / z, loss: pj / z }
}

/** Skills at every game: a Gaussian random walk from `N(0, spread²)` with `drift` per game for each player. */
function skillPaths(k: Key, players: number, games: number, spread: number, drift: number): Float64Array[] {
  const s0 = normal(foldIn(k, 0), [players]).data
  const steps = drift > 0 ? normal(foldIn(k, 1), [games, players]).data : undefined
  const paths: Float64Array[] = []
  let current = Float64Array.from(s0, (v) => spread * v)
  for (let g = 0; g < games; g++) {
    paths.push(current)
    if (steps !== undefined) current = Float64Array.from(current, (v, p) => v + drift * (steps[g * players + p] ?? 0))
  }
  return paths
}

function outcome(u: number, p: { win: number; draw: number }): [number, number] {
  if (u < p.win) return [0, 1]
  if (u < p.win + p.draw) return [0, 0]
  return [1, 0]
}

/**
 * Chess-like pairwise games between players with latent skills on the Elo scale: skills `N(0, spread²)` (optionally
 * drifting as random walks), random pairings, and outcomes by the Davidson-Bradley-Terry model, where `P(i beats j) ∝
 * 10^(s_i / 400)` and draws have weight `drawWeight √(p_i p_j)`. The truth has every player's skill (and its path),
 * the outcome probabilities and the log-likelihood.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { eloGames } from 'mdml-engine/datasets'
 * generate(eloGames, { knobs: { players: 8, games: 40 } }).data.participants.shape // => [40, 2]
 * ```
 */
export const eloGames: DatasetGenerator<ComparisonData> = {
  name: 'eloGames',
  kind: 'comparisons',
  description: 'Pairwise games between players with latent Elo skills, with draws and optional skill drift.',
  knobs: dict({
    players: int(2, 10_000, { label: 'players', description: 'The number of players.', default: 20 }),
    games: int(1, 1_000_000, { label: 'games', description: 'The number of games played.', default: 500 }),
    spread: real(0, 2000, {
      description: 'The standard deviation of the latent skills, in Elo points.',
      default: 200,
      label: 'skill standard deviation (Elo points)',
    }),
    drawWeight: real(0, 10, {
      description: 'The Davidson draw parameter: larger values make draws likelier.',
      default: 0.5,
      label: 'Davidson draw parameter',
    }),
    drift: real(0, 200, {
      description: 'The random-walk step of each skill per game.',
      default: 0,
      label: 'skill random-walk step per game',
    }),
  }),
  generate: (k, knobs) => {
    const players = knobs['players'] as number
    const games = knobs['games'] as number
    const nu = knobs['drawWeight'] as number
    const paths = skillPaths(foldIn(k, 0), players, games, knobs['spread'] as number, knobs['drift'] as number)
    const a = randint(foldIn(k, 1), [games], 0, players).data
    const b = randint(foldIn(k, 2), [games], 0, players - 1).data
    const u = uniform(foldIn(k, 3), [games]).data
    const part = new Int32Array(2 * games)
    const ranks = new Int32Array(2 * games)
    let logLik = 0
    for (let g = 0; g < games; g++) {
      const i = a[g] ?? 0
      const j = (b[g] ?? 0) >= i ? (b[g] ?? 0) + 1 : (b[g] ?? 0)
      const s = paths[g] as Float64Array
      const p = davidson(s[i] ?? 0, s[j] ?? 0, nu)
      const [ri, rj] = outcome(u[g] ?? 0, p)
      part[2 * g] = i
      part[2 * g + 1] = j
      ranks[2 * g] = ri
      ranks[2 * g + 1] = rj
      logLik += Math.log(ri === rj ? p.draw : ri === 0 ? p.win : p.loss)
    }
    const final = paths[games - 1] ?? new Float64Array(players)
    return {
      data: comparisons(fromStorage(part, [games, 2], 'int32'), fromStorage(ranks, [games, 2], 'int32'), {
        times: vector(Array.from({ length: games }, (_, g) => g)),
        meta: stamp('eloGames', eloGames.description, knobs),
      }),
      truth: {
        latents: { skills: vector(final), skillPaths: paths },
        extra: {
          probabilities: (si: number, sj: number) => davidson(si, sj, nu),
          logLikelihood: logLik,
          scale: 'Elo (400 / ln 10 per natural-log unit)',
        },
      },
    }
  },
}

/**
 * A round-robin tournament: every pair of players meets `rounds` times, with Bradley-Terry-Davidson outcomes from
 * fixed latent skills, so ranking methods can be scored against the true order (`truth.extra.order`).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { roundRobin } from 'mdml-engine/datasets'
 * generate(roundRobin, { knobs: { players: 4, rounds: 2 } }).data.participants.shape // => [12, 2]
 * ```
 */
export const roundRobin: DatasetGenerator<ComparisonData> = {
  name: 'roundRobin',
  kind: 'comparisons',
  description: 'A round-robin tournament between players with latent skills.',
  knobs: dict({
    players: int(2, 500, { label: 'players', description: 'The number of players.', default: 10 }),
    rounds: int(1, 100, { label: 'rounds', description: 'How many times every pair plays.', default: 2 }),
    spread: real(0, 2000, {
      label: 'skill spread',
      description: 'The standard deviation of the latent skills, in Elo points.',
      default: 200,
    }),
    drawWeight: real(0, 10, {
      label: 'draw weight',
      description: 'The Davidson draw parameter: larger values make draws likelier.',
      default: 0.5,
    }),
  }),
  generate: (k, knobs) => {
    const players = knobs['players'] as number
    const rounds = knobs['rounds'] as number
    const nu = knobs['drawWeight'] as number
    const skills = Float64Array.from(normal(foldIn(k, 0), [players]).data, (v) => (knobs['spread'] as number) * v)
    const pairs: [number, number][] = []
    for (let r = 0; r < rounds; r++)
      for (let i = 0; i < players; i++) for (let j = i + 1; j < players; j++) pairs.push(r % 2 === 0 ? [i, j] : [j, i])
    const u = uniform(foldIn(k, 1), [pairs.length]).data
    const part = new Int32Array(2 * pairs.length)
    const ranks = new Int32Array(2 * pairs.length)
    for (const [g, [i, j]] of pairs.entries()) {
      const [ri, rj] = outcome(u[g] ?? 0, davidson(skills[i] ?? 0, skills[j] ?? 0, nu))
      part[2 * g] = i
      part[2 * g + 1] = j
      ranks[2 * g] = ri
      ranks[2 * g + 1] = rj
    }
    const order = Array.from({ length: players }, (_, i) => i).sort((p, q) => (skills[q] ?? 0) - (skills[p] ?? 0))
    return {
      data: comparisons(fromStorage(part, [pairs.length, 2], 'int32'), fromStorage(ranks, [pairs.length, 2], 'int32'), {
        meta: stamp('roundRobin', roundRobin.description, knobs),
      }),
      truth: {
        latents: { skills: vector(skills) },
        extra: { order, probabilities: (a: number, b: number) => davidson(a, b, nu) },
      },
    }
  },
}

/** Options of {@link multiplayerRaces}. */
export interface RaceOptions {
  readonly players?: number
  readonly races?: number
  /** Participants per race. */
  readonly size?: number
  /** Skill standard deviation (default 25 / 3, TrueSkill's default sigma). */
  readonly spread?: number
  /** Performance noise standard deviation (default 25 / 6, TrueSkill's beta). */
  readonly beta?: number
  /** Performances closer than this tie (default 0). */
  readonly drawMargin?: number
}

/**
 * Multi-player contests in the TrueSkill (Thurstone) model: each participant's performance is their skill plus
 * `N(0, beta²)` noise and places follow performances (ties within the draw margin). The truth has skills and the
 * performances behind each result.
 *
 * @param key - The random key.
 * @param options - Players, races, participants per race, spreads and draw margin.
 * @returns The contests and their truth.
 * @example
 * ```ts
 * import { multiplayerRaces } from 'mdml-engine/datasets'
 * import { key } from 'mdml-engine/random'
 * multiplayerRaces(key(0), { players: 10, races: 5, size: 4 }).data.ranks.shape // => [5, 4]
 * ```
 */
export function multiplayerRaces(key: Key, options: RaceOptions = {}): Generated<ComparisonData> {
  const players = options.players ?? 20
  const races = options.races ?? 100
  const size = Math.min(options.size ?? 4, players)
  const spread = options.spread ?? 25 / 3
  const beta = options.beta ?? 25 / 6
  const margin = options.drawMargin ?? 0
  const skills = Float64Array.from(normal(foldIn(key, 0), [players]).data, (v) => 25 + spread * v)
  const part = new Int32Array(races * size)
  const ranks = new Int32Array(races * size)
  const perf = new Float64Array(races * size)
  for (let r = 0; r < races; r++) {
    const who = permutation(foldIn(foldIn(key, 1), r), players).data.slice(0, size)
    const noise = normal(foldIn(foldIn(key, 2), r), [size]).data
    const p = Array.from(who, (i, j) => (skills[i] ?? 0) + beta * (noise[j] ?? 0))
    const order = p.map((_, j) => j).sort((a, b) => (p[b] ?? 0) - (p[a] ?? 0))
    let place = 0
    for (const [pos, j] of order.entries()) {
      if (pos > 0 && (p[order[pos - 1] ?? 0] ?? 0) - (p[j] ?? 0) > margin) place = pos
      ranks[r * size + j] = place
    }
    for (let j = 0; j < size; j++) {
      part[r * size + j] = who[j] ?? 0
      perf[r * size + j] = p[j] ?? 0
    }
  }
  return {
    data: comparisons(fromStorage(part, [races, size], 'int32'), fromStorage(ranks, [races, size], 'int32'), {
      meta: {
        name: 'multiplayerRaces',
        description: 'Multi-player contests in the TrueSkill model.',
        source: 'mdml-js generator',
      },
    }),
    truth: {
      latents: { skills: vector(skills), performances: fromStorage(perf, [races, size], 'float64') },
      extra: { beta, spread, drawMargin: margin },
    },
  }
}
