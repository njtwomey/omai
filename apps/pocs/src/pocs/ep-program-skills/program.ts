import { ndtr } from 'omai-engine/numerics/special'
import { epSpecMethod } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, normal, uniform } from 'omai-engine/random'
import { item, toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'

/** What the page changes about the data. */
export interface SkillsSetup {
  readonly seed: number
  readonly players: number
  /** How many times each pair plays. */
  readonly rounds: number
  /** The sd of the true skills. */
  readonly spread: number
}

export const defaultSetup: SkillsSetup = { seed: 3, players: 8, rounds: 3, spread: 1 }

// region simulate
export function simulate({ seed, players, rounds, spread }: SkillsSetup) {
  const k = key(seed)
  const skills = Array.from(toFlat(normal(foldIn(k, 0), [players], { scale: spread })))
  // A round robin: every pair meets `rounds` times, taking turns to be listed first.
  const games: [number, number][] = []
  for (let r = 0; r < rounds; r++)
    for (let i = 0; i < players; i++) for (let j = i + 1; j < players; j++) games.push(r % 2 ? [j, i] : [i, j])
  const u = toFlat(uniform(foldIn(k, 1), [games.length]))
  const win = games.map(([a, b], g) =>
    (u[g] ?? 1) < item(ndtr(((skills[a] ?? 0) - (skills[b] ?? 0)) / Math.SQRT2)) ? 1 : 0,
  )
  return { a: games.map((g) => g[0]), b: games.map((g) => g[1]), win, skills, players }
}
// endregion

// region model
export const tournamentModel = defineModel('tournament', (m) => {
  const player = m.range('player')
  const game = m.range('game')
  // The first and the second player of each game.
  const a = m.data('a', { dims: [game], support: 'index', role: 'index', indexes: player })
  const b = m.data('b', { dims: [game], support: 'index', role: 'index', indexes: player })
  const win = m.data('win', { dims: [game], support: 'binary' })
  // The precision of the skills: how spread out the players are.
  const tau = m.latent('tau', d.gamma(2, 2))
  const skill = m.latent('skill', d.normal(0, ops.div(1, ops.sqrt(tau))), { dims: [player] })
  // Performance noise of 1 on each side.
  const diff = ops.div(skill.at(a).sub(skill.at(b)), Math.SQRT2)
  m.observe('win', d.bernoulli({ probs: ops.probit(diff) }), win)
})
// endregion

// region infer
export function infer(data: ReturnType<typeof simulate>) {
  const games = { a: data.a, b: data.b, win: data.win }
  const bound = tournamentModel.bind(games, { sizes: { player: data.players } })
  const method = epSpecMethod(bound)
  const config = { tol: 1e-8 }
  const run = session(method, { config, seed: 1, steps: 300, mode: 'eager', logs: 'all' })
  // The marginals by site: skill, tau.
  const posterior = method.posterior?.(run.state, run.config)
  if (posterior === undefined) throw new Error('EP gave no posterior')
  return { bound, run, posterior }
}
// endregion

/** The skills as a posterior recovered them: mean, sd and a 90% interval per player. */
export function skillsOf(posterior: ReturnType<typeof infer>['posterior']) {
  const mean = Array.from(toFlat(posterior.mean('skill')))
  const sd = Array.from(toFlat(posterior.stddev('skill')))
  return {
    mean,
    sd,
    lower: mean.map((m, i) => m - 1.645 * (sd[i] ?? 0)),
    upper: mean.map((m, i) => m + 1.645 * (sd[i] ?? 0)),
    tau: { mean: Number(toFlat(posterior.mean('tau'))[0]), sd: Number(toFlat(posterior.stddev('tau'))[0]) },
  }
}
