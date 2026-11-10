import { type ModelRun, modelRunFromTape } from 'omai-engine/exposition'
import { epSpecMethod, nutsMethod } from 'omai-engine/probprog/inference'
import { type BoundModel, toStructure } from 'omai-engine/probprog/model'
import { trueskillModel, trueskillTournament } from 'omai-engine/probprog/models'
import { key } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { AnytimeRunSpec } from '../../shared/useAnytimeRuns.ts'

/** The games on offer: the first `n` of the same tournament, so adding a game keeps the ones already played. */
export const MAX_GAMES = 16

/** A tournament of three players of skill 1, 0 and -1, and what is made of its first `games` games. */
export function tournamentOf(seed: number): ReturnType<typeof trueskillTournament> {
  return trueskillTournament(key(seed), { games: MAX_GAMES })
}

/** A belief about a player: mean and standard deviation. */
export interface Belief {
  readonly mean: number
  readonly sd: number
}

const boundOf = (t: ReturnType<typeof tournamentOf>, games: number): BoundModel => {
  const { definition, data } = trueskillModel(t, games)
  return definition.bind(data)
}

/** EP on the games so far, as a record of the run (messages on the edges of the factor graph, beliefs at each step). */
export function messageRun(t: ReturnType<typeof tournamentOf>, games: number): ModelRun {
  const bound = boundOf(t, games)
  const s = session(epSpecMethod(bound), {
    seed: 1,
    steps: 100_000,
    mode: 'eager',
    // A step is one factor: the messages it sends to the players it joins.
    config: { tol: 1e-9, granularity: 'factor' },
    logs: 'all',
  })
  return modelRunFromTape(toStructure(bound, { view: 'factor' }), { run: s.run() }, { maxSteps: 400, maxMessages: 4 })
}

/** The players' beliefs after 0 to `MAX_GAMES` games: expectation propagation on the first j games. */
export function beliefsAlong(t: ReturnType<typeof tournamentOf>): Belief[][] {
  const prior = t.players.map((): Belief => ({ mean: 0, sd: 1 }))
  return [
    prior,
    ...Array.from({ length: MAX_GAMES }, (_, j) => {
      const bound = boundOf(t, j + 1)
      const ep = epSpecMethod(bound)
      const s = session(ep, { seed: 1, steps: 100_000, mode: 'eager', config: { tol: 1e-9 } })
      const p = ep.posterior?.(s.state, s.config)
      return t.players.map((name): Belief => {
        const site = `skill_${name}`
        return {
          mean: toFlat(p?.mean(site) as never)[0] ?? Number.NaN,
          sd: toFlat(p?.stddev(site) as never)[0] ?? Number.NaN,
        }
      })
    }),
  ]
}

/** NUTS on the games so far, reading each player's belief at a snapshot. */
export function nutsRun(t: ReturnType<typeof tournamentOf>, games: number): AnytimeRunSpec<Belief[]>[] {
  const method = nutsMethod(boundOf(t, games))
  return [
    {
      id: 'nuts',
      label: 'NUTS',
      method,
      config: { numChains: 2, numWarmup: 200, numDraws: 1500 },
      derive: (s, x) => {
        const p = method.posterior?.(x.state as never, s.config as never)
        return t.players.map((name): Belief => {
          const site = `skill_${name}`
          return p === undefined || p.sites.length === 0
            ? { mean: Number.NaN, sd: Number.NaN }
            : { mean: toFlat(p.mean(site))[0] ?? Number.NaN, sd: toFlat(p.stddev(site))[0] ?? Number.NaN }
        })
      },
    },
  ]
}
