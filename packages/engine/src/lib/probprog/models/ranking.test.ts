// The two pieces the proofs of concept of the probabilistic programming group needed of the benchmark models: exact draws
// from the banana to draw the target beside what a method makes of it, and a small TrueSkill tournament written as a model
// with one factor per game, so message passing can be watched game by game.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { epSpecMethod } from '../inference/index.ts'
import { toFactorGraph, toStructure } from '../model/index.ts'
import { banana, bananaDraws, trueskillModel, trueskillTournament } from './index.ts'

describe('bananaDraws', () => {
  const d = bananaDraws(key(5), 20000)
  const mean = (v: Float64Array): number => v.reduce((a, b) => a + b, 0) / v.length
  const sd = (v: Float64Array): number => {
    const m = mean(v)
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
  }

  it('has the banana’s marginals: x0 normal with sd 2, x1 with mean 0 and sd 3', () => {
    expect(Math.abs(mean(d.x0))).toBeLessThan(0.1)
    expect(sd(d.x0)).toBeGreaterThan(1.9)
    expect(sd(d.x0)).toBeLessThan(2.1)
    expect(Math.abs(mean(d.x1))).toBeLessThan(0.15)
    expect(sd(d.x1)).toBeGreaterThan(2.7)
    expect(sd(d.x1)).toBeLessThan(3.3)
  })

  it('is the sampler the benchmark’s truth is made from', () => {
    expect(banana.truth().marginals['x1']?.sd).toBe(3)
    expect(banana.truth().marginals['x0']?.sd).toBe(2)
  })
})

describe('trueskillTournament and trueskillModel', () => {
  const t = trueskillTournament(key(2), { games: 12 })

  it('draws the games from the skills, the same ones for the same key, and the first n of a longer tournament are those of a shorter', () => {
    expect(trueskillTournament(key(2), { games: 12 }).wins).toEqual(t.wins)
    expect(trueskillTournament(key(2), { games: 5 }).pairs).toEqual(t.pairs.slice(0, 5))
    expect(trueskillTournament(key(2), { games: 5 }).wins).toEqual(t.wins.slice(0, 5))
    expect(t.players).toEqual(['A', 'B', 'C'])
  })

  it('is a model with a node per player and an observed node and factor per game', () => {
    const { definition, data } = trueskillModel(t, 6)
    expect(Object.keys(data)).toHaveLength(6)
    const bound = definition.bind(data)
    const graph = toFactorGraph(bound).graph
    expect(graph?.variables.map((v) => v.name).filter((n) => n.startsWith('skill_'))).toEqual([
      'skill_A',
      'skill_B',
      'skill_C',
    ])
    const ids = toStructure(bound, { view: 'factor' }).nodes.map((n) => n.id)
    for (let g = 1; g <= 6; g++) expect(ids).toContain(`factor:win_${g}`)
  })

  it('is rated by expectation propagation in the order of the true skills after enough games', () => {
    const long = trueskillTournament(key(2), { games: 30 })
    const { definition, data } = trueskillModel(long)
    const ep = epSpecMethod(definition.bind(data))
    const s = session(ep, { steps: 100_000, mode: 'eager', config: { tol: 1e-9 } })
    const post = ep.posterior?.(s.state, s.config)
    const mean = long.players.map((p) => toFlat(post?.mean(`skill_${p}`) as never)[0] as number)
    expect(s.status).toBe('converged')
    expect((mean[0] as number) > (mean[1] as number)).toBe(true)
    expect((mean[1] as number) > (mean[2] as number)).toBe(true)
  })
})
