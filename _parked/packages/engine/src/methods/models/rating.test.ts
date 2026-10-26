import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { eloGames, roundRobin } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { bradleyTerry, eloRatings, eloSystem, glickoRatings, trueSkillRatings, trueSkillSystem } from './rating.ts'

describe('rating systems', () => {
  it('Elo and TrueSkill single-game updates match the textbook numbers', () => {
    const elo = eloSystem()
    const [a, b] = elo.update(elo.initial(), elo.initial(), 1, { gamesA: 0, gamesB: 0 })
    expect(a.rating).toBeCloseTo(1516, 10)
    expect(b.rating).toBeCloseTo(1484, 10)
    // The trueskill package: rate_1vs1(Rating(), Rating()).
    const ts = trueSkillSystem()
    const [w, l] = ts.update(ts.initial(), ts.initial(), 1, { gamesA: 0, gamesB: 0 })
    expect(w.mu).toBeCloseTo(29.396, 3)
    expect(w.sigma).toBeCloseTo(7.171, 3)
    expect(l.mu).toBeCloseTo(20.604, 3)
    const [d1, d2] = ts.update(ts.initial(), ts.initial(), 0.5, { gamesA: 0, gamesB: 0 })
    expect(d1.mu).toBeCloseTo(25, 6)
    expect(d2.sigma).toBeCloseTo(6.458, 3)
  })

  it('every system ranks the latent skills of a tournament and records its curves', () => {
    const g = generate(roundRobin, { key: key(1), knobs: { players: 8, rounds: 6, spread: 300 } })
    const skills = Array.from(toFlat(g.truth?.latents?.['skills'] as never)) as number[]
    const best = skills.indexOf(Math.max(...skills))
    for (const r of [eloRatings(g.data), glickoRatings(g.data), trueSkillRatings(g.data), bradleyTerry(g.data)]) {
      expect(r.order.slice(0, 2)).toContain(best)
      const [table] = recordsOf(r.exposition, 'coefficientTable')
      expect(table?.names.length).toBe(8)
      expect(recordsOf(r.exposition, 'learningCurve').length).toBe(1)
    }
    const ts = trueSkillRatings(g.data)
    expect(recordsOf(ts.exposition, 'posteriorGrid')[0]?.series?.length).toBe(3)
    expect(ts.deviations.every((s) => s < 25 / 3)).toBe(true)
  })

  it('Elo on drifting games predicts better than chance', () => {
    const g = generate(eloGames, { key: key(2), knobs: { players: 10, games: 600, spread: 300 } })
    const r = eloRatings(g.data, { k: 'fide' })
    const [curve] = recordsOf(r.exposition, 'learningCurve')
    expect(curve?.validation.at(-1)).toBeGreaterThan(0.6)
    expect(r.history.length).toBe(600)
  })
})
