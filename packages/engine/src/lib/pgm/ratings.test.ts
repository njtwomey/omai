import { describe, expect, it } from 'vitest'
import { gumbel, key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as P from './index.ts'

const worked = [
  { opponent: { r: 1400, rd: 30 }, score: 1 },
  { opponent: { r: 1550, rd: 100 }, score: 0 },
  { opponent: { r: 1700, rd: 300 }, score: 0 },
]

describe('Elo', () => {
  it('is the logistic curve and moves the two ratings by the same amount in opposite directions', () => {
    expect(P.eloExpected(1500, 1500)).toBe(0.5)
    expect(P.eloExpected(1900, 1500)).toBeCloseTo(10 / 11, 12)
    expect(P.eloUpdate(1500, 1500, 1)).toEqual([1516, 1484])
    const [a, b] = P.eloUpdate(1700, 1400, 0.5)
    expect(a + b).toBeCloseTo(3100, 12)
    expect(a).toBeLessThan(1700)
  })
})

describe('Glicko and Glicko-2 worked examples', () => {
  it('reproduces the numbers of Glickman (1999)', () => {
    const p = P.glickoUpdate({ r: 1500, rd: 200 }, worked)
    expect(p.r).toBeCloseTo(1464.1, 1)
    expect(p.rd).toBeCloseTo(151.4, 1)
  })
  it('reproduces the numbers of Glickman (2012)', () => {
    const p = P.glicko2Update({ r: 1500, rd: 200, sigma: 0.06 }, worked, 0.5)
    expect(p.r).toBeCloseTo(1464.06, 1)
    expect(p.rd).toBeCloseTo(151.52, 1)
    expect(p.sigma).toBeCloseTo(0.06, 3)
  })
  it('lets the deviation grow with idle periods and shrink with games', () => {
    expect(P.glickoInflate(50, 10, 0)).toBe(50)
    expect(P.glickoInflate(340, 100, 5)).toBe(350)
    expect(P.glicko2Update({ r: 1500, rd: 100, sigma: 0.06 }, []).rd).toBeGreaterThan(100)
    expect(P.glickoUpdate({ r: 1500, rd: 200 }, worked).rd).toBeLessThan(200)
  })
})

describe('TrueSkill', () => {
  const start = { mu: 25, sigma: 25 / 3 }
  it('reproduces the reference one-versus-one update between two new players', () => {
    const [[w], [l]] = P.trueskillUpdate([start], [start])
    expect(w?.mu).toBeCloseTo(29.396, 3)
    expect(l?.mu).toBeCloseTo(20.604, 3)
    expect(w?.sigma).toBeCloseTo(7.171, 3)
    const [[a], [b]] = P.trueskillUpdate([start], [start], true)
    expect(a?.mu).toBeCloseTo(25, 6)
    expect(b?.mu).toBeCloseTo(25, 6)
    expect(a?.sigma).toBeCloseTo(6.458, 3)
  })
  it('moves more on an upset than on an expected result, and uncertain players more than sure ones', () => {
    const strong = { mu: 30, sigma: 2 }
    const weak = { mu: 20, sigma: 2 }
    const expected = P.trueskillUpdate([strong], [weak])[0][0] as P.TrueSkillRating
    const upset = P.trueskillUpdate([weak], [strong])[0][0] as P.TrueSkillRating
    expect(upset.mu - weak.mu).toBeGreaterThan(expected.mu - strong.mu)
    const unsure = P.trueskillUpdate([{ mu: 25, sigma: 8 }], [start])[0][0] as P.TrueSkillRating
    const sure = P.trueskillUpdate([{ mu: 25, sigma: 1 }], [start])[0][0] as P.TrueSkillRating
    expect(unsure.mu - 25).toBeGreaterThan(sure.mu - 25)
  })
  it('updates teams symmetrically and shrinks every uncertainty', () => {
    const [win, lose] = P.trueskillUpdate([start, start], [start, start])
    expect(win[0]?.mu).toBeCloseTo(win[1]?.mu ?? 0, 12)
    expect((win[0]?.mu ?? 0) - 25).toBeCloseTo(25 - (lose[0]?.mu ?? 0), 12)
    for (const r of [...win, ...lose]) expect(r.sigma).toBeLessThan(start.sigma)
    // A team of two moves each member less than a one-versus-one win does.
    expect((win[0]?.mu ?? 0) - 25).toBeLessThan(29.396 - 25)
  })
})

describe('rating algorithm', () => {
  const n = 8
  const truth = Array.from({ length: n }, (_, i) => (i - 3.5) * 0.7)
  const games: P.RatedGame[] = []
  const ks = Array.from({ length: 1200 }, (_, g) => key(1000 + g))
  ks.forEach((k, g) => {
    const a = g % n
    const b = (a + 1 + (Math.floor(g / n) % (n - 1))) % n
    const [ga, gb] = toFlat(gumbel(k, [2]))
    games.push({
      a,
      b,
      score: (truth[a] ?? 0) + (ga ?? 0) > (truth[b] ?? 0) + (gb ?? 0) ? 1 : 0,
      period: Math.floor(g / 40),
    })
  })
  const rank = (v: readonly number[]): number[] => v.map((x) => v.filter((y) => y < x).length)
  const spearman = (a: readonly number[], b: readonly number[]): number => {
    const ra = rank(a)
    const rb = rank(b)
    const m = (n - 1) / 2
    const cov = ra.reduce((s, x, i) => s + (x - m) * ((rb[i] ?? 0) - m), 0)
    const va = ra.reduce((s, x) => s + (x - m) ** 2, 0)
    return cov / va
  }
  it.each(['elo', 'glicko', 'glicko2', 'trueskill'] as const)(
    '%s orders the players by their true strength',
    (system) => {
      const { algorithm, steps } = P.ratingAlgorithm(system, games, n)
      const run = runAlgorithm(algorithm, { maxSteps: steps })
      expect(run.steps).toBe(steps)
      expect(spearman(run.state.rating, truth)).toBeGreaterThan(0.9)
    },
  )
  it('predicts better than a coin once ratings are learnt, and the uncertainty falls', () => {
    const { algorithm, steps } = P.ratingAlgorithm('trueskill', games, n)
    const run = runAlgorithm(algorithm, { maxSteps: steps })
    expect(run.state.logLoss / games.length).toBeLessThan(Math.log(2))
    expect(Math.max(...run.state.deviation)).toBeLessThan(25 / 3 / 2)
  })
  it('serialises its state', () => {
    const { algorithm } = P.ratingAlgorithm('glicko2', games, n)
    const half = runAlgorithm(algorithm, { maxSteps: 4 })
    expect(algorithm.deserialise?.(algorithm.serialise?.(half.state) as never)).toEqual(half.state)
  })
})
