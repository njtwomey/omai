import { describe, expect, it } from 'vitest'
import { bernoulliBandit, cartPole, nonStationaryBandit } from '../../methods/rl/index.ts'
import {
  cumulativeRegret,
  isBandit,
  klBernoulli,
  klDivergence,
  klExponential,
  klGaussian,
  klPoisson,
  klUcbIndex,
  laiRobbinsBound,
  ucb1Index,
} from './bandits.ts'

describe('bandit helpers', () => {
  it('compute KL divergences of the exponential families', () => {
    expect(klBernoulli(0.2, 0.5)).toBeCloseTo(0.2 * Math.log(0.4) + 0.8 * Math.log(1.6), 14)
    expect(klBernoulli(0, 1)).toBeGreaterThan(30)
    expect(klGaussian(1, 3, 2)).toBe(1)
    expect(klPoisson(1, Math.E)).toBeCloseTo(Math.E - 2, 14)
    expect(klExponential(2, 1)).toBeCloseTo(1 - Math.log(2), 14)
    expect(klDivergence('poisson', 3, 3)).toBe(0)
    expect(klDivergence('exponential', 1, 2)).toBeCloseTo(0.5 - 1 - Math.log(0.5), 14)
  })

  it('compute UCB1 and KL-UCB indices', () => {
    expect(ucb1Index(0.3, 0, 10)).toBe(Infinity)
    expect(ucb1Index(0.5, 4, 100)).toBeCloseTo(0.5 + Math.sqrt((2 * Math.log(100)) / 4), 14)
    const q = klUcbIndex(0.4, 10, 100)
    expect(10 * klBernoulli(0.4, q)).toBeCloseTo(Math.log(100), 8)
    expect(q).toBeGreaterThan(0.4)
    expect(klUcbIndex(0.4, 0, 100)).toBe(Infinity)
    expect(klUcbIndex(0.4, 10, 1)).toBe(0.4)
    const p = klUcbIndex(2, 5, 50, { family: 'poisson' })
    expect(5 * klPoisson(2, p)).toBeCloseTo(Math.log(50), 8)
    const e = klUcbIndex(1, 5, 50, { family: 'exponential' })
    expect(5 * klExponential(1, e)).toBeCloseTo(Math.log(50), 8)
    // KL-UCB is tighter than UCB1 for Bernoulli arms.
    expect(klUcbIndex(0.9, 20, 1000)).toBeLessThan(ucb1Index(0.9, 20, 1000))
  })

  it('compute the Lai-Robbins bound and cumulative regret from the oracle', () => {
    const means = [0.1, 0.5, 0.9]
    const expected = ((0.8 / klBernoulli(0.1, 0.9) + 0.4 / klBernoulli(0.5, 0.9)) * Math.log(1000)) as number
    expect(laiRobbinsBound(means, 1000)).toBeCloseTo(expected, 12)
    const env = bernoulliBandit.create({ probs: means })
    expect(isBandit(env)).toBe(true)
    expect(isBandit(cartPole.create())).toBe(false)
    const regret = cumulativeRegret(env.oracle as never, [{ t: 0 }, { t: 1 }, { t: 2 }], [0, 2, 1])
    expect(regret.map((r) => Number(r.toFixed(12)))).toEqual([0.8, 0.8, 1.2])
    expect(() => cumulativeRegret(env.oracle as never, [], [0])).toThrow(/state before/)
    const drifting = nonStationaryBandit.create()
    expect(isBandit(drifting)).toBe(true)
  })
})
