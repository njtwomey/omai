import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { cumulativeRegret, make, runEpisode } from '../rl/index.ts'
import {
  adversarialBandit,
  bernoulliBandit,
  contextualLinearBandit,
  gaussianBandit,
  nonStationaryBandit,
} from './bandits.ts'

describe('bandit environments', () => {
  it('pay Bernoulli and Gaussian rewards with the oracle means', () => {
    const env = bernoulliBandit.create({ probs: [0.25, 0.75] })
    let total = 0
    for (let i = 0; i < 4000; i++) total += env.step({ t: 0 }, 1, key(i)).reward
    expect(Math.abs(total / 4000 - 0.75)).toBeLessThan(0.03)
    expect(env.oracle?.optimalAction?.({ t: 0 })).toBe(1)
    const g = gaussianBandit.create({ means: [0, 2], sigma: 0.5 })
    let sum = 0
    for (let i = 0; i < 2000; i++) sum += g.step({ t: 0 }, 1, key(i)).reward
    expect(Math.abs(sum / 2000 - 2)).toBeLessThan(0.05)
    expect(gaussianBandit.create({ seed: 3 }).oracle?.optimalExpectedReward({ t: 0 })).toBeGreaterThan(0)
  })

  it('run a horizon of pulls under make and measure regret', () => {
    const env = make(bernoulliBandit, { probs: [0.2, 0.8], horizon: 50 })
    const ep = runEpisode(env, () => 0, key(0))
    expect(ep.length).toBe(50)
    expect(ep.truncated).toBe(true)
    const regret = cumulativeRegret(
      bernoulliBandit.create({ probs: [0.2, 0.8] }).oracle as never,
      ep.states,
      ep.actions,
    )
    expect(regret.at(-1)).toBeCloseTo(50 * 0.6, 10)
  })

  it('show linear contexts whose expected reward the oracle knows', () => {
    const env = contextualLinearBandit.create({ arms: 3, dimension: 2, noise: 0 })
    const r = env.reset(key(0))
    const best = env.oracle?.optimalAction?.(r.state) ?? 0
    const step = env.step(r.state, best, key(1))
    expect(step.reward).toBeCloseTo(env.oracle?.optimalExpectedReward(r.state) ?? 0, 12)
    expect(step.observation).toHaveLength(3)
  })

  it('fix adversarial rewards in advance and let non-stationary means drift', () => {
    const adv = adversarialBandit.create({ arms: 2, period: 5, horizon: 20 })
    const a = adv.step({ t: 0 }, 0, key(0)).reward
    expect(adv.step({ t: 0 }, 0, key(99)).reward).toBe(a)
    expect(adv.oracle?.expectedReward({ t: 7 }, 1)).toBeGreaterThan(adv.oracle?.expectedReward({ t: 7 }, 0) ?? 1)
    const drift = nonStationaryBandit.create({ arms: 3 })
    let s = drift.reset(key(0)).state
    for (let t = 0; t < 100; t++) s = drift.step(s, 0, key(t)).state
    expect(s.means.some((m) => m !== 0)).toBe(true)
    expect(s.t).toBe(100)
  })
})
