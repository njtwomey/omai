import { describe, expect, it } from 'vitest'
import { changePointBandit } from '../environments/index.ts'
import { key } from '../random/index.ts'
import type { Agent } from '../rl/index.ts'
import { runBandit, ucb1 } from './bandits.ts'
import { discountedUcb, slidingWindowUcb } from './non-stationary-bandits.ts'

const pull = (arm: number, reward: number) => ({
  observation: 0,
  action: arm,
  reward,
  nextObservation: 0,
  terminated: false,
  truncated: false,
})

describe('sliding-window UCB', () => {
  it('keeps the counts and sums of exactly the last tau pulls', () => {
    const agent = slidingWindowUcb(changePointBandit.create({ probs: [0.5, 0.5, 0.5] }), { window: 4 })
    const pulls = [
      [0, 1],
      [1, 0],
      [0, 1],
      [2, 1],
      [2, 0],
      [0, 0],
      [1, 1],
    ] as const
    let s = agent.init(key(0))
    for (const [arm, reward] of pulls) s = agent.learn(s, pull(arm, reward), key(0))
    const last = pulls.slice(-4)
    for (let a = 0; a < 3; a++) {
      expect(s.counts[a]).toBe(last.filter(([x]) => x === a).length)
      expect(s.sums[a]).toBe(last.filter(([x]) => x === a).reduce((acc, [, r]) => acc + r, 0))
    }
    expect(s.t).toBe(7)
  })

  it('tries every arm first, then follows the index, and revisits an arm that left the window', () => {
    const env = changePointBandit.create({ probs: [0.5, 0.5] })
    const agent = slidingWindowUcb(env, { window: 3 })
    let s = agent.init(key(0))
    expect(agent.act(s, 0, key(0))).toBe(0)
    s = agent.learn(s, pull(0, 1), key(0))
    expect(agent.act(s, 0, key(0))).toBe(1)
    s = agent.learn(s, pull(1, 0), key(0))
    // Three pulls of arm 0 push arm 1 out of the window; its index is infinite again.
    for (let i = 0; i < 3; i++) s = agent.learn(s, pull(0, 1), key(0))
    expect(s.counts[1]).toBe(0)
    expect(agent.act(s, 0, key(0))).toBe(1)
    expect(() => slidingWindowUcb(env, { window: 0 })).toThrow()
  })
})

describe('discounted UCB', () => {
  it('discounts every past pull at each step', () => {
    const agent = discountedUcb(changePointBandit.create({ probs: [0.5, 0.5] }), { discount: 0.9 })
    let s = agent.init(key(0))
    s = agent.learn(s, pull(0, 1), key(0))
    s = agent.learn(s, pull(1, 0.5), key(0))
    s = agent.learn(s, pull(0, 0), key(0))
    expect(s.counts[0]).toBeCloseTo(0.81 + 1, 12)
    expect(s.sums[0]).toBeCloseTo(0.81, 12)
    expect(s.counts[1]).toBeCloseTo(0.9, 12)
    expect(s.sums[1]).toBeCloseTo(0.45, 12)
    expect(() => discountedUcb(changePointBandit.create(), { discount: 1 })).toThrow()
  })
})

describe('after a change in the arms', () => {
  // Arm 3 is best for 3000 pulls and then collapses; plain UCB holds three thousand pulls of its old mean and must pull
  // it thousands of times to unlearn them, while a window or a discount forgets within its memory.
  const changeAt = 3000
  const horizon = 8000
  const env = changePointBandit.create({
    probs: [0.3, 0.4, 0.5, 0.9],
    after: [0.3, 0.4, 0.5, 0.1],
    changeAt,
    horizon,
  })
  const regretOf = (make: () => Agent<unknown, unknown, number>, seeds: number): { before: number; after: number } => {
    let before = 0
    let total = 0
    for (let seed = 0; seed < seeds; seed++) {
      const r = runBandit(env, make(), { pulls: horizon, seed })
      before += r.regret[changeAt - 1] ?? 0
      total += r.regret[horizon - 1] ?? 0
    }
    return { before: before / seeds, after: (total - before) / seeds }
  }

  it('recovers within its memory where plain UCB is still pulling the old best arm', () => {
    const seeds = 12
    const plain = regretOf(() => ucb1(env) as Agent<unknown, unknown, number>, seeds)
    const windowed = regretOf(() => slidingWindowUcb(env, { window: 400 }) as Agent<unknown, unknown, number>, seeds)
    const discounted = regretOf(() => discountedUcb(env, { discount: 0.999 }) as Agent<unknown, unknown, number>, seeds)
    expect(plain.after).toBeGreaterThan(700)
    expect(windowed.after).toBeLessThan(0.5 * plain.after)
    expect(discounted.after).toBeLessThan(0.65 * plain.after)
    // Forgetting is not free before the change, but it is not ruinous either.
    expect(windowed.before).toBeLessThan(3 * plain.before)
  })
})
