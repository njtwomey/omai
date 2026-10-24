import { describe, expect, it, vi } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { make, trainAgent } from '../../../lib/rl/index.ts'
import { box, discrete } from '../../../lib/space/index.ts'
import { recorder, tracePolicy } from '../../../lib/trace/index.ts'
import {
  adversarialBandit,
  bernoulliBandit,
  contextualLinearBandit,
  gaussianBandit,
  nonStationaryBandit,
} from '../environments/index.ts'
import {
  epsilonGreedyBandit,
  exp3,
  gradientBandit,
  klUcb,
  linearThompson,
  linUcb,
  runBandit,
  thompsonBernoulli,
  thompsonGaussian,
  ucb1,
} from './bandits.ts'
import { randomAgent } from './random.ts'

// Learning runs take seconds alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 60_000 })

const final = (xs: readonly number[]): number => xs[xs.length - 1] ?? 0
const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

describe('stochastic bandits: regret far below a random agent', () => {
  it('UCB1, KL-UCB and Thompson sampling on Bernoulli arms, with KL-UCB beating UCB1', () => {
    const env = bernoulliBandit.create({ probs: [0.1, 0.3, 0.5, 0.7, 0.8] })
    const regret = (agent: Parameters<typeof runBandit>[1]): number =>
      final(runBandit(env, agent, { pulls: 3000 }).regret)
    const random = regret(randomAgent(env))
    const u = regret(ucb1(env))
    const kl = regret(klUcb(env))
    const ts = regret(thompsonBernoulli(env))
    for (const r of [u, kl, ts]) expect(r).toBeLessThan(random / 5)
    expect(kl).toBeLessThan(u / 2)
    expect(ts).toBeLessThan(u / 2)
  })

  it('UCB1, Gaussian KL-UCB and Gaussian Thompson sampling (known and estimated variance) on the 10-armed testbed', () => {
    const env = gaussianBandit.create({ seed: 1 })
    const regret = (agent: Parameters<typeof runBandit>[1]): number =>
      final(runBandit(env, agent, { pulls: 2000 }).regret)
    const random = regret(randomAgent(env))
    for (const agent of [
      ucb1(env),
      klUcb(env, { family: 'gaussian', variance: 1 }),
      thompsonGaussian(env, { variance: 1 }),
      thompsonGaussian(env),
    ])
      expect(regret(agent)).toBeLessThan(random / 10)
  })

  it('epsilon-greedy and the gradient bandit learn, and the baseline helps when rewards are offset (Figure 2.5)', () => {
    const base = gaussianBandit.create({ seed: 2 })
    const means = Array.from({ length: 10 }, (_, a) => base.oracle?.expectedReward({ t: 0 }, a) ?? 0)
    const env = gaussianBandit.create({ means: means.map((m) => m + 4) })
    const regret = (agent: Parameters<typeof runBandit>[1]): number =>
      mean([0, 1, 2].map((seed) => final(runBandit(env, agent, { pulls: 1000, seed }).regret)))
    const random = regret(randomAgent(env))
    const withBaseline = regret(gradientBandit(env, { alpha: 0.1 }))
    const without = regret(gradientBandit(env, { alpha: 0.1, baseline: false }))
    expect(withBaseline).toBeLessThan(random / 4)
    expect(withBaseline).toBeLessThan(without)
    expect(regret(epsilonGreedyBandit(env, { epsilon: 0.1 }))).toBeLessThan(random / 4)
  })
})

describe('adversarial, contextual and drifting bandits', () => {
  it('EXP3 has sublinear regret against an adversary', () => {
    // An adversary that keeps one arm best: random play has linear regret, EXP3's average regret falls.
    const fixed = adversarialBandit.create({ arms: 3, period: 4000, horizon: 4000 })
    const r = runBandit(fixed, exp3(fixed, { gamma: 0.1 }), { pulls: 4000 }).regret
    const random = runBandit(fixed, randomAgent(fixed), { pulls: 4000 }).regret
    expect(final(r) / 4000).toBeLessThan(final(random) / 4000 / 4)
    expect(final(r) / 4000).toBeLessThan((r[999] ?? 0) / 1000)
    // An adversary that cycles the best arm: EXP3 keeps up with the best fixed arm.
    const cycling = adversarialBandit.create({ arms: 3, period: 500, horizon: 4000 })
    const c = runBandit(cycling, exp3(cycling, { gamma: 0.1 }), { pulls: 4000 }).regret
    expect(Math.abs(final(c)) / 4000).toBeLessThan(0.05)
  })

  it('LinUCB (shared and disjoint models) and linear Thompson sampling learn the contextual linear bandit', () => {
    const env = contextualLinearBandit.create()
    const random = final(runBandit(env, randomAgent(env), { pulls: 1000 }).regret)
    for (const agent of [linUcb(env), linUcb(env, { model: 'disjoint' }), linearThompson(env, { v: 0.2 })]) {
      const r = runBandit(env, agent, { pulls: 1000 }).regret
      expect(final(r)).toBeLessThan(random / 20)
      // Almost no regret once the model is learned.
      expect(final(r) - (r[499] ?? 0)).toBeLessThan(1)
    }
  })

  it('a constant step size tracks a non-stationary bandit better than sample averages (Exercise 2.5)', () => {
    const env = nonStationaryBandit.create()
    const lateReward = (agent: Parameters<typeof runBandit>[1]): number =>
      mean([0, 1, 2, 3].map((seed) => mean(runBandit(env, agent, { pulls: 5000, seed }).rewards.slice(2500))))
    expect(lateReward(epsilonGreedyBandit(env, { alpha: 0.1 }))).toBeGreaterThan(
      lateReward(epsilonGreedyBandit(env)) + 0.1,
    )
  })
})

describe('the agent contract', () => {
  it('runs under trainAgent with traced statistics', () => {
    const env = make(bernoulliBandit, { horizon: 100 })
    const agent = exp3(env)
    const rec = recorder()
    const r = trainAgent(env, agent, {
      episodes: 2,
      seed: 0,
      tracer: rec,
      policy: tracePolicy({ metrics: ['meanReward', 'topArmShare', 'entropy'] }),
    })
    expect(r.agent.t).toBe(200)
    const names = new Set(
      rec
        .run()
        .events.filter((e) => e.type === 'metric')
        .map((e) => (e as { name: string }).name),
    )
    expect(names).toEqual(new Set(['meanReward', 'topArmShare', 'entropy']))
    const stats = agent.stats?.(r.agent) ?? {}
    expect(stats['entropy']).toBeGreaterThan(0)
    expect(stats['meanReward']).toBeCloseTo(r.agent.totalReward / 200, 12)
  })

  it('turns fractional rewards into Bernoulli trials and follows epsilon schedules', () => {
    const env = bernoulliBandit.create({ probs: [0.5, 0.5] })
    const ts = thompsonBernoulli(env, { prior: [2, 3] })
    const s = ts.learn(
      ts.init(key(0)),
      { observation: 0, action: 1, reward: 0.5, nextObservation: 0, terminated: false, truncated: false },
      key(1),
    )
    expect((s.alpha[1] ?? 0) + (s.beta[1] ?? 0)).toBe(6)
    expect(ts.greedy(ts.init(key(0)), 0)).toBe(0)
    const greedy = epsilonGreedyBandit(env, { epsilon: () => 0, initialValue: 1 })
    expect(greedy.stats?.(greedy.init(key(0)))['epsilon']).toBe(0)
  })

  it('rejects what it cannot handle', () => {
    expect(() => ucb1({ actionSpace: box(0, 1, [1]) })).toThrow(/discrete/)
    expect(() => exp3({ actionSpace: discrete(2) }, { gamma: 0 })).toThrow(/gamma/)
    expect(() => linUcb({ actionSpace: discrete(2) })).toThrow(/dimension/)
    expect(linUcb({ actionSpace: discrete(2) }, { dimension: 3 }).init(key(0)).dimension).toBe(3)
  })
})
