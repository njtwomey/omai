import { describe, expect, it } from 'vitest'
import { frozenLake, infiniteVariance, makeRandomWalk } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { evaluatePolicy, greedyPolicyOf, make, runEpisode, trainAgent } from '../rl/index.ts'
import { offPolicyMonteCarloControl, offPolicyMonteCarloPrediction } from './off-policy-monte-carlo.ts'

const step = (observation: number, action: number, reward: number, nextObservation: number, terminated: boolean) => ({
  observation,
  action,
  reward,
  nextObservation,
  terminated,
  truncated: false,
})

describe('off-policy Monte Carlo prediction', () => {
  it('weights the returns of a first visit by the ratios of the rest of the episode', () => {
    const env = makeRandomWalk({ n: 3 })
    const target = (): readonly number[] => [0.25, 0.75]
    const behaviour = (): readonly number[] => [0.5, 0.5]
    for (const weighting of ['ordinary', 'weighted'] as const) {
      const agent = offPolicyMonteCarloPrediction(env, { target, behaviour, weighting, gamma: 0.5 })
      let s = agent.init(key(0))
      // Episode 1: state 1 (right), state 2 (right), rewards 0 then 4, ending. Episode 2: state 1 (left), reward 2, ending.
      s = agent.learn(s, step(1, 1, 0, 2, false), key(0))
      s = agent.learn(s, step(2, 1, 4, 2, true), key(0))
      s = agent.learn(s, step(1, 0, 2, 1, true), key(0))
      const rho1 = (0.75 / 0.5) * (0.75 / 0.5)
      const rho2 = 0.25 / 0.5
      const g1 = 0 + 0.5 * 4
      const g2 = 2
      const expected = weighting === 'ordinary' ? (rho1 * g1 + rho2 * g2) / 2 : (rho1 * g1 + rho2 * g2) / (rho1 + rho2)
      expect(s.V[1] ?? Number.NaN, weighting).toBeCloseTo(expected, 12)
      expect(s.V[2] ?? Number.NaN, weighting).toBeCloseTo(weighting === 'ordinary' ? (1.5 * 4) / 1 : 4, 12)
      expect(s.episodes).toBe(2)
      expect(s.maxRatio).toBeCloseTo(rho1, 12)
    }
  })

  it('has weighted importance sampling beat ordinary on the process of infinite variance', () => {
    const env = infiniteVariance.create()
    const runs = 120
    const episodes = 100
    const estimate = (weighting: 'ordinary' | 'weighted', seed: number): number => {
      const agent = offPolicyMonteCarloPrediction(env, { target: () => [1, 0], weighting })
      return trainAgent(env, agent, { episodes, checkpoints: false, seed }).agent.V[0] ?? Number.NaN
    }
    const spread = (xs: readonly number[]): { mean: number; variance: number } => {
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length
      return { mean, variance: xs.reduce((a, x) => a + (x - mean) ** 2, 0) / (xs.length - 1) }
    }
    const ordinary = spread(Array.from({ length: runs }, (_, i) => estimate('ordinary', i)))
    const weighted = spread(Array.from({ length: runs }, (_, i) => estimate('weighted', i)))
    // The true value is 1. Both are consistent, but the ordinary estimates scatter with a variance orders above.
    expect(Math.abs(weighted.mean - 1)).toBeLessThan(0.1)
    expect(weighted.variance).toBeLessThan(0.1)
    expect(ordinary.variance).toBeGreaterThan(10 * weighted.variance)
  })

  it('refuses a behaviour that cannot take the target action', () => {
    const env = infiniteVariance.create()
    const agent = offPolicyMonteCarloPrediction(env, { target: () => [1, 0], behaviour: () => [0, 1] })
    expect(() => agent.learn(agent.init(key(0)), step(0, 0, 0, 0, false), key(0))).toThrow()
  })
})

describe('off-policy Monte Carlo control', () => {
  it('replays the episode from its end and stops at the first step the greedy policy would not take', () => {
    const env = makeRandomWalk({ n: 3 })
    const agent = offPolicyMonteCarloControl(env)
    let s = agent.init(key(0))
    s = agent.learn(s, step(1, 1, 0, 2, false), key(0))
    s = agent.learn(s, step(2, 1, -1, 2, true), key(0))
    // (2, right) gets G = -1, and then the greedy action at state 2 is left (value 0): the replay stops there.
    expect(s.Q[2 * 2 + 1]).toBe(-1)
    expect(s.Q[1 * 2 + 1]).toBe(0)
    expect(s.C[1 * 2 + 1]).toBe(0)
    // A rewarded episode is used whole: each step's value rises to the return and the weight doubles per step back.
    s = agent.init(key(0))
    s = agent.learn(s, step(1, 1, 0, 2, false), key(0))
    s = agent.learn(s, step(2, 1, 1, 2, true), key(0))
    expect([s.Q[2 * 2 + 1], s.Q[1 * 2 + 1]]).toEqual([1, 1])
    expect(s.lastWeight).toBe(2)
  })

  it('learns the way across a frozen lake from random behaviour', () => {
    const env = make(frozenLake, { isSlippery: false })
    for (const weighting of ['weighted'] as const) {
      const agent = offPolicyMonteCarloControl(env, { weighting, gamma: 0.9 })
      const trained = trainAgent(env, agent, { episodes: 4000, checkpoints: false, seed: 3 })
      const greedy = evaluatePolicy(env, greedyPolicyOf(agent, trained.agent), { episodes: 5, key: key(1) })
      expect(greedy.meanReturn, weighting).toBe(1)
      expect(runEpisode(env, greedyPolicyOf(agent, trained.agent), key(2)).length, weighting).toBeLessThanOrEqual(12)
    }
  })
})
