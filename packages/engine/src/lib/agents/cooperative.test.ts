import { describe, expect, it } from 'vitest'
import { twoStepGame } from '../../testing/cooperative.ts'
import { cooperativeNavigation } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { evaluatePolicy, greedyPolicyOf, jointEnv, trainAgent } from '../rl/index.ts'
import { maddpg, qmix, vdn } from './cooperative.ts'

// Learning tests stay small; the generous timeout covers a loaded machine.
const SLOW = { timeout: 30_000 }

describe('value decomposition', () => {
  // The two-step game of the QMIX paper, trained with uniform exploration: QMIX finds the optimum 8, while VDN's
  // additive value cannot represent game 2B and settles for the safe 7.
  for (const [make, best] of [
    [vdn, 7],
    [qmix, 8],
  ] as const) {
    it(`${make.name} learns the two-step game (${best})`, SLOW, () => {
      const env = jointEnv(twoStepGame, { reward: 'mean' })
      const agent = make(twoStepGame, {
        hidden: [16],
        embed: 8,
        epsilonStart: 1,
        epsilonEnd: 1,
        learningStarts: 50,
        learningRate: 5e-3,
        targetUpdateInterval: 50,
        trainFrequency: 2,
      })
      const r = trainAgent(env, agent, { maxSteps: 600, key: key(3), checkpoints: false })
      expect(evaluatePolicy(env, greedyPolicyOf(agent, r.agent), { episodes: 1, key: key(9) }).meanReturn).toBe(best)
      expect(r.agent.updates).toBeGreaterThan(200)
      expect(agent.stats?.(r.agent)['epsilon']).toBe(1)
    })
  }

  it('run on cooperative navigation with discrete moves', SLOW, () => {
    const base = cooperativeNavigation.create({ agents: 2, maxCycles: 5 })
    for (const make of [vdn, qmix]) {
      const agent = make(base, { hidden: [8], embed: 4, learningStarts: 8, batchSize: 8, targetUpdateInterval: 5 })
      const r = trainAgent(jointEnv(base), agent, { maxSteps: 20, key: key(1), checkpoints: false })
      expect(Number.isFinite(r.agent.loss)).toBe(true)
      expect(r.agent.updates).toBe(13)
      const a = agent.act(r.agent, { agent_0: new Array(12).fill(0), agent_1: new Array(12).fill(0) }, key(2))
      expect(Object.keys(a)).toEqual(['agent_0', 'agent_1'])
    }
    expect(() => vdn(cooperativeNavigation.create({ continuous: true }))).toThrow(/discrete/)
  })
})

describe('MADDPG', () => {
  it('improves continuous cooperative navigation within a short budget', SLOW, () => {
    const base = cooperativeNavigation.create({ agents: 2, continuous: true, maxCycles: 10 })
    const env = jointEnv(base)
    const agent = maddpg(base, {
      hidden: [32, 32],
      trainFrequency: 4,
      batchSize: 32,
      learningStarts: 100,
      gamma: 0.9,
      tau: 0.05,
      noise: 0.2,
    })
    const before = evaluatePolicy(env, greedyPolicyOf(agent, agent.init(key(0))), { episodes: 10, key: key(9) })
    const r = trainAgent(env, agent, { maxSteps: 2000, key: key(3), checkpoints: false })
    const after = evaluatePolicy(env, greedyPolicyOf(agent, r.agent), { episodes: 10, key: key(9) })
    expect(after.meanReturn).toBeGreaterThan(before.meanReturn + 2)
    const a = agent.act(r.agent, { agent_0: new Array(12).fill(0), agent_1: new Array(12).fill(0) }, key(1))
    for (const v of a['agent_0'] ?? []) expect(Math.abs(v)).toBeLessThanOrEqual(1)
  })
})
