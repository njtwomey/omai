import { describe, expect, it, vi } from 'vitest'
import { reach } from '../../testing/rl-continuous.ts'
import { cartPole, frozenLake } from '../environments/index.ts'
import { key } from '../random/index.ts'
import {
  type Agent,
  evaluatePolicy,
  greedyPolicyOf,
  make,
  oneHotObservation,
  timeLimit,
  trainAgent,
} from '../rl/index.ts'
import { type ActorCriticAgentState, ddpg, sac, sacDiscrete, td3 } from './actor-critic.ts'

// Learning runs take seconds alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 60_000 })

const env = timeLimit(reach(), 10)
const small = { hidden: [16], batchSize: 32 } as const

function solve(agent: Agent<ActorCriticAgentState, readonly number[], readonly number[]>, steps: number, seed: number) {
  const before = evaluatePolicy(env, greedyPolicyOf(agent, agent.init(key(0))), { episodes: 20, seed: 1 }).meanReturn
  const result = trainAgent(env, agent, { maxSteps: steps, seed, checkpoints: false })
  const after = evaluatePolicy(env, greedyPolicyOf(agent, result.agent), { episodes: 20, seed: 1 }).meanReturn
  return { before, after, state: result.agent }
}

describe('off-policy actor-critic agents on the reach task', () => {
  it('DDPG with Ornstein-Uhlenbeck noise solves it', () => {
    const { before, after, state } = solve(
      ddpg(env, { ...small, noise: 'ornstein-uhlenbeck', noiseSigma: 0.3 }),
      1200,
      1,
    )
    expect(before).toBeLessThan(-5)
    expect(after).toBeGreaterThan(-1)
    expect(Object.keys(state.stats).sort()).toEqual(['actorLoss', 'criticLoss', 'qValue'])
    expect(state.updates).toBe(1200 - 100 + 1)
  })

  it('TD3 solves it, updating the actor once per two critic updates', () => {
    const agent = td3(env, small)
    const { after, state } = solve(agent, 1200, 1)
    expect(after).toBeGreaterThan(-0.7)
    expect(agent.config).toMatchObject({ policyDelay: 2, targetPolicyNoise: 0.2, targetNoiseClip: 0.5 })
    expect(state.critic).toHaveProperty('q2')
  })

  it('SAC solves it and tunes its temperature down from 1', () => {
    const agent = sac(env, { ...small, learningRate: 3e-3 })
    const { after, state } = solve(agent, 800, 2)
    expect(after).toBeGreaterThan(-0.7)
    expect(state.stats['alpha']).toBeLessThan(0.5)
    expect(agent.config).toMatchObject({ targetEntropy: -1, autoAlpha: true })
    // Greedy actions are the squashed means, inside the bounds.
    expect(Math.abs(agent.greedy(state, [2])[0] ?? 2)).toBeLessThanOrEqual(1)
  })

  it('keeps a fixed temperature when one is given', () => {
    const agent = sac(env, { ...small, alpha: 0.1, learningStarts: 32 })
    const s = trainAgent(env, agent, { maxSteps: 64, checkpoints: false }).agent
    expect(s.stats['alpha']).toBeCloseTo(0.1, 12)
  })
})

describe('discrete SAC', () => {
  it('learns the dry FrozenLake with one-hot observations', () => {
    const lake = make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })
    const agent = sacDiscrete(lake as never, {
      hidden: [16],
      batchSize: 32,
      learningRate: 3e-3,
      gamma: 0.9,
      targetEntropyScale: 0.3,
      alphaInit: 0.02,
    })
    const result = trainAgent(lake as never, agent, { maxSteps: 1200, seed: 0, checkpoints: false })
    const greedy = evaluatePolicy(lake as never, greedyPolicyOf(agent, result.agent), { episodes: 1 })
    expect(greedy.meanReturn).toBe(1)
    expect(greedy.lengths[0]).toBe(6)
  })

  it('targets -0.98 log(1 / |A|) by default and raises the temperature while the entropy is below the target', () => {
    const defaults = sacDiscrete(cartPole.create())
    expect((defaults.config as { targetEntropy: number }).targetEntropy).toBeCloseTo(0.98 * Math.log(2), 12)
    // No policy over two actions reaches an entropy above log 2, so the temperature can only rise.
    const options = { hidden: [8], learningStarts: 32, batchSize: 16, alphaInit: 0.01, targetEntropy: 1 }
    const agent = sacDiscrete(cartPole.create(), options)
    const s = trainAgent(make(cartPole), agent, { maxSteps: 100, checkpoints: false }).agent
    expect(s.stats['alpha']).toBeGreaterThan(0.01)
    expect(s.stats['entropy']).toBeLessThanOrEqual(Math.log(2) + 1e-12)
  })

  it('rejects mismatched action spaces', () => {
    expect(() => sacDiscrete(env)).toThrow(/discrete/)
    expect(() => ddpg(cartPole.create())).toThrow(/box/)
  })
})
