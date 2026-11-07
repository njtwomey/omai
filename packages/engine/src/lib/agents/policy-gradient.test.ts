import { describe, expect, it, vi } from 'vitest'
import { continuousBandit, reach } from '../../testing/rl-continuous.ts'
import { meanOf } from '../base/index.ts'
import { cartPole } from '../environments/index.ts'
import { key } from '../random/index.ts'
import {
  type Agent,
  evaluatePolicy,
  gae,
  greedyPolicyOf,
  make,
  type Transition,
  timeLimit,
  trainAgent,
} from '../rl/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder, tracePolicy } from '../trace/index.ts'
import { rowsTensor } from './_common.ts'
import { actorCritic, type PolicyAction, prepareRollout } from './_on-policy.ts'
import { a2c, adaptKlCoefficient, type PolicyGradientState, ppo, reinforce, trpo, trpoStep } from './policy-gradient.ts'

// Learning runs take seconds alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 60_000 })

const first = (xs: readonly number[]): number => meanOf(xs.slice(0, 10))
const last = (xs: readonly number[]): number => meanOf(xs.slice(-10))

function trainCartPole(agent: Agent<PolicyGradientState, readonly number[], number>, steps: number, seed = 0) {
  const env = make(cartPole)
  const result = trainAgent(env, agent, { maxSteps: steps, seed, checkpoints: false })
  const greedy = evaluatePolicy(env, greedyPolicyOf(agent, result.agent), { episodes: 3, seed: 1 }).meanReturn
  return { result, greedy }
}

describe('policy-gradient agents on CartPole', () => {
  it('REINFORCE with a learned baseline improves the returns markedly', () => {
    const { result, greedy } = trainCartPole(reinforce(cartPole.create(), { hidden: [32] }), 3000)
    expect(first(result.returns)).toBeLessThan(25)
    expect(last(result.returns)).toBeGreaterThan(3 * first(result.returns))
    expect(greedy).toBeGreaterThan(100)
    expect(Object.keys(result.agent.stats)).toEqual(['policyLoss', 'valueLoss', 'entropy', 'explainedVariance'])
  })

  it('REINFORCE with no baseline, a running average and whole-episode returns', () => {
    for (const options of [
      { baseline: 'none', normalizeReturns: true },
      { baseline: 'average', rewardToGo: false, episodesPerUpdate: 2 },
    ] as const) {
      const { result } = trainCartPole(reinforce(cartPole.create(), { hidden: [32], ...options }), 2500)
      expect(last(result.returns)).toBeGreaterThan(1.5 * first(result.returns))
    }
  })

  it('A2C improves the returns markedly', () => {
    const agent = a2c<number>(cartPole.create(), { hidden: [32], nSteps: 8, learningRate: 3e-3 })
    const { result, greedy } = trainCartPole(agent, 3000, 2)
    expect(last(result.returns)).toBeGreaterThan(3 * first(result.returns))
    expect(greedy).toBeGreaterThan(100)
  })

  it('TRPO improves the returns markedly with accepted line searches', () => {
    const { result, greedy } = trainCartPole(trpo(cartPole.create(), { hidden: [32], nSteps: 500 }), 3000)
    expect(last(result.returns)).toBeGreaterThan(2.5 * first(result.returns))
    expect(greedy).toBeGreaterThan(200)
    expect(result.agent.stats['lineSearchSuccess']).toBe(1)
    expect(result.agent.stats['approxKl']).toBeLessThan(0.01)
  })

  it('PPO (clipped) improves the returns markedly and traces its statistics', () => {
    const agent = ppo<number>(cartPole.create(), { hidden: [32], learningRate: 3e-3, nEpochs: 4 })
    const rec = recorder()
    const env = make(cartPole)
    const result = trainAgent(env, agent, {
      maxSteps: 3000,
      seed: 0,
      checkpoints: false,
      tracer: rec,
      policy: tracePolicy({ metrics: ['approxKl', 'clipFraction', 'explainedVariance'] }),
    })
    const greedy = evaluatePolicy(env, greedyPolicyOf(agent, result.agent), { episodes: 3, seed: 1 }).meanReturn
    expect(last(result.returns)).toBeGreaterThan(3 * first(result.returns))
    expect(greedy).toBeGreaterThan(150)
    const names = new Set(
      rec
        .run()
        .events.filter((e) => e.type === 'metric')
        .map((e) => (e as { name: string }).name),
    )
    expect(names).toEqual(new Set(['approxKl', 'clipFraction', 'explainedVariance']))
    for (const k of ['policyLoss', 'valueLoss', 'entropy', 'approxKl', 'clipFraction', 'explainedVariance'])
      expect(Number.isFinite(result.agent.stats[k])).toBe(true)
  })

  it('PPO with the adaptive KL penalty improves and adapts its coefficient', () => {
    const agent = ppo<number>(cartPole.create(), { hidden: [32], learningRate: 3e-3, nEpochs: 4, objective: 'kl' })
    const { result, greedy } = trainCartPole(agent, 3000)
    expect(last(result.returns)).toBeGreaterThan(3 * first(result.returns))
    expect(greedy).toBeGreaterThan(150)
    expect(result.agent.klCoef).not.toBe(1)
    expect(result.agent.stats['klCoef']).toBe(result.agent.klCoef)
  })

  it('PPO stops early on the target KL, clips values and anneals the learning rate', () => {
    const options = { hidden: [16], nSteps: 128, learningRate: 3e-2, nEpochs: 10, targetKl: 1e-4 }
    const stopped = trainAgent(make(cartPole), ppo(cartPole.create(), options), { maxSteps: 128, checkpoints: false })
    expect(stopped.agent.stats['earlyStop']).toBe(1)
    const annealed = ppo(cartPole.create(), {
      hidden: [16],
      nSteps: 128,
      clipRangeVf: 0.2,
      annealLearningRate: true,
      totalSteps: 128,
    })
    // At the end of the budget the learning-rate factor is zero, so the update leaves the parameters unchanged.
    const s = trainAgent(make(cartPole), annealed, { maxSteps: 128, checkpoints: false }).agent
    const s0 = annealed.init(key(0))
    expect(toFlat((s.params.policy['net'] as Record<string, Record<string, never>>)['0']?.['weight'] as never)).toEqual(
      toFlat((s0.params.policy['net'] as Record<string, Record<string, never>>)['0']?.['weight'] as never),
    )
    expect(() => ppo(cartPole.create(), { annealLearningRate: true })).toThrow(/totalSteps/)
  })
})

describe('policy-gradient agents on continuous actions', () => {
  it('PPO solves the reach task with state-independent and squashed state-dependent Gaussians', () => {
    const env = timeLimit(reach(), 10)
    for (const options of [{}, { stateDependentStd: true, squash: true }]) {
      const agent = ppo<readonly number[]>(env, { hidden: [16], learningRate: 3e-3, nEpochs: 4, ...options })
      const before = evaluatePolicy(env, greedyPolicyOf(agent, agent.init(key(0))), { episodes: 20, seed: 1 })
      const result = trainAgent(env, agent, { maxSteps: 3000, seed: 0, checkpoints: false })
      const after = evaluatePolicy(env, greedyPolicyOf(agent, result.agent), { episodes: 20, seed: 1 })
      expect(before.meanReturn).toBeLessThan(-4)
      expect(after.meanReturn).toBeGreaterThan(-0.8)
    }
  })

  it('TRPO solves the reach task; REINFORCE and A2C improve on it', () => {
    const env = timeLimit(reach(), 10)
    const greedyAfter = (agent: Agent<PolicyGradientState, readonly number[], readonly number[]>): number => {
      const result = trainAgent(env, agent, { maxSteps: 3000, seed: 0, checkpoints: false })
      return evaluatePolicy(env, greedyPolicyOf(agent, result.agent), { episodes: 20, seed: 1 }).meanReturn
    }
    expect(greedyAfter(trpo(env, { hidden: [16], nSteps: 250 }))).toBeGreaterThan(-0.6)
    expect(greedyAfter(reinforce(env, { hidden: [16], episodesPerUpdate: 4 }))).toBeGreaterThan(-2)
    expect(greedyAfter(a2c(env, { hidden: [16], nSteps: 10, learningRate: 3e-3 }))).toBeGreaterThan(-2)
  })

  it('PPO learns the contextual bandit, and greedy actions stay within the bounds', () => {
    const env = timeLimit(continuousBandit(), 1)
    const agent = ppo<readonly number[]>(env, {
      hidden: [16],
      learningRate: 3e-3,
      nEpochs: 4,
      nSteps: 128,
      batchSize: 32,
    })
    const result = trainAgent(env, agent, { maxSteps: 1500, seed: 1, checkpoints: false })
    expect(
      evaluatePolicy(env, greedyPolicyOf(agent, result.agent), { episodes: 50, seed: 1 }).meanReturn,
    ).toBeGreaterThan(-0.1)
    for (const c of [-1, 1]) expect(Math.abs(agent.greedy(result.agent, [c])[0] ?? 2)).toBeLessThanOrEqual(1)
  })
})

describe('policy-gradient building blocks', () => {
  it('bootstraps truncated steps from the value of their final observation', () => {
    const env = cartPole.create()
    const net = actorCritic(env, { hidden: [4] }, 'test')
    const o = (x: number): readonly number[] => [x, 0, 0, 0]
    const step = (
      x: number,
      done: Partial<Transition<readonly number[], PolicyAction>>,
    ): Transition<readonly number[], PolicyAction> => ({
      observation: o(x),
      action: 0,
      reward: 1,
      nextObservation: o(x + 1),
      terminated: false,
      truncated: false,
      ...done,
    })
    // Truncated after step 1 (next observation o(2)); a new episode starts at o(5) and the rollout ends mid-episode.
    const pending = [step(0, {}), step(1, { truncated: true }), step(5, {}), step(6, {})]
    const batch = prepareRollout(net, net.params, pending, 0.9, 0.8)
    const v = toFlat(net.value(net.params.value, rowsTensor([o(0), o(1), o(5), o(6), o(2), o(7)], 4)))
    const rewards = [1, 1 + 0.9 * (v[4] ?? 0), 1, 1]
    const want = gae(rewards, v.slice(0, 4), [0, 1, 0, 0], v[5] ?? 0, 0.9, 0.8)
    for (const [i, a] of batch.reward.advantages.entries()) expect(a).toBeCloseTo(want.advantages[i] ?? 0, 12)
    expect(batch.reward.values).toEqual(v.slice(0, 4))
  })

  it('takes natural-gradient steps to the trust-region boundary and adapts the KL penalty', () => {
    const fisher = (v: readonly number[]): number[] => [2 * (v[0] ?? 0) + (v[1] ?? 0), (v[0] ?? 0) + 3 * (v[1] ?? 0)]
    const r = trpoStep(fisher, [1, 1], { maxKl: 0.02, damping: 0 })
    const fs = fisher(r.step)
    expect(0.5 * (r.step[0] ?? 0) * (fs[0] ?? 0) + 0.5 * (r.step[1] ?? 0) * (fs[1] ?? 0)).toBeCloseTo(0.02, 12)
    expect(r.direction[0]).toBeCloseTo(0.4, 12)
    expect(r.direction[1]).toBeCloseTo(0.2, 12)
    expect(adaptKlCoefficient(4, 0.1, 0.01)).toBe(8)
    expect(a2c(cartPole.create(), { advantages: 'gae' }).config).toMatchObject({ gaeLambda: 0.95 })
  })

  it('rejects action spaces it cannot handle', () => {
    expect(() =>
      ppo({
        observationSpace: cartPole.create().observationSpace,
        actionSpace: { kind: 'multiBinary', n: 2 } as never,
      }),
    ).toThrow()
  })
})
