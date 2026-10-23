import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { evaluatePolicy, greedyPolicyOf, make, oneHotObservation, trainAgent } from '../../../lib/rl/index.ts'
import { tensor } from '../../../lib/tensor/index.ts'
import { recorder, tracePolicy } from '../../../lib/trace/index.ts'
import { frozenLake } from '../environments/index.ts'
import { dqn, qNetwork, qValues } from './value-based.ts'

// FrozenLake without ice, one-hot observations: the shortest path takes six steps and pays 1.
const lake = () =>
  make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })

const base = {
  hidden: [32],
  learningRate: 5e-3,
  gamma: 0.9,
  batchSize: 32,
  learningStarts: 64,
  targetUpdateInterval: 50,
  explorationSteps: 800,
  epsilonEnd: 0.05,
  bufferSize: 2000,
} as const

describe('DQN', () => {
  it('learns the dry lake within a short budget and reports its statistics', () => {
    const env = lake()
    const agent = dqn(env as never, base)
    const rec = recorder()
    const result = trainAgent(env as never, agent, {
      maxSteps: 1500,
      seed: 0,
      tracer: rec,
      policy: tracePolicy({ metrics: ['epsilon', 'loss'] }),
    })
    const greedy = evaluatePolicy(env as never, greedyPolicyOf(agent, result.agent), { episodes: 1, seed: 1 })
    expect(greedy.meanReturn).toBe(1)
    expect(greedy.lengths[0]).toBe(6)
    const metrics = rec
      .run()
      .events.filter((e) => e.type === 'metric')
      .map((e) => (e as { name: string }).name)
    expect(new Set(metrics)).toEqual(new Set(['epsilon', 'loss']))
  }, 30_000)

  for (const [name, variant] of [
    ['double + duelling + n-step', { double: true, dueling: true, nStep: 3 }],
    ['prioritised replay with soft targets', { prioritized: true, tau: 0.05 }],
    ['noisy networks', { noisy: true, sigma0: 0.5 }],
  ] as const) {
    it(`learns with ${name}`, () => {
      const env = lake()
      const agent = dqn(env as never, { ...base, ...variant })
      const result = trainAgent(env as never, agent, { maxSteps: 2000, seed: 2 })
      expect(evaluatePolicy(env as never, greedyPolicyOf(agent, result.agent), { episodes: 1 }).meanReturn).toBe(1)
      // Learning runs take a second or two alone; the timeout leaves room for a loaded machine.
    }, 30_000)
  }

  it('builds duelling, noisy and distributional networks of the right shapes', () => {
    const x = tensor([
      [1, 0],
      [0, 1],
    ])
    expect(
      qNetwork(2, 3, { hidden: [4] }, key(0)).apply(qNetwork(2, 3, { hidden: [4] }, key(0)).params, x).shape,
    ).toEqual([2, 3])
    const dist = qNetwork(2, 3, { hidden: [4], atoms: 5, dueling: true, noisy: true }, key(1))
    expect(dist.apply(dist.params, x, key(2)).shape).toEqual([2, 3, 5])
    const env = lake()
    const s = dqn(env as never, { hidden: [4] }).init(key(0))
    expect(qValues(env as never, s, new Array<number>(16).fill(0), { hidden: [4] })).toHaveLength(4)
    expect(() =>
      dqn({ observationSpace: env.observationSpace, actionSpace: { kind: 'box', low: 0, high: 1, shape: [1] } }),
    ).toThrow(/discrete/)
  })
})
