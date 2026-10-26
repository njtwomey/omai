import { describe, expect, it } from 'vitest'
import { tabularExpert } from '../../testing/rl-offline.ts'
import { cliffWalking, frozenLake } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { evaluatePolicy, greedyPolicyOf, type MakeConfig, make, oneHotObservation, trainAgent } from '../rl/index.ts'
import { recorder, runAlgorithm, tracePolicy } from '../trace/index.ts'
import {
  behaviourCloning,
  behaviourCloningLearner,
  collectDataset,
  datasetEpisodes,
  offlineAlgorithm,
} from './offline.ts'
import { bcq, cql, cqlLearner, offlineDqn } from './offline-q-learning.ts'

const oneHot: MakeConfig = { wrappers: [(e) => oneHotObservation(e as never) as never] }
// The dry 4 x 4 lake and cliff walking (capped at 50 steps), with one-hot observations.
const lake = () => make(frozenLake, { isSlippery: false }, oneHot)
const cliff = () => make(cliffWalking, {}, { ...oneHot, maxEpisodeSteps: 50 })
const options = { hidden: [32], steps: 300, batchSize: 32, learningRate: 3e-3, gamma: 0.99 } as const

describe('datasets', () => {
  it('collects whole episodes with a behaviour policy and splits them again', () => {
    const env = lake()
    const data = collectDataset(env as never, tabularExpert(env as never, 0.99, 0.3), { episodes: 5, seed: 3 })
    const episodes = datasetEpisodes(data)
    expect(episodes).toHaveLength(5)
    for (const e of episodes) {
      const last = e[e.length - 1]
      expect(last?.terminated === true || last?.truncated === true).toBe(true)
      expect(e.slice(0, -1).every((t) => !t.terminated && !t.truncated)).toBe(true)
    }
    expect(collectDataset(env as never, () => 0, { steps: 150 }).length).toBeGreaterThanOrEqual(150)
    expect(collectDataset(env as never, () => 0, { episodes: 2 })).toEqual(
      collectDataset(env as never, () => 0, { episodes: 2 }),
    )
  })

  it('runs a learner as a traced algorithm over minibatches', () => {
    const env = lake()
    const data = collectDataset(env as never, tabularExpert(env as never, 0.99, 0.3), { episodes: 10 })
    const rec = recorder()
    const result = runAlgorithm(
      offlineAlgorithm(cqlLearner(env as never, { hidden: [8] }), data, { steps: 5, batchSize: 8 }),
      {
        maxSteps: 100,
        tracer: rec,
        policy: tracePolicy({ metrics: ['loss', 'regulariser'] }),
      },
    )
    expect(result.steps).toBe(5)
    expect(result.status).toBe('completed')
    const names = rec
      .run()
      .events.filter((e) => e.type === 'metric')
      .map((e) => (e as { name: string }).name)
    expect(new Set(names)).toEqual(new Set(['loss', 'regulariser']))
    expect(() => offlineAlgorithm(behaviourCloningLearner(env as never), [])).toThrow(/empty/)
  })
})

describe('offline learners on discrete tasks', () => {
  it('behaviour cloning recovers the expert from noisy demonstrations, as an agent too', { timeout: 60_000 }, () => {
    const env = lake()
    const data = collectDataset(env as never, tabularExpert(env as never, 0.99, 0.3), { episodes: 50 })
    const r = behaviourCloning(env as never, data, options)
    expect(evaluatePolicy(env as never, r.policy as never, { episodes: 1 }).meanReturn).toBe(1)
    expect(evaluatePolicy(env as never, greedyPolicyOf(r.agent, r.state) as never, { episodes: 1 }).meanReturn).toBe(1)
    // The agent view works with the online machinery and ignores what it is shown.
    const online = trainAgent(env as never, r.agent as never, { maxSteps: 20, seed: 1 })
    expect(online.agent).toBe(r.state)
    expect(r.agent.act(r.state, data[0]?.observation as never, key(0))).toBeTypeOf('number')
  })

  it('CQL and BCQ follow the data where offline DQN extrapolates off the cliff', { timeout: 60_000 }, () => {
    const env = cliff()
    const data = collectDataset(env as never, tabularExpert(env as never, 0.99, 0.1), { episodes: 20 })
    const score = (p: (o: never, k: never) => number) =>
      evaluatePolicy(env as never, p as never, { episodes: 1 }).meanReturn
    const conservative = score(cql(env as never, data, options).policy as never)
    const constrained = score(bcq(env as never, data, options).policy as never)
    const naive = score(offlineDqn(env as never, data, options).policy as never)
    expect(conservative).toBe(-13)
    expect(constrained).toBe(-13)
    expect(naive).toBeLessThan(-100)
  })

  it('CQL solves the lake from noisy demonstrations', { timeout: 60_000 }, () => {
    const env = lake()
    const data = collectDataset(env as never, tabularExpert(env as never, 0.99, 0.3), { episodes: 50 })
    const r = cql(env as never, data, { ...options, alpha: 1 })
    expect(evaluatePolicy(env as never, r.policy as never, { episodes: 1 }).meanReturn).toBe(1)
    expect(r.state.regulariser).toBeGreaterThan(0)
  })
})
