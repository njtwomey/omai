import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { make, oneHotObservation } from '../../../lib/rl/index.ts'
import { reach1d, reachExpert, tabularExpert } from '../../../testing/rl-offline.ts'
import { frozenLake } from '../environments/index.ts'
import {
  decisionTransformer,
  decisionTransformerAgent,
  decisionTransformerLearner,
  evaluateInContext,
} from './decision-transformer.ts'
import { collectDataset } from './offline.ts'

const model = { context: 4, embedding: 16, heads: 2, batchSize: 16, learningRate: 3e-3 } as const

describe('decision transformers', () => {
  it('conditioned on the best return, act like the successful episodes of a noisy dataset', { timeout: 60_000 }, () => {
    const env = make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })
    const data = collectDataset(env as never, tabularExpert(env as never, 0.99, 0.5), { episodes: 50 })
    const r = decisionTransformer(env as never, data, { ...model, steps: 150 })
    expect(r.state.targetReturn).toBe(1)
    expect(evaluateInContext(env as never, r.agent as never, { episodes: 2 }).meanReturn).toBe(1)
    expect(r.state.loss).toBeLessThan(1.2)
  })

  it('keeps the episode context and the return-to-go, and handles continuous actions', () => {
    const env = make(reach1d)
    const data = collectDataset(env, reachExpert(0.3), { episodes: 4 })
    const learner = decisionTransformerLearner<readonly number[], readonly number[]>(env as never, {
      ...model,
      embedding: 8,
    })
    let s = learner.init(key(0), data)
    expect(s.maxTimestep).toBe(10)
    for (let i = 0; i < 3; i++) s = learner.update(s, data.slice(0, 4), key(i))
    expect(Number.isFinite(s.loss)).toBe(true)
    const agent = decisionTransformerAgent<readonly number[], readonly number[]>(env as never, s, {
      ...model,
      embedding: 8,
      targetReturn: -2,
    })
    let c = agent.init(key(0))
    const t = data[0]
    if (t === undefined) throw new Error('no data')
    for (let i = 0; i < 6; i++) c = agent.learn(c, t, key(i))
    expect(c.history).toHaveLength(3)
    expect(c.t).toBe(6)
    expect(c.returnToGo).toBeCloseTo(-2 - 6 * t.reward, 12)
    const a = agent.act(c, t.observation, key(9))
    expect(a).toHaveLength(1)
    expect(Math.abs(a[0] ?? 2)).toBeLessThanOrEqual(1)
    expect(agent.learn(c, { ...t, truncated: true }, key(0)).history).toHaveLength(0)
  })
})
