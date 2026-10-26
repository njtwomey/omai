import { describe, expect, it } from 'vitest'
import { reach1d, reachExpert, tabularExpert } from '../../testing/rl-offline.ts'
import { frozenLake } from '../environments/index.ts'
import { key } from '../random/index.ts'
import { evaluatePolicy, make, oneHotObservation } from '../rl/index.ts'
import { item, tensor } from '../tensor/index.ts'
import { behaviourCloning, collectDataset } from './offline.ts'
import { expectileLoss, iql, td3bc } from './offline-actor-critic.ts'

const options = { hidden: [32, 32], steps: 300, batchSize: 32, learningRate: 3e-3 } as const

describe('offline actor-critic methods', () => {
  it('the expectile loss weights positive residuals by tau', () => {
    expect(item(expectileLoss(tensor([2, -2]), 0.75))).toBe(2)
    expect(item(expectileLoss(tensor([1, -1]), 0.5))).toBe(0.5)
  })

  it('IQL extracts the expert from noisy demonstrations on the lake', { timeout: 60_000 }, () => {
    const env = make(frozenLake, { isSlippery: false }, { wrappers: [(e) => oneHotObservation(e as never) as never] })
    const data = collectDataset(env as never, tabularExpert(env as never, 0.99, 0.3), { episodes: 50 })
    const r = iql(env as never, data, { ...options, hidden: [32], gamma: 0.99 })
    expect(evaluatePolicy(env as never, r.policy as never, { episodes: 1 }).meanReturn).toBe(1)
    expect(r.agent.act(r.state, data[0]?.observation as never, key(0))).toBeTypeOf('number')
  })

  it('BC, Gaussian BC, IQL and TD3+BC beat a noisy behaviour policy on a continuous reach task', {
    timeout: 60_000,
  }, () => {
    const env = make(reach1d)
    const behaviour = reachExpert(0.5)
    const data = collectDataset(env, behaviour, { episodes: 30 })
    const score = (p: unknown) => evaluatePolicy(env, p as never, { episodes: 20 }).meanReturn
    const baseline = score(behaviour)
    const idle = score(() => [0])
    const results = {
      bc: score(behaviourCloning(env as never, data, options).policy),
      gaussian: score(behaviourCloning(env as never, data, { ...options, loss: 'gaussian' }).policy),
      iql: score(iql(env as never, data, options).policy),
      td3bc: score(td3bc(env as never, data, options).policy),
    }
    for (const [name, g] of Object.entries(results)) {
      expect(g, name).toBeGreaterThan(baseline)
      expect(g, name).toBeGreaterThan(idle + 3)
    }
    expect(results.td3bc).toBeGreaterThan(-0.6)
  })
})
