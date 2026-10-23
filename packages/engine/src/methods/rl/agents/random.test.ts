import { describe, expect, it } from 'vitest'
import { key } from '../../../lib/random/index.ts'
import { make, trainAgent } from '../../../lib/rl/index.ts'
import { contains } from '../../../lib/space/index.ts'
import { blackjack, cartPole, cliffWalking, pendulum } from '../environments/index.ts'
import { constantAgent, randomAgent } from './random.ts'

describe('random and constant agents', () => {
  it('draws uniform discrete actions from the step key', () => {
    const agent = randomAgent(cliffWalking.create())
    const counts = [0, 0, 0, 0]
    for (let i = 0; i < 4000; i++) {
      const a = agent.act(null, 0, key(i))
      counts[a] = (counts[a] ?? 0) + 1
    }
    for (const c of counts) expect(Math.abs(c - 1000)).toBeLessThan(120)
    expect(agent.act(null, 0, key(7))).toBe(agent.act(null, 0, key(7)))
    expect(agent.greedy(null, 0)).toBe(agent.greedy(null, 5))
  })

  it('samples box actions inside their bounds', () => {
    const env = pendulum.create()
    const agent = randomAgent(env)
    for (let i = 0; i < 20; i++) expect(contains(env.actionSpace, agent.act(null, [1, 0, 0], key(i)))).toBe(true)
  })

  it('runs under trainAgent and never changes its state', () => {
    const env = make(cartPole)
    const random = trainAgent(env, randomAgent(env), { episodes: 3, seed: 0 })
    expect(random.returns).toHaveLength(3)
    expect(random.agent).toBeNull()
    const stick = constantAgent(0)
    const r = trainAgent(blackjack.create(), stick, { episodes: 5, seed: 1 })
    expect(r.lengths.every((n) => n === 1)).toBe(true)
    expect(stick.greedy(null, [12, 3, 0])).toBe(0)
    expect(stick.learn(null, {} as never, key(0))).toBeNull()
  })
})
