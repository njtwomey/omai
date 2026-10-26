import { describe, expect, it } from 'vitest'
import { riverSwim } from '../environments/index.ts'
import { key } from '../random/index.ts'
import {
  evaluatePolicy,
  greedyPolicyOf,
  make,
  oneHotObservation,
  policyAgent,
  type Transition,
  trainAgent,
} from '../rl/index.ts'
import { box, discrete } from '../space/index.ts'
import { type BonusModule, countBonusModule, icmModule, intrinsicReward, rndModule } from './exploration.ts'
import { dqn } from './value-based.ts'

// RiverSwim with one-hot observations and little epsilon-greedy exploration: resting on the near bank pays 0.005 a
// step, swimming against the current to the far bank pays 1 a step.
const river = () => make(riverSwim, {}, { wrappers: [(e) => oneHotObservation(e as never) as never] })
const base = {
  hidden: [32],
  learningRate: 5e-3,
  gamma: 0.95,
  batchSize: 32,
  learningStarts: 64,
  targetUpdateInterval: 100,
  explorationSteps: 1000,
  epsilonStart: 0.1,
  epsilonEnd: 0.01,
  bufferSize: 5000,
} as const

const oneHot = (i: number, n = 6): number[] => Array.from({ length: n }, (_, j) => (j === i ? 1 : 0))
const step = (s: number, a: number, s2: number): Transition<readonly number[], number> => ({
  observation: oneHot(s),
  action: a,
  reward: 0,
  nextObservation: oneHot(s2),
  terminated: false,
  truncated: false,
})

// Feed one transition repeatedly, returning the bonuses it earned.
function repeatBonus<B>(
  m: BonusModule<B, readonly number[], number>,
  t: Transition<readonly number[], number>,
  n: number,
) {
  let s = m.init(key(0))
  const bonuses: number[] = []
  for (let i = 0; i < n; i++) {
    const u = m.update(s, t, key(i + 1))
    s = u.state
    bonuses.push(u.bonus)
  }
  return { state: s, bonuses }
}

describe('intrinsic rewards', () => {
  it('DQN with a count bonus reaches the far bank where epsilon-greedy DQN settles for the near one', {
    timeout: 60_000,
  }, () => {
    const env = river()
    const run = (agent: Parameters<typeof trainAgent>[1]) => {
      const r = trainAgent(env as never, agent, { maxSteps: 2000, seed: 0 })
      return {
        training: r.state.returns.reduce((a, b) => a + b, 0) + r.state.episodeReturn,
        greedy: evaluatePolicy(env as never, greedyPolicyOf(agent, r.agent as never), { episodes: 3 }).meanReturn,
      }
    }
    const plain = run(dqn(env as never, base))
    const counted = run(intrinsicReward(dqn(env as never, base), countBonusModule(env as never), { scale: 0.1 }))
    expect(counted.training).toBeGreaterThan(10 * plain.training)
    expect(counted.greedy).toBeGreaterThan(40)
    expect(plain.greedy).toBeLessThan(1)
  })

  it('counts states, pairs or SimHash codes and decays its scale', () => {
    const env = river()
    const pairs = repeatBonus(countBonusModule(env as never, { beta: 2 }), step(0, 1, 1), 4)
    expect(pairs.bonuses).toEqual([2, 2 / Math.SQRT2, 2 / Math.sqrt(3), 1])
    const states = countBonusModule(env as never, { on: 'next-state', hash: { bits: 8 } })
    const a = states.update(states.init(key(0)), step(0, 0, 3), key(1)).state
    expect(states.bonus(a, step(5, 1, 3))).toBeCloseTo(1 / Math.SQRT2, 15)
    expect(states.bonus(a, step(5, 1, 4))).toBe(1)
    const tabular = countBonusModule({ observationSpace: discrete(3), actionSpace: discrete(2) }, { on: 'state' })
    expect(tabular.update(tabular.init(key(0)), { ...step(0, 0, 0), observation: 2 } as never, key(1)).bonus).toBe(1)

    const wrapped = intrinsicReward(
      policyAgent(() => 1),
      countBonusModule(env as never),
      { scale: 0.5, decay: 0.5 },
    )
    let s = wrapped.init(key(0))
    for (let i = 0; i < 3; i++) s = wrapped.learn(s, step(0, 1, 1), key(i))
    expect(wrapped.stats?.(s)).toEqual({ distinct: 1, intrinsic: 1 / Math.sqrt(3), intrinsicScale: 0.125 })
    expect(wrapped.act(s, oneHot(0), key(0))).toBe(1)
  })

  it('RND bonuses fall on a repeatedly visited state and stay high on novel ones', () => {
    const env = river()
    for (const options of [{}, { normalizeObservations: false, normalizeRewards: false }]) {
      const m = rndModule(env as never, { features: 8, hidden: [16], batchSize: 8, learningRate: 1e-2, ...options })
      const { state, bonuses } = repeatBonus(m, step(0, 1, 2), 150)
      expect(bonuses[149] ?? 0).toBeLessThan(0.2 * (bonuses[10] ?? 0))
      expect(m.bonus(state, step(0, 1, 5))).toBeGreaterThan(5 * m.bonus(state, step(0, 1, 2)))
      expect(m.stats?.(state)['rndLoss']).toBeGreaterThan(0)
    }
  })

  it('ICM bonuses fall on a repeated transition, for discrete and continuous actions', () => {
    const env = river()
    const m = icmModule(env as never, { features: 8, hidden: [16], batchSize: 8, learningRate: 1e-2 })
    const { state, bonuses } = repeatBonus(m, step(1, 1, 2), 150)
    expect(bonuses[149] ?? 0).toBeLessThan(0.2 * (bonuses[10] ?? 0))
    expect(m.stats?.(state)['inverseLoss']).toBeLessThan(0.7)
    const continuous = icmModule<readonly number[], readonly number[]>(
      { observationSpace: box(0, 1, [6]), actionSpace: box(-1, 1, [2]) },
      { features: 8, hidden: [16], batchSize: 8, learningRate: 1e-2 },
    )
    const t = { ...step(1, 0, 2), action: [0.5, -0.5] }
    let c = continuous.init(key(0))
    const first = continuous.bonus(c, t)
    for (let i = 0; i < 100; i++) c = continuous.update(c, t, key(i)).state
    expect(continuous.bonus(c, t)).toBeLessThan(0.2 * first)
  })

  it('works with any agent through trainAgent and reports the bonus', () => {
    const env = river()
    const agent = intrinsicReward(
      dqn(env as never, { ...base, learningStarts: 32 }),
      rndModule(env as never, { features: 8 }),
    )
    const r = trainAgent(env as never, agent, { maxSteps: 100, seed: 1 })
    const stats = agent.stats?.(r.agent) ?? {}
    expect(Object.keys(stats).sort()).toEqual(
      ['epsilon', 'intrinsic', 'intrinsicScale', 'loss', 'qValue', 'rndError', 'rndLoss', 'tdError'].sort(),
    )
    expect(agent.metrics?.['intrinsic']).toBeDefined()
  })
})
