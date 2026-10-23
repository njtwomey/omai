import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { box } from '../space/index.ts'
import { add, concat, mul, type Value } from '../tensor/index.ts'
import {
  countBonus,
  countTable,
  countVisit,
  curiosityBonus,
  pseudoCount,
  rndBonus,
  rndLoss,
  simHash,
  visits,
} from './exploration.ts'
import { type DynamicsModel, ensemblePredict, ensembleSample, modelEnv, modelTargets } from './models.ts'

// next = obs + w * action, reward = obs; probabilistic members also predict a log-variance.
const linear: DynamicsModel<number> = {
  observationSize: 1,
  actionSize: 1,
  predict: (w, o, a) => ({ mean: concat([mul(a, w), o], -1) }),
}
const noisy: DynamicsModel<number> = {
  ...linear,
  predict: (w, o, a) => ({
    mean: concat([mul(a, w), o], -1),
    logVar: mul(add(mul(o, 0), Math.log(0.01)), concat([mul(a, 0), mul(a, 0)], -1)) as Value,
  }),
}

describe('learned dynamics models', () => {
  it('decompose ensemble uncertainty into epistemic and aleatoric parts', () => {
    const r = ensemblePredict({ model: linear, members: [1, 3] }, [[0], [1]], [[1], [2]])
    expect(r.mean).toEqual([
      [2, 0],
      [4, 1],
    ])
    expect(r.epistemic[1]?.[0]).toBe(4)
    expect(r.aleatoric[0]).toEqual([0, 0])
    expect(() => ensemblePredict({ model: linear, members: [] }, [[0]], [[0]])).toThrow(/no members/)
  })

  it('sample next states by trajectory sampling and build targets', () => {
    const samples = ensembleSample(
      { model: linear, members: [1, 2, 3] },
      [[0], [0], [0], [0]],
      [[1], [1], [1], [1]],
      key(0),
    )
    for (const s of samples) expect([1, 2, 3]).toContain(s.observation[0])
    const fixed = ensembleSample({ model: linear, members: [1, 2, 3] }, [[0]], [[1]], key(0), { member: 2 })
    expect(fixed[0]?.observation).toEqual([3])
    const det = ensembleSample({ model: noisy, members: [1] }, [[0]], [[1]], key(0), { deterministic: true })
    expect(det[0]?.observation).toEqual([1])
    const t = {
      observation: [1, 2],
      action: 3,
      reward: 4,
      nextObservation: [2, 0],
      terminated: false,
      truncated: false,
    }
    expect(modelTargets([t])).toEqual({ observations: [[1, 2]], actions: [[3]], targets: [[1, -2, 4]] })
  })

  it('wrap an ensemble as an environment with known reward and termination', () => {
    const env = modelEnv(
      { model: linear, members: [1] },
      {
        observationSpace: box(-10, 10, [1]),
        actionSpace: box(-1, 1, [1]),
        initial: () => [0],
        reward: (_, __, next) => -Math.abs((next[0] ?? 0) - 1),
        terminated: (o) => (o[0] ?? 0) >= 1,
        maxEpisodeSteps: 3,
      },
    )
    const s0 = env.reset(key(0)).state
    const s1 = env.step(s0, [0.5], key(1))
    expect(s1.reward).toBe(-0.5)
    const s2 = env.step(s1.state, [0.5], key(2))
    expect(s2.terminated).toBe(true)
    expect(env.step(env.step(s1.state, [0], key(3)).state, [0], key(4)).truncated).toBe(true)
  })
})

describe('exploration bonuses', () => {
  it('count visits and pay bonuses that decay with the count', () => {
    let t = countTable()
    for (const s of ['a', 'a', 'b']) t = countVisit(t, s)
    expect([visits(t, 'a'), visits(t, 'b'), visits(t, 'c'), t.total]).toEqual([2, 1, 0, 3])
    expect(countBonus(0)).toBe(1)
    expect(countBonus(9, 3)).toBe(1)
    expect(pseudoCount(0.5, 0.5)).toBe(Infinity)
  })

  it('hash nearby continuous observations together', () => {
    const h = simHash(key(1), 3, 12)
    expect(h([1, 0, 0])).toHaveLength(12)
    expect(h([1, 0, 0])).not.toBe(h([-1, 0, 0]))
    expect(() => simHash(key(0), 2, 0)).toThrow(/bit/)
  })

  it('compute RND and curiosity bonuses and losses', () => {
    const rnd = { target: (x: Value) => mul(x, 2), predictor: (w: number, x: Value) => mul(x, w) }
    expect(rndBonus(rnd, 2, [[1, 2]])).toEqual([0])
    expect(rndBonus(rnd, 0, [[1], [2]])).toEqual([4, 16])
    const icm = {
      encode: (_: null, x: Value) => x,
      forward: (_: null, f: Value, a: Value) => add(f, a),
      inverse: (_: null, f: Value) => f,
    }
    expect(curiosityBonus(icm, null, [[0], [1]], [[1], [1]], [[1], [0]], 2)).toEqual([0, 4])
    expect(rndLoss(rnd, 2, [[1]]).shape).toEqual([])
  })
})
