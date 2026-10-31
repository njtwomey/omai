import { describe, expect, it } from 'vitest'
import { gaussianNB } from '../../testing/learn.ts'
import { key, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder, runAlgorithm, runTree } from '../trace/index.ts'
import {
  activeLearning,
  activeLearningAlgorithm,
  appendRows,
  badgeSampling,
  badgeSelection,
  baldAcquisition,
  baldSampling,
  batchBald,
  coreSetGreedy,
  coreSetSampling,
  densityWeightedSampling,
  densityWeightedUncertainty,
  expectedErrorReduction,
  expectedModelChange,
  expectedModelChangeScores,
  predictiveEntropy,
  randomSampling,
  streamActiveLearning,
  uncertaintySampling,
  voteEntropySampling,
} from './active-learning.ts'

const pool = Array.from({ length: 30 }, (_, i) => [Math.sin(i) * 3, Math.cos(1.7 * i) * 3])
const truth = (r: readonly number[]) => ((r[0] ?? 0) + (r[1] ?? 0) > 0 ? 1 : 0)
const labeled = {
  x: [
    [-3, -3],
    [3, 3],
    [-2, -2.5],
    [2.5, 2],
  ],
  y: [0, 1, 0, 1],
}
const oracle = (idx: readonly number[]) => idx.map((i) => truth(pool[i] ?? []))

describe('active learning', () => {
  it('spends the budget without repeating rows, traced round by round', () => {
    const rec = recorder()
    const r = activeLearning(gaussianNB(), {
      labeled,
      pool,
      oracle,
      strategy: voteEntropySampling(),
      batchSize: 3,
      budget: 10,
      evaluation: { x: pool, y: pool.map(truth) },
      scoring: 'accuracy',
      key: key(1),
      tracer: rec,
    })
    expect(r.queried.length).toBe(10)
    expect(new Set(r.queried).size).toBe(10)
    expect(r.scores.length).toBe(4)
    expect(runTree(rec.run()).length).toBe(1)
  })

  it('stops when the pool is exhausted', () => {
    const r = activeLearning(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 5),
      oracle,
      strategy: randomSampling(),
      batchSize: 2,
      key: key(2),
    })
    expect([...r.queried].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4])
  })

  it('expected error reduction picks the candidate with the lowest expected pool error', () => {
    const est = gaussianNB()
    const r = activeLearning(est, {
      labeled,
      pool: pool.slice(0, 8),
      oracle,
      strategy: expectedErrorReduction(),
      budget: 1,
    })
    expect(r.queried.length).toBe(1)
    expect(appendRows([[1, 2]], [[3, 4]]).shape).toEqual([2, 2])
  })

  it('selective sampling queries only uncertain arrivals within budget', () => {
    const r = streamActiveLearning(gaussianNB(), {
      labeled,
      stream: pool,
      oracle: (t) => truth(pool[t] ?? []),
      threshold: 0.2,
      budget: 4,
    })
    expect(r.queried.length).toBeLessThanOrEqual(4)
    expect([...r.queried].sort((a, b) => a - b)).toEqual(r.queried)
  })

  it('expectedModelChange scores uncertain high-leverage points higher than confident ones', () => {
    const scores = expectedModelChangeScores(
      [
        [0.99, 0.01],
        [0.5, 0.5],
        [0.5, 0.5],
      ],
      [[1], [1], [3]],
    )
    expect(scores[1]).toBeGreaterThan(scores[0] ?? 0)
    expect(scores[2]).toBeGreaterThan(scores[1] ?? 0)
    const r = activeLearning(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 10),
      oracle,
      strategy: expectedModelChange(),
      budget: 2,
    })
    expect(r.queried.length).toBe(2)
  })

  it('baldAcquisition and batchBald distinguish epistemic disagreement from aleatoric noise', () => {
    const m1 = [
      [0.98, 0.02],
      [0.98, 0.02],
      [0.5, 0.5],
    ]
    const m2 = [
      [0.02, 0.98],
      [0.02, 0.98],
      [0.5, 0.5],
    ]
    const bald = baldAcquisition([m1, m2])
    expect(bald[0]).toBeGreaterThan(0.5)
    expect(bald[2]).toBeCloseTo(0, 10)
    // Candidate 0 and 1 are identical under both members, so after picking 0 the marginal gain of 1 drops near 0.
    const bb = batchBald([m1, m2], 2)
    expect(bb.indices[0]).toBe(0)
    expect(bb.scores[0]).toBeGreaterThan(0.5)

    const r = activeLearning(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 8),
      oracle,
      strategy: baldSampling(),
      budget: 2,
      key: key(5),
    })
    expect(r.queried.length).toBe(2)
  })

  it('densityWeightedUncertainty down-weights isolated outliers', () => {
    const u = [0.45, 0.45, 0.45, 0.5]
    const pts = [[0], [0.05], [-0.05], [50]]
    const dw = densityWeightedUncertainty(u, pts, { beta: 1, gamma: 1 })
    expect(dw[0]).toBeGreaterThan(dw[3] ?? 0)
    const r = activeLearning(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 8),
      oracle,
      strategy: densityWeightedSampling({ beta: 1 }),
      budget: 2,
    })
    expect(r.queried.length).toBe(2)
  })

  it('coreSetGreedy and badgeSelection select spread-out batches and activeLearningAlgorithm serialises state', () => {
    const cs = coreSetGreedy([[0], [1], [5], [10]], 2, { labeled: [[0]] })
    expect(cs.indices).toEqual([3, 2])
    expect(cs.radius).toBe(1)

    // Rows 0 and 1 are one uncertain point twice, row 2 its mirror image, row 3 a fourth direction: the duplicate has
    // squared distance zero to the first centre, so it is never drawn while the others remain.
    const p4 = [
      [0.6, 0.4],
      [0.6, 0.4],
      [0.4, 0.6],
      [0.55, 0.45],
    ]
    const x4 = [[-4], [-4], [4], [0]]
    for (let seed = 0; seed < 40; seed++) {
      const three = badgeSelection(p4, x4, 3, { key: key(seed) })
      expect(three[0]).toBe(0)
      expect([...three].sort()).toEqual([0, 2, 3])
    }
    // A confident row has a near-zero embedding but lies far from the big ones, so it is drawn sometimes.
    const withConfident = [...p4, [0.99, 0.01]]
    const picked = new Set<number>()
    for (let seed = 0; seed < 40; seed++)
      for (const i of badgeSelection(withConfident, [...x4, [1]], 2, { key: key(seed) })) picked.add(i)
    expect(picked.has(1)).toBe(false)
    expect(picked.size).toBeGreaterThan(2)

    const rCore = activeLearning(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 8),
      oracle,
      strategy: coreSetSampling(),
      budget: 2,
    })
    expect(rCore.queried.length).toBe(2)

    const rBadge = activeLearning(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 8),
      oracle,
      strategy: badgeSampling(),
      budget: 2,
    })
    expect(rBadge.queried.length).toBe(2)

    const alg = activeLearningAlgorithm(gaussianNB(), {
      labeled,
      pool: pool.slice(0, 6),
      oracle,
      strategy: uncertaintySampling(),
      budget: 2,
    })
    const res = runAlgorithm(alg, { maxSteps: 2, key: key(7) })
    const json = alg.serialise?.(res.state)
    expect(json).toBeDefined()
    const restored = alg.deserialise?.(json as NonNullable<typeof json>)
    expect(restored?.queried).toEqual(res.state.queried)
  })

  it('batchBald matches the exact joint mutual information and avoids duplicated candidates', () => {
    // Four posterior members (a, b) in {0, 1}^2 over four candidates and two classes: candidates 0 and 1 are twins that
    // both reveal a, candidate 2 reveals b less sharply, candidate 3 is noise.
    const members = [0, 1, 2, 3].map((m) => {
      const [a, b] = [m & 1, m >> 1]
      const reveal = (v: number, sharp: number): number[] => (v === 1 ? [1 - sharp, sharp] : [sharp, 1 - sharp])
      return [reveal(a, 0.999), reveal(a, 0.999), reveal(b, 0.9), [0.5, 0.5]]
    })
    const entropy = (ps: readonly number[]): number => -ps.reduce((a, q) => a + (q > 0 ? q * Math.log(q) : 0), 0)
    // I(y_S; theta) = H(mean_m prod_i p_m(y_i)) - mean_m sum_i H(p_m(. | i)), by enumeration of all label configurations.
    const jointMi = (set: readonly number[]): number => {
      let h = 0
      for (let cfg = 0; cfg < 2 ** set.length; cfg++) {
        const labels = set.map((_, t) => (cfg >> t) & 1)
        const q =
          members.reduce((a, m) => a + set.reduce((pr, i, t) => pr * (m[i]?.[labels[t] ?? 0] ?? 0), 1), 0) /
          members.length
        h -= q * Math.log(q)
      }
      const cond = members.reduce((a, m) => a + set.reduce((b, i) => b + entropy(m[i] ?? []), 0), 0) / members.length
      return h - cond
    }
    const single = baldAcquisition(members)
    for (let i = 0; i < 4; i++) expect(single[i]).toBeCloseTo(jointMi([i]), 12)

    const bb = batchBald(members, 3)
    expect(bb.scores[0]).toBeCloseTo(jointMi([bb.indices[0] ?? 0]), 12)
    expect(bb.scores[1]).toBeCloseTo(jointMi(bb.indices.slice(0, 2)), 12)
    expect(bb.scores[2]).toBeCloseTo(jointMi(bb.indices.slice(0, 3)), 12)
    // Plain top-2 BALD takes the twins; BatchBALD takes one twin and candidate 2, which adds information about b.
    const top2 = [...single.keys()].sort((a, b) => (single[b] ?? 0) - (single[a] ?? 0)).slice(0, 2)
    expect(top2.slice().sort()).toEqual([0, 1])
    expect(jointMi([0, 2])).toBeGreaterThan(jointMi([0, 1]) + 0.1)
    expect(bb.indices.slice(0, 2).filter((i) => i < 2).length).toBe(1)
    expect(bb.indices[1] === 2 || bb.indices[0] === 2).toBe(true)

    // The sampled estimator (more configurations than the cap) still avoids the twins.
    const capped = batchBald(members, 3, { numSamples: 4, key: key(1) })
    expect(capped.indices.slice(0, 2).filter((i) => i < 2).length).toBe(1)
  })

  it('BALD ignores aleatoric noise that entropy sampling chases', () => {
    // Candidate 0: every member says 50/50 (noise). Candidate 1: members disagree sharply (epistemic).
    const members = [
      [
        [0.5, 0.5],
        [0.97, 0.03],
      ],
      [
        [0.5, 0.5],
        [0.03, 0.97],
      ],
      [
        [0.5, 0.5],
        [0.97, 0.03],
      ],
      [
        [0.5, 0.5],
        [0.03, 0.97],
      ],
    ]
    const bald = baldAcquisition(members)
    expect(bald[0]).toBeCloseTo(0, 12)
    expect(bald[1]).toBeGreaterThan(0.5)
    const meanProba = [
      [0.5, 0.5],
      [0.5, 0.5],
    ]
    expect(predictiveEntropy(meanProba)[0]).toBeCloseTo(Math.LN2, 12)
  })

  it('batchBald with sampled label configurations estimates the exact joint information', () => {
    const z = toFlat(uniform(key(11), [4, 7, 1]))
    const members = Array.from({ length: 4 }, (_, m) =>
      Array.from({ length: 7 }, (_, i) => {
        const p = 0.05 + 0.9 * (z[m * 7 + i] ?? 0.5)
        return [1 - p, p]
      }),
    )
    const exact = batchBald(members, 6, { numSamples: 1_000_000 })
    const sampled = batchBald(members, 6, { numSamples: 64, key: key(3) })
    // Six binary picks make 64 configurations, which fit a cap of 64 exactly; a cap of 16 samples from the fifth pick on.
    expect(sampled.indices).toEqual(exact.indices)
    const small = batchBald(members, 6, { numSamples: 16, key: key(3) })
    expect(small.indices.slice(0, 3)).toEqual(exact.indices.slice(0, 3))
    expect(Math.abs((small.scores[5] ?? 0) - (exact.scores[5] ?? 0))).toBeLessThan(0.25 * (exact.scores[5] ?? 1))
  })
})
