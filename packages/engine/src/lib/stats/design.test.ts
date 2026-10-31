import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  aliasStructure,
  blockedDesign,
  fractionalFactorialDesign,
  fullFactorialDesign,
  latinSquareDesign,
  matchedPairsDesign,
  minimizationAlgorithm,
  minimizationAllocation,
  oneWayAnova,
  permutedBlockRandomisation,
  simpleRandomisation,
  stratifiedRandomisation,
  twoWayAnova,
} from './design.ts'

const counts = (arms: ArrayLike<number>, k: number): number[] => {
  const c = new Array<number>(k).fill(0)
  for (let i = 0; i < arms.length; i++) c[arms[i] ?? 0] = (c[arms[i] ?? 0] ?? 0) + 1
  return c
}

describe('randomisation', () => {
  it('exact simple randomisation hits the quotas and is reproducible', () => {
    const a = simpleRandomisation(key(3), 10, { nArms: 3, probabilities: [0.5, 0.3, 0.2], exact: true })
    expect(counts(a.data, 3)).toEqual([5, 3, 2])
    expect(
      Array.from(simpleRandomisation(key(3), 10, { nArms: 3, probabilities: [0.5, 0.3, 0.2], exact: true }).data),
    ).toEqual(Array.from(a.data))
  })

  it('independent simple randomisation follows the allocation probabilities', () => {
    const a = simpleRandomisation(key(1), 20000, { probabilities: [0.25, 0.75] })
    expect((counts(a.data, 2)[0] ?? 0) / 20000).toBeCloseTo(0.25, 1)
  })

  it('permuted blocks balance every block, including a 2:1 ratio', () => {
    const arms = permutedBlockRandomisation(key(5), 24, { ratios: [2, 1], blockSize: 6 })
    for (let b = 0; b < 4; b++) expect(counts(arms.data.slice(b * 6, b * 6 + 6), 2)).toEqual([4, 2])
    expect(blockedDesign(key(5), 24, { ratios: [2, 1], blockSize: 6 }).data).toEqual(arms.data)
    expect(() => permutedBlockRandomisation(key(0), 8, { blockSize: 3 })).toThrow()
  })

  it('stratified randomisation balances within each stratum', () => {
    const strata = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    const arms = stratifiedRandomisation(key(2), strata, { blockSize: 2 })
    for (const s of [0, 1]) {
      const inStratum = Array.from(arms.data).filter((_, i) => strata[i] === s)
      expect(counts(inStratum, 2)).toEqual([inStratum.length / 2, inStratum.length / 2])
    }
  })
})

describe('minimization', () => {
  it('keeps the marginal imbalance small and beats simple randomisation on the same covariates', () => {
    const n = 60
    const cov = Array.from({ length: n }, (_, i) => [i % 2, (i * 7) % 3, (i % 5) % 2])
    const res = minimizationAllocation(cov, { key: key(1), pBiased: 1 })
    expect(res.assignments.shape).toEqual([n])
    // Deterministic minimisation keeps each level's two arms within one unit of each other.
    expect(res.imbalance).toBeLessThanOrEqual(3 * 3)
    expect(res.imbalanceHistory.shape).toEqual([n])
    const simple = simpleRandomisation(key(1), n, { exact: true })
    let simpleImbalance = 0
    for (let j = 0; j < 3; j++) {
      const byLevel = new Map<number, number[]>()
      for (let i = 0; i < n; i++) {
        const lvl = cov[i]?.[j] ?? 0
        const c = byLevel.get(lvl) ?? [0, 0]
        c[simple.data[i] ?? 0] = (c[simple.data[i] ?? 0] ?? 0) + 1
        byLevel.set(lvl, c)
      }
      for (const c of byLevel.values()) simpleImbalance += Math.abs((c[0] ?? 0) - (c[1] ?? 0))
    }
    expect(res.imbalance).toBeLessThanOrEqual(simpleImbalance)
  })

  it('is an Algorithm whose state serialises and whose run resumes to the same allocation', () => {
    const cov = [[0], [0], [1], [1], [0], [1], [0], [1], [1], [0]]
    const algo = minimizationAlgorithm(cov, { pBiased: 0.8 })
    const full = runAlgorithm(algo, { seed: 4, maxSteps: 10 })
    expect(full.status).toBe('completed')
    expect(full.steps).toBe(10)
    const part = runAlgorithm(algo, { seed: 4, maxSteps: 4 })
    const json = algo.serialise?.(part.state)
    expect(json).toBeDefined()
    const restored = algo.deserialise?.(json ?? null)
    const resumed = runAlgorithm(algo, { seed: 4, maxSteps: 6, start: { state: restored, step: 4 } })
    expect(Array.from(resumed.state.assignments)).toEqual(Array.from(full.state.assignments))
  })
})

describe('matched pairs', () => {
  it('finds the minimum-weight perfect matching, not the greedy one', () => {
    // Greedy takes the closest pair (1, 2) first and then pays 6 for (0, 3); the optimum is (0,1) + (2,3) = 2 + 2.
    const res = matchedPairsDesign([0, 2, 4.1, 6.1], { distance: 'euclidean' })
    expect(res.pairs).toEqual([
      [0, 1],
      [2, 3],
    ])
    const treated = Array.from(res.assignments.data)
    expect((treated[0] ?? 0) + (treated[1] ?? 0)).toBe(1)
    expect((treated[2] ?? 0) + (treated[3] ?? 0)).toBe(1)
    expect(res.meanDistance).toBeGreaterThan(0)
  })
})

describe('factorial designs', () => {
  it('builds the full factorial grid', () => {
    const x = fullFactorialDesign(3)
    expect(x.shape).toEqual([8, 3])
    // Every column is balanced and every pair of columns is orthogonal.
    for (let a = 0; a < 3; a++) {
      let s = 0
      for (let r = 0; r < 8; r++) s += x.data[r * 3 + a] ?? 0
      expect(s).toBe(0)
      for (let b = a + 1; b < 3; b++) {
        let dot = 0
        for (let r = 0; r < 8; r++) dot += (x.data[r * 3 + a] ?? 0) * (x.data[r * 3 + b] ?? 0)
        expect(dot).toBe(0)
      }
    }
    expect(fullFactorialDesign([2, 3]).shape).toEqual([6, 2])
  })

  it('derives the defining relation, resolution and aliases of a 2^(5-2) design', () => {
    const info = aliasStructure(5, ['D=AB', 'E=AC'])
    expect([...info.definingRelation].sort()).toEqual(['ABD', 'ACE', 'BCDE', 'I'].sort())
    expect(info.resolution).toBe(3)
    expect(info.aliases['A']).toEqual(expect.arrayContaining(['BD', 'CE']))
    const half = aliasStructure(4, ['D=ABC'])
    expect(half.resolution).toBe(4)
    expect(half.aliases['AB']).toEqual(['CD'])
  })

  it('generates a fraction whose generated column is the product of its base columns', () => {
    const ff = fractionalFactorialDesign(4, ['D=ABC'])
    expect(ff.design.shape).toEqual([8, 4])
    for (let r = 0; r < 8; r++) {
      const row = Array.from(ff.design.data.slice(r * 4, r * 4 + 4))
      expect(row[3]).toBe((row[0] ?? 0) * (row[1] ?? 0) * (row[2] ?? 0))
    }
  })

  it('draws Latin squares: every symbol once per row and column', () => {
    for (const k of [undefined, key(9)]) {
      const sq = latinSquareDesign(5, k === undefined ? {} : { key: k })
      for (let i = 0; i < 5; i++) {
        expect(new Set(Array.from(sq.data.slice(i * 5, i * 5 + 5))).size).toBe(5)
        expect(new Set(Array.from({ length: 5 }, (_, r) => sq.data[r * 5 + i])).size).toBe(5)
      }
    }
  })
})

describe('ANOVA', () => {
  it('reproduces the hand-computed one-way table', () => {
    const res = oneWayAnova([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ])
    expect(res.ssBetween).toBeCloseTo(54, 12)
    expect(res.ssWithin).toBeCloseTo(6, 12)
    expect(res.statistic).toBeCloseTo(27, 12)
    expect(res.etaSquared).toBeCloseTo(0.9, 12)
  })

  it('two-way types agree on a balanced design and the terms add up to the total', () => {
    const y = [2, 3, 4, 5.5, 6, 7.5, 8, 9, 3, 2.5, 5, 4.5, 7, 6.5, 9, 10]
    const a = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    const b = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    const t1 = twoWayAnova(y, a, b, { ssType: 'I' })
    const t2 = twoWayAnova(y, a, b, { ssType: 'II' })
    const t3 = twoWayAnova(y, a, b, { ssType: 'III' })
    expect(t1.factorA.ss).toBeCloseTo(t3.factorA.ss, 10)
    expect(t2.factorB.ss).toBeCloseTo(t3.factorB.ss, 10)
    const sum = t1.factorA.ss + t1.factorB.ss + (t1.interaction?.ss ?? 0) + t1.residual.ss
    expect(sum).toBeCloseTo(t1.total.ss, 10)
  })
})
