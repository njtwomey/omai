import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { foldIn, key } from '../random/index.ts'
import { add, matmul, tensor, toFlat, transpose, type Value } from '../tensor/index.ts'
import {
  conditionDpp,
  dppLogProbability,
  expectedSize,
  greedyMap,
  inclusionProbability,
  kDppLogProbability,
  marginalKernel,
  sampleDpp,
  sampleKDpp,
} from './dpp.ts'

const L = [
  [2, 0.8, 0.1, 0],
  [0.8, 1.5, 0.3, 0.2],
  [0.1, 0.3, 1, 0.6],
  [0, 0.2, 0.6, 0.8],
]

function subsets(n: number): number[][] {
  const out: number[][] = []
  for (let m = 0; m < 1 << n; m++) out.push(Array.from({ length: n }, (_, i) => i).filter((i) => (m >> i) & 1))
  return out
}

describe('L-ensembles', () => {
  it('subset probabilities sum to one and marginals agree with them', () => {
    const all = subsets(4)
    const p = all.map((a) => Math.exp(toFlat(dppLogProbability(L, a))[0] ?? 0))
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    // P(1, 2 in Y) by summing over supersets
    const both = all.reduce((s, a, i) => (a.includes(1) && a.includes(2) ? s + (p[i] ?? 0) : s), 0)
    expect(toFlat(inclusionProbability(L, [1, 2]))[0]).toBeCloseTo(both, 12)
    const sizes = all.reduce((s, a, i) => s + a.length * (p[i] ?? 0), 0)
    expect(expectedSize(L)).toBeCloseTo(sizes, 12)
    // repulsion: P(i, j) <= P(i) P(j)
    const k = toFlat(marginalKernel(L))
    expect(toFlat(inclusionProbability(L, [0, 1]))[0]).toBeLessThan((k[0] ?? 0) * (k[5] ?? 0))
  })

  it('exact samples have the marginal inclusion frequencies', () => {
    const runs = 3000
    const hits = new Float64Array(4)
    let pair = 0
    for (let r = 0; r < runs; r++) {
      const y = sampleDpp(foldIn(key(1), r), L)
      for (const i of y) hits[i] = (hits[i] ?? 0) + 1
      if (y.includes(2) && y.includes(3)) pair += 1
    }
    const k = toFlat(marginalKernel(L))
    for (let i = 0; i < 4; i++) {
      const want = k[i * 5] ?? 0
      expect(Math.abs((hits[i] ?? 0) / runs - want)).toBeLessThan(4 * Math.sqrt((want * (1 - want)) / runs))
    }
    const want = toFlat(inclusionProbability(L, [2, 3]))[0] ?? 0
    expect(Math.abs(pair / runs - want)).toBeLessThan(4 * Math.sqrt((want * (1 - want)) / runs))
  })

  it('differentiate their probabilities in the kernel', () => {
    const f = (b: Value): Value =>
      dppLogProbability(
        add(
          matmul(b, transpose(b)),
          tensor([
            [0.1, 0],
            [0, 0.1],
          ]),
        ),
        [0],
      )
    expectConformant(
      f,
      [
        tensor([
          [1, 0.3],
          [0.2, 0.7],
        ]),
      ],
      { vmap: false },
    )
  })
})

describe('k-DPPs, MAP and conditioning', () => {
  it('k-DPP samples have size k and frequencies proportional to det L_A', () => {
    const runs = 3000
    const freq = new Map<string, number>()
    for (let r = 0; r < runs; r++) {
      const y = sampleKDpp(foldIn(key(2), r), L, 2)
      expect(y.length).toBe(2)
      freq.set(y.join(','), (freq.get(y.join(',')) ?? 0) + 1)
    }
    const want = Math.exp(toFlat(kDppLogProbability(L, [0, 2]))[0] ?? 0)
    expect(Math.abs((freq.get('0,2') ?? 0) / runs - want)).toBeLessThan(4 * Math.sqrt((want * (1 - want)) / runs))
    const total = subsets(4)
      .filter((a) => a.length === 2)
      .reduce((s, a) => s + Math.exp(toFlat(kDppLogProbability(L, a))[0] ?? 0), 0)
    expect(total).toBeCloseTo(1, 12)
    expect(() =>
      sampleKDpp(
        key(0),
        [
          [1, 1],
          [1, 1],
        ],
        2,
      ),
    ).toThrow()
  })

  it('greedy MAP finds the best subset of a well-separated kernel', () => {
    const best = subsets(4)
      .filter((a) => a.length === 2)
      .map((a) => ({ a, v: toFlat(dppLogProbability(L, a))[0] ?? 0 }))
      .reduce((x, y) => (y.v > x.v ? y : x))
    expect([...greedyMap(L, 2)].sort()).toEqual(best.a)
    // without k, stop when no item increases det: every chosen gain exceeded one
    expect(
      greedyMap([
        [0.5, 0],
        [0, 0.4],
      ]),
    ).toEqual([])
  })

  it('conditioning on exclusion restricts and on inclusion matches the conditional law', () => {
    const c = conditionDpp(L, { include: [0] })
    expect(c.items).toEqual([1, 2, 3])
    // P(Y = {0, 2} | 0 in Y) = det(L^A_{2}) / det(L^A + I)
    const direct =
      Math.exp(toFlat(dppLogProbability(L, [0, 2]))[0] ?? 0) / (toFlat(inclusionProbability(L, [0]))[0] ?? 1)
    expect(Math.exp(toFlat(dppLogProbability(c.kernel, [1]))[0] ?? 0)).toBeCloseTo(direct, 10)
    expect(conditionDpp(L, { exclude: [3] }).items).toEqual([0, 1, 2])
    expect(() => conditionDpp(L, { include: [1], exclude: [1] })).toThrow()
  })
})
