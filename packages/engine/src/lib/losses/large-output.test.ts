import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { key } from '../random/index.ts'
import { exp, item, sum, tensor, toArray } from '../tensor/index.ts'
import * as L from './index.ts'

describe('Huffman coding and the hierarchical softmax', () => {
  const counts = [40, 7, 3, 25, 1, 9, 15]

  it('builds a prefix code with optimal weighted length', () => {
    const coding = L.huffmanCoding(counts)
    const words = coding.codes.map((c) => c.join(''))
    for (const [i, a] of words.entries()) {
      for (const [j, b] of words.entries()) if (i !== j) expect(b.startsWith(a)).toBe(false)
    }
    // The optimal cost equals the sum of all merged weights.
    const pool = [...counts]
    let merged = 0
    while (pool.length > 1) {
      pool.sort((a, b) => a - b)
      const s = (pool.shift() ?? 0) + (pool.shift() ?? 0)
      merged += s
      pool.push(s)
    }
    expect(coding.codes.reduce((acc, c, i) => acc + c.length * (counts[i] ?? 0), 0)).toBe(merged)
    expect(() => L.huffmanCoding([1])).toThrow()
  })

  it('breaks ties towards the lower node id (the tree from information.huffmanTree)', () => {
    expect(L.huffmanCoding([1, 1, 1, 1])).toEqual({
      nClasses: 4,
      paths: [
        [2, 0],
        [2, 0],
        [2, 1],
        [2, 1],
      ],
      codes: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
    })
    expect(L.huffmanCoding([2, 1, 1]).codes).toEqual([[0], [1, 0], [1, 1]])
  })

  it('gives probabilities that sum to one and agree with the loss', () => {
    const coding = L.huffmanCoding(counts)
    const h = tensor([
      [0.2, -0.5, 1],
      [1.5, 0.3, -0.2],
    ])
    const w = tensor(Array.from({ length: 6 }, (_, i) => [Math.sin(i), Math.cos(2 * i), 0.3 * i - 1]))
    const b = tensor([0.1, -0.2, 0.3, 0, 0.5, -0.4])
    const lp = L.hierarchicalSoftmaxLogProbs(h, w, coding, { nodeBias: b })
    for (const row of toArray(exp(lp)) as number[][]) expect(row.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 14)
    const loss = toArray(L.hierarchicalSoftmax(h, w, [3, 4], coding, { nodeBias: b, reduction: 'none' })) as number[]
    const rows = toArray(lp) as number[][]
    expect(loss[0]).toBeCloseTo(-(rows[0]?.[3] ?? 0), 13)
    expect(loss[1]).toBeCloseTo(-(rows[1]?.[4] ?? 0), 13)
    expectConformant((hh, ww, bb) => L.hierarchicalSoftmax(hh, ww, [3, 4], coding, { nodeBias: bb }), [h, w, b])
  })
})

describe('sampled softmax', () => {
  const h = tensor([
    [0.2, -0.5],
    [1, 0.3],
  ])
  const w = tensor([
    [1, 0],
    [0, 1],
    [1, 1],
    [-1, 0.5],
  ])

  it('equals the full cross-entropy when every other class is a uniform candidate', () => {
    const full = L.crossEntropy(tensor([[0, 0, 0, 0]]), [0])
    const s = L.sampledSoftmax([[0, 0]], w, [0], { sampled: [1, 2, 3], sampler: 'uniform' })
    expect(item(s)).toBeCloseTo(item(full), 14)
  })

  it('masks accidental hits', () => {
    const masked = L.sampledSoftmax([[0, 0]], w, [0], { sampled: [1, 0], sampler: 'uniform' })
    // With the hit masked, only one competitor remains: log(1 + 1) after equal corrections.
    expect(item(masked)).toBeCloseTo(Math.LN2, 14)
    const kept = L.sampledSoftmax([[0, 0]], w, [0], {
      sampled: [1, 0],
      sampler: 'uniform',
      removeAccidentalHits: false,
    })
    expect(item(kept)).toBeCloseTo(Math.log(3), 14)
  })

  it('draws candidates with a key, reproducibly', () => {
    const a = L.sampleClasses(key(3), 100, 200)
    expect(L.sampleClasses(key(3), 100, 200)).toEqual(a)
    // Log-uniform proposals favour small class indices.
    expect(a.filter((c) => c < 10).length).toBeGreaterThan(a.filter((c) => c >= 90).length)
    expect(L.sampleClasses(key(4), 5, 20, 'uniform').every((c) => c >= 0 && c < 5)).toBe(true)
    expect(item(sum(L.sampledSoftmax(h, w, [1, 2], { key: key(1), nSampled: 3, reduction: 'none' })))).toBeGreaterThan(
      0,
    )
    expect(() => L.sampledSoftmax(h, w, [1, 2])).toThrow()
  })

  it('passes the conformance check', () => {
    expectConformant(
      (hh, ww) => L.sampledSoftmax(hh, ww, [1, 2], { sampled: [0, 3, 2], biases: [0.1, 0, -0.2, 0.3] }),
      [h, w],
    )
  })
})
