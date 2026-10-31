import { describe, expect, it } from 'vitest'
import { pairwiseCoupling } from './probability.ts'

describe('pairwiseCoupling', () => {
  it('returns a distribution that favours the class that wins its pairs', () => {
    const p = pairwiseCoupling([
      [0, 0.8, 0.9],
      [0.2, 0, 0.6],
      [0.1, 0.4, 0],
    ])
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(p[0] as number).toBeGreaterThan(p[1] as number)
    expect(p[1] as number).toBeGreaterThan(p[2] as number)
  })

  it('is uniform for indifferent pairs and reduces to the pair for two classes', () => {
    expect(
      pairwiseCoupling([
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
      ]).map((v) => Math.round(v * 1e6) / 1e6),
    ).toEqual([1 / 3, 1 / 3, 1 / 3].map((v) => Math.round(v * 1e6) / 1e6))
    const two = pairwiseCoupling([
      [0, 0.7],
      [0.3, 0],
    ])
    expect(two[0] as number).toBeCloseTo(0.7, 2)
  })

  it('rejects ragged input', () => {
    expect(() => pairwiseCoupling([[0, 1]])).toThrow(/k x k/)
  })
})
