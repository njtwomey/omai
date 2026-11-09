import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { type Biclusters, consensusScore } from './bicluster-score.ts'

const a: Biclusters = {
  rows: [
    [1, 1, 0, 0],
    [0, 0, 1, 1],
  ],
  columns: [
    [1, 0, 0],
    [0, 1, 1],
  ],
}

describe('consensusScore', () => {
  it('is 1 for the same set in any order, 0 for disjoint sets, and divides by the larger set', () => {
    const reversed: Biclusters = { rows: [...a.rows].reverse(), columns: [...a.columns].reverse() }
    expect(consensusScore(a, reversed)).toBe(1)
    const elsewhere: Biclusters = { rows: [[0, 0, 0, 1]], columns: [[1, 0, 0]] }
    expect(consensusScore(a, elsewhere)).toBe(0)
    // One of two biclusters found exactly: 1 / max(2, 1).
    expect(consensusScore(a, { rows: [a.rows[0] ?? []], columns: [a.columns[0] ?? []] })).toBe(0.5)
    expect(consensusScore({ rows: [], columns: [] }, a)).toBe(0)
  })

  it('takes booleans and a custom similarity', () => {
    const b: Biclusters = { rows: [[true, true, false, false]], columns: [[true, false, false]] }
    expect(consensusScore(b, b)).toBe(1)
    expect(consensusScore(b, b, { similarity: () => 0.25 })).toBe(0.25)
    expect(() => consensusScore({ rows: [[1]], columns: [] }, b)).toThrow(/row indicators/)
  })

  it('is symmetric and within [0, 1] for any pair of sets', () => {
    const set = fc.integer({ min: 1, max: 4 }).chain((k) =>
      fc.record({
        rows: fc.array(
          fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 6, maxLength: 6 }).filter((r) => r.some(Boolean)),
          { minLength: k, maxLength: k },
        ),
        columns: fc.array(
          fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 5, maxLength: 5 }).filter((r) => r.some(Boolean)),
          { minLength: k, maxLength: k },
        ),
      }),
    )
    fc.assert(
      fc.property(set, set, (p, q) => {
        const s = consensusScore(p, q)
        expect(s).toBeGreaterThanOrEqual(0)
        expect(s).toBeLessThanOrEqual(1 + 1e-12)
        expect(consensusScore(q, p)).toBeCloseTo(s, 12)
        expect(consensusScore(p, p)).toBeCloseTo(1, 12)
      }),
      { numRuns: 60 },
    )
  })
})
