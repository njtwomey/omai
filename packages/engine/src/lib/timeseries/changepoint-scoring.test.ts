import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  changepointCovering,
  changepointHausdorff,
  changepointPrecisionRecall,
  matchChangepoints,
} from './changepoint-scoring.ts'

const cps = (n: number): fc.Arbitrary<number[]> =>
  fc.uniqueArray(fc.integer({ min: 1, max: n - 1 }), { maxLength: 6 }).map((a) => a.sort((x, y) => x - y))

/** The largest matching by trying every assignment (small sets). */
function bruteMatching(truth: readonly number[], detected: readonly number[], margin: number): number {
  const best = (i: number, used: ReadonlySet<number>): number => {
    if (i === truth.length) return 0
    let top = best(i + 1, used)
    for (const [j, d] of detected.entries())
      if (!used.has(j) && Math.abs((truth[i] ?? 0) - d) <= margin)
        top = Math.max(top, 1 + best(i + 1, new Set([...used, j])))
    return top
  }
  return best(0, new Set())
}

/** The covering from labels per step: for each true segment, its best Jaccard with a detected segment. */
function bruteCovering(truth: readonly number[], detected: readonly number[], n: number): number {
  const label = (cs: readonly number[]): number[] =>
    Array.from({ length: n }, (_, t) => cs.filter((c) => c <= t).length)
  const [a, b] = [label(truth), label(detected)]
  let total = 0
  for (const seg of new Set(a)) {
    const inA = a.map((v) => v === seg)
    let best = 0
    for (const other of new Set(b)) {
      const inB = b.map((v) => v === other)
      const both = inA.filter((v, t) => v && inB[t]).length
      const either = inA.filter((v, t) => v || inB[t]).length
      best = Math.max(best, both / either)
    }
    total += inA.filter(Boolean).length * best
  }
  return total / n
}

describe('scoring detected changepoints', () => {
  it('matches each true changepoint to at most one detection within the margin', () => {
    expect(matchChangepoints([50, 100], [48, 52, 101], 5)).toEqual([
      { truth: 50, detected: 48 },
      { truth: 100, detected: 101 },
    ])
    expect(matchChangepoints([10], [5, 15], 5)).toEqual([{ truth: 10, detected: 5 }])
    expect(matchChangepoints([10], [16], 5)).toEqual([])
    expect(() => matchChangepoints([1], [1], -1)).toThrow()
    expect(() => matchChangepoints([Number.NaN], [1], 1)).toThrow()
  })

  it('the matching is a maximum matching (property: brute force)', () => {
    fc.assert(
      fc.property(cps(60), cps(60), fc.integer({ min: 0, max: 8 }), (truth, detected, margin) => {
        const pairs = matchChangepoints(truth, detected, margin)
        const distinct = new Set(pairs.map((p) => p.detected)).size === pairs.length
        return distinct && pairs.length === bruteMatching(truth, detected, margin)
      }),
      { numRuns: 200 },
    )
  })

  it('precision, recall and F1', () => {
    const s = changepointPrecisionRecall([50, 100, 150], [48, 52, 101], 5)
    expect([s.truePositives, s.falsePositives, s.falseNegatives]).toEqual([2, 1, 1])
    expect(s.precision).toBeCloseTo(2 / 3, 12)
    expect(s.recall).toBeCloseTo(2 / 3, 12)
    expect(s.f1).toBeCloseTo(2 / 3, 12)
    expect(changepointPrecisionRecall([], [], 3)).toMatchObject({ precision: 1, recall: 1, f1: 1 })
    expect(changepointPrecisionRecall([5], [], 3)).toMatchObject({ precision: 1, recall: 0, f1: 0 })
    expect(changepointPrecisionRecall([], [5], 3)).toMatchObject({ precision: 0, recall: 1, f1: 0 })
  })

  it('the Hausdorff distance is symmetric, zero only on equal sets, and the ends make "nothing found" finite', () => {
    expect(changepointHausdorff([50, 100], [48, 101])).toBe(2)
    expect(changepointHausdorff([50, 100], [], 120)).toBe(50)
    expect(changepointHausdorff([], [])).toBe(0)
    expect(changepointHausdorff([5], [])).toBe(Number.POSITIVE_INFINITY)
    fc.assert(
      fc.property(cps(60), cps(60), (a, b) => {
        const d = changepointHausdorff(a, b, 60)
        return d === changepointHausdorff(b, a, 60) && (d > 0 || a.join() === b.join()) && d <= 60
      }),
    )
  })

  it('the covering is 1 for the right segmentation and equals the pointwise formula (property)', () => {
    expect(changepointCovering([50], [50], 100)).toBe(1)
    expect(changepointCovering([50], [], 100)).toBe(0.5)
    expect(changepointCovering([], [], 10)).toBe(1)
    // One true segment cut in two: each half has Jaccard overlap 1/2 with the whole, so the covering is 1/2.
    expect(changepointCovering([], [50], 100)).toBe(0.5)
    expect(() => changepointCovering([0], [], 10)).toThrow()
    expect(() => changepointCovering([], [], 0)).toThrow()
    fc.assert(
      fc.property(cps(40), cps(40), (truth, detected) => {
        const c = changepointCovering(truth, detected, 40)
        return Math.abs(c - bruteCovering(truth, detected, 40)) < 1e-12 && c > 0 && c <= 1
      }),
      { numRuns: 200 },
    )
  })
})
