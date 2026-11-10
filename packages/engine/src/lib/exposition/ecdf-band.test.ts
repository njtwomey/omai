import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { bandCoverage, ecdfBand, uniformGrid } from './ecdf-band.ts'

/** The exact probability by listing every assignment of the n values to the cells the grid cuts the unit interval into. */
function brute(n: number, z: readonly number[], lower: readonly number[], upper: readonly number[]): number {
  const edges = [0, ...z, 1]
  const cells = edges.slice(1).map((e, i) => e - (edges[i] ?? 0))
  let total = 0
  const assign = (i: number, counts: number[], p: number): void => {
    if (i === n) {
      let acc = 0
      for (const [j, zj] of z.entries()) {
        acc += counts[j] ?? 0
        void zj
        if (acc < (lower[j] ?? 0) || acc > (upper[j] ?? n)) return
      }
      total += p
      return
    }
    for (const [c, w] of cells.entries()) {
      counts[c] = (counts[c] ?? 0) + 1
      assign(i + 1, counts, p * w)
      counts[c] = (counts[c] ?? 0) - 1
    }
  }
  assign(0, new Array<number>(cells.length).fill(0), 1)
  return total
}

describe('bandCoverage', () => {
  it('is the exact probability: it matches the enumeration of every assignment', () => {
    const z = [0.3, 0.6, 0.85]
    for (const [lo, hi] of [
      [
        [0, 1, 2],
        [2, 3, 4],
      ],
      [
        [1, 2, 3],
        [1, 3, 4],
      ],
      [
        [0, 0, 0],
        [4, 4, 4],
      ],
    ] as const)
      expect(bandCoverage(5, z, lo, hi)).toBeCloseTo(brute(5, z, lo, hi), 12)
  })

  it('is 1 when the limits allow every count, and 0 when they exclude every one', () => {
    expect(bandCoverage(6, [0.2, 0.7], [0, 0], [6, 6])).toBeCloseTo(1, 12)
    expect(bandCoverage(6, [0.2, 0.7], [7, 7], [8, 8])).toBe(0)
  })

  it('rejects a grid that is not inside (0, 1) and increasing, and limits of the wrong length', () => {
    expect(() => bandCoverage(3, [0.5, 0.4], [0, 0], [3, 3])).toThrow(OmaiError)
    expect(() => bandCoverage(3, [0, 0.4], [0, 0], [3, 3])).toThrow(OmaiError)
    expect(() => bandCoverage(3, [0.5], [0, 0], [3, 3])).toThrow(expect.objectContaining({ code: 'shape' }))
  })
})

describe('ecdfBand', () => {
  it('holds with at least the stated probability, and not by a wide margin (the band is searched, not padded)', () => {
    for (const [n, k, level] of [
      [40, 9, 0.95],
      [200, 20, 0.99],
      [25, 24, 0.9],
    ] as const) {
      const z = uniformGrid(k)
      const b = ecdfBand(n, z, level)
      const lo = b.lower.map((v) => Math.round(v * n))
      const hi = b.upper.map((v) => Math.round(v * n))
      const p = bandCoverage(n, z, lo, hi)
      expect(p).toBeGreaterThanOrEqual(level - 1e-9)
      expect(p).toBeLessThan(level + 0.03)
    }
  })

  it('is wider than the pointwise band at the same level, and wider for a higher level', () => {
    const z = uniformGrid(15)
    const a = ecdfBand(100, z, 0.9)
    const b = ecdfBand(100, z, 0.99)
    const mid = 7
    // The pointwise 90% band at z = 1/2 is about 0.5 +- 1.645 * 0.05.
    expect((a.upper[mid] ?? 0) - (a.lower[mid] ?? 0)).toBeGreaterThan(2 * 1.645 * 0.05)
    expect((b.upper[mid] ?? 0) - (b.lower[mid] ?? 0)).toBeGreaterThan((a.upper[mid] ?? 0) - (a.lower[mid] ?? 0))
    expect(b.gamma).toBeLessThan(a.gamma)
  })

  it('brackets the grid, is symmetric about the diagonal, and narrows with more values', () => {
    const z = uniformGrid(9)
    const b = ecdfBand(80, z, 0.95)
    for (const [i, zi] of z.entries()) {
      expect(b.lower[i]).toBeLessThanOrEqual(zi)
      expect(b.upper[i]).toBeGreaterThanOrEqual(zi)
      // By the symmetry of the uniform distribution the limits mirror each other (up to the discreteness of the counts).
      const j = z.length - 1 - i
      expect(Math.abs((b.upper[i] ?? 0) - (1 - (b.lower[j] ?? 0)))).toBeLessThan(2 / 80)
    }
    const wide = (x: typeof b): number => Math.max(...x.upper.map((u, i) => u - (x.lower[i] ?? 0)))
    expect(wide(ecdfBand(800, z, 0.95))).toBeLessThan(wide(b))
  })

  it('rejects a level outside (0, 1), no values, and a bad grid', () => {
    expect(() => ecdfBand(10, [0.5], 1)).toThrow(OmaiError)
    expect(() => ecdfBand(0, [0.5])).toThrow(OmaiError)
    expect(() => ecdfBand(10, [1.2])).toThrow(OmaiError)
  })
})
