import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { isMdmlError, minOf } from '../base/index.ts'
import type { Tensor } from '../tensor/index.ts'
import { halton, key, latinHypercube, poissonDisk, scrambledSobol, sobol, uniform } from './index.ts'

const seed = fc.integer({ min: 0, max: 2 ** 31 })

function rows(x: Tensor): number[][] {
  const [n = 0, d = 0] = x.shape
  return Array.from({ length: n }, (_, i) => Array.from(x.data.slice(i * d, (i + 1) * d)))
}

/** Whether the first 2^m points of two columns form a (0, m, 2)-net: every dyadic box of area 2^-m holds one point. */
function isNet(points: number[][], m: number, a: number, b: number): boolean {
  for (let p = 0; p <= m; p++) {
    const q = m - p
    const seen = new Set<number>()
    for (const row of points) seen.add(Math.floor((row[a] ?? 0) * 2 ** p) * 2 ** q + Math.floor((row[b] ?? 0) * 2 ** q))
    if (seen.size !== 2 ** m) return false
  }
  return true
}

/** Whether each column of the first 2^m points puts one point in every interval [i / 2^m, (i + 1) / 2^m). */
function isStratified(points: number[][], m: number): boolean {
  const d = points[0]?.length ?? 0
  for (let j = 0; j < d; j++) {
    if (new Set(points.map((row) => Math.floor((row[j] ?? 0) * 2 ** m))).size !== 2 ** m) return false
  }
  return true
}

describe('sobol', () => {
  it('starts at the origin, then 1/2, as scipy', () => {
    expect(Array.from(sobol(2, 3).data)).toEqual([0, 0, 0, 0.5, 0.5, 0.5])
  })

  it('skip gives the same points as dropping the leading rows', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 300 }), fc.integer({ min: 1, max: 40 }), (skip, n) => {
        const all = rows(sobol(skip + n, 4))
        expect(rows(sobol(n, 4, { skip }))).toEqual(all.slice(skip))
      }),
      { numRuns: 30 },
    )
  })

  it('the first two dimensions form (0, m, 2)-nets and every dimension is stratified', () => {
    for (let m = 1; m <= 8; m++) {
      const x = rows(sobol(2 ** m, 12))
      expect(isNet(x, m, 0, 1)).toBe(true)
      expect(isStratified(x, m)).toBe(true)
    }
  })

  it('reaches the last index 2^bits - 1 and rejects one more', () => {
    expect(sobol(1, 2, { skip: 2 ** 30 - 1 }).data.every((v) => v >= 0 && v < 1)).toBe(true)
    expect(Array.from(sobol(4, 1, { bits: 2 }).data)).toEqual([0, 0.5, 0.75, 0.25])
    expect(() => sobol(5, 1, { bits: 2 })).toThrow(/at most 2\^2/)
  })

  it('rejects unsupported dimensions and bit counts', () => {
    for (const f of [() => sobol(1, 0), () => sobol(1, 1112), () => sobol(1, 2, { bits: 33 })]) {
      try {
        f()
        expect.unreachable()
      } catch (e) {
        expect(isMdmlError(e, 'value')).toBe(true)
      }
    }
  })
})

describe('scrambledSobol', () => {
  it('lies in [0, 1), is deterministic per key and differs between keys', () => {
    fc.assert(
      fc.property(seed, seed, (s, t) => {
        const a = scrambledSobol(key(s), 32, 5)
        expect(a.data.every((v) => v >= 0 && v < 1)).toBe(true)
        expect(Array.from(scrambledSobol(key(s), 32, 5).data)).toEqual(Array.from(a.data))
        if (s !== t) expect(Array.from(scrambledSobol(key(t), 32, 5).data)).not.toEqual(Array.from(a.data))
      }),
      { numRuns: 20 },
    )
  })

  it('keeps the net and stratification properties of the unscrambled sequence', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 1, max: 8 }), (s, m) => {
        const x = rows(scrambledSobol(key(s), 2 ** m, 6))
        expect(isNet(x, m, 0, 1)).toBe(true)
        expect(isStratified(x, m)).toBe(true)
      }),
      { numRuns: 25 },
    )
  })

  it('scrambles each dimension independently of the others, and skip drops leading rows', () => {
    const narrow = rows(scrambledSobol(key(3), 16, 2))
    const wide = rows(scrambledSobol(key(3), 16, 7)).map((row) => row.slice(0, 2))
    expect(wide).toEqual(narrow)
    expect(rows(scrambledSobol(key(3), 6, 2, { skip: 10 }))).toEqual(narrow.slice(10))
  })

  it('integrates a smooth function far better than independent uniforms', () => {
    // f(x) = prod(1 + (x_j - 1/2)) has integral 1 over the unit cube.
    const f = (row: number[]): number => row.reduce((p, x) => p * (1 + (x - 0.5)), 1)
    let qmcError = 0
    let mcError = 0
    for (let s = 0; s < 10; s++) {
      const q = rows(scrambledSobol(key(s), 256, 4))
      const u = rows(uniform(key(100 + s), [256, 4]))
      qmcError += (q.reduce((a, r) => a + f(r), 0) / 256 - 1) ** 2
      mcError += (u.reduce((a, r) => a + f(r), 0) / 256 - 1) ** 2
    }
    expect(qmcError).toBeLessThan(mcError / 20)
  })
})

describe('halton', () => {
  it('skips the origin by default and matches scipy with skip 0', () => {
    expect(rows(halton(3, 4, { skip: 0 }))[0]).toEqual([0, 0, 0, 0])
    expect(rows(halton(5, 4, { skip: 0 })).slice(1)).toEqual(rows(halton(4, 4)))
  })

  it('works beyond the 25th prime and rejects a bad skip', () => {
    // The 30th prime is 113, so the second point's last coordinate is 1 / 113.
    expect(rows(halton(1, 30))[0]?.[29]).toBeCloseTo(1 / 113, 15)
    expect(() => halton(1, 2, { skip: -1 })).toThrow(/skip/)
  })
})

describe('latinHypercube', () => {
  it('puts one point in every stratum of every axis, centred on request', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 1, max: 30 }), fc.boolean(), (s, n, centered) => {
        const x = rows(latinHypercube(key(s), n, 3, { centered }))
        for (let j = 0; j < 3; j++) {
          const cells = x.map((r) => Math.floor((r[j] ?? 0) * n))
          expect(new Set(cells).size).toBe(n)
          if (centered) for (const r of x) expect(((r[j] ?? 0) * n) % 1).toBeCloseTo(0.5, 12)
        }
      }),
      { numRuns: 30 },
    )
  })
})

function nearest(points: number[][], x: readonly number[]): number {
  return minOf(points.map((p) => Math.hypot(...p.map((v, j) => v - (x[j] ?? 0)))))
}

describe('poissonDisk', () => {
  it('keeps every pair at least the radius apart, inside the box', () => {
    fc.assert(
      fc.property(
        seed,
        fc.integer({ min: 1, max: 3 }),
        fc.double({ min: 0.2, max: 0.5, noNaN: true }),
        fc.constantFrom<'volume' | 'surface'>('volume', 'surface'),
        (s, d, r, hypersphere) => {
          const x = poissonDisk(key(s), d, r, { hypersphere, lower: new Array(d).fill(-0.5) })
          const m = x.shape[0] ?? 0
          expect(x.data.every((v) => v >= -0.5 && v <= 1)).toBe(true)
          let closest = Infinity
          for (let i = 0; i < m; i++) {
            for (let j = 0; j < i; j++) {
              let s2 = 0
              for (let t = 0; t < d; t++) s2 += ((x.data[i * d + t] ?? 0) - (x.data[j * d + t] ?? 0)) ** 2
              if (s2 < closest) closest = s2
            }
          }
          expect(Math.sqrt(closest)).toBeGreaterThanOrEqual(r)
        },
      ),
      { numRuns: 15 },
    )
  })

  it('fills the box: almost no point of the box is farther than the radius from the sample', () => {
    for (const [d, r] of [
      [1, 0.02],
      [2, 0.05],
      [3, 0.15],
    ] as const) {
      const x = rows(poissonDisk(key(d), d, r))
      const probes = rows(uniform(key(50 + d), [2000, d]))
      const gaps = probes.filter((p) => nearest(x, p) >= r).length
      expect(gaps / probes.length).toBeLessThan(0.01)
      // And nothing is farther than twice the radius: a candidate could always be placed there.
      expect(probes.every((p) => nearest(x, p) < 2 * r)).toBe(true)
    }
  })

  it('stops at n points, is deterministic per key, and validates its arguments', () => {
    expect(poissonDisk(key(1), 2, 0.05, { n: 17 }).shape).toEqual([17, 2])
    expect(poissonDisk(key(1), 2, 0.05, { n: 0 }).shape).toEqual([0, 2])
    expect(Array.from(poissonDisk(key(2), 2, 0.1).data)).toEqual(Array.from(poissonDisk(key(2), 2, 0.1).data))
    expect(() => poissonDisk(key(0), 2, 0)).toThrow(/radius/)
    expect(() => poissonDisk(key(0), 2, 0.1, { lower: [0, 0], upper: [1, 0] })).toThrow(/bounds/)
    expect(() => poissonDisk(key(0), 8, 0.01)).toThrow(/cells/)
  })
})
