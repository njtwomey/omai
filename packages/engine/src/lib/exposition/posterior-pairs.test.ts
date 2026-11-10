import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { corrcoef, gaussianKde, mean, std } from '../stats/index.ts'
import { toFlat } from '../tensor/index.ts'
import { drawsUpTo, pairIndex, pairsFrameAt, posteriorPairs } from './posterior-pairs.ts'

/** A deterministic correlated pair: a = u, b = rho u + sqrt(1 - rho^2) w, u and w from a low-discrepancy sequence mapped to normals. */
function correlated(n: number, rho: number, shift = 0): { a: number[]; b: number[]; c: number[] } {
  const norm = (u: number): number => {
    // Acklam-free: a logistic quantile scaled to unit variance is close enough for a test of the statistics.
    return (Math.log(u / (1 - u)) * Math.sqrt(3)) / Math.PI
  }
  const a: number[] = []
  const b: number[] = []
  const c: number[] = []
  for (let i = 0; i < n; i++) {
    const u = ((i * 0.6180339887) % 1) * 0.998 + 0.001
    const w = ((i * 0.7548776662 + 0.3) % 1) * 0.998 + 0.001
    a.push(norm(u) + shift)
    b.push(rho * norm(u) + Math.sqrt(1 - rho * rho) * norm(w))
    c.push(norm(w) * 2)
  }
  return { a, b, c }
}

describe('posteriorPairs', () => {
  const draws = correlated(600, 0.7)
  const r = posteriorPairs([{ name: 'm', draws }], { maxPoints: 100, frames: 5, gridPoints: 40 })

  it('keeps every coordinate, a shared grid and the points thinned evenly in step order', () => {
    expect(r.coordinates.map((c) => c.name)).toEqual(['a', 'b', 'c'])
    expect(r.coordinates.every((c) => c.grid.length === 40)).toBe(true)
    const m = r.methods[0]
    expect(m?.points.steps).toHaveLength(100)
    expect(m?.points.steps[0]).toBe(0)
    expect(m?.points.steps.at(-1)).toBe(599)
    expect(m?.points.values.map((v) => v.length)).toEqual([100, 100, 100])
    expect([...(m?.points.steps ?? [])].sort((x, y) => x - y)).toEqual(m?.points.steps)
    // The i-th point is the draw at the evenly spaced index.
    expect(m?.points.values[0]?.[50]).toBe(draws.a[Math.floor((50 * 599) / 99)])
  })

  it('has the cursors evenly over the steps, the last at the end, and finds the frame of a step', () => {
    expect(r.cursors).toHaveLength(5)
    expect(r.cursors.at(-1)).toBe(599)
    expect(r.cursors[0]).toBeCloseTo(119.8, 6)
    expect(pairsFrameAt(r, 50)).toBe(-1)
    expect(pairsFrameAt(r, 130)).toBe(0)
    expect(pairsFrameAt(r, 1e9)).toBe(4)
    expect(drawsUpTo([0, 1, 1, 4], 1)).toBe(3)
  })

  it('computes the means, sds and correlations of the draws up to each cursor, as stats does', () => {
    const f = r.methods[0]?.frames
    for (const frame of [1, 4]) {
      const n = f?.count[frame] as number
      expect(n).toBe(Math.floor(r.cursors[frame] as number) + 1)
      const cols = [draws.a, draws.b, draws.c].map((d) => d.slice(0, n))
      const corr = toFlat(corrcoef(cols))
      cols.forEach((c, j) => {
        expect(f?.mean[frame]?.[j]).toBeCloseTo(mean(c), 9)
        expect(f?.sd[frame]?.[j]).toBeCloseTo(std(c, { ddof: 1 }), 9)
      })
      expect(f?.correlation[frame]?.[pairIndex(3, 0, 1)]).toBeCloseTo(corr[1] as number, 9)
      expect(f?.correlation[frame]?.[pairIndex(3, 0, 2)]).toBeCloseTo(corr[2] as number, 9)
      expect(f?.correlation[frame]?.[pairIndex(3, 1, 2)]).toBeCloseTo(corr[5] as number, 9)
    }
    expect(f?.correlation[4]?.[0]).toBeGreaterThan(0.5)
  })

  it('has densities that integrate to one and agree with gaussianKde (Scott) on the same grid', () => {
    const f = r.methods[0]?.frames
    const g = r.coordinates[0]?.grid ?? []
    const step = (g[1] as number) - (g[0] as number)
    const dens = f?.density[4]?.slice(0, 40) ?? []
    expect(dens.reduce((s, v) => s + v, 0) * step).toBeGreaterThan(0.95)
    expect(dens.reduce((s, v) => s + v, 0) * step).toBeLessThan(1.02)
    // 600 draws are thinned to 500 for the density, evenly; stats on that same subsample agrees.
    const sub = Array.from({ length: 500 }, (_, i) => draws.a[Math.floor((i * 599) / 499)] as number)
    const ref = toFlat(gaussianKde(sub).evaluate(g))
    dens.forEach((v, i) => {
      expect(v).toBeCloseTo(ref[i] as number, 6)
    })
  })

  it('shows nothing but the count before two draws have arrived', () => {
    const early = posteriorPairs([{ name: 'm', draws, steps: draws.a.map((_, i) => i + 100) }], { frames: 600 })
    const f = early.methods[0]?.frames
    // The first cursor is just after the first draw's step (the steps start at 100).
    expect(f?.count[0]).toBe(1)
    const single = posteriorPairs([{ name: 'm', draws: { a: [1, 2, 3, 4], b: [2, 1, 4, 3] } }], { frames: 4 })
    expect(single.cursors).toEqual([0.75, 1.5, 2.25, 3])
    expect(single.methods[0]?.frames.count).toEqual([1, 2, 3, 4])
    expect(single.methods[0]?.frames.density[0]?.every((v) => v === 0)).toBe(true)
    expect(single.methods[0]?.frames.correlation[0]).toEqual([0])
  })

  it('puts the steps in one unit across the methods: a shorter method holds its last state', () => {
    const long = correlated(400, 0.2)
    const short = correlated(100, 0.2, 1)
    const two = posteriorPairs(
      [
        { name: 'long', draws: long },
        { name: 'short', draws: short },
      ],
      { frames: 4, coordinates: ['a', 'b'] },
    )
    expect(two.stepRange).toEqual([0, 399])
    expect(two.methods[1]?.stepRange).toEqual([0, 99])
    expect(two.methods[1]?.frames.count).toEqual([100, 100, 100, 100])
    expect(two.methods[0]?.frames.count).toEqual([100, 200, 300, 400])
  })

  it('takes the steps within each chain by default and interleaves the chains', () => {
    const chain = [0, 1, 0, 1, 0, 1]
    const x = [10, 20, 11, 21, 12, 22]
    const two = posteriorPairs([{ name: 'm', draws: { x, y: x }, chain }], { frames: 3 })
    expect(two.methods[0]?.points.steps).toEqual([0, 0, 1, 1, 2, 2])
    expect(two.methods[0]?.points.values[0]).toEqual([10, 20, 11, 21, 12, 22])
    expect(two.methods[0]?.frames.count).toEqual([2, 4, 6])
  })

  it('keeps the divergent draws apart, thinned to the limit, and counts them all', () => {
    const d = correlated(300, 0.5)
    const flags = d.a.map((_, i) => (i % 10 === 0 ? 1 : 0))
    const two = posteriorPairs([{ name: 'm', draws: d, divergent: flags }], {
      maxDivergent: 8,
      coordinates: ['a', 'b'],
    })
    const m = two.methods[0]
    expect(m?.numDivergent).toBe(30)
    expect(m?.divergent.steps).toHaveLength(8)
    expect(m?.divergent.steps.every((s) => s % 10 === 0)).toBe(true)
    expect(m?.divergent.values[0]).toHaveLength(8)
  })

  it('shares one range per coordinate covering every method and the reference value, and records the reference', () => {
    const far = posteriorPairs(
      [
        { name: 'p', draws: correlated(200, 0.3) },
        { name: 'q', draws: correlated(200, 0.3, 10) },
      ],
      { coordinates: ['a', 'b'], truth: { a: -6 }, referenceLabel: 'MAP' },
    )
    const a = far.coordinates[0]
    expect(a?.range[0]).toBeLessThan(-6)
    expect(a?.range[1]).toBeGreaterThan(10)
    expect(a?.truth).toBe(-6)
    expect(far.coordinates[1]?.truth).toBeUndefined()
    expect(far.referenceLabel).toBe('MAP')
  })

  it('survives a coordinate with no spread (a spike, not an error)', () => {
    const flat = posteriorPairs([{ name: 'm', draws: { a: new Array<number>(50).fill(2), b: correlated(50, 0).a } }])
    expect(flat.methods[0]?.frames.sd.at(-1)?.[0]).toBe(0)
    expect(flat.methods[0]?.frames.correlation.at(-1)).toEqual([0])
    expect(
      flat.methods[0]?.frames.density
        .at(-1)
        ?.slice(0, 60)
        .some((v) => v > 0),
    ).toBe(true)
  })

  it('is small, plain and cloneable: no typed arrays, and a size set by the options', () => {
    const json = JSON.stringify(r)
    expect(structuredClone(r)).toEqual(JSON.parse(json))
    // 100 points x 3 coordinates + 5 frames x (3 x 40 density + extras): a few thousand numbers.
    expect(json.length).toBeLessThan(40_000)
  })

  it('rejects what cannot be drawn', () => {
    expect(() => posteriorPairs([])).toThrow(OmaiError)
    expect(() => posteriorPairs([{ name: 'm', draws: { a: [1, 2, 3] } }])).toThrow(
      expect.objectContaining({ code: 'shape' }),
    )
    expect(() => posteriorPairs([{ name: 'm', draws: { a: [1, 2, 3], b: [1, 2] } }])).toThrow(
      expect.objectContaining({ code: 'shape' }),
    )
    expect(() =>
      posteriorPairs([{ name: 'm', draws: { a: [1, 2, 3], b: [1, 2, 3] } }], { coordinates: ['a', 'z'] }),
    ).toThrow(expect.objectContaining({ code: 'value' }))
    expect(() => posteriorPairs([{ name: 'm', draws: { a: [1], b: [1] } }])).toThrow(
      expect.objectContaining({ code: 'value' }),
    )
    expect(() => posteriorPairs([{ name: 'm', draws: { a: [1, 2, 3], b: [1, Number.NaN, 3] } }])).toThrow(OmaiError)
    const nine = Object.fromEntries(Array.from({ length: 9 }, (_, i) => [`x${i}`, [1, 2, 3]]))
    expect(() => posteriorPairs([{ name: 'm', draws: nine }], { coordinates: Object.keys(nine) })).toThrow(
      expect.objectContaining({ code: 'shape' }),
    )
    expect(() => posteriorPairs([{ name: 'm', draws: { a: [1, 2, 3], b: [1, 2, 4] }, steps: [0, 1] }])).toThrow(
      expect.objectContaining({ code: 'shape' }),
    )
  })

  it('has a pair index that numbers the upper triangle by rows', () => {
    const k = 5
    const seen: number[] = []
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) seen.push(pairIndex(k, i, j))
    expect(seen).toEqual(Array.from({ length: 10 }, (_, i) => i))
  })
})
