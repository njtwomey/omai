import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const values = fc.array(
  fc.integer({ min: -1000, max: 1000 }).map((v) => v / 10),
  { minLength: 2, maxLength: 60 },
)
const RULES: readonly S.BinRule[] = ['auto', 'fd', 'scott', 'sturges', 'doane', 'rice', 'sqrt', 'knuth']

describe('histograms', () => {
  it('count every observation once and integrate to one as densities, for every rule', () => {
    fc.assert(
      fc.property(values, fc.constantFrom(...RULES), (x, bins) => {
        const h = S.histogram(x, { bins })
        expect(toFlat(h.counts).reduce((s, v) => s + v, 0)).toBe(x.length)
        const e = toFlat(h.edges)
        for (let i = 1; i < e.length; i++) expect(e[i] ?? 0).toBeGreaterThan(e[i - 1] ?? 0)
        const d = toFlat(S.histogram(x, { bins, density: true }).counts)
        let area = 0
        for (let i = 0; i < d.length; i++) area += (d[i] ?? 0) * ((e[i + 1] ?? 0) - (e[i] ?? 0))
        expect(area).toBeCloseTo(1, 10)
      }),
    )
  })

  it('agrees between the binned count and the histogram', () => {
    fc.assert(
      fc.property(values, fc.integer({ min: 1, max: 12 }), (x, bins) => {
        const counts = toFlat(S.binnedStatistic(x, x, { statistic: 'count', bins }).statistic)
        expect(counts).toEqual(toFlat(S.histogram(x, { bins }).counts))
      }),
    )
  })

  it('two-dimensional counts sum to the in-range observations and marginalise to the 1-D histogram', () => {
    const x = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    const y = [3, 1, 2, 0, 1, 2, 3]
    const h = S.histogram2d(x, y, { bins: [3, 2] })
    const c = toFlat(h.counts)
    expect(c.reduce((s, v) => s + v, 0)).toBe(7)
    const rows = [0, 1, 2].map((r) => (c[2 * r] ?? 0) + (c[2 * r + 1] ?? 0))
    expect(rows).toEqual(toFlat(S.histogram(x, { bins: 3 }).counts))
  })

  it('validates its inputs', () => {
    expect(() => S.histogram([1, 2], { bins: 0 })).toThrow()
    expect(() => S.histogram([1, 2], { bins: [2, 1] })).toThrow()
    expect(() => S.histogram([1, 2], { bins: 'auto', weights: [1, 1] })).toThrow()
    expect(toFlat(S.histogramBinEdges([5, 5]))).toHaveLength(11)
  })
})
