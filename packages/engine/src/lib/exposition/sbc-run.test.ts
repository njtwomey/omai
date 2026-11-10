import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { sbc } from '../probprog/validation/sbc.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { bandCoverage } from './ecdf-band.ts'
import { sbcRun, sbcTable } from './sbc-run.ts'

// m ~ N(0, 1), y ~ N(m, 1): the posterior is N(y / 2, 1 / 2). `width` scales the posterior's sd: 1 is exact.
function run(width: number, shift = 0): ReturnType<typeof sbc> {
  return sbc({
    key: key(3),
    replications: 150,
    numRanks: 19,
    nullSimulations: 400,
    simulate: (k) => {
      const m = item(normal(foldIn(k, 0), []))
      return { data: m + item(normal(foldIn(k, 1), [])), truth: { m } }
    },
    fit: (y, k) => ({ m: Array.from(normal(k, [19], { loc: y / 2 + shift, scale: width * Math.SQRT1_2 }).data) }),
  })
}

describe('sbcRun', () => {
  it('takes the kernel result as it is, and its ECDF difference is the ECDF of the ranks minus the uniform CDF', () => {
    const result = run(1)
    const r = sbcRun(result)
    const q = r.quantities[0]
    expect(r.kind).toBe('sbcRun')
    expect([r.replications, r.numRanks, r.passes]).toEqual([150, 19, true])
    expect(q?.histogram).toEqual(result.quantities[0]?.histogram)
    const ranks = result.quantities[0]?.ranks ?? []
    const z = q?.ecdfDifference.z ?? []
    expect(z).toHaveLength(19)
    for (const [j, zj] of z.entries()) {
      const at = ranks.filter((v) => v <= j).length / 150
      expect(q?.ecdfDifference.diff[j]).toBeCloseTo(at - zj, 12)
    }
    // The statistic of the kernel's test is the largest |difference| over every cell.
    const peak = Math.max(...(q?.ecdfDifference.diff ?? []).map(Math.abs))
    expect(peak).toBeCloseTo(result.quantities[0]?.ecdf.statistic ?? Number.NaN, 12)
  })

  it('draws the bands the exact computation gives: the ECDF band holds at the level, the histogram band is binomial', () => {
    const r = sbcRun(run(1), { level: 0.99 })
    const q = r.quantities[0]
    const n = r.replications
    const z = q?.ecdfDifference.z ?? []
    const lo = (q?.ecdfDifference.lower ?? []).map((d, j) => Math.round((d + (z[j] ?? 0)) * n))
    const hi = (q?.ecdfDifference.upper ?? []).map((d, j) => Math.round((d + (z[j] ?? 0)) * n))
    expect(bandCoverage(n, z, lo, hi)).toBeGreaterThanOrEqual(0.99 - 1e-9)
    // 20 equally likely cells of 150 ranks: 7.5 expected in each, a binomial band around it.
    expect(q?.histogram).toHaveLength(20)
    expect(q?.expected.every((e) => Math.abs(e - 7.5) < 1e-12)).toBe(true)
    expect((q?.band.lower[0] ?? 99) < 7.5 && (q?.band.upper[0] ?? 0) > 7.5).toBe(true)
    expect(r.level).toBe(0.99)
  })

  it('shows a method that is too confident as a U-shaped histogram and an ECDF difference that leaves its band', () => {
    const tight = run(0.4)
    expect(tight.passes).toBe(false)
    const r = sbcRun(tight)
    const q = r.quantities[0]
    expect(q?.passes).toBe(false)
    expect(q?.diagnosis).toContain('U-shaped')
    expect(q?.ecdfDifference.outside).toBeGreaterThan(0)
    // The ends of the histogram are above the band, the middle below.
    expect((q?.histogram[0] ?? 0) > (q?.band.upper[0] ?? 0)).toBe(true)
    expect(Math.min(...(q?.histogram.slice(6, 14) ?? []))).toBeLessThan(q?.band.lower[10] ?? 0)
  })

  it('reads a biased method as a slope', () => {
    const q = sbcRun(run(1, 0.8)).quantities[0]
    expect(q?.passes).toBe(false)
    expect(q?.diagnosis).toContain('biased')
  })

  it('thins the grid of a long rank range, keeps it increasing inside (0, 1), and checks its input', () => {
    const ranks = Array.from({ length: 40 }, (_, i) => (i * 97) % 1000)
    const q = {
      name: 'a',
      ranks,
      numRanks: 999,
      histogram: [10, 10, 10, 10],
      chiSquare: { statistic: 0, pValue: 1 },
      ecdf: { statistic: 0, pValue: 1 },
      pValue: 1,
      diagnosis: 'uniform',
      passes: true,
    }
    const r = sbcRun({ replications: 40, numRanks: 999, alpha: 0.05, quantities: [q], passes: true }, { points: 30 })
    const z = r.quantities[0]?.ecdfDifference.z ?? []
    expect(z.length).toBeLessThanOrEqual(30)
    expect(z.every((v, i) => v > 0 && v < 1 && (i === 0 || v > (z[i - 1] ?? 0)))).toBe(true)
    const bad = { replications: 40, numRanks: 999, alpha: 0.05, passes: true }
    expect(() => sbcRun({ ...bad, quantities: [{ ...q, ranks: [...ranks.slice(1), 1000] }] })).toThrow(OmaiError)
    expect(() => sbcRun({ ...bad, quantities: [{ ...q, ranks: ranks.slice(1) }] })).toThrow(OmaiError)
    expect(() => sbcRun({ ...bad, quantities: [{ ...q, histogram: [10, 10, 10, 9] }] })).toThrow(OmaiError)
    expect(() => sbcRun({ ...bad, quantities: [q] }, { level: 1 })).toThrow(OmaiError)
  })

  it('tabulates the verdict, a failing quantity marked bad with its reading', () => {
    const t = sbcTable(sbcRun(run(0.4)))
    expect(t.columns).toHaveLength(6)
    expect(t.marks?.[0]).toMatchObject({ row: 0, col: 5, status: 'bad' })
    expect(t.caption).toContain('NOT calibrated')
    expect(sbcTable(sbcRun(run(1))).marks?.[0]?.status).toBe('ok')
  })
})
