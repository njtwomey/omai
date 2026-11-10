import { type SbcRun, sbcRun } from 'omai-engine/exposition'
import { sbc } from 'omai-engine/probprog/validation'
import { foldIn, key, normal } from 'omai-engine/random'
import { item } from 'omai-engine/tensor'
import { describe, expect, it } from 'vitest'
import { chosenCalibrated, sbcEcdfView, sbcRankView } from './sbc-run.ts'

// m ~ N(0, 1), y ~ N(m, 1): the exact posterior is N(y / 2, 1 / 2); `width` scales its sd (1 is exact).
function record(names: readonly string[], widths: readonly number[]): SbcRun {
  const result = sbc({
    key: key(5),
    replications: 100,
    numRanks: 19,
    nullSimulations: 200,
    simulate: (k) => {
      const truth: Record<string, number> = {}
      const y: number[] = []
      for (const [i, n] of names.entries()) {
        const m = item(normal(foldIn(k, 2 * i), []))
        truth[n] = m
        y.push(m + item(normal(foldIn(k, 2 * i + 1), [])))
      }
      return { data: y, truth }
    },
    fit: (y, k) =>
      Object.fromEntries(
        names.map((n, i) => [
          n,
          Array.from(normal(foldIn(k, i), [19], { loc: (y[i] ?? 0) / 2, scale: (widths[i] ?? 1) * Math.SQRT1_2 }).data),
        ]),
      ),
  })
  return sbcRun(result)
}

const rec = record(['a', 'b', 'c'], [1, 0.35, 1])

describe('calibration charts', () => {
  it('chooses the failing quantities first, then the rest in order, or the ones named', () => {
    expect(rec.quantities.map((q) => q.passes)).toEqual([true, false, true])
    expect(chosenCalibrated(rec).map((q) => q.name)).toEqual(['b', 'a', 'c'])
    expect(chosenCalibrated(rec, { max: 1 }).map((q) => q.name)).toEqual(['b'])
    expect(chosenCalibrated(rec, { quantities: ['c', 'zz'] }).map((q) => q.name)).toEqual(['c'])
  })

  it('draws a rank panel per quantity: the histogram as plateaus, the expected count and the band', () => {
    const o = sbcRankView(rec, { quantities: ['b'] })
    expect((o.grid as unknown[]).length).toBe(1)
    const series = o.series as { name?: string; data?: unknown[]; type?: string }[]
    const hist = series.find((s) => s.name === 'b')
    // Twenty bins, two points each.
    expect(hist?.data).toHaveLength(40)
    expect(series.some((s) => s.name === '__expected')).toBe(true)
    expect(series.some((s) => s.name === '__band')).toBe(true)
    expect(JSON.stringify(o)).toContain('U-shaped')
  })

  it('draws the ECDF difference pinned at both corners, inside its band, with the points outside counted in the title', () => {
    const o = sbcEcdfView(rec, { quantities: ['b'] })
    const line = (o.series as { name?: string; data?: [number, number][] }[]).find((s) => s.name === 'b')
    expect(line?.data?.[0]).toEqual([0, 0])
    expect(line?.data?.at(-1)).toEqual([1, 0])
    expect(line?.data).toHaveLength(21)
    expect(JSON.stringify(o)).toMatch(/\d+ outside the band/)
    expect(JSON.stringify(sbcEcdfView(rec, { quantities: ['a'] }))).not.toContain('outside the band')
  })

  it('draws small multiples of a calibration campaign in two columns from four quantities on', () => {
    const many = record(['p', 'q', 'r', 's', 't'], [1, 1, 1, 1, 1])
    expect((sbcRankView(many).grid as unknown[]).length).toBe(5)
    expect(new Set((sbcRankView(many).grid as { left: number }[]).map((g) => g.left)).size).toBe(2)
  })
})
