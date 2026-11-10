import { chainDiagnostics } from 'omai-engine/exposition'
import { key, normal } from 'omai-engine/random'
import { chainEssCurves } from 'omai-engine/sampling'
import { describe, expect, it } from 'vitest'
import { autocorrelationView, ecdfView, energyView, essView, hasChart, pairView } from './chain-charts.ts'
import { traceView } from './chain-diagnostics.ts'

type Datum = [number, number] | { value: [number, number] }
/** The coordinates of a datum: line series hold pairs, point series objects with the pair as `value`. */
const xy = (d: Datum | undefined): [number, number] | undefined => (Array.isArray(d) ? d : d?.value)

const iid = (seed: number, n: number, shift = 0): number[] => Array.from(normal(key(seed), [n]).data, (v) => v + shift)

function make(n = 200): ReturnType<typeof chainDiagnostics> {
  const draws = {
    a: [iid(1, n), iid(2, n), iid(3, n)],
    b: [iid(4, n), iid(5, n), iid(6, n)],
    c: [iid(7, n), iid(8, n), iid(9, n)],
  }
  const divergent = [0, 1, 2].map((c) =>
    Array.from({ length: n }, (_, i) => (c === 0 && (i === 5 || i === 150) ? 1 : 0)),
  )
  return chainDiagnostics(draws, {
    divergent,
    energy: [iid(10, n), iid(11, n), iid(12, n)],
    essCurves: chainEssCurves(draws, { probs: [0.1, 0.5, 0.9], points: 4 }),
  })
}

const rec = make()

describe('which charts a record can draw', () => {
  it('knows the optional parts: the ECDF and autocorrelation by default, the rest when given, a pair plot from two quantities', () => {
    const bare = chainDiagnostics({
      a: [
        [1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1],
      ],
    })
    expect(
      ['ecdf', 'autocorrelation', 'essQuantile', 'essEvolution', 'energy', 'pairs'].map((c) =>
        hasChart(bare, c as never),
      ),
    ).toEqual([true, true, false, false, false, false])
    expect(
      ['ecdf', 'autocorrelation', 'essQuantile', 'essEvolution', 'energy', 'pairs'].map((c) =>
        hasChart(rec, c as never),
      ),
    ).toEqual([true, true, true, true, true, true])
  })

  it('says what is missing, in the words of the option that adds it', () => {
    const bare = chainDiagnostics({
      a: [
        [1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1],
      ],
    })
    expect(() => energyView(bare)).toThrow(
      expect.objectContaining({ code: 'value', message: expect.stringContaining('energy') }),
    )
    expect(() => essView(bare)).toThrow(expect.objectContaining({ code: 'value' }))
    expect(() => ecdfView(chainDiagnostics({ a: [[1, 2, 1, 2, 1, 2]] }, { ecdfPoints: 0 }))).toThrow(
      expect.objectContaining({ code: 'value' }),
    )
    expect(() => pairView(bare)).toThrow(expect.objectContaining({ code: 'value' }))
  })
})

describe('the charts', () => {
  it('draws the ECDF difference pinned at the corners with the band, a line per chain', () => {
    const o = ecdfView(rec, { quantities: ['a'] })
    const series = o.series as { name?: string; data?: [number, number][] }[]
    const chain = series.filter((s) => s.name?.startsWith('chain'))
    expect(chain.map((s) => s.name)).toEqual(['chain 0', 'chain 1', 'chain 2'])
    expect(chain[0]?.data).toHaveLength(42)
    expect([chain[0]?.data?.[0], chain[0]?.data?.at(-1)]).toEqual([
      [0, 0],
      [1, 0],
    ])
    expect(series.some((s) => s.name === '__band')).toBe(true)
  })

  it('draws the autocorrelation from lag zero at one, a line per chain and the mean', () => {
    const o = autocorrelationView(rec, { quantities: ['b'] })
    const series = o.series as { name?: string; data?: [number, number][] }[]
    expect(series.find((s) => s.name === 'mean')?.data?.[0]).toEqual([0, 1])
    expect(series.find((s) => s.name === 'chain 1')?.data).toHaveLength(41)
  })

  it('draws the ESS by quantile, or bulk and tail as the draws accumulate, with the line of 400', () => {
    const q = essView(rec, { quantities: ['a'] })
    expect((q.series as { name?: string; data?: unknown[] }[]).find((s) => s.name === 'ESS')?.data).toHaveLength(3)
    expect(JSON.stringify(q)).toContain('markLine')
    const e = essView(rec, { quantities: ['a'], curve: 'evolution' })
    expect((e.series as { name?: string }[]).map((s) => s.name)).toEqual(expect.arrayContaining(['bulk', 'tail']))
  })

  it('draws the energy distributions on common bins with the E-BFMI in the title', () => {
    const o = energyView(rec)
    const series = o.series as { name?: string; data?: unknown[] }[]
    expect(series.find((s) => s.name === 'energy')?.data).toHaveLength(60)
    expect(series.find((s) => s.name === 'energy change')?.data).toHaveLength(60)
    expect(JSON.stringify(o)).toMatch(/E-BFMI \d/)
  })

  it('draws the pair plot as a lower triangle with a density on the diagonal and the divergent transitions in red', () => {
    const o = pairView(rec)
    // Three quantities: six cells.
    expect((o.grid as unknown[]).length).toBe(6)
    const series = o.series as { name?: string; data?: Datum[] }[]
    const div = series.filter((s) => s.name === 'divergent')
    expect(div).toHaveLength(3)
    expect(div[0]?.data).toHaveLength(2)
    const a = rec.quantities[0]
    expect(xy(div[0]?.data?.[0])).toEqual([a?.atDivergences?.[0], rec.quantities[1]?.atDivergences?.[0]])
    // A thinned scatter: no more than the cap per chain.
    const thin = pairView(rec, { maxPoints: 20 })
    expect(
      (thin.series as { name?: string; data?: unknown[] }[])
        .filter((s) => s.name === 'chain 0')
        .every((s) => (s.data?.length ?? 0) <= 20),
    ).toBe(true)
    expect((pairView(rec, { quantities: ['a', 'c'] }).grid as unknown[]).length).toBe(3)
  })
})

describe('the trace of a thinned run', () => {
  it('marks a divergence at its draw and its value, whatever the thinning', () => {
    const big = chainDiagnostics(
      { a: [iid(1, 1500), iid(2, 1500)] },
      {
        divergent: [
          new Array<number>(1500).fill(0).map((_, i) => (i === 1203 ? 1 : 0)),
          new Array<number>(1500).fill(0),
        ],
        maxDraws: 100,
      },
    )
    const o = traceView(big)
    const series = o.series as { name?: string; data?: Datum[] }[]
    expect(xy(series.find((s) => s.name === 'divergent')?.data?.[0])?.[0]).toBe(1203)
    expect(xy(series.find((s) => s.name === 'divergent')?.data?.[0])?.[1]).toBe(big.quantities[0]?.atDivergences?.[0])
    // The thinned trace spans the draws: its last point is the last draw.
    expect(xy(series.find((s) => s.name === 'chain 0')?.data?.at(-1))?.[0]).toBe(1499)
    expect(series.find((s) => s.name === 'chain 0')?.data).toHaveLength(100)
  })
})
