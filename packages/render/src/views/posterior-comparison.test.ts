import { normal } from 'omai-engine/dists'
import { posteriorComparison } from 'omai-engine/exposition'
import { agree, distributionOracle } from 'omai-engine/probprog/validation'
import { key } from 'omai-engine/random'
import { item } from 'omai-engine/tensor'
import { describe, expect, it } from 'vitest'
import { posteriorComparisonView } from './posterior-comparison.ts'

const exact = normal(0.6, Math.SQRT1_2)
const draws = (seed: number, scale = 1, shift = 0): number[] =>
  Array.from({ length: 800 }, (_, i) => 0.6 + shift + scale * (item(exact.sample(key(seed * 10_000 + i), [])) - 0.6))
const oracle = { m: distributionOracle('m', exact), s: distributionOracle('s', exact) }
const approximations = [
  { name: 'nuts', draws: { m: draws(1), s: draws(2) } },
  { name: 'narrow', draws: { m: draws(3, 0.5), s: draws(4, 0.5) } },
]
const record = posteriorComparison(
  [
    ...approximations,
    { name: 'vi', normal: { m: { mean: 0.6, sd: Math.SQRT1_2 }, s: { mean: 0.6, sd: Math.SQRT1_2 } } },
  ],
  { truth: oracle, table: agree(approximations, { oracle }), quantilePoints: 80, gridPoints: 70 },
)

type Datum = [number, number] | { value: [number, number] }
const xy = (d: Datum | undefined): [number, number] | undefined => (Array.isArray(d) ? d : d?.value)
const named = (o: { series?: unknown }, name: string): { name?: string; data?: Datum[] }[] =>
  (o.series as { name?: string; data?: Datum[] }[]).filter((s) => s.name === name)

describe('posterior comparison charts', () => {
  it('draws the oracle as a density from the slope of its CDF and a curve per method in its own colour', () => {
    const o = posteriorComparisonView(record, { quantities: ['m'] })
    expect((o.grid as unknown[]).length).toBe(1)
    const truth = named(o, 'oracle')[0]
    expect(truth?.data).toHaveLength(70)
    // The oracle N(0.6, 0.5) has density 0.564 at its mean: the slope of its CDF finds it.
    const peak = Math.max(...(truth?.data ?? []).map((d) => xy(d)?.[1] ?? 0))
    expect(peak).toBeGreaterThan(0.54)
    expect(peak).toBeLessThan(0.58)
    expect(['nuts', 'narrow', 'vi'].map((n) => named(o, n).length)).toEqual([1, 1, 1])
    // The narrow method's bell is taller than the oracle's; the normal one is as tall.
    const top = (n: string): number => Math.max(...(named(o, n)[0]?.data ?? []).map((d) => xy(d)?.[1] ?? 0))
    expect(top('narrow')).toBeGreaterThan(top('nuts') * 1.5)
    expect(top('vi')).toBeCloseTo(0.564, 1)
  })

  it('draws the ECDFs over one another, and the difference from the oracle with its largest gap near the kernel KS', () => {
    const e = posteriorComparisonView(record, { quantities: ['m'], chart: 'ecdf' })
    expect(named(e, 'nuts')[0]?.data?.length).toBeGreaterThan(10)
    const d = posteriorComparisonView(record, { quantities: ['m'], chart: 'difference' })
    const gap = (n: string): number => Math.max(...(named(d, n)[0]?.data ?? []).map((p) => Math.abs(xy(p)?.[1] ?? 0)))
    const ks = (n: string): number =>
      record.agreement.find((r) => r.method === n && r.quantity === 'm')?.ks ?? Number.NaN
    expect(gap('narrow')).toBeGreaterThan(gap('nuts') * 2)
    // The quantiles are an 80-point sample of the draws: the gap is the kernel's KS to that resolution.
    expect(Math.abs(gap('narrow') - ks('narrow'))).toBeLessThan(0.03)
  })

  it('lays quantities out in small multiples and chooses those named', () => {
    expect((posteriorComparisonView(record).grid as unknown[]).length).toBe(2)
    expect((posteriorComparisonView(record, { quantities: ['s', 'zz'] }).grid as unknown[]).length).toBe(1)
  })

  it('has no difference without an oracle, and says so', () => {
    const free = posteriorComparison(approximations)
    expect(() => posteriorComparisonView(free, { chart: 'difference' })).toThrow(
      expect.objectContaining({ code: 'value' }),
    )
    expect((posteriorComparisonView(free).grid as unknown[]).length).toBe(2)
  })
})
