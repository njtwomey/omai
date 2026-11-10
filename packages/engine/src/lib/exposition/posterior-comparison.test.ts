import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { normal } from '../dists/index.ts'
import { agree } from '../probprog/validation/agree.ts'
import { distributionOracle } from '../probprog/validation/oracle.ts'
import { key } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { comparisonTable, posteriorComparison } from './posterior-comparison.ts'

// The exact posterior N(0.6, 0.5); an iid sample of it, a sample that is too narrow, and a shifted one.
const exact = normal(0.6, Math.SQRT1_2)
const iid = (seed: number, scale = 1, shift = 0): number[] =>
  Array.from({ length: 1500 }, (_, i) => 0.6 + shift + scale * (item(exact.sample(key(seed * 10_000 + i), [])) - 0.6))
const oracle = { m: distributionOracle('m', exact) }
const approximations = [
  { name: 'iid', draws: { m: iid(1) } },
  { name: 'narrow', draws: { m: iid(2, 0.5) } },
  { name: 'shifted', draws: { m: iid(3, 1, 1.2) } },
]
const table = agree(approximations, { oracle })

describe('posteriorComparison', () => {
  const r = posteriorComparison(approximations, { truth: oracle, table, quantilePoints: 100, gridPoints: 60 })

  it('keeps the kernel table, the oracle on a grid that holds every method, and equal-mass quantiles of each method', () => {
    expect(r.kind).toBe('posteriorComparison')
    expect([r.oracle, r.methods]).toEqual(['oracle', ['iid', 'narrow', 'shifted']])
    expect(r.agreement).toEqual(table.rows)
    const q = r.quantities[0]
    expect(q?.truth?.x).toHaveLength(60)
    expect(q?.truth?.cdf[0]).toBeLessThan(0.01)
    expect(q?.truth?.cdf.at(-1)).toBeGreaterThan(0.99)
    expect(q?.truth?.cdf.every((v, i, a) => i === 0 || v >= (a[i - 1] ?? 0))).toBe(true)
    // The shifted method's draws reach past the oracle's 99.95% point: the grid reaches them.
    expect(q?.truth?.x.at(-1)).toBeGreaterThanOrEqual(Math.max(...(q?.methods[2]?.quantiles ?? [])))
    for (const m of q?.methods ?? []) expect(m.quantiles).toHaveLength(100)
  })

  it('has the quantiles of the draws: their median and spread, and the moments of the draws', () => {
    const iidQ = r.quantities[0]?.methods[0]
    const sorted = [...(approximations[0]?.draws.m ?? [])].sort((a, b) => a - b)
    expect(iidQ?.quantiles?.[50]).toBeCloseTo(((sorted[749] ?? 0) + (sorted[750] ?? 0)) / 2, 1)
    expect(iidQ?.mean).toBeCloseTo(table.rows[0]?.mean ?? Number.NaN, 12)
    expect(iidQ?.sd).toBeCloseTo(table.rows[0]?.sd ?? Number.NaN, 12)
  })

  it('weights the quantiles: a draw with the weight of all the others is the median', () => {
    const w = posteriorComparison([{ name: 'w', draws: { x: [0, 1, 2, 3] }, weights: [1, 1, 1, 97] }], {
      quantilePoints: 10,
    })
    expect(w.quantities[0]?.methods[0]?.quantiles?.[5]).toBeGreaterThan(2.5)
    expect(w.quantities[0]?.methods[0]?.mean).toBeCloseTo(2.94, 10)
  })

  it('takes a normal as a method, and an oracle-free comparison of draws and normals', () => {
    const v = posteriorComparison([{ name: 'vi', normal: { m: { mean: 0.5, sd: 0.8 } } }, approximations[0] as never], {
      truth: oracle,
    })
    expect(v.quantities[0]?.methods[0]).toMatchObject({ name: 'vi', normal: { mean: 0.5, sd: 0.8 } })
    expect(v.quantities[0]?.methods[0]?.quantiles).toBeUndefined()
    const free = posteriorComparison(approximations.slice(0, 1))
    expect(free.quantities[0]?.truth).toBeUndefined()
  })

  it('flags the cells beyond tolerance: the narrow method in the sd ratio, the shifted in the mean, the exact in none', () => {
    const t = comparisonTable(r)
    expect(t.columns).toEqual(['method', 'quantity', 'mean error (sd)', 'sd ratio', 'KS', 'W1 (sd)', 'KL', 'ESS'])
    const at = (row: number, col: number): string | undefined =>
      t.marks?.find((m) => m.row === row && m.col === col)?.status
    expect(at(0, 2)).toBeUndefined()
    expect(at(0, 3)).toBeUndefined()
    expect(at(1, 3)).toBe('bad')
    expect(at(2, 2)).toBe('bad')
    expect(at(2, 4)).toBe('bad')
    expect(t.caption).toContain('compared with oracle')
    const loose = comparisonTable(
      posteriorComparison(approximations, {
        truth: oracle,
        table,
        tolerances: { sdRatio: [0.1, 10], meanError: 5, meanZ: 1e9, ks: 1 },
      }),
    )
    expect(loose.marks).toEqual([])
  })

  it('shows a measure the kernel could not compute as missing, not as an error', () => {
    const row = { ...table.rows[0], meanZ: Number.POSITIVE_INFINITY, kl: Number.NaN } as (typeof table.rows)[number]
    const odd = posteriorComparison(approximations.slice(0, 1), { table: { ...table, rows: [row] } })
    expect(comparisonTable(odd).cells[0]?.[6]).toBeNull()
  })

  it('checks its input', () => {
    expect(() => posteriorComparison([])).toThrow(expect.objectContaining({ code: 'shape' }))
    expect(() => posteriorComparison(approximations, { quantities: ['nope'] })).toThrow(OmaiError)
    expect(() => posteriorComparison([{ name: 'a', draws: { m: [1] } }])).toThrow(OmaiError)
  })
})
