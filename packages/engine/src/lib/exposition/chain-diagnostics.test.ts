import { describe, expect, it } from 'vitest'
import { OmaiError } from '../base/index.ts'
import { key, normal } from '../random/index.ts'
import { autocorrelation, bfmi, chainEssCurves } from '../sampling/index.ts'
import { toArray } from '../tensor/index.ts'
import { chainDiagnostics, chainsFromTape, pooledRanks, rankCounts } from './chain-diagnostics.ts'

const iid = (seed: number, n: number, shift = 0): number[] => Array.from(normal(key(seed), [n]).data, (v) => v + shift)

/** An AR(1) chain: strongly autocorrelated when `rho` is near 1. */
function ar1(seed: number, n: number, rho: number): number[] {
  const e = iid(seed, n)
  const out: number[] = []
  let x = 0
  for (const v of e) {
    x = rho * x + Math.sqrt(1 - rho * rho) * v
    out.push(x)
  }
  return out
}

describe('pooled ranks', () => {
  it('rank all the draws together, ties by their mean rank, and the histograms are made of them', () => {
    expect(
      pooledRanks([
        [3, 1],
        [2, 2],
      ]),
    ).toEqual([
      [3, 0],
      [1.5, 1.5],
    ])
    expect(
      rankCounts(
        [
          [1, 3, 5, 7],
          [2, 4, 6, 8],
        ],
        2,
      ),
    ).toEqual([
      [2, 2],
      [2, 2],
    ])
    expect(
      rankCounts(
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ],
        2,
      ),
    ).toEqual([
      [4, 0],
      [0, 4],
    ])
  })
})

describe('autocorrelation of the chains', () => {
  it('is the library autocorrelation of each chain at the lags asked for, and the mean over chains', () => {
    const chains = [ar1(1, 300, 0.8), ar1(2, 300, 0.8), ar1(3, 300, 0.2)]
    const r = chainDiagnostics({ x: chains }, { acfLags: 25 })
    const acf = r.quantities[0]?.acf
    expect(acf?.lags).toBe(25)
    const lib = toArray(autocorrelation(chains)) as number[][]
    for (const [c, row] of lib.entries())
      for (let k = 0; k <= 25; k++) expect(acf?.chains[c]?.[k]).toBeCloseTo(row[k] as number, 12)
    expect(acf?.mean[3]).toBeCloseTo(((lib[0]?.[3] ?? 0) + (lib[1]?.[3] ?? 0) + (lib[2]?.[3] ?? 0)) / 3, 12)
    // The slow chains decay like rho^k, the quick one is near zero after one lag.
    expect(acf?.chains[0]?.[5]).toBeGreaterThan(0.2)
    expect(Math.abs(acf?.chains[2]?.[5] ?? 1)).toBeLessThan(0.2)
  })

  it('is capped at half the draws, can be left out, and a constant chain is 1 then 0', () => {
    expect(chainDiagnostics({ x: [iid(1, 20)] }).quantities[0]?.acf?.lags).toBe(10)
    expect(chainDiagnostics({ x: [iid(1, 20)] }, { acfLags: 0 }).quantities[0]?.acf).toBeUndefined()
    const flat = chainDiagnostics({ x: [new Array<number>(30).fill(2)] }, { acfLags: 3 }).quantities[0]?.acf
    expect(flat?.chains[0]).toEqual([1, 0, 0, 0])
  })
})

describe('ECDF difference of the chains', () => {
  it('stays inside the simultaneous band for chains of one distribution, and leaves it for chains that disagree', () => {
    const agree = chainDiagnostics({ x: [iid(11, 400), iid(12, 400), iid(13, 400), iid(14, 400)] })
    const e = agree.ecdf
    expect(e?.z).toHaveLength(40)
    expect([e?.level, e?.draws]).toEqual([0.95, 400])
    const outside = (r: typeof agree): number =>
      (r.quantities[0]?.ecdfDiff ?? []).filter((row) =>
        row.some((d, j) => d < (r.ecdf?.lower[j] ?? 0) - 1e-12 || d > (r.ecdf?.upper[j] ?? 0) + 1e-12),
      ).length
    expect(outside(agree)).toBe(0)
    const apart = chainDiagnostics({ x: [iid(11, 400), iid(12, 400), iid(13, 400, 1.5), iid(14, 400, -1.5)] })
    expect(outside(apart)).toBeGreaterThanOrEqual(2)
    // A chain that sits high has fewer low ranks: its ECDF lies below the uniform CDF.
    const high = apart.quantities[0]?.ecdfDiff?.[2] ?? []
    expect(Math.min(...high)).toBeLessThan(-0.1)
  })

  it('is the ECDF of the pooled ranks at the grid, thinned to the most draws asked for, and can be left out', () => {
    const chains = [iid(1, 60), iid(2, 60)]
    const r = chainDiagnostics({ x: chains }, { ecdfPoints: 3 })
    const ranks = pooledRanks(chains)
    const z = r.ecdf?.z ?? []
    expect(z).toEqual([0.25, 0.5, 0.75])
    for (const [c, diff] of (r.quantities[0]?.ecdfDiff ?? []).entries())
      for (const [j, zj] of z.entries()) {
        const at = (ranks[c] ?? []).filter((v) => (v + 0.5) / 120 <= zj).length / 60
        expect(diff[j]).toBeCloseTo(at - zj, 12)
      }
    expect(chainDiagnostics({ x: [iid(1, 3000)] }).ecdf?.draws).toBe(2000)
    const none = chainDiagnostics({ x: chains }, { ecdfPoints: 0 })
    expect([none.ecdf, none.quantities[0]?.ecdfDiff]).toEqual([undefined, undefined])
  })
})

describe('energy of the chains', () => {
  it('has the E-BFMI of the library per chain, and two densities on common bins', () => {
    // A chain whose energy hardly moves between draws has a low E-BFMI: here a slow random walk.
    const walk = ar1(5, 400, 0.98)
    const free = iid(6, 400)
    const r = chainDiagnostics({ x: [iid(1, 400), iid(2, 400)] }, { energy: [walk, free] })
    const lib = toArray(bfmi([walk, free])) as number[]
    expect(r.energy?.bfmi[0]).toBeCloseTo(lib[0] as number, 12)
    expect(r.energy?.bfmi[1]).toBeCloseTo(lib[1] as number, 12)
    expect(r.energy?.bfmi[0]).toBeLessThan(0.3)
    expect(r.energy?.bfmi[1]).toBeGreaterThan(1.5)
    const e = r.energy
    expect(e?.edges).toHaveLength(31)
    // Densities integrate to one.
    const w = ((e?.edges[1] ?? 0) - (e?.edges[0] ?? 0)) as number
    expect((e?.marginal ?? []).reduce((s, v) => s + v * w, 0)).toBeCloseTo(1, 10)
    expect((e?.transition ?? []).reduce((s, v) => s + v * w, 0)).toBeCloseTo(1, 10)
    expect(chainDiagnostics({ x: [iid(1, 10)] }).energy).toBeUndefined()
  })

  it('reads the energy of a tape beside the draws, the warm-up left out', () => {
    const t = (step: number, name: string, shape: number[], values: number[]) => ({
      type: 'tensor',
      step,
      name,
      shape,
      values,
    })
    const events = [
      { type: 'metric', step: 1, name: 'mcmc/stage', value: 1 },
      t(1, 'mcmc/position', [2, 1], [9, 9]),
      t(1, 'mcmc/energy', [2], [99, 99]),
      { type: 'metric', step: 2, name: 'mcmc/stage', value: 2 },
      t(2, 'mcmc/position', [2, 1], [0, 1]),
      t(2, 'mcmc/energy', [2], [3, 4]),
      { type: 'metric', step: 3, name: 'mcmc/stage', value: 2 },
      t(3, 'mcmc/position', [2, 1], [2, 3]),
      t(3, 'mcmc/energy', [2], [5, 6]),
    ]
    const read = chainsFromTape({ events })
    expect(read.energy).toEqual([
      [3, 5],
      [4, 6],
    ])
    expect(read.draws['position[0]']).toEqual([
      [0, 2],
      [1, 3],
    ])
  })
})

describe('ESS curves', () => {
  it('are kept by quantity name beside the draws and checked finite by the record', () => {
    const chains = [iid(1, 200), iid(2, 200)]
    const essCurves = chainEssCurves({ x: chains }, { probs: [0.1, 0.5, 0.9], points: 5 })
    const r = chainDiagnostics({ x: chains, y: chains }, { essCurves })
    expect(r.quantities[0]?.essCurves?.quantile.probs).toEqual([0.1, 0.5, 0.9])
    expect(r.quantities[1]?.essCurves).toBeUndefined()
    expect(() =>
      chainDiagnostics(
        { x: chains },
        {
          essCurves: {
            x: { quantile: { probs: [0.5], ess: [Number.NaN] }, evolution: { draws: [], bulk: [], tail: [] } },
          },
        },
      ),
    ).toThrow(OmaiError)
  })
})

describe('the record with the new data', () => {
  it('keeps the old fields as they were', () => {
    const r = chainDiagnostics(
      {
        mu: [
          [1, 2, 3, 4],
          [2, 3, 4, 5],
        ],
      },
      { bins: 2, summary: [{ site: 'mu', name: 'mu', rhat: 1.01, ess: 300 }] },
    )
    // Pooled ranks 0, 1.5, 3.5, 5.5 and 1.5, 3.5, 5.5, 7 of eight draws, in two bins of four ranks.
    expect([r.quantities[0]?.rhat, r.quantities[0]?.mean, r.quantities[0]?.ranks.counts]).toEqual([
      1.01,
      3,
      [
        [2, 2],
        [1, 3],
      ],
    ])
  })
})
