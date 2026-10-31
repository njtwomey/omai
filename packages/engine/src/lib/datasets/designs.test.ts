import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { carryoverExperiment, discontinuityDesign, networkExperiment, panelEffects } from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)
const col = (x: Tensor, j: number): number[] => {
  const [n = 0, d = 1] = x.shape
  return Array.from({ length: n }, (_, i) => Number(x.data[i * d + j] ?? 0))
}

/** Ordinary least squares of y on the given columns (plus an intercept). */
function ols(y: readonly number[], cols: readonly (readonly number[])[]): number[] {
  const rows = y.map((_, i) => [1, ...cols.map((c) => c[i] ?? 0)])
  return toFlat(lstsq(tensor(rows), tensor([...y]))) as number[]
}

describe('panelEffects', () => {
  it('keeps the truth consistent with the data: the observed outcome is the potential outcome of the arm', () => {
    const { data, truth } = generate(panelEffects, {
      key: key(1),
      knobs: { units: 60, periods: 8, timing: 'staggered', growth: 0.5 },
    })
    const treated = col(data.x, 2)
    const y = toFlat(data.y)
    const { control, treated: t1 } = truth.effects?.potentialOutcomes ?? { control: undefined, treated: undefined }
    const c = toFlat(control as Tensor)
    const t = toFlat(t1 as Tensor)
    const ite = toFlat(truth.effects?.ite as Tensor)
    y.forEach((v, i) => {
      expect(v).toBe(treated[i] === 1 ? t[i] : c[i])
      expect((t[i] ?? 0) - (c[i] ?? 0)).toBeCloseTo(ite[i] ?? 0, 12)
    })
    const att = mean(ite.filter((_, i) => treated[i] === 1))
    expect(truth.effects?.att).toBeCloseTo(att, 12)
    // Staggered adopters adopt at several periods, and the effect grows with event time.
    const adoption = (truth.extra?.['adoptionPeriod'] as number[]).filter((a) => a >= 0)
    expect(new Set(adoption).size).toBeGreaterThan(1)
    const byEvent = truth.extra?.['eventTimeEffects'] as Record<string, number>
    expect(byEvent['2'] ?? 0).toBeGreaterThan(byEvent['0'] ?? 0)
  })

  it('a two-by-two difference in differences recovers the effect when trends are parallel and misses it by the trend otherwise', () => {
    const did = (trend: number): number => {
      const { data, truth } = generate(panelEffects, {
        key: key(2),
        knobs: { units: 4000, periods: 6, adopters: 0.5, effect: 2, trend, noise: 0.5, selection: 1 },
      })
      const adopter = col(data.x, 3)
      const period = col(data.x, 1)
      const y = toFlat(data.y)
      const adoptionAt = Math.floor(6 / 2)
      const cell = (a: number, post: boolean): number =>
        mean(
          y.filter(
            (_, i) =>
              adopter[i] === a && (post ? (period[i] ?? 0) === adoptionAt : (period[i] ?? 0) === adoptionAt - 1),
          ),
        )
      expect(truth.extra?.['differentialTrend']).toBe(trend)
      return cell(1, true) - cell(1, false) - (cell(0, true) - cell(0, false))
    }
    expect(did(0)).toBeCloseTo(2, 0)
    expect(Math.abs(did(0) - 2)).toBeLessThan(0.15)
    // Adopters' untreated outcome gains `trend` more per period, so the estimate is off by the trend.
    expect(did(0.8) - 2).toBeGreaterThan(0.6)
  })

  it('never-adopters have no effect and are marked -1', () => {
    const { truth } = generate(panelEffects, { key: key(3), knobs: { units: 30, periods: 6, adopters: 0 } })
    expect((truth.extra?.['adoptionPeriod'] as number[]).every((a) => a === -1)).toBe(true)
    expect(truth.effects?.ate).toBe(0)
  })
})

describe('discontinuityDesign', () => {
  /** Local linear estimate of the jump at the cutoff in a window, from each side separately. */
  function jump(x: Tensor, cutoff: number, bandwidth: number, outcome: (i: number) => number): number {
    const r = col(x, 0)
    const side = (above: boolean): number => {
      const idx = r
        .map((_v, i) => i)
        .filter(
          (i) => (above ? (r[i] ?? 0) >= cutoff : (r[i] ?? 0) < cutoff) && Math.abs((r[i] ?? 0) - cutoff) < bandwidth,
        )
      const b = ols(idx.map(outcome), [idx.map((i) => (r[i] ?? 0) - cutoff)])
      return b[0] ?? 0
    }
    return side(true) - side(false)
  }

  it('has the planted jump in a sharp design', () => {
    const { data, truth } = generate(discontinuityDesign, {
      key: key(4),
      knobs: { n: 40000, effect: 3, cutoff: 0.5, curvature: 0.2 },
    })
    const y = toFlat(data.y)
    const est = jump(data.x, 0.5, 0.8, (i) => y[i] ?? 0)
    expect(Math.abs(est - 3)).toBeLessThan(0.25)
    expect(truth.extra?.['outcomeJump']).toBe(3)
    // Treatment is exactly the indicator of the running variable above the cutoff.
    const r = col(data.x, 0)
    const d = col(data.x, 1)
    expect(d.every((v, i) => v === ((r[i] ?? 0) >= 0.5 ? 1 : 0))).toBe(true)
  })

  it('a fuzzy design has a smaller outcome jump, and the Wald ratio recovers the effect', () => {
    const { data, truth } = generate(discontinuityDesign, {
      key: key(5),
      knobs: { n: 80000, effect: 2, design: 'fuzzy', compliance: 0.6, curvature: 0 },
    })
    const y = toFlat(data.y)
    const treated = col(data.x, 1)
    const outcomeJump = jump(data.x, 0, 1, (i) => y[i] ?? 0)
    const firstStage = jump(data.x, 0, 1, (i) => treated[i] ?? 0)
    expect(truth.extra?.['firstStageJump']).toBeCloseTo(0.6, 12)
    expect(Math.abs(firstStage - 0.6)).toBeLessThan(0.05)
    expect(Math.abs(outcomeJump - 1.2)).toBeLessThan(0.25)
    expect(Math.abs(outcomeJump / firstStage - 2)).toBeLessThan(0.5)
  })

  it('leaves the smooth function behind once the effect is removed, whatever the kink', () => {
    fc.assert(
      fc.property(fc.double({ min: -2, max: 2, noNaN: true }), fc.integer({ min: 0, max: 20 }), (kink, seed) => {
        const { data, truth } = generate(discontinuityDesign, {
          key: key(seed),
          knobs: { n: 3000, effect: 1.5, slopeChange: kink, noise: 0.2 },
        })
        const f = truth.extra?.['smoothFunction'] as (v: number) => number
        const y = toFlat(data.y)
        const r = col(data.x, 0)
        const d = col(data.x, 1)
        const resid = y.map((v, i) => v - 1.5 * (d[i] ?? 0) - f(r[i] ?? 0))
        expect(Math.abs(mean(resid))).toBeLessThan(0.05)
      }),
      { numRuns: 8 },
    )
  })
})

describe('networkExperiment', () => {
  it('a regression on own treatment and exposure recovers the direct effect and the spillover', () => {
    const { data, truth } = generate(networkExperiment, {
      key: key(6),
      knobs: { units: 6000, direct: 1, spillover: 2, noise: 0.5 },
    })
    const y = toFlat(data.y)
    const b = ols(y, [col(data.x, 0), col(data.x, 1), col(data.x, 3)])
    expect(Math.abs((b[1] ?? 0) - 1)).toBeLessThan(0.1)
    expect(Math.abs((b[2] ?? 0) - 2)).toBeLessThan(0.3)
    expect(truth.extra?.['totalEffect']).toBe(3)
    expect(truth.effects?.ate).toBe(3)
  })

  it('cluster randomisation puts neighbours in the same arm, raising the exposure of the treated', () => {
    const gap = (randomisation: 'unit' | 'cluster'): number => {
      const { data } = generate(networkExperiment, {
        key: key(7),
        knobs: { units: 4000, randomisation, clusterSize: 100, rewiring: 0.02 },
      })
      const t = col(data.x, 0)
      const e = col(data.x, 1)
      return mean(e.filter((_, i) => t[i] === 1)) - mean(e.filter((_, i) => t[i] === 0))
    }
    expect(Math.abs(gap('unit'))).toBeLessThan(0.03)
    expect(gap('cluster')).toBeGreaterThan(0.5)
  })

  it('exposure is the share of treated neighbours, computed from the edges in the truth', () => {
    const { data, truth } = generate(networkExperiment, { key: key(8), knobs: { units: 100, neighbours: 3 } })
    const edges = truth.extra?.['edges'] as [number, number][]
    const t = col(data.x, 0)
    const e = col(data.x, 1)
    const nb: number[][] = Array.from({ length: 100 }, () => [])
    for (const [a, b] of edges) {
      nb[a]?.push(b)
      nb[b]?.push(a)
    }
    nb.forEach((list, i) => {
      expect(e[i]).toBeCloseTo(list.length === 0 ? 0 : mean(list.map((j) => t[j] ?? 0)), 12)
    })
  })
})

describe('carryoverExperiment', () => {
  it('the naive difference in the second experiment is off by the stated bias, and unbiased without stickiness', () => {
    const naive = (stickiness: number): { diff: number; bias: number } => {
      const { data, truth } = generate(carryoverExperiment, {
        key: key(9),
        knobs: { users: 60000, firstEffect: 2, secondEffect: 0.5, persistence: 0.5, stickiness },
      })
      const y = toFlat(data.y)
      const t2 = col(data.x, 1)
      return {
        diff: mean(y.filter((_, i) => t2[i] === 1)) - mean(y.filter((_, i) => t2[i] === 0)),
        bias: truth.extra?.['naiveBias'] as number,
      }
    }
    const sticky = naive(0.8)
    expect(sticky.bias).toBeCloseTo(0.8, 12)
    expect(Math.abs(sticky.diff - (0.5 + sticky.bias))).toBeLessThan(0.05)
    const fresh = naive(0)
    expect(Math.abs(fresh.diff - 0.5)).toBeLessThan(0.05)
    // Adjusting for the first assignment removes the bias.
    const { data } = generate(carryoverExperiment, {
      key: key(9),
      knobs: { users: 60000, firstEffect: 2, secondEffect: 0.5, persistence: 0.5, stickiness: 0.8 },
    })
    const b = ols(toFlat(data.y), [col(data.x, 1), col(data.x, 0), col(data.x, 2)])
    expect(Math.abs((b[1] ?? 0) - 0.5)).toBeLessThan(0.05)
  })
})
