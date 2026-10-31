import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { simulateArma } from '../../testing/series.ts'
import { multiply } from './_matrix.ts'
import * as T from './index.ts'

const tree: T.HierarchyNode = {
  name: 'total',
  children: [
    { name: 'a', children: [{ name: 'a1' }, { name: 'a2' }] },
    { name: 'b', children: [{ name: 'b1' }, { name: 'b2' }, { name: 'b3' }] },
  ],
}
const { S, names, bottom } = T.treeSummingMatrix(tree)

describe('summing matrices', () => {
  it('builds S for a tree, breadth first', () => {
    expect(names).toEqual(['total', 'a', 'b', 'a1', 'a2', 'b1', 'b2', 'b3'])
    expect(bottom).toEqual(['a1', 'a2', 'b1', 'b2', 'b3'])
    expect(S[0]).toEqual([1, 1, 1, 1, 1])
    expect(S[1]).toEqual([1, 1, 0, 0, 0])
    expect(S[2]).toEqual([0, 0, 1, 1, 1])
    expect(S[7]).toEqual([0, 0, 0, 0, 1])
  })

  it('builds S for crossed groupings', () => {
    const g = T.groupedSummingMatrix([
      ['x', 'x', 'y', 'y'],
      ['p', 'q', 'p', 'q'],
    ])
    expect(g.names.slice(0, 5)).toEqual(['total', 'g0:x', 'g0:y', 'g1:p', 'g1:q'])
    expect(g.S[3]).toEqual([1, 0, 1, 0])
    expect(g.S).toHaveLength(9)
  })
})

describe('reconciliation', () => {
  const base = [[100, 42, 55, 20, 25, 18, 20, 15]]
  const residuals = Array.from({ length: 60 }, (_, t) =>
    names.map((_, i) => simulateArma(80 + i, 60, [0.3], [])[t] ?? 0),
  )
  const methods: T.ReconcileMethod[] = [
    'bottom-up',
    'ols',
    'wls-structural',
    'wls-variance',
    'mint-sample',
    'mint-shrink',
  ]

  it('makes forecasts coherent by every method', () => {
    expect(T.isCoherent(base, S)).toBe(false)
    for (const m of methods) {
      const r = T.reconcile(base, S, m, residuals)
      expect(T.isCoherent(r.reconciled, S)).toBe(true)
    }
  })

  it('leaves coherent forecasts unchanged, since G S = I (unbiasedness)', () => {
    const coherent = [multiply(S, [[3], [4], [5], [6], [7]]).map((r) => r[0] ?? 0)]
    for (const m of methods.filter((x) => x !== 'bottom-up')) {
      const r = T.reconcile(coherent, S, m, residuals)
      r.reconciled[0]?.forEach((v, i) => {
        expect(v).toBeCloseTo(coherent[0]?.[i] ?? 0, 9)
      })
      const gs = multiply(r.G, S)
      gs.forEach((row, i) => {
        row.forEach((v, j) => {
          expect(v).toBeCloseTo(i === j ? 1 : 0, 9)
        })
      })
    }
  })

  it('is the least-squares projection for ols, and keeps the bottom forecasts for bottom-up', () => {
    const small = [
      [1, 1],
      [1, 0],
      [0, 1],
    ]
    const r = T.reconcile([[10, 4, 5]], small, 'ols')
    expect(r.bottom[0]?.[0]).toBeCloseTo(13 / 3, 12)
    expect(r.bottom[0]?.[1]).toBeCloseTo(16 / 3, 12)
    const bu = T.reconcile([[10, 4, 5]], small, 'bottom-up')
    expect(bu.reconciled[0]).toEqual([9, 4, 5])
  })

  it('shrinks the covariance towards its diagonal and needs residuals for MinT', () => {
    const { covariance, lambda } = T.shrinkCovariance(residuals)
    expect(lambda).toBeGreaterThanOrEqual(0)
    expect(lambda).toBeLessThanOrEqual(1)
    const sample = (i: number, j: number) =>
      residuals.reduce((acc, e) => acc + (e[i] ?? 0) * (e[j] ?? 0), 0) / residuals.length
    expect(covariance[2]?.[2]).toBeCloseTo(sample(2, 2), 12)
    expect(covariance[0]?.[1]).toBeCloseTo((1 - lambda) * sample(0, 1), 12)
    expect(() => T.reconcile(base, S, 'mint-shrink')).toThrow()
    expect(() => T.reconcile([[1, 2]], S, 'ols')).toThrow()
  })

  it('reduces the squared error of incoherent base forecasts for the truth under OLS (a projection)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 1, max: 10, noNaN: true }), { minLength: 5, maxLength: 5 }),
        fc.array(fc.double({ min: -1, max: 1, noNaN: true }), { minLength: 8, maxLength: 8 }),
        (b, noise) => {
          const truth = multiply(
            S,
            b.map((v) => [v]),
          ).map((r) => r[0] ?? 0)
          const noisy = truth.map((v, i) => v + (noise[i] ?? 0))
          const r = T.reconcile([noisy], S, 'ols').reconciled[0] ?? []
          const err = (x: readonly number[]) => x.reduce((s, v, i) => s + (v - (truth[i] ?? 0)) ** 2, 0)
          expect(err(r)).toBeLessThanOrEqual(err(noisy) + 1e-9)
        },
      ),
      { numRuns: 30 },
    )
  })

  it('splits a total by historical proportions', () => {
    const history = [
      [1, 3],
      [2, 2],
      [3, 1],
    ]
    expect(T.topDownProportions(history, 'proportion-averages')).toEqual([0.5, 0.5])
    expect(T.topDownProportions(history).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
})
