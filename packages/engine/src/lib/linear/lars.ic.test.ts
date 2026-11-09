import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { larsPath, lassoLarsIC } from './lars.ts'

const make = (n: number, d: number, seed: number): { x: number[][]; y: number[] } => {
  const x = Array.from({ length: n }, (_, i) =>
    Array.from({ length: d }, (_, j) => Math.sin(seed + i * (j + 1.7)) + 0.2 * j),
  )
  const y = x.map((r, i) => 3 * (r[0] ?? 0) - 2 * (r[3] ?? 0) + 1 + 0.05 * Math.cos(seed * 2 + i * 3.1))
  return { x, y }
}

describe('lassoLarsIC', () => {
  const data = make(60, 6, 1)

  it('picks the sparse model: the true support, a criterion lowest at the chosen knot of the lasso path', () => {
    const f = lassoLarsIC({ criterion: 'bic' }).fit(data)
    expect(f.params.coef.map((c) => c !== 0)).toEqual([true, false, false, true, false, false])
    expect(f.params.criterion.length).toBe(f.params.path.alphas.length)
    expect(f.params.criterion[f.params.bestKnot]).toBe(Math.min(...f.params.criterion))
    expect(f.params.alpha).toBe(f.params.path.alphas[f.params.bestKnot])
    expect(f.params.degreesOfFreedom[0]).toBe(0)
    // The path is the lasso path of larsPath.
    expect(f.params.path.alphas).toEqual(larsPath(data, { method: 'lasso' }).alphas)
    expect(f.score?.(data)).toBeGreaterThan(0.99)
  })

  it('the criterion follows its definition at a knot', () => {
    const f = lassoLarsIC({ criterion: 'aic', noiseVariance: 0.5 }).fit(data)
    const k = f.params.bestKnot
    const coef = f.params.path.coefs[k] ?? []
    const fitted = data.x.map((r) => r.reduce((a, v, j) => a + v * (coef[j] ?? 0), 0))
    const yMean = data.y.reduce((a, v) => a + v, 0) / data.y.length
    const xMean = (data.x[0] ?? []).map((_, j) => data.x.reduce((a, r) => a + (r[j] ?? 0), 0) / data.x.length)
    const rss = data.x.reduce((a, r, i) => {
      const centred = r.reduce((s, v, j) => s + (v - (xMean[j] ?? 0)) * (coef[j] ?? 0), 0)
      return a + ((data.y[i] ?? 0) - yMean - centred) ** 2
    }, 0)
    expect(fitted.length).toBe(60)
    const df = coef.filter((v) => v !== 0).length
    expect(f.params.criterion[k]).toBeCloseTo(60 * Math.log(2 * Math.PI * 0.5) + rss / 0.5 + 2 * df, 8)
  })

  it('cannot estimate the noise with no more rows than features, but takes it as given', () => {
    const wide = make(10, 30, 2)
    expect(() => lassoLarsIC().fit(wide)).toThrow()
    expect(() => lassoLarsIC({ noiseVariance: 0.1 }).fit(wide)).not.toThrow()
  })

  it('a session adds knots and fittedAt is the best knot so far', () => {
    const s = lassoLarsIC({ criterion: 'bic' }).session(data, { logs: 'all' })
    s.advance({ steps: 2 })
    expect(s.fittedAt(2).params.path.alphas.length).toBe(3)
    expect(toFlat(s.fittedAt(2).predict(data.x)).length).toBe(60)
  })

  it('the BIC never keeps more variables than the AIC (n >= 8)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 20, max: 60 }),
        fc.integer({ min: 2, max: 7 }),
        fc.integer({ min: 0, max: 500 }),
        (n, d, seed) => {
          const dset = make(n, d, seed)
          const aic = lassoLarsIC({ criterion: 'aic' }).fit(dset).params
          const bic = lassoLarsIC({ criterion: 'bic', noiseVariance: aic.noiseVariance }).fit(dset).params
          expect(bic.degreesOfFreedom[bic.bestKnot] ?? 0).toBeLessThanOrEqual(aic.degreesOfFreedom[aic.bestKnot] ?? 0)
        },
      ),
      { numRuns: 30 },
    )
  })
})
