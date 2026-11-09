import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { marchenkoPastur, marchenkoPasturEdges } from '../dists/index.ts'
import { key, normal } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { item, toArray } from '../tensor/index.ts'
import { symmetricEigen, toCorrelation } from './_matrix.ts'
import { denoiseCorrelation, detoneCorrelation } from './spectral.ts'

const seed = fc.integer({ min: 0, max: 2 ** 31 })

const noiseCorrelation = (s: number, n: number, p: number): number[][] =>
  toCorrelation(toArray(empiricalCovariance(normal(key(s), [n, p]))) as number[][]).correlation

describe('the eigenvalues of a large sample correlation matrix of white noise', () => {
  it('lie in the Marchenko-Pastur support and follow its distribution function', () => {
    const p = 80
    const n = 800
    const { lower, upper } = marchenkoPasturEdges(p / n)
    const law = marchenkoPastur(p / n)
    fc.assert(
      fc.property(seed, (s) => {
        const values = symmetricEigen(noiseCorrelation(s, n, p)).values
        expect(Math.min(...values)).toBeGreaterThan(lower - 0.1)
        expect(Math.max(...values)).toBeLessThan(upper + 0.1)
        // Kolmogorov-Smirnov distance of the eigenvalues to the limiting law.
        const ascending = [...values].sort((a, b) => a - b)
        let worst = 0
        for (const [i, v] of ascending.entries()) {
          const f = item(law.cdf(v))
          worst = Math.max(worst, Math.abs(f - i / p), Math.abs(f - (i + 1) / p))
        }
        expect(worst).toBeLessThan(0.08)
      }),
      { numRuns: 12 },
    )
  })
})

describe('denoising and detoning keep a correlation matrix valid', () => {
  it('denoise keeps the trace, the unit diagonal, the symmetry and positive semidefiniteness', () => {
    fc.assert(
      fc.property(
        seed,
        fc.integer({ min: 3, max: 14 }),
        fc.integer({ min: 4, max: 80 }),
        fc.constantFrom('constant', 'target', 'zero'),
        (s, p, n, method) => {
          const c = noiseCorrelation(s, n, p)
          const r = denoiseCorrelation(c, { sampleSize: n, method })
          let trace = 0
          for (let i = 0; i < p; i++) {
            trace += r.correlation[i]?.[i] ?? 0
            expect(r.correlation[i]?.[i]).toBeCloseTo(1, 10)
            for (let j = 0; j < p; j++) expect(r.correlation[i]?.[j]).toBeCloseTo(r.correlation[j]?.[i] ?? 0, 12)
          }
          expect(trace).toBeCloseTo(p, 10)
          expect(Math.min(...symmetricEigen(r.correlation).values)).toBeGreaterThan(-1e-9)
          expect(r.nSignal).toBeGreaterThanOrEqual(0)
          expect(r.nSignal).toBeLessThanOrEqual(p)
        },
      ),
      { numRuns: 40 },
    )
  })

  it('denoise leaves its own output unchanged when the constant bulk is already flat', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 4, max: 10 }), (s, p) => {
        const n = 60
        const once = denoiseCorrelation(noiseCorrelation(s, n, p), { sampleSize: n })
        const twice = denoiseCorrelation(once.correlation, { sampleSize: n, variance: once.variance })
        for (let i = 0; i < p; i++) expect(twice.correlation[i]?.[i]).toBeCloseTo(1, 10)
        expect(Math.min(...symmetricEigen(twice.correlation).values)).toBeGreaterThan(-1e-9)
      }),
      { numRuns: 20 },
    )
  })

  it('detone keeps the unit diagonal and symmetry and leaves p - m positive eigenvalues', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 4, max: 12 }), fc.integer({ min: 1, max: 3 }), (s, p, m) => {
        const r = detoneCorrelation(noiseCorrelation(s, 50, p), { components: m })
        for (let i = 0; i < p; i++) {
          expect(r.correlation[i]?.[i]).toBe(1)
          for (let j = 0; j < p; j++) expect(r.correlation[i]?.[j]).toBeCloseTo(r.correlation[j]?.[i] ?? 0, 12)
        }
        expect(r.removed).toHaveLength(m)
        expect(r.removed).toEqual(r.eigenvalues.slice(0, m))
      }),
      { numRuns: 30 },
    )
  })
})
