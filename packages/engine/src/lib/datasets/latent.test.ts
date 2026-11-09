import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { inv, sqrtm, svdvals } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import { fromStorage, matmul, type Tensor, toFlat, transpose } from '../tensor/index.ts'
import { mixedSources, sparseCodedSignal, sparseDictionary, toySuperposition, twoViews } from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length)
const column = (t: Tensor, j: number): number[] => {
  const [n = 0, d = 1] = t.shape
  return Array.from({ length: n }, (_, i) => Number(t.data[i * d + j] ?? 0))
}

describe('mixedSources', () => {
  it('mixes the sources by the recorded matrix, and the inverse recovers them', () => {
    const { data, truth } = generate(mixedSources, { key: key(1), knobs: { n: 500, sources: 3 } })
    const a = (truth.extra ?? {})['mixing'] as Tensor
    const recovered = matmul(data.x, transpose(inv(a)))
    toFlat(recovered).forEach((v, i) => {
      expect(v).toBeCloseTo(data.y.data[i] ?? 0, 9)
    })
    const sv = toFlat(svdvals(a))
    expect((sv[0] ?? 0) / (sv[2] ?? 1)).toBeLessThan(10.01)
  })

  it('has unit-variance uncorrelated sources whose kurtosis tells the kinds apart', () => {
    const kurt = (kind: 'laplace' | 'uniform'): number => {
      const { data } = generate(mixedSources, { key: key(2), knobs: { n: 20000, sources: 2, kind } })
      const s = column(data.y, 0)
      const s2 = column(data.y, 1)
      expect(Math.abs(mean(s.map((v) => v * v)) - 1)).toBeLessThan(0.05)
      expect(Math.abs(mean(s.map((v, i) => v * (s2[i] ?? 0))))).toBeLessThan(0.03)
      return mean(s.map((v) => v ** 4)) - 3
    }
    expect(kurt('laplace')).toBeGreaterThan(2)
    expect(kurt('uniform')).toBeLessThan(-1)
  })
})

describe('twoViews', () => {
  /** Sample canonical correlations of two data matrices, via the whitened cross-covariance. */
  function sampleCorrelations(x: Tensor, y: Tensor): number[] {
    const [n = 0, da = 1] = x.shape
    const db = y.shape[1] ?? 1
    const cov = (u: Tensor, v: Tensor, p: number, q: number): Tensor => {
      const out = new Float64Array(p * q)
      const mu = Array.from({ length: p }, (_, j) => mean(column(u, j)))
      const mv = Array.from({ length: q }, (_, j) => mean(column(v, j)))
      for (let i = 0; i < n; i++)
        for (let a = 0; a < p; a++)
          for (let b = 0; b < q; b++)
            out[a * q + b] =
              (out[a * q + b] ?? 0) +
              (((u.data[i * p + a] ?? 0) - (mu[a] ?? 0)) * ((v.data[i * q + b] ?? 0) - (mv[b] ?? 0))) / n
      return fromStorage(out, [p, q], 'float64')
    }
    const white = matmul(matmul(inv(sqrtm(cov(x, x, da, da))), cov(x, y, da, db)), inv(sqrtm(cov(y, y, db, db))))
    return toFlat(svdvals(white))
  }

  it('has sample canonical correlations near the population ones', () => {
    const { data, truth } = generate(twoViews, {
      key: key(3),
      knobs: { n: 40000, latent: 2, viewA: 4, viewB: 3, noise: 1 },
    })
    const population = (truth.extra ?? {})['canonicalCorrelations'] as number[]
    expect(population).toHaveLength(2)
    const sample = sampleCorrelations(data.x, data.y)
    population.forEach((rho, i) => {
      expect(Math.abs((sample[i] ?? 0) - rho)).toBeLessThan(0.03)
      expect(rho).toBeGreaterThan(0)
      expect(rho).toBeLessThan(1)
    })
    // Beyond the shared factors the correlations vanish.
    expect(sample[2] ?? 1).toBeLessThan(0.05)
  })

  it('more private noise means weaker canonical correlations', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 15 }), (seed) => {
        const rho = (noise: number): number => {
          const { truth } = generate(twoViews, { key: key(seed), knobs: { n: 20, noise } })
          return ((truth.extra ?? {})['canonicalCorrelations'] as number[])[0] ?? 0
        }
        expect(rho(3)).toBeLessThan(rho(0.3))
      }),
      { numRuns: 10 },
    )
  })
})

describe('toySuperposition', () => {
  it('has the stated sparsity and geometric importances, and the targets equal the features', () => {
    const { data, truth } = generate(toySuperposition, {
      key: key(4),
      knobs: { n: 5000, features: 10, sparsity: 0.8, decay: 0.7 },
    })
    const zeros = mean(Array.from(data.x.data, (v) => (v === 0 ? 1 : 0)))
    expect(Math.abs(zeros - 0.8)).toBeLessThan(0.02)
    const importance = (truth.extra ?? {})['importance'] as number[]
    expect(importance[3]).toBeCloseTo(0.7 ** 3, 12)
    expect(Array.from(data.y.data)).toEqual(Array.from(data.x.data))
    const loss = (truth.extra ?? {})['weightedLoss'] as (r: ArrayLike<number>) => number
    expect(loss(data.x.data)).toBe(0)
    expect(loss(new Float64Array(5000 * 10))).toBeCloseTo((truth.extra ?? {})['meanSquare'] as number, 12)
    expect((truth.extra ?? {})['meanSquare'] as number).toBeGreaterThan(0)
  })
})

describe('sparseDictionary', () => {
  it('writes sparse codes through unit directions, exactly without noise', () => {
    const { data, truth } = generate(sparseDictionary, {
      key: key(5),
      knobs: { n: 3000, features: 5, dim: 7, sparsity: 0.7, noise: 0 },
    })
    const dictionary = (truth.extra ?? {})['dictionary'] as number[][]
    expect(dictionary).toHaveLength(5)
    for (const row of dictionary) expect(Math.hypot(...row)).toBeCloseTo(1, 12)
    const codes = data.y
    const zeros = mean(Array.from(codes.data, (v) => (v === 0 ? 1 : 0)))
    expect(Math.abs(zeros - 0.7)).toBeLessThan(0.02)
    for (const v of codes.data) expect(v === 0 || (v >= 0.5 && v <= 1.5)).toBe(true)
    // x is codes times the dictionary.
    const d = fromStorage(Float64Array.from(dictionary.flat()), [5, 7], 'float64')
    const x = toFlat(matmul(codes, d))
    for (let i = 0; i < x.length; i++) expect(x[i]).toBeCloseTo(data.x.data[i] ?? 0, 12)
    expect((truth.extra ?? {})['coherence'] as number).toBeLessThan(1)
  })
})

describe('sparseCodedSignal', () => {
  it('writes every signal from exactly k atoms of a unit-norm dictionary, exactly without noise', () => {
    const { data, truth } = generate(sparseCodedSignal, {
      key: key(3),
      knobs: { n: 200, components: 9, dim: 12, nonzeros: 3, noise: 0 },
    })
    const extra = truth.extra ?? {}
    const dictionary = extra['dictionary'] as number[][]
    const supports = extra['supports'] as number[][]
    for (const row of dictionary) expect(Math.hypot(...row)).toBeCloseTo(1, 12)
    const codes = toFlat(data.y)
    for (let i = 0; i < 200; i++) {
      const row = codes.slice(i * 9, (i + 1) * 9)
      expect(row.filter((v) => v !== 0)).toHaveLength(3)
      expect(supports[i]).toEqual(row.flatMap((v, j) => (v === 0 ? [] : [j])))
      for (const v of row) expect(v === 0 || (Math.abs(v) >= 0.5 && Math.abs(v) <= 1.5)).toBe(true)
    }
    expect(codes.some((v) => v < 0)).toBe(true)
    const d = fromStorage(Float64Array.from(dictionary.flat()), [9, 12], 'float64')
    const x = toFlat(matmul(data.y, d))
    for (let i = 0; i < x.length; i++) expect(x[i]).toBeCloseTo(data.x.data[i] ?? 0, 12)
    expect(extra['coherence'] as number).toBeLessThan(1)
  })

  it('can make all coefficients positive and is keyed', () => {
    const knobs = { n: 50, components: 6, dim: 8, nonzeros: 2, signs: 'positive' } as const
    const a = generate(sparseCodedSignal, { key: key(1), knobs })
    expect(toFlat(a.data.y).every((v) => v >= 0)).toBe(true)
    expect(toFlat(generate(sparseCodedSignal, { key: key(1), knobs }).data.x)).toEqual(toFlat(a.data.x))
  })
})
