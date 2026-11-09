import { describe, expect, it } from 'vitest'
import { lookup } from '../base/index.ts'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import {
  blockCovarianceData,
  datasetRegistry,
  factorCovarianceData,
  spikedCovarianceData,
  toeplitzCovarianceData,
  wishartCovarianceData,
} from './index.ts'

const truthOf = (t: { readonly extra?: Readonly<Record<string, unknown>> }): Record<string, unknown> => t.extra ?? {}
const maxAbs = (a: Tensor, b: Tensor): number => Math.max(...toFlat(a).map((v, i) => Math.abs(v - (toFlat(b)[i] ?? 0))))

describe('the generators with a known covariance', () => {
  const cases = [
    ['factorCovarianceData', factorCovarianceData, { n: 40000, p: 6, factors: 2 }],
    ['spikedCovarianceData', spikedCovarianceData, { n: 40000, p: 6, spikes: 2, strength: 8 }],
    ['blockCovarianceData', blockCovarianceData, { n: 40000, p: 6, blocks: 2 }],
    ['toeplitzCovarianceData', toeplitzCovarianceData, { n: 40000, p: 6 }],
    ['wishartCovarianceData', wishartCovarianceData, { n: 40000, p: 6, df: 12 }],
  ] as const

  for (const [name, g, knobs] of cases) {
    it(`${name}: the sample covariance converges to the truth, which is a valid covariance`, () => {
      const { data, truth } = generate(g, { key: key(3), knobs })
      const extra = truthOf(truth)
      expect(data.x.shape).toEqual([knobs.n, knobs.p])
      const cov = extra['covariance'] as Tensor
      expect(maxAbs(empiricalCovariance(data.x), cov)).toBeLessThan(0.12)
      const eigenvalues = extra['eigenvalues'] as number[]
      expect(eigenvalues).toHaveLength(knobs.p)
      expect([...eigenvalues].sort((a, b) => b - a)).toEqual(eigenvalues)
      expect(eigenvalues[eigenvalues.length - 1] ?? -1).toBeGreaterThan(-1e-9)
      const trace = (toArray(cov as never) as number[][]).reduce((a, row, i) => a + (row[i] ?? 0), 0)
      expect(eigenvalues.reduce((a, v) => a + v, 0)).toBeCloseTo(trace, 8)
      const corr = toArray(extra['correlation'] as never) as number[][]
      for (let i = 0; i < knobs.p; i++) expect(corr[i]?.[i]).toBeCloseTo(1, 12)
    })

    it(`${name}: is reproducible from the key and records its recipe`, () => {
      const a = generate(g, { key: key(5), knobs: { ...knobs, n: 30 } })
      const b = generate(g, { key: key(5), knobs: { ...knobs, n: 30 } })
      expect(toFlat(a.data.x)).toEqual(toFlat(b.data.x))
      expect(a.data.meta.recipe?.generator).toBe(name)
    })

    it(`${name}: is in the registry with a tabular modality`, () => {
      const entry = lookup(datasetRegistry, name)
      expect(entry.modality).toBe('tabular')
      expect(entry.real).toBe(false)
    })
  }

  it('the precision is the inverse of the covariance', () => {
    for (const [, g, knobs] of cases) {
      const extra = truthOf(generate(g, { key: key(1), knobs: { ...knobs, n: 20 } }).truth)
      const c = toArray(extra['covariance'] as never) as number[][]
      const pr = toArray(extra['precision'] as never) as number[][]
      const prod = c.flatMap((row, i) =>
        row.map((_, j) => row.reduce((sum, v, k) => sum + v * (pr[k]?.[j] ?? 0), 0) - (i === j ? 1 : 0)),
      )
      expect(Math.max(...prod.map(Math.abs))).toBeLessThan(1e-8)
    }
  })
})

describe('factorCovarianceData', () => {
  it('has the covariance Lambda Lambda^T + Psi and factor scores as targets', () => {
    const { data, truth } = generate(factorCovarianceData, { key: key(2), knobs: { n: 50, p: 5, factors: 2 } })
    const extra = truthOf(truth)
    const l = toArray(extra['loadings'] as never) as number[][]
    const psi = extra['uniquenesses'] as number[]
    const cov = toArray(extra['covariance'] as never) as number[][]
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 5; j++)
        expect(cov[i]?.[j]).toBeCloseTo(
          (l[i] ?? []).reduce((s, v, k) => s + v * (l[j]?.[k] ?? 0), 0) + (i === j ? (psi[i] ?? 0) : 0),
          12,
        )
    expect(data.y.shape).toEqual([50, 2])
  })

  it('rejects more factors than variables', () => {
    expect(() => generate(factorCovarianceData, { knobs: { n: 10, p: 3, factors: 5 } })).toThrow(/factors/)
  })
})

describe('spikedCovarianceData', () => {
  it('has the spike eigenvalues at the top of the spectrum and the BBP truth of its aspect ratio', () => {
    const { truth } = generate(spikedCovarianceData, {
      key: key(4),
      knobs: { n: 100, p: 25, spikes: 2, strength: 6, spread: 0.5 },
    })
    const extra = truthOf(truth)
    const eig = extra['eigenvalues'] as number[]
    expect(extra['spikes']).toEqual([6, 3.5])
    expect(eig[0]).toBeCloseTo(6, 10)
    expect(eig[1]).toBeCloseTo(3.5, 10)
    expect(eig[2]).toBeCloseTo(1, 10)
    expect(extra['bbpThreshold']).toBe(1.5)
    expect(extra['detectable']).toEqual([true, true])
    const limits = extra['limitingEigenvalues'] as number[]
    expect(limits[0]).toBeCloseTo(6 * (1 + 0.25 / 5), 12)
  })

  it('marks a spike below the threshold as undetectable', () => {
    const extra = truthOf(generate(spikedCovarianceData, { knobs: { n: 100, p: 100, spikes: 1, strength: 1.5 } }).truth)
    expect(extra['bbpThreshold']).toBe(2)
    expect(extra['detectable']).toEqual([false])
    expect(extra['limitingOverlaps']).toEqual([0])
  })

  it('needs no spike', () => {
    const { data, truth } = generate(spikedCovarianceData, { knobs: { n: 10, p: 4, spikes: 0 } })
    expect(data.y.shape).toEqual([10, 0])
    expect(truthOf(truth)['eigenvalues']).toEqual([1, 1, 1, 1])
  })
})

describe('blockCovarianceData', () => {
  it('correlates within blocks by `within` and across by `between`', () => {
    const extra = truthOf(
      generate(blockCovarianceData, { knobs: { n: 10, p: 7, blocks: 3, within: 0.7, between: 0.2 } }).truth,
    )
    expect(extra['blockOf']).toEqual([0, 0, 0, 1, 1, 2, 2])
    const c = toArray(extra['covariance'] as never) as number[][]
    expect(c[0]?.[1]).toBe(0.7)
    expect(c[0]?.[3]).toBe(0.2)
    expect(c[2]?.[2]).toBe(1)
  })

  it('rejects a between-block correlation above the within-block one', () => {
    expect(() =>
      generate(blockCovarianceData, { knobs: { n: 10, p: 6, blocks: 2, within: 0.2, between: 0.5 } }),
    ).toThrow(/between/)
  })
})

describe('toeplitzCovarianceData', () => {
  it('has the geometric autocorrelation of an AR(1) and a tridiagonal precision', () => {
    const extra = truthOf(generate(toeplitzCovarianceData, { knobs: { n: 10, p: 5, rho: 0.5 } }).truth)
    expect(extra['autocorrelation']).toEqual([1, 0.5, 0.25, 0.125, 0.0625])
    const pr = toArray(extra['precision'] as never) as number[][]
    expect(Math.abs(pr[0]?.[2] ?? 1)).toBeLessThan(1e-12)
    expect(pr[0]?.[1]).toBeCloseTo(-0.5 / 0.75, 12)
  })

  it('has only lag one for a moving average, and positive long-memory correlations', () => {
    const ma = truthOf(
      generate(toeplitzCovarianceData, { knobs: { n: 10, p: 4, profile: 'moving-average', rho: 0.5 } }).truth,
    )
    expect(ma['autocorrelation']).toEqual([1, 0.4, 0, 0])
    const lm = truthOf(
      generate(toeplitzCovarianceData, { knobs: { n: 10, p: 6, profile: 'long-memory', hurst: 0.9 } }).truth,
    )
    const r = lm['autocorrelation'] as number[]
    expect(r.every((v, i) => i === 0 || (v > 0 && v < (r[i - 1] ?? 0)))).toBe(true)
  })
})

describe('wishartCovarianceData', () => {
  it('has a singular covariance when df is below p', () => {
    const extra = truthOf(generate(wishartCovarianceData, { knobs: { n: 10, p: 6, df: 3 } }).truth)
    expect(extra['conditionNumber']).toBe(Number.POSITIVE_INFINITY)
    expect(extra['populationRatio']).toBe(2)
  })
})
