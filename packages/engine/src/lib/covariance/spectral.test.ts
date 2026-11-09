import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blockCovarianceData, spikedCovarianceData } from '../datasets/index.ts'
import { marchenkoPasturEdges } from '../dists/index.ts'
import { key, normal } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { type Tensor, toArray, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { frobeniusDistance, symmetricEigen, toCorrelation } from './_matrix.ts'
import { covarianceEstimator } from './shrinkage.ts'
import {
  bbpThreshold,
  denoiseCorrelation,
  detoneCorrelation,
  eigenSpectrum,
  noiseVariance,
  SpectralCleaning,
  spectralCleaningAlgorithm,
  spectralCleaningEstimator,
  spikedEigenvalueLimit,
  spikedEigenvectorOverlap,
} from './spectral.ts'

const sampleCorrelation = (x: Tensor): number[][] =>
  toCorrelation(toArray(empiricalCovariance(x)) as number[][]).correlation
const truthMatrix = (t: { readonly extra?: Readonly<Record<string, unknown>> }, name: string): number[][] =>
  toArray(t.extra?.[name] as never) as number[][]
const trace = (a: readonly (readonly number[])[]): number => a.reduce((s, r, i) => s + (r[i] ?? 0), 0)

describe('eigenSpectrum', () => {
  it('returns descending eigenvalues with matching unit eigenvectors', () => {
    const s = eigenSpectrum([
      [2, 1],
      [1, 2],
    ])
    expect(s.eigenvalues[0]).toBeCloseTo(3, 12)
    expect(s.eigenvalues[1]).toBeCloseTo(1, 12)
    expect(Math.abs(s.eigenvectors[0]?.[0] ?? 0)).toBeCloseTo(Math.SQRT1_2, 12)
    expect(Math.abs((s.eigenvectors[0]?.[0] ?? 0) - (s.eigenvectors[0]?.[1] ?? 1))).toBeLessThan(1e-12)
  })

  it('rejects a non-square matrix', () => {
    expect(() => eigenSpectrum([[1, 2, 3]])).toThrow(/square/)
  })
})

describe('the spiked-model companions', () => {
  it('puts the BBP threshold at variance (1 + sqrt(ratio))', () => {
    expect(bbpThreshold(0.25)).toBe(1.5)
    expect(bbpThreshold(1, 2)).toBe(4)
    expect(() => bbpThreshold(-1)).toThrow(/ratio/)
  })

  it('moves a spike to the bulk edge at the threshold and above it by 1 + variance ratio / (spike - variance)', () => {
    const y = 0.25
    const edge = marchenkoPasturEdges(y).upper
    expect(spikedEigenvalueLimit(bbpThreshold(y), y)).toBeCloseTo(edge, 12)
    expect(spikedEigenvalueLimit(0.8, y)).toBe(edge)
    expect(spikedEigenvalueLimit(5, y)).toBeCloseTo(5 * (1 + 0.25 / 4), 14)
    expect(spikedEigenvalueLimit(1e9, y)).toBeCloseTo(1e9, -3)
    let last = edge
    for (const l of [1.6, 2, 3, 6, 12]) {
      const v = spikedEigenvalueLimit(l, y)
      expect(v).toBeGreaterThan(last)
      expect(v).toBeGreaterThan(l)
      last = v
    }
  })

  it('has an overlap that is 0 at the threshold, rises to 1 and is 0 below', () => {
    expect(spikedEigenvectorOverlap(1.5, 0.25)).toBe(0)
    expect(spikedEigenvectorOverlap(1.2, 0.25)).toBe(0)
    expect(spikedEigenvectorOverlap(1.5001, 0.25)).toBeLessThan(0.01)
    expect(spikedEigenvectorOverlap(100, 0.25)).toBeGreaterThan(0.99)
    expect(spikedEigenvectorOverlap(5, 0.25)).toBeCloseTo((1 - 0.25 / 16) / (1 + 0.25 / 4), 14)
  })

  it('predicts the top sample eigenvalue of simulated spiked data above and below the threshold', () => {
    const run = (strength: number): number => {
      const { data } = generate(spikedCovarianceData, { key: key(11), knobs: { n: 900, p: 300, spikes: 1, strength } })
      return eigenSpectrum(toArray(empiricalCovariance(data.x)) as number[][]).eigenvalues[0] ?? 0
    }
    const y = 300 / 900
    expect(Math.abs(run(5) - spikedEigenvalueLimit(5, y))).toBeLessThan(0.25)
    expect(Math.abs(run(1.2) - marchenkoPasturEdges(y).upper)).toBeLessThan(0.12)
    // The spike below the threshold is invisible: its sample eigenvalue is no larger than for pure noise.
    expect(run(1.2)).toBeLessThan(marchenkoPasturEdges(y).upper + 0.12)
  })
})

describe('noiseVariance', () => {
  it('recovers the variance of white noise, through spikes, for a ratio below and above 1', () => {
    for (const [n, p, variance] of [
      [800, 200, 4],
      [200, 400, 0.5],
    ] as const) {
      const z = normal(key(5), [n, p]).data
      const x = Float64Array.from(z, (v) => Math.sqrt(variance) * v)
      // Plant two strong directions.
      for (let i = 0; i < n; i++) {
        const s = z[i] ?? 0
        x[i * p] = (x[i * p] ?? 0) + 6 * s
        x[i * p + 1] = (x[i * p + 1] ?? 0) + 6 * s
      }
      let c = new Float64Array(p * p)
      for (let i = 0; i < n; i++)
        for (let a = 0; a < p; a++)
          for (let b = 0; b < p; b++)
            c[a * p + b] = (c[a * p + b] ?? 0) + ((x[i * p + a] ?? 0) * (x[i * p + b] ?? 0)) / n
      c = Float64Array.from(c)
      const rows = Array.from({ length: p }, (_, a) => Array.from(c.subarray(a * p, (a + 1) * p)))
      const est = noiseVariance(symmetricEigen(rows).values, p / n)
      expect(Math.abs(est / variance - 1)).toBeLessThan(0.08)
    }
  })

  it('is the identity variance of an identity spectrum at a vanishing ratio', () => {
    expect(noiseVariance([1, 1, 1, 1], 1e-9)).toBeCloseTo(1, 6)
  })
})

describe('denoiseCorrelation', () => {
  const n = 120
  const planted = (): { c: number[][]; truth: number[][] } => {
    const { data, truth } = generate(spikedCovarianceData, {
      key: key(2),
      knobs: { n, p: 40, spikes: 2, strength: 12, spread: 0.7 },
    })
    return { c: sampleCorrelation(data.x), truth: toCorrelation(truthMatrix(truth, 'covariance')).correlation }
  }

  it('keeps the trace and the unit diagonal for every replacement', () => {
    const { c } = planted()
    for (const method of ['constant', 'target', 'zero'] as const) {
      const r = denoiseCorrelation(c, { sampleSize: n, method, margin: 0.15 })
      expect(trace(r.correlation)).toBeCloseTo(40, 10)
      for (let i = 0; i < 40; i++) expect(r.correlation[i]?.[i]).toBeCloseTo(1, 12)
    }
  })

  it('keeps the signal, replaces the bulk by its mean and preserves the eigenvalue sum', () => {
    const { c } = planted()
    const r = denoiseCorrelation(c, { sampleSize: n, margin: 0.15 })
    expect(r.nSignal).toBe(2)
    expect(r.cleanedEigenvalues.slice(0, 2)).toEqual(r.eigenvalues.slice(0, 2))
    const bulk = r.cleanedEigenvalues.slice(2)
    expect(Math.max(...bulk) - Math.min(...bulk)).toBeLessThan(1e-12)
    expect(r.cleanedEigenvalues.reduce((a, v) => a + v, 0)).toBeCloseTo(
      r.eigenvalues.reduce((a, v) => a + v, 0),
      10,
    )
    expect(r.edge).toBeCloseTo(marchenkoPasturEdges(40 / n, r.variance).upper, 12)
  })

  it('is closer to the true correlation than the sample matrix and is positive definite', () => {
    const { c, truth } = planted()
    const r = denoiseCorrelation(c, { sampleSize: n, margin: 0.15 })
    expect(frobeniusDistance(r.correlation, truth)).toBeLessThan(frobeniusDistance(c, truth))
    expect(Math.min(...symmetricEigen(r.correlation).values)).toBeGreaterThan(0)
  })

  it('sets the bulk to a target or to zero (a singular matrix) when asked', () => {
    const { c } = planted()
    const t = denoiseCorrelation(c, { sampleSize: n, method: 'target', target: 0.5, margin: 0.15 })
    expect(t.cleanedEigenvalues.slice(2).every((v) => v === 0.5)).toBe(true)
    const z = denoiseCorrelation(c, { sampleSize: n, method: 'zero', margin: 0.15 })
    expect(z.cleanedEigenvalues.slice(2).every((v) => v === 0)).toBe(true)
    expect(Math.min(...symmetricEigen(z.correlation).values)).toBeLessThan(1e-9)
  })

  it('leaves a matrix with every eigenvalue above the edge unchanged', () => {
    const c = [
      [1, 0.9],
      [0.9, 1],
    ]
    const r = denoiseCorrelation(c, { ratio: 0.01, variance: 0.01 })
    expect(r.nSignal).toBe(2)
    expect(frobeniusDistance(r.correlation, c)).toBeLessThan(1e-12)
  })

  it('needs an aspect ratio or a sample size and a positive variance', () => {
    expect(() => denoiseCorrelation([[1]])).toThrow(/ratio/)
    expect(() => denoiseCorrelation([[1]], { ratio: 1, variance: -1 })).toThrow(/variance/)
  })
})

describe('detoneCorrelation', () => {
  it('removes the market mode of an equicorrelation matrix, leaving a unit diagonal', () => {
    const c = [
      [1, 0.5, 0.5],
      [0.5, 1, 0.5],
      [0.5, 0.5, 1],
    ]
    const r = detoneCorrelation(c)
    expect(r.removed[0]).toBeCloseTo(2, 12)
    expect(r.correlation[0]?.[1]).toBeCloseTo(-0.5, 12)
    for (let i = 0; i < 3; i++) expect(r.correlation[i]?.[i]).toBe(1)
  })

  it('removes the correlation the blocks share, leaves a unit diagonal and one null direction per mode', () => {
    const { data } = generate(blockCovarianceData, {
      key: key(9),
      knobs: { n: 2000, p: 12, blocks: 3, within: 0.7, between: 0.35 },
    })
    const c = sampleCorrelation(data.x)
    const across = (m: readonly (readonly number[])[]): number => {
      let total = 0
      let count = 0
      for (let i = 0; i < 12; i++)
        for (let j = 0; j < 12; j++)
          if (Math.floor(i / 4) !== Math.floor(j / 4)) {
            total += m[i]?.[j] ?? 0
            count++
          }
      return total / count
    }
    expect(across(c)).toBeGreaterThan(0.3)
    for (const m of [1, 2]) {
      const r = detoneCorrelation(c, { components: m })
      // Removing the market mode takes the shared positive correlation away (it overshoots to a small negative one).
      expect(across(r.correlation)).toBeLessThan(0.05)
      for (let i = 0; i < 12; i++) expect(r.correlation[i]?.[i]).toBe(1)
      expect(symmetricEigen(r.correlation).values.filter((v) => Math.abs(v) < 1e-9).length).toBeGreaterThanOrEqual(m)
    }
    expect(() => detoneCorrelation(c, { components: 12 })).toThrow(/components/)
  })
})

describe('spectralCleaningEstimator', () => {
  const { data } = generate(spikedCovarianceData, { key: key(8), knobs: { n: 100, p: 30, spikes: 2, strength: 15 } })

  it('fits a covariance with the sample standard deviations and a precision', () => {
    const fit = spectralCleaningEstimator({ margin: 0.15 }).fit({ x: data.x })
    const cov = fit.params.covariance
    const emp = toArray(empiricalCovariance(data.x)) as number[][]
    for (let i = 0; i < 30; i++) expect(cov[i]?.[i]).toBeCloseTo(emp[i]?.[i] ?? 0, 10)
    expect(fit.params.nSignal).toBe(2)
    expect(fit.params.location).toHaveLength(30)
    const prod = cov.map((row) =>
      (fit.params.precision[0] ?? []).map((_, j) =>
        row.reduce((s, v, k) => s + v * (fit.params.precision[k]?.[j] ?? 0), 0),
      ),
    )
    expect(prod[0]?.[0]).toBeCloseTo(1, 6)
    expect(prod[0]?.[1]).toBeCloseTo(0, 6)
  })

  it('scores held-out data better than the empirical covariance when the sample is short', () => {
    const train = generate(spikedCovarianceData, { key: key(1), knobs: { n: 45, p: 30, spikes: 2, strength: 15 } }).data
      .x
    const test = generate(spikedCovarianceData, { key: key(1), knobs: { n: 600, p: 30, spikes: 2, strength: 15 } }).data
    const cleaned = spectralCleaningEstimator({ margin: 0.15 }).fit({ x: train }).score({ x: test.x })
    const empirical = covarianceEstimator().fit({ x: train }).score({ x: test.x })
    expect(cleaned).toBeGreaterThan(empirical)
  })

  it('exposes the spectrum, the matrices before and after and the anomaly scores, and a trace', () => {
    const fit = spectralCleaningEstimator({ margin: 0.15 }).fit({ x: data.x })
    expect(fit.exposition.map((r) => r.kind)).toEqual(['curves', 'valueGrids', 'anomalyScores'])
    const out = runAlgorithm(spectralCleaningAlgorithm(data.x, { margin: 0.15 }), { maxSteps: 3 })
    expect(out.steps).toBe(1)
    expect(out.state.cleanedCorrelation[3]?.[3]).toBeCloseTo(1, 12)
    expect(out.state.nSignal).toBe(2)
  })

  it('detones and works as a class', () => {
    const m = new SpectralCleaning({ method: 'denoise-detone', components: 1 }).fit(data.x).model
    expect(m.params.covariance).toHaveLength(30)
    expect(toFlat(m.mahalanobis(data.x))).toHaveLength(100)
  })
})
