import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { spikedCovarianceData, wishartCovarianceData } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { empiricalCovariance, ledoitWolf } from '../stats/index.ts'
import { type Tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { conditionOf, frobeniusDistance, symmetricEigen } from './_matrix.ts'
import {
  NonlinearShrinkageEstimator,
  nonlinearShrinkageAlgorithm,
  nonlinearShrinkageEstimator,
  quadraticInverseShrinkage,
} from './nonlinear-shrinkage.ts'
import { covarianceEstimator } from './shrinkage.ts'

const truth = (t: { readonly extra?: Readonly<Record<string, unknown>> }): number[][] =>
  toArray(t.extra?.['covariance'] as Tensor) as number[][]
const spiked = (n: number, p: number, seed = 1): { x: Tensor; cov: number[][] } => {
  const { data, truth: t } = generate(spikedCovarianceData, { key: key(seed), knobs: { n, p, spikes: 3, strength: 8 } })
  return { x: data.x, cov: truth(t) }
}

describe('quadraticInverseShrinkage', () => {
  it('keeps the eigenvectors and the trace, the covariance having the shrunk eigenvalues', () => {
    const { x } = spiked(200, 40)
    const r = quadraticInverseShrinkage(x)
    const eig = symmetricEigen(r.covariance).values
    ;[...r.shrunkEigenvalues]
      .sort((a, b) => b - a)
      .forEach((v, i) => {
        expect(eig[i]).toBeCloseTo(v, 8)
      })
    expect(r.shrunkEigenvalues.reduce((a, v) => a + v, 0)).toBeCloseTo(
      r.sampleEigenvalues.reduce((a, v) => a + v, 0),
      10,
    )
    expect(r.ratio).toBeCloseTo(40 / 199, 12)
    // The sample eigenvalues are those of the unbiased sample covariance.
    const s = toArray(empiricalCovariance(x)) as number[][]
    expect(r.sampleEigenvalues[0]).toBeCloseTo((symmetricEigen(s).values[0] ?? 0) * (200 / 199), 10)
  })

  it('leaves the spectrum nearly as it is when there are far more observations than variables', () => {
    const { x } = spiked(20000, 5)
    const r = quadraticInverseShrinkage(x)
    r.shrunkEigenvalues.forEach((v, i) => {
      expect(Math.abs(v / (r.sampleEigenvalues[i] ?? 1) - 1)).toBeLessThan(0.15)
    })
  })

  it('bends the spectrum: the largest eigenvalues fall and the smallest rise, so the condition number drops', () => {
    const { x } = spiked(100, 80)
    const r = quadraticInverseShrinkage(x)
    expect(r.shrunkEigenvalues[0]).toBeLessThan(r.sampleEigenvalues[0] ?? 0)
    expect(r.shrunkEigenvalues[79]).toBeGreaterThan(r.sampleEigenvalues[79] ?? 0)
    expect(conditionOf(r.shrunkEigenvalues)).toBeLessThan(0.5 * conditionOf(r.sampleEigenvalues))
  })

  it('is closer to the true covariance than the sample covariance and the Ledoit-Wolf shrinkage', () => {
    for (const [n, p] of [
      [200, 100],
      [80, 100],
    ] as const) {
      const { x, cov } = spiked(n, p)
      const sample = toArray(empiricalCovariance(x)) as number[][]
      const lw = toArray(ledoitWolf(x).covariance) as number[][]
      const qis = quadraticInverseShrinkage(x).covariance
      expect(frobeniusDistance(qis, cov)).toBeLessThan(frobeniusDistance(sample, cov))
      expect(frobeniusDistance(qis, cov)).toBeLessThan(frobeniusDistance(lw, cov))
    }
  })

  it('is closer than the sample covariance for a Wishart-distributed covariance too', () => {
    const { data, truth: t } = generate(wishartCovarianceData, { key: key(2), knobs: { n: 150, p: 60, df: 70 } })
    const sample = toArray(empiricalCovariance(data.x)) as number[][]
    expect(frobeniusDistance(quadraticInverseShrinkage(data.x).covariance, truth(t))).toBeLessThan(
      frobeniusDistance(sample, truth(t)),
    )
  })

  it('is invertible when there are more variables than observations, with the null eigenvalues all equal', () => {
    const { x, cov } = spiked(30, 100)
    const r = quadraticInverseShrinkage(x)
    const nullEigenvalues = r.shrunkEigenvalues.slice(29)
    expect(Math.min(...r.shrunkEigenvalues)).toBeGreaterThan(0)
    expect(Math.max(...nullEigenvalues) - Math.min(...nullEigenvalues)).toBeLessThan(1e-12)
    const sample = toArray(empiricalCovariance(x)) as number[][]
    expect(frobeniusDistance(r.covariance, cov)).toBeLessThan(frobeniusDistance(sample, cov))
  })

  it('divides by n instead of n - 1 for centred data', () => {
    const { x } = spiked(50, 10)
    const a = quadraticInverseShrinkage(x, { assumeCentered: true })
    expect(a.location).toEqual(new Array(10).fill(0))
    expect(a.ratio).toBeCloseTo(10 / 50, 12)
  })

  it('needs at least three observations', () => {
    expect(() =>
      quadraticInverseShrinkage([
        [1, 2],
        [3, 4],
      ]),
    ).toThrow(/at least/)
  })
})

describe('nonlinearShrinkageEstimator', () => {
  const { x } = spiked(120, 30, 3)

  it('fits the covariance, a precision that inverts it, the verbs and a spectrum record', () => {
    const fit = nonlinearShrinkageEstimator().fit({ x })
    expect(fit.params.covariance).toEqual(quadraticInverseShrinkage(x).covariance)
    const prod = fit.params.covariance.map((row) =>
      row.reduce((s, v, k) => s + v * (fit.params.precision[k]?.[0] ?? 0), 0),
    )
    expect(prod[0]).toBeCloseTo(1, 6)
    expect(prod[1]).toBeCloseTo(0, 6)
    expect(Number.isFinite(fit.score({ x }))).toBe(true)
    expect(fit.exposition.map((r) => r.kind)).toEqual(['curves', 'anomalyScores'])
  })

  it('scores held-out data better than the empirical covariance when the sample is short', () => {
    const train = spiked(40, 30, 5).x
    const test = spiked(800, 30, 5).x
    const q = nonlinearShrinkageEstimator().fit({ x: train }).score({ x: test })
    const e = covarianceEstimator().fit({ x: train }).score({ x: test })
    expect(q).toBeGreaterThan(e)
  })

  it('runs as a one-step algorithm recording the conditioning, and as a class', () => {
    const out = runAlgorithm(nonlinearShrinkageAlgorithm(x), { maxSteps: 3 })
    expect(out.steps).toBe(1)
    expect(out.state.bandwidth).toBeGreaterThan(0)
    expect(new NonlinearShrinkageEstimator().fit(x).model.params.ratio).toBeCloseTo(30 / 119, 12)
  })
})
