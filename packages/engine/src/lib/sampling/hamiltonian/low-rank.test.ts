import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { numericTarget } from '../target.ts'
import { windowAdaptationAlgorithm, windowAdaptationResult } from './adaptation.ts'
import { hmcKernel } from './hmc.ts'
import { lowRankInverseMassMatrix } from './low-rank.ts'

/** A covariance with scales 1, 2, ... and a correlation of 0.9 between the first two coordinates: the whitened matrix has two non-unit directions. */
function structured(dim: number): Float64Array {
  const cov = new Float64Array(dim * dim)
  for (let a = 0; a < dim; a++)
    for (let b = 0; b < dim; b++) cov[a * dim + b] = (1 + a) * (1 + b) * (a === b ? 1 : a + b === 1 ? 0.9 : 0)
  return cov
}

describe('lowRankInverseMassMatrix', () => {
  it('is the diagonal at rank 0, the covariance at full rank, and exact at the rank of the structure', () => {
    const dim = 5
    const cov = structured(dim)
    const diag = lowRankInverseMassMatrix(cov, dim, 0)
    expect(
      Array.from(diag)
        .filter((_, k) => k % (dim + 1) !== 0)
        .every((v) => v === 0),
    ).toBe(true)
    const full = lowRankInverseMassMatrix(cov, dim, dim)
    for (const [k, v] of full.entries()) expect(v).toBeCloseTo(cov[k] ?? 0, 9)
    const two = lowRankInverseMassMatrix(cov, dim, 2)
    for (const [k, v] of two.entries()) expect(v).toBeCloseTo(cov[k] ?? 0, 9)
    // One direction is not enough to recover the structure, but the matrix stays symmetric with the right diagonal.
    const one = lowRankInverseMassMatrix(cov, dim, 1)
    expect(Math.abs((one[1] ?? 0) - (one[dim] ?? 0))).toBeLessThan(1e-12)
    expect(one[1]).not.toBeCloseTo(cov[1] ?? 0, 3)
    expect(Array.from({ length: dim }, (_, i) => one[i * dim + i])).toEqual(
      Array.from({ length: dim }, (_, i) => cov[i * dim + i]).map((v, i) => (i < 2 ? one[i * dim + i] : v)),
    )
    expect(() => lowRankInverseMassMatrix(cov, dim, 6)).toThrow()
    expect(() => lowRankInverseMassMatrix(Float64Array.of(0, 0, 0, 1), 2, 1)).toThrow()
  })
})

describe('window adaptation with a low-rank mass matrix', () => {
  it('learns the strong correlation of a posterior with the one direction it keeps', () => {
    // Three coordinates, the first two with correlation 0.95.
    const rho = 0.95
    const det = 1 - rho * rho
    const t = numericTarget(
      (x) =>
        -0.5 * (((x[0] ?? 0) ** 2 - 2 * rho * (x[0] ?? 0) * (x[1] ?? 0) + (x[1] ?? 0) ** 2) / det + (x[2] ?? 0) ** 2),
      3,
      {
        grad: (x) => [-((x[0] ?? 0) - rho * (x[1] ?? 0)) / det, -((x[1] ?? 0) - rho * (x[0] ?? 0)) / det, -(x[2] ?? 0)],
      },
    )
    const alg = windowAdaptationAlgorithm(
      3,
      (p) => hmcKernel(t, { ...p, numIntegrationSteps: 6 }),
      normal(key(1), [3]).data,
      {
        numWarmup: 400,
        dense: true,
        lowRank: 1,
      },
    )
    const m = windowAdaptationResult(runAlgorithm(alg, { key: key(2), maxSteps: 400 }).state).inverseMassMatrix
    // The learned matrix has the correlation between the first two coordinates and none elsewhere.
    const corr = (m[1] ?? 0) / Math.sqrt((m[0] ?? 1) * (m[4] ?? 1))
    expect(corr).toBeGreaterThan(0.7)
    expect(Math.abs((m[2] ?? 0) / Math.sqrt((m[0] ?? 1) * (m[8] ?? 1)))).toBeLessThan(0.3)
  })
})
