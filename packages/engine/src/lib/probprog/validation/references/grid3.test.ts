import { describe, expect, it } from 'vitest'
import { posterior3 } from './grid3.ts'

/** The log density, up to a constant, of the correlation of eight pairs near 0.97 from a unit-variance bivariate normal, under a uniform prior. */
const logRho = (rho: number): number => {
  const [n, sxx, syy, sxy] = [8, 8.4, 8.1, 8.0]
  return -(n / 2) * Math.log(1 - rho * rho) - (sxx - 2 * rho * sxy + syy) / (2 * (1 - rho * rho))
}

/** The mean and sd of that correlation by a fine grid in the logit (twice the Fisher transform), independent of the grid under test. */
function exactRho(): { mean: number; sd: number } {
  const N = 40000
  let [z, m1, m2] = [0, 0, 0]
  for (let i = 0; i < N; i++) {
    const s = -30 + (60 * (i + 0.5)) / N
    const rho = Math.tanh(s / 2)
    const w = Math.exp(logRho(rho) + Math.log((1 - rho * rho) / 2))
    z += w
    m1 += w * rho
    m2 += w * rho * rho
  }
  return { mean: m1 / z, sd: Math.sqrt(m2 / z - (m1 / z) ** 2) }
}

describe('posterior3', () => {
  const boxes = [
    { lo: 1e-3, hi: 12, min: 1e-9, max: Number.POSITIVE_INFINITY },
    { lo: 1e-3, hi: 12, min: 1e-9, max: Number.POSITIVE_INFINITY },
    { lo: -0.995, hi: 0.995, min: -0.9999999, max: 0.9999999 },
  ] as const
  // two gamma variables and the correlation, independent
  const density = (a: number, b: number, rho: number): number => 2 * Math.log(a) - 2 * a + Math.log(b) - b + logRho(rho)

  it('recovers independent marginals: gammas and a skewed correlation, equal steps or on the warped scales', () => {
    const { mean, sd } = exactRho()
    const warped = posterior3(
      [
        { ...boxes[0], warp: 'log' },
        { ...boxes[1], warp: 'log' },
        { ...boxes[2], warp: 'correlation' },
      ],
      density,
    )
    // Gamma(3, 2) and Gamma(2, 1)
    expect(warped.marginal(0).mean).toBeCloseTo(1.5, 2)
    expect(warped.marginal(0).sd).toBeCloseTo(Math.sqrt(3) / 2, 2)
    expect(warped.marginal(1).mean).toBeCloseTo(2, 2)
    expect(warped.marginal(1).sd).toBeCloseTo(Math.SQRT2, 2)
    expect(warped.marginal(2).mean).toBeCloseTo(mean, 3)
    expect(warped.marginal(2).sd / sd).toBeCloseTo(1, 2)
    // the density is in the coordinate itself: it integrates to one over the correlation's range
    const rho = warped.marginal(2)
    let area = 0
    const M = 4000
    for (let i = 0; i < M; i++) area += (rho.pdf(-1 + (2 * (i + 0.5)) / M) * 2) / M
    expect(area).toBeCloseTo(1, 2)
    const plain = posterior3([...boxes], density)
    // the unwarped grid is the one the warp was made for: its spread of the strong correlation is not as good
    expect(Math.abs(plain.marginal(2).sd / sd - 1)).toBeGreaterThan(Math.abs(warped.marginal(2).sd / sd - 1))
  })

  it('the log evidence is the same on either scale', () => {
    const plain = posterior3([...boxes], density)
    const warped = posterior3(
      [
        { ...boxes[0], warp: 'log' },
        { ...boxes[1], warp: 'log' },
        { ...boxes[2], warp: 'correlation' },
      ],
      density,
    )
    expect(warped.logEvidence).toBeCloseTo(plain.logEvidence, 1)
  })

  it('a mixture keeps at most the nodes it is given', () => {
    const grid = posterior3([...boxes], density, { mixtureNodes: 50 })
    const m = grid.mixture((a, _b, rho) => ({ mean: a * rho, variance: 0.1 }))
    expect(Number.isFinite(m.mean)).toBe(true)
    expect(Number.isFinite(m.sd)).toBe(true)
  })
})
