import { describe, expect, it } from 'vitest'
import { littleMcarTest, missingnessProfile } from './missing.ts'

describe('missingness diagnostics and Little MCAR test', () => {
  it('summarizes column, row, and pattern missingness accurately', () => {
    const x = [
      [1, 10, 100],
      [2, Number.NaN, 200],
      [3, 30, 300],
      [Number.NaN, Number.NaN, 400],
      [5, Number.NaN, 500],
    ]
    const prof = missingnessProfile(x)
    expect(prof.rows).toBe(5)
    expect(prof.cols).toBe(3)
    expect(prof.totalCells).toBe(15)
    expect(prof.missingCells).toBe(4)
    expect(prof.missingFraction).toBeCloseTo(4 / 15, 12)
    expect(prof.completeCaseCount).toBe(2)
    expect(prof.completeCaseFraction).toBeCloseTo(0.4, 12)
    expect(Array.from(prof.missingPerColumn.data)).toEqual([1, 3, 0])
    expect(Array.from(prof.missingPerRow.data)).toEqual([0, 1, 0, 2, 1])
    expect(prof.patterns).toHaveLength(3)
    expect(prof.patterns[0]?.key).toBe('101')
    expect(prof.patterns[0]?.count).toBe(2)
  })

  it('does not reject MCAR when missingness is independent of observed values, and rejects strongly under MAR', () => {
    // Balanced MCAR dataset: missingness in column 1 occurs symmetrically across low and high values of column 0
    const mcarData = [
      [1.0, 2.0],
      [2.0, 4.1],
      [3.0, 5.9],
      [4.0, 8.0],
      [5.0, 10.1],
      [6.0, 11.9],
      [2.0, Number.NaN],
      [5.0, Number.NaN],
    ]
    const mcarRes = littleMcarTest(mcarData)
    expect(mcarRes.df).toBe(1)
    expect(mcarRes.statistic).toBeCloseTo(0, 5)
    expect(mcarRes.pValue).toBeGreaterThan(0.95)
    expect(mcarRes.patterns).toHaveLength(2)

    // Strongly MAR dataset: column 1 is missing exclusively when column 0 is very large
    const marData = [
      [0.1, 1.0],
      [0.2, 1.1],
      [0.3, 0.9],
      [0.4, 1.0],
      [0.5, 1.2],
      [0.6, 0.8],
      [10.0, Number.NaN],
      [10.2, Number.NaN],
      [10.4, Number.NaN],
      [10.6, Number.NaN],
      [10.8, Number.NaN],
      [11.0, Number.NaN],
    ]
    const marRes = littleMcarTest(marData)
    expect(marRes.df).toBe(1)
    expect(marRes.statistic).toBeGreaterThan(8)
    expect(marRes.pValue).toBeLessThan(0.01)

    // Complete dataset has df = 0, statistic = 0, pValue = 1
    const completeRes = littleMcarTest([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    expect(completeRes.df).toBe(0)
    expect(completeRes.statistic).toBe(0)
    expect(completeRes.pValue).toBe(1)
  })

  it('estimates the EM moments of a monotone pattern in closed form (Anderson 1957)', () => {
    const complete = [
      [1.0, 2.0],
      [2.0, 4.1],
      [3.0, 5.9],
      [4.5, 8.0],
      [5.0, 10.1],
      [6.0, 11.9],
    ]
    const partial = [2.5, 7.0, 0.5]
    const data = [...complete, ...partial.map((v) => [v, Number.NaN])]
    const res = littleMcarTest(data, { tol: 1e-13, maxIter: 5000 })
    const nc = complete.length
    const n = data.length
    const x1 = complete.map((r) => r[0] ?? 0)
    const x2 = complete.map((r) => r[1] ?? 0)
    const m1c = x1.reduce((a, b) => a + b, 0) / nc
    const m2c = x2.reduce((a, b) => a + b, 0) / nc
    const s11c = x1.reduce((a, b) => a + (b - m1c) ** 2, 0) / nc
    const s22c = x2.reduce((a, b) => a + (b - m2c) ** 2, 0) / nc
    const s12c = x1.reduce((a, b, i) => a + (b - m1c) * ((x2[i] ?? 0) - m2c), 0) / nc
    const all1 = [...x1, ...partial]
    const mu1 = all1.reduce((a, b) => a + b, 0) / n
    const s11 = all1.reduce((a, b) => a + (b - mu1) ** 2, 0) / n
    const beta = s12c / s11c
    const mu2 = m2c + beta * (mu1 - m1c)
    const s22 = s22c - beta * beta * s11c + beta * beta * s11
    expect(res.pooledMean.data[0]).toBeCloseTo(mu1, 8)
    expect(res.pooledMean.data[1]).toBeCloseTo(mu2, 8)
    expect(res.pooledCovariance.data[0]).toBeCloseTo(s11, 8)
    expect(res.pooledCovariance.data[1]).toBeCloseTo(beta * s11, 8)
    expect(res.pooledCovariance.data[3]).toBeCloseTo(s22, 8)
    // d0^2 is the sum over the two patterns of m_j (ybar_j - mu)' Sigma_OO^-1 (ybar_j - mu).
    const d2 = ((partial.reduce((a, b) => a + b, 0) / partial.length - mu1) ** 2 / s11) * partial.length
    const comp = res.patterns.find((p) => p.key === '11')
    const part = res.patterns.find((p) => p.key === '10')
    expect(part?.distanceSquared).toBeCloseTo(d2, 8)
    // The complete pattern uses the full 2 x 2 covariance.
    expect(comp?.distanceSquared).toBeGreaterThan(0)
    expect(res.statistic).toBeCloseTo((comp?.distanceSquared ?? 0) + d2, 10)
  })
})
