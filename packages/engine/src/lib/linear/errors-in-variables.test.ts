import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { deming, totalLeastSquares } from './errors-in-variables.ts'
import { olsSummary } from './inference.ts'

const moments = (x: number[], y: number[]): { sxx: number; syy: number; sxy: number } => {
  const n = x.length
  const mx = x.reduce((a, v) => a + v, 0) / n
  const my = y.reduce((a, v) => a + v, 0) / n
  let sxx = 0
  let syy = 0
  let sxy = 0
  for (let i = 0; i < n; i++) {
    sxx += ((x[i] ?? 0) - mx) ** 2
    syy += ((y[i] ?? 0) - my) ** 2
    sxy += ((x[i] ?? 0) - mx) * ((y[i] ?? 0) - my)
  }
  return { sxx, syy, sxy }
}

const n = 200
const z = toFlat(normal(key(11), [3 * n]))
const truth = Array.from({ length: n }, (_, i) => (i / n) * 10)
const xs = truth.map((v, i) => v + 0.7 * (z[i] ?? 0))
const ys = truth.map((v, i) => 1 + 2 * v + 0.7 * (z[n + i] ?? 0))
const x = xs.map((v) => [v])

describe('deming', () => {
  it('has the closed-form slope of the variance ratio', () => {
    const { sxx, syy, sxy } = moments(xs, ys)
    for (const delta of [0.25, 1, 4]) {
      const slope = (syy - delta * sxx + Math.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy)
      const f = deming({ delta }).fit({ x, y: ys })
      expect(f.params.coef[0]).toBeCloseTo(slope, 9)
      const mx = xs.reduce((a, v) => a + v, 0) / n
      const my = ys.reduce((a, v) => a + v, 0) / n
      expect(f.params.intercept).toBeCloseTo(my - slope * mx, 9)
    }
  })

  it('tends to least squares as the target error dominates and to inverse regression as it vanishes', () => {
    const { sxx, syy, sxy } = moments(xs, ys)
    expect(deming({ delta: 1e8 }).fit({ x, y: ys }).params.coef[0]).toBeCloseTo(sxy / sxx, 5)
    expect(deming({ delta: 1e-8 }).fit({ x, y: ys }).params.coef[0]).toBeCloseTo(syy / sxy, 5)
  })

  it('removes the attenuation of the least-squares slope that feature noise causes', () => {
    const ols = olsSummary({ x, y: ys }).table.estimates[0] ?? 0
    const tls = deming({ delta: 1 }).fit({ x, y: ys }).params.coef[0] ?? 0
    expect(ols).toBeLessThan(1.95)
    expect(Math.abs(tls - 2)).toBeLessThan(Math.abs(ols - 2))
    expect(tls).toBeCloseTo(2, 1)
  })

  it('rejects a non-positive ratio', () => {
    expect(() => deming({ delta: 0 })).toThrow(/delta/)
  })
})

describe('totalLeastSquares', () => {
  it('recovers a noise-free hyperplane in several features, whatever the error scales', () => {
    const rows = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), (i % 7) / 7])
    const y = rows.map((r) => 0.5 + 2 * (r[0] ?? 0) - 3 * (r[1] ?? 0) + (r[2] ?? 0))
    for (const errorX of [1, [1, 5, 0.2]]) {
      const f = totalLeastSquares({ errorX, errorY: 2 }).fit({ x: rows, y })
      expect(f.params.coef[0]).toBeCloseTo(2, 8)
      expect(f.params.coef[1]).toBeCloseTo(-3, 8)
      expect(f.params.coef[2]).toBeCloseTo(1, 8)
      expect(f.params.intercept).toBeCloseTo(0.5, 8)
      expect(f.params.orthogonalSumOfSquares).toBeLessThan(1e-18)
    }
  })

  it('scales the slope inversely with the unit of the feature', () => {
    const a = totalLeastSquares().fit({ x, y: ys }).params.coef[0] ?? 0
    const b = totalLeastSquares({ errorX: 100 }).fit({ x: xs.map((v) => [v * 100]), y: ys }).params.coef[0] ?? 0
    expect(b * 100).toBeCloseTo(a, 8)
  })

  it('rejects a vertical fit, bad errors and mismatched error lists', () => {
    expect(() => totalLeastSquares().fit({ x: [[1], [1], [1], [1]], y: [0, 1, 2, 3] })).toThrow(/vertical/)
    expect(() => totalLeastSquares({ errorY: 0 }).fit({ x, y: ys })).toThrow(/positive/)
    expect(() => totalLeastSquares({ errorX: [1, 2] }).fit({ x, y: ys })).toThrow(/feature errors/)
  })
})
