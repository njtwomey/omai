import { describe, expect, it } from 'vitest'
import * as I from './index.ts'

describe('rate-distortion', () => {
  it('rate-distortion matches the binary closed form and is convex and decreasing', () => {
    const hamming = [
      [0, 1],
      [1, 0],
    ]
    for (const p of [0.5, 0.3]) {
      for (const beta of [1.5, 3]) {
        const point = I.rateDistortion([p, 1 - p], hamming, beta, { base: 2 })
        if (point.distortion < Math.min(p, 1 - p))
          expect(point.rate).toBeCloseTo(I.binaryRateDistortion(p, point.distortion, { base: 2 }), 8)
      }
    }
    const source = [0.2, 0.3, 0.5]
    const d = [
      [0, 1, 4],
      [1, 0, 1],
      [4, 1, 0],
    ]
    const curve = I.rateDistortionCurve(source, d, [0.25, 0.5, 1, 2, 4, 8])
    for (let i = 1; i < curve.length; i++) {
      const a = curve[i - 1] as I.RateDistortionPoint
      const b = curve[i] as I.RateDistortionPoint
      expect(b.distortion).toBeLessThanOrEqual(a.distortion + 1e-12)
      expect(b.rate).toBeGreaterThanOrEqual(a.rate - 1e-12)
    }
    // The slope of R(D) is -beta: compare with a finite difference of neighbouring points.
    const left = I.rateDistortion(source, d, 1.99)
    const right = I.rateDistortion(source, d, 2.01)
    expect((right.rate - left.rate) / (right.distortion - left.distortion)).toBeCloseTo(-2, 2)
    expect(I.binaryRateDistortion(0.3, 0.4)).toBe(0)
    expect(() => I.rateDistortionAlgorithm([0.5, 0.5], hamming, 0)).toThrow()
    expect(I.rateDistortion(source, d, 2, { maxIter: 1 }).converged).toBe(false)
    expect(I.rateDistortion(source, d, 2).converged).toBe(true)
  })

  it('reverse water-filling spends the distortion budget and gives the Gaussian rate', () => {
    const { allocation, total } = I.reverseWaterFilling([4, 1, 0.25], 1.25, { base: 2 })
    expect(allocation.reduce((a, b) => a + b, 0)).toBeCloseTo(1.25, 12)
    expect(total).toBeCloseTo(
      I.gaussianRateDistortion(4, 0.5, { base: 2 }) + I.gaussianRateDistortion(1, 0.5, { base: 2 }),
      12,
    )
    expect(I.reverseWaterFilling([2, 3], 5).total).toBe(0)
    expect(I.gaussianRateDistortion(1, 2)).toBe(0)
    expect(() => I.reverseWaterFilling([1], 2)).toThrow()
    expect(() => I.gaussianCapacity(-1)).toThrow()
    expect(() => I.gaussianRateDistortion(0, 1)).toThrow()
  })
})
