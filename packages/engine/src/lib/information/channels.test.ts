import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { checkPrimitive } from '../autodiff/index.ts'
import { item, tensor, toFlat, type Value } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import * as I from './index.ts'

const row = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: 0.01, max: 1, noNaN: true }), { minLength: n, maxLength: n }).map((w) => {
    const s = w.reduce((a, b) => a + b, 0)
    return w.map((v) => v / s)
  })
const channel = (nx: number, ny: number): fc.Arbitrary<number[][]> =>
  fc.array(row(ny), { minLength: nx, maxLength: nx })
const compose = (a: readonly (readonly number[])[], b: readonly (readonly number[])[]): number[][] =>
  a.map((r) => (b[0] ?? []).map((_, j) => r.reduce((s, v, k) => s + v * (b[k]?.[j] ?? 0), 0)))

describe('channels', () => {
  it('Blahut-Arimoto agrees with the closed-form capacities', () => {
    for (const p of [0.01, 0.1, 0.3, 0.5]) {
      const bsc = I.binarySymmetricChannel(p, { base: 2 })
      expect(I.channelCapacity(bsc.transition, { base: 2 }).capacity).toBeCloseTo(bsc.capacity, 10)
      const bec = I.binaryErasureChannel(p, { base: 2 })
      expect(I.channelCapacity(bec.transition, { base: 2 }).capacity).toBeCloseTo(bec.capacity, 10)
      const z = I.zChannel(p, { base: 2 })
      expect(I.channelCapacity(z.transition, { base: 2 }).capacity).toBeCloseTo(z.capacity, 10)
    }
    expect(I.zChannel(1).capacity).toBe(0)
    const result = I.channelCapacity(I.zChannel(0.5).transition)
    // The optimal input of the Z channel sends a 1 with probability 2/5 when p = 1/2.
    expect(toFlat(result.input)[1]).toBeCloseTo(0.4, 8)
    expect(result.lower).toBeLessThanOrEqual(result.upper)
    expect(() => I.binarySymmetricChannel(1.5)).toThrow()
    expect(() => I.channelCapacity([[0.5, 0.4]])).toThrow()
    expect(() => I.channelCapacity([1, 0])).toThrow()
  })

  it('records the bounds closing in', () => {
    const rec = recorder()
    I.channelCapacity(I.zChannel(0.3).transition, { tracer: rec })
    const gaps = rec.events
      .filter((e) => e.type === 'metric' && e.name === 'gap')
      .map((e) => (e as { value: number }).value)
    expect(gaps.length).toBeGreaterThan(2)
    expect(gaps[gaps.length - 1]).toBeLessThan(1e-10)
    for (let i = 1; i < gaps.length; i++) expect(gaps[i] ?? 0).toBeLessThanOrEqual((gaps[i - 1] ?? 0) + 1e-15)
    expect(() => I.channelCapacity(I.zChannel(0.3).transition, { maxIter: 2 })).toThrow()
    expect(() =>
      I.blahutArimotoAlgorithm(
        [
          [1, 0],
          [0, 1],
        ],
        { init: [1] },
      ),
    ).toThrow()
  })

  it('capacity bounds every input and is reduced by processing (property)', () => {
    fc.assert(
      fc.property(channel(3, 4), channel(4, 3), row(3), (w1, w2, p) => {
        const options = { tol: 1e-7 }
        const c1 = I.channelCapacity(w1, options).capacity
        expect(item(I.channelMutualInformation(p, w1))).toBeLessThanOrEqual(c1 + 1e-7)
        expect(c1).toBeLessThanOrEqual(Math.log(3) + 1e-9)
        const c12 = I.channelCapacity(compose(w1, w2), options).capacity
        expect(c12).toBeLessThanOrEqual(Math.min(c1, I.channelCapacity(w2, options).capacity) + 1e-7)
      }),
      { numRuns: RUNS },
    )
  })

  it('channel mutual information is differentiable', () => {
    const f = (p: Value, w: Value): Value => I.channelMutualInformation(p, w)
    const w = tensor([
      [0.7, 0.2, 0.1],
      [0.1, 0.3, 0.6],
    ])
    expect(checkPrimitive(f, [tensor([0.4, 0.6]), w], { vmap: false }).failures).toEqual([])
    expect(() => I.channelMutualInformation([0.5, 0.5], [[1, 0]])).toThrow()
  })

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
    expect(() => I.rateDistortion(source, d, 2, { maxIter: 1 })).toThrow()
  })

  it('water-filling satisfies its optimality conditions', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0.1, max: 5, noNaN: true }), { minLength: 1, maxLength: 8 }),
        fc.double({ min: 0, max: 20, noNaN: true }),
        (noise, power) => {
          const { allocation, level, total } = I.waterFilling(noise, power)
          expect(allocation.reduce((a, b) => a + b, 0)).toBeCloseTo(power, 9)
          noise.forEach((n, i) => {
            const a = allocation[i] ?? 0
            if (a > 0) expect(a + n).toBeCloseTo(level, 9)
            else expect(n).toBeGreaterThanOrEqual(level - 1e-9)
          })
          // Any other split (here: proportional to 1 / N) does no better.
          const inv = noise.map((n) => 1 / n)
          const s = inv.reduce((a, b) => a + b, 0)
          const other = noise.reduce((acc, n, i) => acc + 0.5 * Math.log1p((((inv[i] ?? 0) / s) * power) / n), 0)
          expect(total).toBeGreaterThanOrEqual(other - 1e-9)
        },
      ),
      { numRuns: RUNS },
    )
    expect(() => I.waterFilling([], 1)).toThrow()
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
