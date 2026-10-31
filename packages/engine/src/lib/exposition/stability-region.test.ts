import { describe, expect, it } from 'vitest'
import {
  explicitRungeKutta,
  realAxisLimit,
  sampleStabilityRegion,
  stabilityModulus,
  stabilityRegion,
  stableArea,
} from './stability-region.ts'

const window = { real: [-4, 1], imaginary: [-3.5, 3.5] } as const
const FACTORIAL = [1, 1, 2, 6, 24]
const taylor = (p: number): number[] => Array.from({ length: p + 1 }, (_, k) => 1 / (FACTORIAL[k] ?? 1))

describe('the stability region record', () => {
  it('builds the stability polynomial of a Runge-Kutta tableau', () => {
    expect(explicitRungeKutta([[0]], [1]).numerator).toEqual([1, 1])
    expect(
      explicitRungeKutta(
        [
          [0, 0],
          [0.5, 0],
        ],
        [0, 1],
      ).numerator,
    ).toEqual([1, 1, 0.5])
    const rk4 = explicitRungeKutta(
      [
        [0, 0, 0, 0],
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1, 0],
      ],
      [1 / 6, 1 / 3, 1 / 3, 1 / 6],
    )
    for (const [k, c] of rk4.numerator.entries()) expect(c).toBeCloseTo(taylor(4)[k] ?? 0, 14)
    expect(() => explicitRungeKutta([[0]], [1, 1])).toThrow(/2 by 2/)
  })

  it('forward Euler is stable exactly in the unit disc about -1', () => {
    const euler = { numerator: [1, 1] }
    expect(stabilityModulus(euler, [-1, 0])).toBe(0)
    expect(stabilityModulus(euler, [-1, 1])).toBeCloseTo(1, 12)
    expect(stabilityModulus(euler, [-2.1, 0])).toBeGreaterThan(1)
    expect(stabilityModulus(euler, [0.1, 0])).toBeGreaterThan(1)
    const r = sampleStabilityRegion('forward Euler', euler, window, { n: 501 })
    expect(Math.abs(stableArea(r) - Math.PI)).toBeLessThan(0.03)
    expect(realAxisLimit(r)).toBeCloseTo(-2, 1)
  })

  it('the region of RK4 reaches -2.785 on the real axis and is larger than those of lower order', () => {
    const area = (p: number): number =>
      stableArea(sampleStabilityRegion('rk', { numerator: taylor(p) }, window, { n: 501 }))
    const r4 = sampleStabilityRegion('RK4', { numerator: taylor(4) }, window, { n: 801 })
    expect(realAxisLimit(r4)).toBeCloseTo(-2.785, 1)
    expect(area(2)).toBeGreaterThan(area(1))
    expect(area(4)).toBeGreaterThan(area(2))
    // The region of RK4 reaches 2.83 up the imaginary axis (it is stable on the segment up to about 2.828i).
    expect(stabilityModulus({ numerator: taylor(4) }, [0, 2.8])).toBeLessThan(1)
    expect(stabilityModulus({ numerator: taylor(4) }, [0, 2.9])).toBeGreaterThan(1)
    expect(stableArea(r4)).toBeGreaterThan(12)
    expect(stableArea(r4)).toBeLessThan(13.5)
  })

  it('implicit methods use a denominator: backward Euler is stable outside the disc about 1', () => {
    const backward = { numerator: [1], denominator: [1, -1] }
    expect(stabilityModulus(backward, [-1, 0])).toBe(0.5)
    expect(stabilityModulus(backward, [1, 0])).toBe(Number.POSITIVE_INFINITY)
    expect(stabilityModulus(backward, [0.5, 0])).toBe(2)
    expect(stabilityModulus(backward, [-100, 100])).toBeLessThan(0.01)
    const trapezoid = { numerator: [1, 0.5], denominator: [1, -0.5] }
    expect(stabilityModulus(trapezoid, [0, 5])).toBeCloseTo(1, 12)
  })

  it('checks the grid and the marks', () => {
    expect(() => stabilityRegion({ method: 'm', x: [0], y: [0, 1], modulus: [[1], [1]] })).toThrow(/x needs/)
    expect(() =>
      stabilityRegion({
        method: 'm',
        x: [1, 0],
        y: [0, 1],
        modulus: [
          [1, 1],
          [1, 1],
        ],
      }),
    ).toThrow(/not increasing/)
    expect(() => stabilityRegion({ method: 'm', x: [0, 1], y: [0, 1], modulus: [[1, 1]] })).toThrow(/2 rows/)
    expect(() =>
      stabilityRegion({
        method: 'm',
        x: [0, 1],
        y: [0, 1],
        modulus: [
          [1, 1],
          [1, 1],
        ],
        marks: [{ at: [Number.NaN, 0] }],
      }),
    ).toThrow(/marks\[0\]/)
  })
})
