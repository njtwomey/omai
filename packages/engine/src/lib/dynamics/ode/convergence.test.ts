import { describe, expect, it } from 'vitest'
import { neg } from '../../tensor/index.ts'
import { DORMAND_PRINCE, TSITOURAS } from './adaptive.ts'
import {
  aStability,
  boundaryLocus,
  convergenceOrder,
  orderConditions,
  rootedTrees,
  stabilityFunction,
  stabilityRegion,
  workPrecision,
} from './convergence.ts'
import { EULER, HEUN, RK4 } from './explicit.ts'
import { BACKWARD_EULER, IMPLICIT_MIDPOINT, SDIRK2 } from './implicit.ts'

describe('convergence and stability analysis', () => {
  it('counts rooted trees (OEIS A000081)', () => {
    expect(rootedTrees(8).map((t) => t.length)).toEqual([1, 1, 2, 4, 9, 20, 48, 115])
    expect(() => rootedTrees(0)).toThrow()
  })

  it('reports the order of tableaus and their residuals', () => {
    expect(orderConditions(DORMAND_PRINCE, 6).order).toBe(5)
    expect(orderConditions(TSITOURAS, 6, 1e-9).order).toBe(5)
    expect(orderConditions(HEUN, 4).residuals[2]).toBeGreaterThan(1e-3)
  })

  it('stability functions are the truncated exponential for explicit methods', () => {
    const z = { re: -0.7, im: 0.4 }
    const r = stabilityFunction(RK4, z)
    // 1 + z + z^2/2 + z^3/6 + z^4/24
    let re = 1
    let im = 0
    let pr = 1
    let pi = 0
    for (let k = 1; k <= 4; k++) {
      ;[pr, pi] = [(pr * z.re - pi * z.im) / k, (pr * z.im + pi * z.re) / k]
      re += pr
      im += pi
    }
    expect(r.re).toBeCloseTo(re, 12)
    expect(r.im).toBeCloseTo(im, 12)
    const region = stabilityRegion(EULER, { re: [-3, 1], im: [-2, 2], n: 9 })
    // z = -1 (the centre of Euler's disc) is stable; z = 0.5 is not.
    expect(region.stable[4]?.[4]).toBe(true)
    expect(region.stable[4]?.[7]).toBe(false)
  })

  it('A-stability and L-stability', () => {
    expect(aStability(EULER).aStable).toBe(false)
    expect(aStability(IMPLICIT_MIDPOINT).aStable).toBe(true)
    expect(aStability(IMPLICIT_MIDPOINT).lStable).toBe(false)
    expect(aStability(SDIRK2).lStable).toBe(true)
    expect(aStability(BACKWARD_EULER).atInfinity).toBeLessThan(1e-6)
  })

  it('observed orders and work-precision data', () => {
    const s = convergenceOrder((h) => h ** 3 * (1 + h), [0.1, 0.05, 0.025, 0.0125])
    expect(s.order).toBeCloseTo(3, 1)
    expect(s.localOrders.length).toBe(3)
    expect(() => convergenceOrder((h) => h, [0.1])).toThrow()
    const points = workPrecision((_t, y) => neg(y), [0, 1], [1], [Math.exp(-1)], ['RK23', 'DOP853'], [1e-3, 1e-7])
    const dop = points.filter((p) => p.method === 'DOP853')
    expect((dop[1]?.error ?? 1) < (dop[0]?.error ?? 0)).toBe(true)
  })

  it('the boundary locus of backward Euler is the circle |z - 1| = 1', () => {
    const locus = boundaryLocus([-1, 1], [0, 1], 8)
    locus.re.forEach((x, i) => {
      expect(Math.hypot(x - 1, locus.im[i] ?? 0)).toBeCloseTo(1, 12)
    })
  })
})
