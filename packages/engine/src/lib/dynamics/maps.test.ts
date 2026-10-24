import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import {
  bifurcationDiagram,
  circleMap,
  cobweb,
  henonMap,
  logisticMap,
  lyapunovExponents,
  orbit,
  standardMap,
  tentMap,
} from './maps.ts'

describe('iterated maps', () => {
  it('Lyapunov exponents match the closed forms', () => {
    expect(Math.abs((lyapunovExponents(tentMap(1.9), [0.3], { n: 3000 })[0] ?? 0) - Math.log(1.9))).toBeLessThan(1e-9)
    const [l1, l2] = lyapunovExponents(henonMap(), [0.1, 0.1], { n: 3000 })
    // det J = -b, so l1 + l2 = ln 0.3 exactly; l1 is about 0.42.
    expect(Math.abs((l1 ?? 0) + (l2 ?? 0) - Math.log(0.3))).toBeLessThan(1e-9)
    expect(Math.abs((l1 ?? 0) - 0.42)).toBeLessThan(0.03)
    // A stable fixed point has a negative exponent ln|f'(x*)| = ln|2 - r|.
    expect(Math.abs((lyapunovExponents(logisticMap(2.5), [0.3], { n: 2000 })[0] ?? 0) - Math.log(0.5))).toBeLessThan(
      1e-3,
    )
    // The area-preserving standard map has exponents summing to zero.
    const [s1, s2] = lyapunovExponents(standardMap(1.5), [0.5, 0.2], { n: 1000 })
    expect(Math.abs((s1 ?? 0) + (s2 ?? 0))).toBeLessThan(1e-9)
  })

  it('period doubling of the logistic map', () => {
    const periods = [2.9, 3.2, 3.5].map((r) => {
      const d = bifurcationDiagram(logisticMap, [r], [0.4], { transient: 2000, keep: 32 })
      return new Set(d.x.map((v) => v.toFixed(5))).size
    })
    expect(periods).toEqual([1, 2, 4])
  })

  it('orbits are traced and cobwebs alternate between graph and diagonal', () => {
    const rec = recorder()
    const o = orbit(henonMap(), [0, 0], { n: 5, tracer: rec })
    expect(o.shape).toEqual([5, 2])
    expect(rec.run().events.length).toBeGreaterThan(0)
    const web = cobweb(logisticMap(3), 0.2, 3)
    expect(web.length).toBe(7)
    expect(web[2]?.[0]).toBe(web[2]?.[1])
    expect(toFlat(circleMap(0.25, 0)(orbit(circleMap(0.25, 0), [0], { n: 1 })))).toEqual([0.25])
    // An orbit that leaves to infinity stops with a diverged run.
    expect(orbit(logisticMap(10), [0.5], { n: 50 }).shape[0]).toBeLessThan(50)
  })
})
