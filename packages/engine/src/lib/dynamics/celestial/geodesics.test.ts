import { describe, expect, it } from 'vitest'
import { tensor, toFlat } from '../../tensor/index.ts'
import { messages, metricSeries, policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { flowAlgorithm } from '../ode/flow.ts'
import {
  criticalImpactParameter,
  deflectionAngle,
  flammParaboloid,
  lightRayAlgorithm,
  measuredDeflection,
  nullInvariant,
  photonSphereRadius,
  schwarzschildNullGeodesicField,
  weakFieldDeflection,
} from './geodesics.ts'

describe('the Binet field and its constants', () => {
  it('has the photon sphere at 3 M as an equilibrium and the critical impact parameter 3 sqrt(3) M', () => {
    const M = 1.7
    expect(photonSphereRadius(M)).toBe(5.1)
    expect(criticalImpactParameter(M)).toBeCloseTo(3 * Math.sqrt(3) * M, 14)
    const out = toFlat(schwarzschildNullGeodesicField(M)(0, tensor([1 / (3 * M), 0])))
    expect(out[0]).toBe(0)
    expect(out[1]).toBeCloseTo(0, 15)
    // The critical parameter is where 1/b^2 equals the invariant's value at the photon sphere.
    expect(nullInvariant([1 / (3 * M), 0], M)).toBeCloseTo(1 / criticalImpactParameter(M) ** 2, 14)
  })

  it('conserves (du/dphi)^2 + u^2 - 2 M u^3 along the flow', () => {
    const M = 1
    const start = [0.01, Math.sqrt(1 / 64 - 0.0001 + 2e-6)]
    const alg = flowAlgorithm(schwarzschildNullGeodesicField(M), start, { dt: 0.001, steps: 8000 })
    const end = runAlgorithm(alg, { maxSteps: 9000 }).state
    expect(nullInvariant(end.y, M)).toBeCloseTo(nullInvariant(start, M), 10)
  })
})

describe('the deflection angle', () => {
  it('approaches the weak-field value 4 M / b, twice the Newtonian 2 M / b, and its second-order correction', () => {
    const M = 1
    for (const b of [200, 1000, 5000]) {
      expect(deflectionAngle(b, M) / weakFieldDeflection(b, M)).toBeCloseTo(1, 1)
    }
    const b = 400
    const exact = deflectionAngle(b, M)
    // The second-order term accounts for the remainder, up to O((M / b)^3).
    expect(Math.abs(exact - weakFieldDeflection(b, M, 2))).toBeLessThan(0.05 * (exact - weakFieldDeflection(b, M)))
    expect(weakFieldDeflection(100, 1) / (2 / 100)).toBe(2)
  })

  it('falls as the ray passes farther away and grows without bound at the critical impact parameter', () => {
    const M = 1
    const bc = criticalImpactParameter(M)
    const values = [1.01, 1.1, 1.5, 2, 5, 20].map((x) => deflectionAngle(x * bc, M))
    for (let i = 1; i < values.length; i++) expect(values[i] as number).toBeLessThan(values[i - 1] as number)
    expect(values[0]).toBeGreaterThan(Math.PI)
    // The strong-field logarithm of Bozza: alpha = -ln(b / b_c - 1) + ln(216 (7 - 4 sqrt 3)) - pi.
    const strong = (eps: number): number => -Math.log(eps) + Math.log(216 * (7 - 4 * Math.sqrt(3))) - Math.PI
    expect(deflectionAngle(bc * 1.001, M)).toBeCloseTo(strong(0.001), 1)
    expect(deflectionAngle(bc * 1.01, M)).toBeCloseTo(strong(0.01), 1)
    expect(deflectionAngle(bc, M)).toBe(Number.POSITIVE_INFINITY)
    expect(deflectionAngle(2, M)).toBe(Number.POSITIVE_INFINITY)
  })

  it('scales: the deflection depends on b / M alone', () => {
    expect(deflectionAngle(60, 3)).toBeCloseTo(deflectionAngle(20, 1), 12)
    expect(() => deflectionAngle(-1, 1)).toThrow()
  })
})

describe('a ray as a stepped algorithm', () => {
  it('measures the deflection that the quadrature gives', () => {
    const M = 1
    for (const b of [8, 20, 100]) {
      const alg = lightRayAlgorithm(M, b, { dphi: 0.002, startRadius: 400 * b })
      const { state } = runAlgorithm(alg, { maxSteps: 60000 })
      expect(state.fate).toBe('escaped')
      expect(measuredDeflection(state, b, M, 400 * b)).toBeCloseTo(deflectionAngle(b, M), 5)
    }
  })

  it('has its closest approach where the turning point of the cubic puts it', () => {
    const b = 9
    const { state } = runAlgorithm(lightRayAlgorithm(1, b, { dphi: 0.001 }), { maxSteps: 60000 })
    // r_min from 1/b^2 = 1/r^2 - 2/r^3 (M = 1) outside the photon sphere, found independently by bisection (the right side
    // falls from its peak at r = 3 as r grows).
    let lo = 3
    let hi = 9
    for (let i = 0; i < 100; i++) {
      const mid = 0.5 * (lo + hi)
      if (1 / (mid * mid) - 2 / mid ** 3 > 1 / (b * b)) lo = mid
      else hi = mid
    }
    expect(state.closest).toBeCloseTo(0.5 * (lo + hi), 4)
  })

  it('captures a ray inside 3 sqrt(3) M with an event, and lets one just outside it swing round', () => {
    const rec = recorder()
    const caught = runAlgorithm(lightRayAlgorithm(1, 5.0), { maxSteps: 60000, tracer: rec, policy: policies.teaching })
    expect(caught.state.fate).toBe('captured')
    expect(caught.status).toBe('stopped')
    expect(messages(rec.run(), 'ray/captured').length).toBe(1)
    expect(Math.min(...metricSeries(rec.run(), 'ray/radius').values)).toBeLessThan(2.1)
    const near = runAlgorithm(lightRayAlgorithm(1, 5.3, { dphi: 0.001 }), { maxSteps: 120000 })
    expect(near.state.fate).toBe('escaped')
    expect(measuredDeflection(near.state, 5.3, 1, 200 * 5.3)).toBeGreaterThan(Math.PI)
    // It came within a hair of the photon sphere.
    expect(near.state.closest).toBeLessThan(3.6)
    expect(near.state.closest).toBeGreaterThan(3)
  })

  it('logs the impact parameter, the weak-field value and the bend so far, which tends to the deflection', () => {
    const b = 25
    const rec = recorder()
    const { state } = runAlgorithm(lightRayAlgorithm(1, b, { dphi: 0.002, startRadius: 5000 }), {
      maxSteps: 60000,
      tracer: rec,
    })
    const r = rec.run()
    expect(new Set(metricSeries(r, 'ray/impactParameter').values)).toEqual(new Set([b]))
    expect(metricSeries(r, 'ray/weakField').values[0]).toBe(weakFieldDeflection(b, 1))
    const bend = metricSeries(r, 'ray/deflection').values
    expect(Math.abs((bend.at(-1) as number) - deflectionAngle(b, 1))).toBeLessThan(0.01)
    expect(state.fate).toBe('escaped')
  })

  it('serialises and resumes to the same fate', () => {
    const alg = lightRayAlgorithm(1, 12, { dphi: 0.004, startRadius: 1200 })
    const straight = runAlgorithm(alg, { maxSteps: 60000 }).state
    const half = runAlgorithm(alg, { maxSteps: 700 }).state
    let s = alg.deserialise?.(alg.serialise?.(half) as never) as typeof half
    const ctx = { log: () => undefined, emit: () => undefined, wants: () => false } as never
    while (alg.done?.(s, s.step) === undefined) s = alg.step(s, ctx)
    expect(s.fate).toBe(straight.fate)
    expect(s.exitAngle).toBe(straight.exitAngle)
    expect(() => lightRayAlgorithm(1, 10, { startRadius: 20 })).toThrow()
    expect(() => lightRayAlgorithm(0, 10)).toThrow()
  })
})

describe("Flamm's paraboloid", () => {
  it('starts at the horizon and has the proper radial length element dr / sqrt(1 - 2 M / r)', () => {
    const M = 1.3
    expect(flammParaboloid(M, 2 * M)).toBe(0)
    for (const r of [3, 5, 12]) {
      const h = 1e-6
      const dz = (flammParaboloid(M, r + h) - flammParaboloid(M, r - h)) / (2 * h)
      expect(Math.sqrt(1 + dz * dz)).toBeCloseTo(1 / Math.sqrt(1 - (2 * M) / r), 6)
    }
    expect(() => flammParaboloid(1, 1.5)).toThrow()
  })
})
