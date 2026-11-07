import { describe, expect, it } from 'vitest'
import type { Vec3 } from '../../base/index.ts'
import { mul, pow, reshape, sliceAxis, stack, sum, toFlat, type Value } from '../../tensor/index.ts'
import { solveIvp } from '../ode/index.ts'
import {
  elementsFromState,
  escapeSpeed,
  keplerInvariants,
  keplerState,
  meanMotion,
  type OrbitalElements,
  orbitalPeriod,
  propagateKepler,
  semiLatusRectum,
  stateFromElements,
  visViva,
} from './elements.ts'

const MU = 1.7

const close = (u: readonly number[], v: readonly number[], tol: number): void => {
  for (const [i, x] of u.entries()) expect(Math.abs(x - (v[i] as number))).toBeLessThan(tol)
}

/** r'' = -mu r / |r|^3 as a first-order field on [r, v]. */
const twoBodyField =
  (mu: number) =>
  (_t: number, y: Value): Value => {
    const r = reshape(sliceAxis(y, 0, 0, 3), [3])
    const v = reshape(sliceAxis(y, 0, 3, 6), [3])
    const r3 = pow(sum(mul(r, r)), 1.5)
    return reshape(stack([v, mul(mul(r, pow(r3, -1)), -mu)]), [6])
  }

const stateArray = (r: Vec3, v: Vec3): number[] => [...r, ...v]

describe('elements and state vectors', () => {
  const cases: OrbitalElements[] = [
    { a: 2, e: 0.3, i: 0.4, raan: 1.1, argp: 2.2, nu: 0.7 },
    { a: 1, e: 0.9, i: 2.9, raan: 5, argp: 0.3, nu: 4.5 },
    { a: 5, e: 0.01, i: 1.0, raan: 0.2, argp: 6, nu: 3 },
    { a: -3, e: 1.7, i: 0.8, raan: 3, argp: 1, nu: -0.9 },
    { a: -0.5, e: 4, i: 1.5, raan: 0.1, argp: 0.2, nu: 0.4 },
    { a: Number.POSITIVE_INFINITY, e: 1, p: 2, i: 0.5, raan: 1, argp: 2, nu: 1.3 },
  ]

  it('round-trips the elements through the state vector for ellipses, parabolas and hyperbolas', () => {
    for (const el of cases) {
      const { r, v } = stateFromElements(MU, el)
      const back = elementsFromState(MU, r, v)
      expect(back.e).toBeCloseTo(el.e, 10)
      expect(back.i).toBeCloseTo(el.i, 10)
      expect(back.raan).toBeCloseTo(el.raan % (2 * Math.PI), 9)
      expect(back.argp).toBeCloseTo(el.argp % (2 * Math.PI), 9)
      if (Number.isFinite(el.a)) expect(back.a / el.a).toBeCloseTo(1, 10)
      else expect(back.a).toBe(Number.POSITIVE_INFINITY)
      const nu = el.e < 1 ? ((el.nu % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) : el.nu
      expect(back.nu).toBeCloseTo(nu, 9)
    }
  })

  it('has the speed and radius of the vis-viva equation and a state of the right energy', () => {
    for (const el of cases.slice(0, 5)) {
      const { r, v } = stateFromElements(MU, el)
      const rn = Math.hypot(...r)
      expect(Math.hypot(...v)).toBeCloseTo(visViva(MU, rn, el.a), 10)
      expect(keplerInvariants(MU, r, v).energy).toBeCloseTo(-MU / (2 * el.a), 10)
    }
  })

  it('gives a parabola exactly the escape speed', () => {
    const el = cases[5] as OrbitalElements
    const { r, v } = stateFromElements(MU, el)
    expect(Math.hypot(...v)).toBeCloseTo(escapeSpeed(MU, Math.hypot(...r)), 12)
    expect(keplerInvariants(MU, r, v).energy).toBeCloseTo(0, 12)
  })

  it('points the eccentricity vector at periapsis with length e', () => {
    const el = cases[0] as OrbitalElements
    const peri = stateFromElements(MU, { ...el, nu: 0 })
    const inv = keplerInvariants(MU, peri.r, peri.v)
    expect(Math.hypot(...inv.eccentricity)).toBeCloseTo(el.e, 12)
    const dir = peri.r.map((x) => x / Math.hypot(...peri.r))
    close(
      inv.eccentricity.map((x) => x / el.e),
      dir,
      1e-12,
    )
  })

  it('uses the stated conventions where the angles are undefined', () => {
    // Circular and inclined: the true anomaly is the argument of latitude, argp = 0.
    const inclined = stateFromElements(1, { a: 1, e: 0, i: 0.5, raan: 1, argp: 0, nu: 2 })
    const a = elementsFromState(1, inclined.r, inclined.v)
    expect([a.argp, a.nu, a.raan]).toEqual([0, expect.closeTo(2, 9), expect.closeTo(1, 9)])
    // Equatorial and eccentric: the node is 0 and argp is measured from x.
    const equatorial = stateFromElements(1, { a: 1, e: 0.2, i: 0, raan: 0, argp: 1.3, nu: 0.5 })
    const b = elementsFromState(1, equatorial.r, equatorial.v)
    expect([b.raan, b.argp, b.nu]).toEqual([0, expect.closeTo(1.3, 9), expect.closeTo(0.5, 9)])
    // Retrograde equatorial (i = pi) closes the same loop.
    const retro = stateFromElements(1, { a: 1, e: 0.2, i: Math.PI, raan: 0, argp: 1.3, nu: 0.5 })
    const c = elementsFromState(1, retro.r, retro.v)
    expect([c.i, c.argp, c.nu]).toEqual([expect.closeTo(Math.PI, 9), expect.closeTo(1.3, 9), expect.closeTo(0.5, 9)])
    // Circular and equatorial: the true longitude.
    const flat = stateFromElements(1, { a: 1, e: 0, i: 0, raan: 0, argp: 0, nu: 4 })
    expect(elementsFromState(1, flat.r, flat.v).nu).toBeCloseTo(4, 9)
  })

  it('rejects rectilinear orbits, a parabola without p and a true anomaly beyond the asymptote', () => {
    expect(() => elementsFromState(1, [1, 0, 0], [1, 0, 0])).toThrow()
    expect(() => semiLatusRectum({ a: Number.POSITIVE_INFINITY, e: 1, i: 0, raan: 0, argp: 0, nu: 0 })).toThrow()
    expect(() => stateFromElements(1, { a: -1, e: 2, i: 0, raan: 0, argp: 0, nu: 3 })).toThrow()
    expect(() => stateFromElements(0, cases[0] as OrbitalElements)).toThrow()
  })
})

describe('analytic propagation', () => {
  it('returns to the start after one period', () => {
    const el: OrbitalElements = { a: 3, e: 0.6, i: 0.3, raan: 0.4, argp: 1, nu: 2 }
    const T = orbitalPeriod(MU, el.a)
    expect(T).toBeCloseTo((2 * Math.PI) / meanMotion(MU, el), 12)
    const start = stateFromElements(MU, el)
    const end = keplerState(MU, el, T)
    close(stateArray(end.r, end.v), stateArray(start.r, start.v), 1e-10)
    const back = keplerState(MU, el, -T)
    close(stateArray(back.r, back.v), stateArray(start.r, start.v), 1e-10)
  })

  it('conserves the energy, the angular momentum and the eccentricity vector', () => {
    for (const el of [
      { a: 3, e: 0.6, i: 0.3, raan: 0.4, argp: 1, nu: 2 },
      { a: -2, e: 1.5, i: 0.3, raan: 0.4, argp: 1, nu: -1 },
    ]) {
      const s0 = stateFromElements(MU, el)
      const i0 = keplerInvariants(MU, s0.r, s0.v)
      for (const dt of [0.37, 1.9, 5]) {
        const s = keplerState(MU, el, dt)
        const i1 = keplerInvariants(MU, s.r, s.v)
        expect(i1.energy).toBeCloseTo(i0.energy, 10)
        close(i1.h, i0.h, 1e-10)
        close(i1.eccentricity, i0.eccentricity, 1e-10)
      }
    }
  })

  it("agrees with a numerical integration of r'' = -mu r / r^3 for an ellipse, a parabola and a hyperbola", () => {
    const els: OrbitalElements[] = [
      { a: 2, e: 0.5, i: 0.3, raan: 1, argp: 2, nu: 0.5 },
      { a: Number.POSITIVE_INFINITY, e: 1, p: 1.5, i: 0.3, raan: 1, argp: 2, nu: -1 },
      { a: -2, e: 1.8, i: 0.3, raan: 1, argp: 2, nu: -0.5 },
    ]
    for (const el of els) {
      const s0 = stateFromElements(MU, el)
      const times = [0.5, 1.7, 4]
      const sol = solveIvp(twoBodyField(MU), [0, 4], stateArray(s0.r, s0.v), {
        method: 'DOP853',
        rtol: 1e-12,
        atol: 1e-13,
        tEval: times,
      })
      const ys = toFlat(sol.y as Value)
      times.forEach((t, k) => {
        const exact = keplerState(MU, el, t)
        close(stateArray(exact.r, exact.v), ys.slice(6 * k, 6 * k + 6), 1e-8)
      })
    }
  })

  it('advances only the true anomaly', () => {
    const el: OrbitalElements = { a: 2, e: 0.5, i: 0.3, raan: 1, argp: 2, nu: 0.5 }
    const later = propagateKepler(MU, el, 1)
    expect({ ...later, nu: el.nu }).toEqual(el)
    expect(later.nu).not.toBe(el.nu)
  })

  it('sweeps equal areas in equal times (Kepler’s second law)', () => {
    const el: OrbitalElements = { a: 1, e: 0.8, i: 0, raan: 0, argp: 0, nu: 0 }
    const area = (t0: number, t1: number): number => {
      const n = 4000
      let a = 0
      for (let k = 0; k < n; k++) {
        const s0 = keplerState(MU, el, t0 + ((t1 - t0) * k) / n)
        const s1 = keplerState(MU, el, t0 + ((t1 - t0) * (k + 1)) / n)
        a += 0.5 * (s0.r[0] * s1.r[1] - s0.r[1] * s1.r[0])
      }
      return a
    }
    const periapsisPass = area(-0.2, 0.2)
    const apoapsisPass = area(orbitalPeriod(MU, 1) / 2 - 0.2, orbitalPeriod(MU, 1) / 2 + 0.2)
    expect(apoapsisPass).toBeCloseTo(periapsisPass, 6)
    const s0 = stateFromElements(MU, el)
    const h = keplerInvariants(MU, s0.r, s0.v).hNorm
    expect(periapsisPass).toBeCloseTo((h / 2) * 0.4, 6)
  })

  it('errs for orbits with no period', () => {
    expect(() => orbitalPeriod(1, -1)).toThrow()
  })
})
