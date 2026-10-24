import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  BURGERS,
  burgersRiemann,
  conservationLaw,
  godunovFlux,
  LIMITERS,
  linearFlux,
  rusanovFlux,
} from './finite-volume.ts'

describe('finite volumes', () => {
  it('limiters are TVD (0 <= phi(r) <= min(2r, 2), phi(1) = 1)', () => {
    for (const [name, phi] of Object.entries(LIMITERS)) {
      for (const r of [-1, 0, 0.25, 0.5, 1, 2, 5]) {
        const v = phi(r)
        expect(v).toBeGreaterThanOrEqual(0)
        if (r > 0) expect(v).toBeLessThanOrEqual(Math.min(2 * r, 2) + 1e-12)
      }
      if (name !== 'none') expect(phi(1)).toBeCloseTo(1, 12)
    }
  })

  it('Godunov and Rusanov fluxes are consistent', () => {
    fc.assert(
      fc.property(fc.double({ min: -3, max: 3, noNaN: true }), (u) => {
        return (
          Math.abs(godunovFlux(BURGERS, u, u) - BURGERS.f(u)) < 1e-12 &&
          Math.abs(rusanovFlux(BURGERS, u, u) - BURGERS.f(u)) < 1e-12
        )
      }),
    )
    expect(godunovFlux(linearFlux(-2), 1, 3)).toBe(-6)
  })

  it('Burgers shock and rarefaction converge to the exact Riemann solutions', () => {
    for (const [uL, uR] of [
      [1, 0],
      [0, 1],
      [-1, 1],
    ] as const) {
      for (const scheme of ['godunov', 'muscl'] as const) {
        const s = conservationLaw((x) => (x < 0.5 ? uL : uR), [0, 1], 400, {
          flux: BURGERS,
          tEnd: 0.25,
          scheme,
          boundary: 'outflow',
          limiter: 'vanLeer',
        })
        expect(s.status).toBe('success')
        const u = Array.from(s.u.data).slice(-400)
        let err = 0
        s.x.forEach((x, i) => {
          err += Math.abs((u[i] ?? 0) - burgersRiemann(uL, uR, x, 0.25, 0.5)) / 400
        })
        expect(err).toBeLessThan(scheme === 'godunov' ? 0.01 : 0.006)
      }
    }
  })

  it('periodic runs conserve mass exactly and do not increase total variation (property)', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.1, max: 2, noNaN: true }), (amp) => {
        const s = conservationLaw((x) => amp * Math.sin(2 * Math.PI * x), [0, 1], 64, {
          flux: BURGERS,
          tEnd: 0.3,
          scheme: 'muscl',
          limiter: 'minmod',
        })
        const tv = s.totalVariation
        const mass = s.mass
        const tvOk = tv.every((v, i) => i === 0 || v <= (tv[i - 1] ?? 0) + 1e-10)
        return tvOk && Math.abs((mass.at(-1) ?? 0) - (mass[0] ?? 0)) < 1e-12
      }),
      { numRuns: 8 },
    )
  })

  it('validates inputs', () => {
    expect(() => conservationLaw([1], [0, 1], 1, { flux: BURGERS, tEnd: 1 })).toThrow()
    expect(() => conservationLaw([1, 2], [0, 1], 3, { flux: BURGERS, tEnd: 1 })).toThrow()
  })
})
