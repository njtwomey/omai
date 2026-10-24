import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import {
  amplificationFactor,
  btcs,
  courantNumber,
  crankNicolsonScheme,
  dispersion,
  ftcs,
  laxFriedrichsScheme,
  laxWendroffScheme,
  stableTimeStep,
  upwindScheme,
  vonNeumann,
} from './stability.ts'

describe('stability analysis', () => {
  it('FTCS is stable exactly up to r = 1/2', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.01, max: 1.5, noNaN: true }),
        (r) => vonNeumann(ftcs(r)).stable === r <= 0.5 + 1e-12,
      ),
    )
  })

  it('implicit heat schemes are unconditionally stable, CN preserves high-mode amplitude at large r', () => {
    for (const r of [0.1, 1, 10, 100]) {
      expect(vonNeumann(btcs(r)).stable).toBe(true)
      expect(vonNeumann(crankNicolsonScheme(r)).stable).toBe(true)
    }
    const g = amplificationFactor(crankNicolsonScheme(100), Math.PI)
    expect(Math.abs(g.re)).toBeGreaterThan(0.99)
  })

  it('advection schemes are stable for |c| <= 1 and their closed-form factors match', () => {
    for (const make of [upwindScheme, laxFriedrichsScheme, laxWendroffScheme]) {
      expect(vonNeumann(make(0.8)).stable).toBe(true)
      expect(vonNeumann(make(1.1)).stable).toBe(false)
    }
    const c = 0.6
    const theta = 1.1
    const lw = amplificationFactor(laxWendroffScheme(c), theta)
    expect(lw.re).toBeCloseTo(1 - c * c * (1 - Math.cos(theta)), 12)
    expect(lw.im).toBeCloseTo(-c * Math.sin(theta), 12)
  })

  it('dispersion: Lax-Wendroff lags, upwind dissipates', () => {
    const lw = dispersion(laxWendroffScheme(0.5), 0.5, 1)
    expect(lw.relativePhaseSpeed).toBeLessThan(1)
    const up = dispersion(upwindScheme(0.5), 0.5, 1)
    expect(up.amplitude).toBeLessThan(1)
  })

  it('CFL helpers', () => {
    expect(courantNumber(-2, 0.5, 2)).toBe(0.5)
    expect(stableTimeStep({ dx: 0.1, velocity: 2, diffusivity: 0.01, safety: 0.9 })).toBeCloseTo(0.045, 12)
    expect(stableTimeStep({ dx: 0.1 })).toBe(Number.POSITIVE_INFINITY)
    expect(() => courantNumber(1, 1, 0)).toThrow()
  })
})
