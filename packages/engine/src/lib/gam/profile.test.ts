/**
 * Tests for profiling a family parameter: data are simulated from a negative binomial and a Tweedie with a known
 * parameter and the profile of the REML score recovers it; the minimiser has the lowest score of the evaluations.
 */

import { describe, expect, it } from 'vitest'
import { negativeBinomial, tweedie } from '../dists/families/index.ts'
import { foldIn, key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { GAM_DEFAULTS, profileFamilyParameter, s } from './index.ts'

const n = 400
const x = Array.from({ length: n }, (_, i) => [(i + 0.5) / n])
const mean = (t: number): number => Math.exp(0.8 + 0.9 * Math.sin(2 * Math.PI * t))

describe('profiling the family parameter', () => {
  it('recovers the dispersion of a negative binomial', { timeout: 120000 }, () => {
    const alpha = 0.5
    const family = negativeBinomial(alpha)
    const y = x.map((r, i) => toFlat(family.distribution(mean(r[0] ?? 0)).sample(foldIn(key(4), i), []))[0] ?? 0)
    const fit = profileFamilyParameter(
      x,
      y,
      { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] },
      {
        make: (a) => negativeBinomial(a),
        bounds: [0.02, 5],
        log: true,
      },
    )
    expect(fit.parameter).toBeGreaterThan(0.3)
    expect(fit.parameter).toBeLessThan(0.8)
    expect(fit.criterion).toBeLessThanOrEqual(Math.min(...fit.evaluations.map((e) => e.criterion)) + 1e-6)
    expect(fit.state.converged).toBe(true)
  })

  it('recovers the power of a Tweedie, with the scale estimated', { timeout: 300000 }, () => {
    const power = 1.5
    const family = tweedie(power)
    const y = x.map((r, i) => toFlat(family.distribution(mean(r[0] ?? 0), 1.2).sample(foldIn(key(6), i), []))[0] ?? 0)
    const fit = profileFamilyParameter(
      x,
      y,
      { ...GAM_DEFAULTS, terms: [s(0, { k: 8 })] },
      {
        make: (p) => tweedie(p),
        bounds: [1.1, 1.9],
        tol: 0.01,
      },
    )
    expect(fit.parameter).toBeGreaterThan(1.3)
    expect(fit.parameter).toBeLessThan(1.7)
    expect(fit.criterion).toBeLessThanOrEqual(Math.min(...fit.evaluations.map((e) => e.criterion)) + 1e-6)
  })
})
