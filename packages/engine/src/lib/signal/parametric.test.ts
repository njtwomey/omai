import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { key, normal } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

describe('parametric estimation', () => {
  it('recovers AR coefficients and gives the matching spectrum', () => {
    const e = toFlat(normal(key(11), [6000]))
    const x = new Array<number>(e.length).fill(0)
    for (let t = 2; t < e.length; t++) x[t] = 0.5 * (x[t - 1] ?? 0) - 0.25 * (x[t - 2] ?? 0) + (e[t] ?? 0)
    for (const m of [S.yuleWalker(x, 2), S.burg(x, 2)]) {
      const ar = toFlat(m.ar)
      expect(ar[0]).toBeCloseTo(0.5, 1)
      expect(ar[1]).toBeCloseTo(-0.25, 1)
    }
    // The AR spectrum integrates to the process variance.
    const spec = S.arSpectrum({ ar: [0.5, -0.25], sigma2: 1 }, { n: 2001 })
    const v = toFlat(spec.values)
    let integral = 0
    for (let i = 1; i < v.length; i++) integral += (((v[i] ?? 0) + (v[i - 1] ?? 0)) / 2) * (0.5 / 2000)
    // gamma_0 = (1 - phi2) / ((1 + phi2) ((1 - phi2)^2 - phi1^2)) for unit innovations.
    const [p1, p2] = [0.5, -0.25]
    expect(integral).toBeCloseTo((1 - p2) / ((1 + p2) * ((1 - p2) ** 2 - p1 * p1)), 2)
  })

  it('estimates frequencies by subspace methods and fits exponentials exactly', () => {
    const noise = toFlat(normal(key(4), [200]))
    const x = noise.map((w, n) => Math.cos(2 * Math.PI * 0.07 * n) + 0.7 * Math.sin(2 * Math.PI * 0.21 * n) + 0.05 * w)
    const f1 = S.music(x, 4, { m: 20 }).frequencies
    const f2 = S.esprit(x, 4, { m: 20 })
    for (const f of [f1, f2]) {
      expect(f[0]).toBeCloseTo(0.07, 3)
      expect(f[1]).toBeCloseTo(0.21, 3)
    }
    const damped = Array.from({ length: 30 }, (_, n) => 2 * 0.95 ** n * Math.cos(0.4 * n + 0.3))
    const r = S.prony(damped, 2)
    for (const d of r.damping) expect(d).toBeCloseTo(Math.log(0.95), 9)
    expect(Math.abs(r.frequencies[0] ?? 0)).toBeCloseTo(0.4 / (2 * Math.PI), 9)
  })

  it('differentiates Yule-Walker, Levinson-Durbin and AR spectra', () => {
    expectConformant((x) => S.yuleWalker(x, 2).ar, [tensor([1, 2, 0.5, -1, 0.3, 2, 1, -0.7, 0.2, 0.4])])
    expectConformant((r) => S.levinsonDurbin(r).ar, [tensor([2, 0.8, 0.3, 0.1])])
    expectConformant((a) => S.arSpectrum({ ar: a, sigma2: 1.5 }, { n: 5 }).values, [tensor([0.4, -0.2])])
  })
})
