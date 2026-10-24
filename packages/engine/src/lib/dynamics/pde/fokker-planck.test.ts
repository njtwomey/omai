import { describe, expect, it } from 'vitest'
import { fokkerPlanck, scharfetterGummel, stationaryDensity, stationaryDensityDiscrete } from './fokker-planck.ts'

describe('Fokker-Planck', () => {
  const ou = { drift: (x: number) => -2 * x, sigma: () => 1 }

  it('conserves mass, stays positive and matches the OU moments', () => {
    const ev = fokkerPlanck(ou, (x) => Math.exp(-((x - 1) ** 2) / 0.02), [-4, 4], 400, { dt: 0.005, tEnd: 0.5 })
    const p = Array.from(ev.p.data).slice(-400)
    expect(Math.abs(p.reduce((a, v) => a + v * 0.02, 0) - 1)).toBeLessThan(1e-12)
    expect(p.every((v) => v >= 0)).toBe(true)
    const t = 0.5
    expect(Math.abs((ev.mean.at(-1) ?? 0) - Math.exp(-2 * t))).toBeLessThan(5e-3)
    const v = 0.25 * (1 - Math.exp(-4 * t)) + 0.01 * Math.exp(-4 * t)
    expect(Math.abs((ev.variance.at(-1) ?? 0) - v)).toBeLessThan(5e-3)
  })

  it('the discrete and continuous stationary densities agree, and the generator conserves mass', () => {
    const disc = stationaryDensityDiscrete(ou, [-3, 3], 120)
    const cont = stationaryDensity(ou, [-3, 3], 2001)
    const at = (x: number): number => cont.p[Math.round(((x + 3) / 6) * 2000)] ?? 0
    disc.x.forEach((x, i) => {
      expect(Math.abs((disc.p[i] ?? 0) - at(x))).toBeLessThan(5e-3)
    })
    const g = scharfetterGummel({ drift: (x) => Math.sin(x), sigma: (x) => 1 + 0.5 * Math.cos(x) }, [-2, 2], 30)
    const colSums = new Float64Array(30)
    g.rows.forEach((row) => {
      for (const [j, w] of row) colSums[j] = (colSums[j] ?? 0) + w
    })
    for (const s of colSums) expect(Math.abs(s)).toBeLessThan(1e-10)
  })

  it('validates inputs', () => {
    expect(() => scharfetterGummel(ou, [0, 1], 1)).toThrow()
    expect(() => scharfetterGummel({ drift: () => 0, sigma: () => 0 }, [0, 1], 4)).toThrow()
    expect(() => fokkerPlanck(ou, () => 0, [0, 1], 4, { dt: 0.1, tEnd: 1 })).toThrow()
  })
})
