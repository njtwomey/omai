import { describe, expect, it } from 'vitest'
import { periodicCopies, periodicGrid, threeCopiesBound, wrappedNormalLogPdf, wrapToPeriod } from './periodic.ts'

describe('wrapToPeriod', () => {
  it('maps into [0, period)', () => {
    expect(wrapToPeriod(25, 24)).toBe(1)
    expect(wrapToPeriod(-0.5, 24)).toBe(23.5)
    expect(wrapToPeriod(24, 24)).toBe(0)
    expect(wrapToPeriod(0, 24)).toBe(0)
  })
  it('never returns the period itself, even for a tiny negative value', () => {
    expect(wrapToPeriod(-1e-20, 24)).toBeLessThan(24)
  })
  it('rejects a non-positive period', () => {
    expect(() => wrapToPeriod(1, 0)).toThrow()
  })
})

describe('periodicCopies and periodicGrid', () => {
  it('has 2k + 1 copies, a period apart', () => {
    expect(periodicCopies(5, 24, 2)).toEqual([-43, -19, 5, 29, 53])
  })
  it('the torus grid has (2k + 1)^d points and contains the point itself', () => {
    const g = periodicGrid([1, 2, 3], [24, 7, 12])
    expect(g.length).toBe(27)
    expect(g).toContainEqual([1, 2, 3])
    expect(g).toContainEqual([-23, 9, 15])
    expect(periodicGrid([1, 2], [24, 7], 2).length).toBe(25)
  })
})

describe('wrappedNormalLogPdf', () => {
  it('integrates to one over one period', () => {
    const n = 4000
    let s = 0
    for (let i = 0; i < n; i++) s += Math.exp(wrappedNormalLogPdf(((i + 0.5) * 24) / n, 23, 3, 24)) * (24 / n)
    expect(s).toBeCloseTo(1, 10)
  })
  it('is periodic and symmetric about the mean around the circle', () => {
    expect(wrappedNormalLogPdf(23.5, 0, 1, 24)).toBeCloseTo(wrappedNormalLogPdf(0.5, 0, 1, 24), 12)
    expect(wrappedNormalLogPdf(3, 7, 2, 24)).toBeCloseTo(wrappedNormalLogPdf(27, 7, 2, 24), 12)
  })
  it('three copies against many: the error stays under the bound, over means and points', () => {
    for (const sigma of [2, 3, 4, 6, 9, 12]) {
      let worst = 0
      for (let m = 0; m < 24; m += 2)
        for (let x = 0; x < 24; x += 0.5) {
          const full = Math.exp(wrappedNormalLogPdf(x, m, sigma, 24, 30))
          const three = Math.exp(wrappedNormalLogPdf(x, m, sigma, 24, 1))
          worst = Math.max(worst, (full - three) / full)
        }
      expect(worst).toBeLessThanOrEqual(threeCopiesBound(sigma, 24))
    }
  })
})

describe('threeCopiesBound', () => {
  it('matches the values quoted for a day: 8e-11 at 3 h, under 0.005 at 6 h, about 0.45 at 12 h', () => {
    expect(threeCopiesBound(3, 24)).toBeCloseTo(7.6e-11, 12)
    expect(threeCopiesBound(6, 24)).toBeLessThan(0.005)
    expect(threeCopiesBound(12, 24)).toBeCloseTo(0.447, 2)
  })
})
