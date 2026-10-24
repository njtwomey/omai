import { describe, expect, it } from 'vitest'
import { type Tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const at = (sys: S.Lti, w: number): { re: number; im: number } => {
  const h = S.freqresp(sys, [w]).h
  return { re: toFlat(h.re as Tensor)[0] ?? Number.NaN, im: toFlat(h.im as Tensor)[0] ?? Number.NaN }
}

describe('margins', () => {
  const loops = [
    S.tf([6], [1, 6, 11, 6]),
    S.tf([10, 10], [1, 6, 5, 0]),
    S.tf([0.1, 0.05], [1, -1.4, 0.45], { dt: 0.1 }),
  ]

  it('sit where |L| = 1 and arg L = -180 degrees', () => {
    for (const g of loops) {
      const m = S.stabilityMargins(g)
      if (Number.isFinite(m.wgc)) {
        const h = at(g, m.wgc)
        expect(Math.hypot(h.re, h.im)).toBeCloseTo(1, 9)
        expect((Math.atan2(h.im, h.re) * 180) / Math.PI + 180).toBeCloseTo(m.pm < 0 ? m.pm + 360 : m.pm, 6)
      }
      if (Number.isFinite(m.wpc)) {
        const h = at(g, m.wpc)
        expect(h.im).toBeCloseTo(0, 9)
        expect(1 / Math.abs(h.re)).toBeCloseTo(m.gm, 8)
      }
    }
  })

  it('give a delay margin that brings the phase to -180 degrees', () => {
    const g = S.tf([10, 10], [1, 6, 5, 0])
    const m = S.stabilityMargins(g)
    const h = at(g, m.wdm)
    const phase = Math.atan2(h.im, h.re) - m.wdm * m.dm
    expect(Math.cos(phase)).toBeCloseTo(-1, 9)
    expect(S.margin(g).wcp).toBe(m.wgc)
  })

  it('report infinite margins without crossings', () => {
    const m = S.stabilityMargins(S.tf([0.5], [1, 1]))
    expect(m.gm).toBe(Number.POSITIVE_INFINITY)
    expect(m.pm).toBe(Number.POSITIVE_INFINITY)
    expect(Number.isNaN(m.wgc)).toBe(true)
    expect(() => S.stabilityMargins(S.tf([1, 2, 3], [1, 2], { dt: 1 }))).toThrow()
    expect(() => S.allMargins(S.ss([[-1]], [[1, 1]], [[1]]))).toThrow()
  })
})
