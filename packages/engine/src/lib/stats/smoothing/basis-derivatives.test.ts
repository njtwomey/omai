/**
 * The exact derivatives of the bases equal Richardson-extrapolated central differences of their values (to 1e-6, away
 * from knots and centres where the high derivatives of piecewise or singular kernels jump), the B-spline derivatives
 * sum to zero (partition of unity), and a basis without a closed form says so.
 */

import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { bs, cc, cr, duchon, gp, ns, ps, type SmoothBasis, tp, ts } from './bases.ts'

const x = Array.from({ length: 14 }, (_, i) => i / 13)
// Points between the data (the knots and centres of the bases), with room for the stencil.
const at = [0.045, 0.118, 0.25, 0.41, 0.55, 0.68, 0.83, 0.93]

function richardson(basis: SmoothBasis, order: 1 | 2): number[] {
  const f = (q: number[]): number[] => Array.from(toFlat(basis.evaluate(q)))
  const k = (basis.X.shape[1] as number) ?? 0
  const stencil = (h: number): number[] => {
    const a = f(at.map((t) => t - h))
    const b = f(at)
    const c = f(at.map((t) => t + h))
    return a.map(
      (v, i) => (order === 1 ? (c[i] ?? 0) - v : v - 2 * (b[i] ?? 0) + (c[i] ?? 0)) / (order === 1 ? 2 * h : h * h),
    )
  }
  const h = 2e-3
  const coarse = stencil(h)
  const fine = stencil(h / 2)
  void k
  return fine.map((v, i) => (4 * v - (coarse[i] ?? 0)) / 3)
}

const cases: [string, () => SmoothBasis][] = [
  ['ps', () => ps(x, { k: 8 })],
  ['bs', () => bs(x, { df: 8 })],
  ['bs with intercept', () => bs(x, { df: 8, intercept: true })],
  ['cr', () => cr(x, { k: 7 })],
  ['cc', () => cc(x, { k: 7 })],
  ['tp', () => tp(x, { k: 7 })],
  ['ts', () => ts(x, { k: 7 })],
  ['duchon', () => duchon(x, { k: 7, m: 2, s: 0.25 })],
  ['gp matern 3/2', () => gp(x, { k: 7 })],
  ['gp matern 5/2', () => gp(x, { k: 7, kernel: 'matern52' })],
  ['gp squared exponential', () => gp(x, { k: 7, kernel: 'squared-exponential' })],
]

describe('exact basis derivatives', () => {
  for (const [name, make] of cases) {
    for (const order of [1, 2] as const) {
      it(`${name}, order ${order}: equal to Richardson-extrapolated differences`, () => {
        const basis = make()
        expect(basis.derivative).toBeDefined()
        const exact = Array.from(toFlat(basis.derivative?.(at, order) as never))
        const reference = richardson(basis, order)
        expect(exact.length).toBe(reference.length)
        let worst = 0
        exact.forEach((v, i) => {
          worst = Math.max(worst, Math.abs(v - (reference[i] ?? 0)) / (1 + Math.abs(v)))
        })
        expect(worst).toBeLessThan(1e-6)
      })
    }
  }

  it('the derivatives of a B-spline basis with its intercept sum to zero', () => {
    const basis = bs(x, { df: 8, intercept: true })
    for (const order of [1, 2] as const) {
      const d = Array.from(toFlat(basis.derivative?.(at, order) as never))
      const k = d.length / at.length
      at.forEach((_, r) => {
        let s = 0
        for (let c = 0; c < k; c++) s += d[r * k + c] ?? 0
        expect(Math.abs(s)).toBeLessThan(1e-9)
      })
    }
  })

  it('a basis without a closed form has no derivative', () => {
    expect(ns(x, { df: 5 }).derivative).toBeUndefined()
  })
})
