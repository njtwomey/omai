import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import * as T from './index.ts'

const series = fc.array(fc.double({ min: -100, max: 100, noNaN: true, noDefaultInfinity: true }), {
  minLength: 20,
  maxLength: 40,
})

describe('differencing', () => {
  it('differences once, twice and at a lag', () => {
    expect(T.difference([1, 4, 9, 16])).toEqual([3, 5, 7])
    expect(T.difference([1, 4, 9, 16], 2)).toEqual([2, 2])
    expect(T.difference([1, 2, 4, 8, 16], 1, 2)).toEqual([3, 6, 12])
    expect(T.difference([3, 1, 2], 0)).toEqual([3, 1, 2])
  })

  it('has the polynomial of a mixed regular and seasonal difference', () => {
    expect(T.differencePolynomial(1, 1, 4)).toEqual([1, -1, 0, 0, -1, 1])
    const y = Array.from({ length: 20 }, (_, t) => Math.sin(t) + t * t)
    const poly = T.differencePolynomial(2, 1, 5)
    const direct = T.difference(T.difference(y, 2), 1, 5)
    const byPolynomial = direct.map((_, i) => {
      const t = i + 7
      return poly.reduce((s, c, j) => s + c * (y[t - j] ?? 0), 0)
    })
    byPolynomial.forEach((v, i) => {
      expect(v).toBeCloseTo(direct[i] ?? 0, 9)
    })
  })

  it('integrates what it differenced, at any order and lag', () => {
    fc.assert(
      fc.property(series, fc.integer({ min: 1, max: 3 }), fc.integer({ min: 1, max: 4 }), (y, order, lag) => {
        const z = T.difference(y, order, lag)
        const back = T.integrate(z, y.slice(0, order * lag), order, lag)
        expect(back).toHaveLength(y.length)
        back.forEach((v, i) => {
          expect(v).toBeCloseTo(y[i] ?? 0, 6)
        })
      }),
    )
  })

  it('refuses bad arguments', () => {
    expect(() => T.difference([1, 2], -1)).toThrow()
    expect(() => T.integrate([1, 2], [1, 2], 1)).toThrow()
  })
})
