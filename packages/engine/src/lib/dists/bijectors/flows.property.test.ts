import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { jacobian } from '../../autodiff/index.ts'
import { slogdet } from '../../numerics/linalg/index.ts'
import { matmul, tanh, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as B from './index.ts'

const number = (lo: number, hi: number): fc.Arbitrary<number> => fc.double({ min: lo, max: hi, noNaN: true })
const vector = (n: number, lo: number, hi: number): fc.Arbitrary<number[]> =>
  fc.array(number(lo, hi), { minLength: n, maxLength: n })
const flat = (v: Value): number[] => Array.from(toFlat(v))

/** The round trip, and the log-determinant against the autodiff Jacobian of one row. */
function expectBijective(b: B.Bijector, x: number[], digits = 7): void {
  const y = b.forward(tensor(x))
  const back = flat(b.inverse(y))
  for (const [i, v] of back.entries()) expect(v).toBeCloseTo(x[i] ?? Number.NaN, digits)
  const logDet = flat(b.forwardLogDetJacobian(tensor(x)))[0] ?? Number.NaN
  const reference = flat(slogdet(jacobian((v: Value) => b.forward(v))(tensor(x))).logabsdet)[0] ?? Number.NaN
  expect(logDet).toBeCloseTo(reference, digits)
}

describe('flow layers on random parameters', () => {
  it('planar flows are bijective with the right log-determinant', () => {
    fc.assert(
      fc.property(vector(3, -2, 2), vector(3, -2, 2), number(-1, 1), vector(3, -2, 2), (u, w, b, x) => {
        fc.pre(w.reduce((a, v) => a + v * v, 0) > 0.05)
        expectBijective(B.planar(u, w, b), x)
      }),
      { numRuns: RUNS },
    )
  })

  it('radial flows are bijective with the right log-determinant', () => {
    fc.assert(
      fc.property(vector(3, -1, 1), number(0.2, 2), number(-3, 3), vector(3, -2, 2), (c, a, beta, x) => {
        fc.pre(Math.hypot(...x.map((v, i) => v - (c[i] ?? 0))) > 0.05)
        expectBijective(B.radial(c, a, beta), x)
      }),
      { numRuns: RUNS },
    )
  })

  it('rational-quadratic splines are bijective, monotone and have the right log-derivative, inside and outside the bound', () => {
    fc.assert(
      fc.property(vector(3 * 17, -2, 2), vector(3, -5, 5), (raw, x) => {
        const spline = B.rationalQuadraticSpline(B.splineParametersFromVector(tensor(raw), 6))
        const y = flat(spline.forward(tensor(x)))
        const back = flat(spline.inverse(tensor(y)))
        for (const [i, v] of back.entries()) expect(v).toBeCloseTo(x[i] ?? Number.NaN, 7)
        // The log-derivative is the log of the finite-difference slope.
        const h = 1e-6
        const shifted = flat(spline.forward(tensor(x.map((v) => v + h))))
        const slopes = flat(spline.forwardLogDetJacobian(tensor(x)))
        for (const [i, v] of shifted.entries())
          expect(Math.log((v - (y[i] ?? 0)) / h || 1e-300)).toBeCloseTo(slopes[i] ?? Number.NaN, 3)
      }),
      { numRuns: RUNS },
    )
  })

  it('LU linear layers are bijective for any invertible matrix and the log-determinant is the sum of the log-diagonal', () => {
    fc.assert(
      fc.property(vector(9, -1, 1), vector(3, -2, 2), (entries, x) => {
        // Diagonally dominant, hence invertible.
        const m = [0, 1, 2].map((r) => [0, 1, 2].map((c) => (entries[3 * r + c] ?? 0) + (r === c ? 3 : 0)))
        const layer = B.luLinear(B.luParameters(tensor(m)))
        expectBijective(layer, x, 8)
        expect(flat(layer.forwardLogDetJacobian(tensor(x)))[0]).toBeCloseTo(
          flat(slogdet(tensor(m)).logabsdet)[0] ?? Number.NaN,
          8,
        )
      }),
      { numRuns: RUNS },
    )
  })

  it('sinh-arcsinh and Kumaraswamy bijectors are bijective with the right log-derivative', () => {
    fc.assert(
      fc.property(number(-1, 1), number(0.3, 3), number(-3, 3), (eps, delta, x) => {
        expectBijective(B.sinhArcsinh(eps, delta), [x], 8)
      }),
      { numRuns: RUNS },
    )
    fc.assert(
      fc.property(number(0.4, 4), number(0.4, 4), number(0.05, 0.95), (a, b, x) => {
        expectBijective(B.kumaraswamyCdf(a, b), [x], 7)
      }),
      { numRuns: RUNS },
    )
  })

  it('residual flows with a contracting residual are bijective with the right log-determinant', () => {
    fc.assert(
      fc.property(vector(9, -2, 2), vector(3, -2, 2), (entries, x) => {
        const w = B.lipschitzNormalise(tensor([entries.slice(0, 3), entries.slice(3, 6), entries.slice(6, 9)]), 0.6)
        expectBijective(
          B.residualFlow((v) => tanh(matmul(v, w)), { terms: 60 }),
          x,
          7,
        )
      }),
      { numRuns: Math.min(RUNS, 10) },
    )
  })
})
