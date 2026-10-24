import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { add, eye, matmul, mul, sub, type Tensor, tensor, toFlat, transpose, type Value } from '../../tensor/index.ts'
import * as L from './index.ts'

// Entries on a grid of hundredths: realistic matrices without subnormal or extreme values.
const entry = fc.integer({ min: -150, max: 150 }).map((k) => k / 100)
const matrix = (m: number, n: number): fc.Arbitrary<number[][]> =>
  fc.array(fc.array(entry, { minLength: n, maxLength: n }), { minLength: m, maxLength: m })
const maxAbs = (v: Value): number => toFlat(v as Tensor).reduce((acc, x) => Math.max(acc, Math.abs(x)), 0)
const gram = (m: number[][]): Value => add(matmul(tensor(m), transpose(tensor(m))), mul(eye(m.length), 0.5))

describe('matrix equations leave small residuals', () => {
  it('continuous and discrete Lyapunov, and Sylvester', () => {
    fc.assert(
      fc.property(matrix(3, 3), matrix(3, 3), matrix(2, 2), matrix(3, 2), (a0, q0, b0, c0) => {
        const a = sub(mul(tensor(a0), 0.5), mul(eye(3), 2))
        const q = gram(q0)
        const x = L.solveContinuousLyapunov(a, q)
        expect(maxAbs(sub(add(matmul(a, x), matmul(x, transpose(a))), q))).toBeLessThan(1e-9 * (1 + maxAbs(x)))
        const contraction = mul(tensor(a0), 0.2)
        const xd = L.solveDiscreteLyapunov(contraction, q)
        expect(maxAbs(add(sub(matmul(contraction, matmul(xd, transpose(contraction))), xd), q))).toBeLessThan(
          1e-9 * (1 + maxAbs(xd)),
        )
        const b = add(tensor(b0), mul(eye(2), 3))
        const xs = L.solveSylvester(a, b, tensor(c0))
        expect(maxAbs(sub(add(matmul(a, xs), matmul(xs, b)), tensor(c0)))).toBeLessThan(1e-9 * (1 + maxAbs(xs)))
      }),
      { numRuns: 30 },
    )
  })

  it('continuous and discrete algebraic Riccati equations', () => {
    fc.assert(
      fc.property(matrix(3, 3), matrix(3, 2), matrix(3, 3), matrix(2, 2), (a0, b0, q0, r0) => {
        const a = tensor(a0)
        const b = add(
          tensor(b0),
          tensor([
            [1, 0],
            [0, 1],
            [0.5, 0.5],
          ]),
        )
        const q = gram(q0)
        const r = gram(r0)
        const x = L.solveCare(a, b, q, r)
        const bt = transpose(b)
        const care = add(
          sub(add(matmul(transpose(a), x), matmul(x, a)), matmul(x, matmul(b, L.solve(r, matmul(bt, x))))),
          q,
        )
        expect(maxAbs(care)).toBeLessThan(1e-7 * (1 + maxAbs(x)))
        const ad = mul(a, 0.5)
        const xd = L.solveDare(ad, b, q, r)
        const gain = L.solve(add(r, matmul(bt, matmul(xd, b))), matmul(bt, matmul(xd, ad)))
        const dare = add(
          sub(matmul(transpose(ad), matmul(xd, ad)), matmul(transpose(ad), matmul(xd, matmul(b, gain)))),
          sub(q, xd),
        )
        expect(maxAbs(dare)).toBeLessThan(1e-7 * (1 + maxAbs(xd)))
      }),
      { numRuns: 30 },
    )
  })
})
