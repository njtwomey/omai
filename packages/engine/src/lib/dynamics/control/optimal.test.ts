import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { item, reshape, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { dlqr, lqr, riccatiOde, riccatiRecursion } from './optimal.ts'

const A = [
  [0, 1],
  [-2, -0.3],
]
const B = [[0], [1]]
const Q = [
  [1, 0],
  [0, 1],
]
const Ad = A.map((r, i) => r.map((v, j) => (i === j ? 1 : 0) + 0.1 * v))
const Bd = [[0], [0.1]]

describe('optimal control', () => {
  it('LQR closes a stable loop', () => {
    expect(lqr(A, B, Q, [[0.5]]).E.re.every((v) => v < 0)).toBe(true)
    const d = dlqr(Ad, Bd, Q, [[1]])
    expect(d.E.re.every((v, i) => Math.hypot(v, d.E.im[i] ?? 0) < 1)).toBe(true)
  })

  it('finite-horizon Riccati solutions converge to the algebraic ones', () => {
    const d = dlqr(Ad, Bd, Q, [[1]])
    const rec = riccatiRecursion(
      Ad,
      Bd,
      Q,
      [[1]],
      [
        [0, 0],
        [0, 0],
      ],
      400,
    )
    const k0 = toFlat(rec.gains[0] as Value)
    toFlat(d.K).forEach((v, i) => {
      expect(Math.abs(v - (k0[i] ?? 0))).toBeLessThan(1e-8)
    })
    const c = lqr(A, B, Q, [[0.5]])
    const ode = riccatiOde(
      A,
      B,
      Q,
      [[0.5]],
      [
        [0, 0],
        [0, 0],
      ],
      30,
    )
    const p = toFlat(ode.P)
    toFlat(c.S).forEach((v, i) => {
      expect(Math.abs(v - (p[i] ?? 0))).toBeLessThan(1e-6)
    })
  })

  it('the Riccati solution is differentiable in the weights', () => {
    // a = b = q = 1: S = (1 + sqrt(1 + 4 r)) / 2, so dS/dr = 1 / sqrt(1 + 4 r).
    const S = (r: Value): Value => sum(dlqr([[1]], [[1]], [[1]], reshape(r, [1, 1])).S)
    expect(Math.abs(item(grad(S)(tensor(2))) - 1 / 3)).toBeLessThan(1e-8)
  })
})
