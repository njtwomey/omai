import { describe, expect, it } from 'vitest'
import { checkPrimitive, grad, jvp } from '../autodiff/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import {
  add,
  eye,
  fromStorage,
  matmul,
  mul,
  pow,
  reshape,
  square,
  sub,
  sum,
  swapaxes,
  type Tensor,
  tanh,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import * as O from './index.ts'

/** Newton's method for x^3 + x = theta, elementwise, to machine precision. */
const cubicRoot = (t: Tensor): Tensor =>
  fromStorage(
    Float64Array.from(t.data as Float64Array, (th) => {
      let x = 0
      for (let k = 0; k < 60; k++) x -= (x ** 3 + x - th) / (3 * x * x + 1)
      return x
    }),
    t.shape,
    'float64',
  )
const cubic: O.ImplicitResidual = (x, t) => sub(add(pow(x, 3), x), t as Value)

describe('implicit differentiation', () => {
  it('root of an equation: reverse and forward mode agree with finite differences', () => {
    for (const linearSolver of ['dense', 'gmres'] as const) {
      const root = O.implicitRoot(cubic, (t) => cubicRoot(t), { linearSolver })
      const theta = tensor([0.5, -1.2, 2])
      expect(checkPrimitive((t) => root(t), [theta]).failures).toEqual([])
      // dx/dtheta = 1 / (3 x^2 + 1) elementwise.
      const x = toFlat(root(theta))
      const g = toFlat(grad((t: Value) => sum(root(t)))(theta))
      for (const [i, xi] of x.entries()) expect(g[i]).toBeCloseTo(1 / (3 * xi * xi + 1), 12)
      if (linearSolver !== 'dense') continue
      // Second order (dense solves only): d2x/dtheta2 = -6 x / (3 x^2 + 1)^3, by reverse over reverse.
      const g2 = toFlat(grad((t: Value) => sum(grad((u: Value) => sum(root(u)))(t)))(theta))
      for (const [i, xi] of x.entries()) expect(g2[i]).toBeCloseTo((-6 * xi) / (3 * xi * xi + 1) ** 3, 9)
    }
  })

  it('several parameters, shaped solutions and second-order use', () => {
    // F(X, A, B) = A X - B for a matrix X: X* = A^{-1} B.
    const F: O.ImplicitResidual = (x, a, b) => sub(matmul(a as Value, x), b as Value)
    const xs = O.implicitRoot(F, (a, b) => solve(a as Tensor, b as Tensor))
    const a = tensor([
      [3, 1],
      [0.5, 2],
    ])
    const b = tensor([
      [1, 0],
      [2, 1],
    ])
    expect(checkPrimitive((av, bv) => xs(av, bv), [a, b]).failures).toEqual([])
    // Against differentiating the direct solve.
    const direct = toFlat(grad((av: Value) => sum(square(solve(av, b))))(a))
    const implicit = toFlat(grad((av: Value) => sum(square(xs(av, b))))(a))
    for (const [i, v] of direct.entries()) expect(implicit[i]).toBeCloseTo(v, 12)
    // Forward mode through the tangent rule.
    const t = tensor([
      [1, 0],
      [0, 0],
    ])
    const fwd = toFlat(jvp((av: Value) => xs(av, b), a, t).tangent)
    const ref = toFlat(jvp((av: Value) => solve(av, b), a, t).tangent)
    for (const [i, v] of ref.entries()) expect(fwd[i]).toBeCloseTo(v, 12)
  })

  it('fixed points: an equilibrium layer x = tanh(W x + theta)', () => {
    const w = tensor([
      [0.2, -0.1, 0.05],
      [0.1, 0.3, -0.2],
      [-0.15, 0.05, 0.1],
    ])
    const T: O.ImplicitResidual = (x, t) => tanh(add(reshape(matmul(w, reshape(x, [-1, 1])), [-1]), t as Value))
    const layer = O.implicitFixedPoint(T, undefined, { x0: [0, 0, 0] })
    const theta = tensor([0.3, -0.5, 0.8])
    const x = layer(theta)
    expect(toFlat(sub(x, T(x, theta))).every((v) => Math.abs(v) < 1e-12)).toBe(true)
    expect(checkPrimitive((t) => layer(t), [theta]).failures).toEqual([])
    const krylov = O.implicitFixedPoint(T, undefined, { x0: [0, 0, 0], linearSolver: 'gmres' })
    expect(checkPrimitive((t) => krylov(t), [theta]).failures).toEqual([])
    expect(() => O.implicitFixedPoint(T)(theta)).toThrow()
  })

  it('argmin layers: ridge regression against its closed form', () => {
    const A = tensor([
      [1, 2],
      [0.5, -1],
      [2, 0.3],
    ])
    const y = tensor([1, -0.5, 2])
    const f: O.ImplicitResidual = (x, lam) =>
      add(sum(square(sub(reshape(matmul(A, reshape(x, [-1, 1])), [-1]), y))), mul(sum(square(x)), lam as Value))
    const closed = (lam: Value): Value => {
      const at = swapaxes(A, 0, 1)
      return solve(add(matmul(at, A), mul(eye(2), lam)), reshape(matmul(at, reshape(y, [-1, 1])), [-1]))
    }
    for (const linearSolver of ['dense', 'cg'] as const) {
      const xs = O.implicitArgmin(f, (lam) => closed(lam), { linearSolver })
      const lam = tensor(0.7)
      expect(toFlat(xs(lam)).map((v) => Math.round(v * 1e12))).toEqual(
        toFlat(closed(lam)).map((v) => Math.round(v * 1e12)),
      )
      const gi = toFlat(grad((l: Value) => sum(xs(l)))(lam))
      const gc = toFlat(grad((l: Value) => sum(closed(l)))(lam))
      expect(gi[0]).toBeCloseTo(gc[0] ?? Number.NaN, 10)
      expect(checkPrimitive((l) => xs(l), [lam]).failures).toEqual([])
    }
    expect(() => O.implicitArgmin(f)(tensor(1))).toThrow()
  })

  it('rejects a residual of the wrong size', () => {
    const bad = O.implicitRoot(
      (x) => sum(x),
      () => tensor([1, 2]),
    )
    expect(() => bad(tensor([1]))).toThrow()
  })
})
