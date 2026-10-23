import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import {
  cos,
  item,
  matmul,
  mul,
  neg,
  sin,
  sliceAxis,
  squeeze,
  sub,
  tanh,
  tensor,
  toArray,
  type Value,
} from '../tensor/index.ts'
import * as L from './index.ts'

const points = tensor([
  [0.1, 0.7],
  [0.5, -0.3],
  [1.2, 0.4],
])
const col = (x: Value, k: number): Value => squeeze(sliceAxis(x, 1, k, k + 1), 1)
// u(x, y) = sin(x) cos(y): grad (cos x cos y, -sin x sin y), d2u/dx2 = d2u/dy2 = -u, Laplacian -2u.
const u = (x: Value): Value => mul(sin(col(x, 0)), cos(col(x, 1)))

describe('physics-informed losses', () => {
  it('differentiates fields to second order', () => {
    const d = L.fieldDerivatives(u, points)
    const xs = toArray(points) as number[][]
    const value = xs.map(([a = 0, b = 0]) => Math.sin(a) * Math.cos(b))
    expect(toArray(d.value)).toEqual(value.map((v) => expect.closeTo(v, 14)))
    expect(toArray(d.gradient)).toEqual(
      xs.map(([a = 0, b = 0]) => [
        expect.closeTo(Math.cos(a) * Math.cos(b), 14),
        expect.closeTo(-Math.sin(a) * Math.sin(b), 14),
      ]),
    )
    expect(toArray(d.laplacian as Value)).toEqual(value.map((v) => expect.closeTo(-2 * v, 14)))
    expect(L.fieldDerivatives(u, points, { order: 1 }).laplacian).toBeUndefined()
  })

  it('makes the residual vanish for an exact solution and differentiates in network weights', () => {
    // The Helmholtz equation u_xx + u_yy + 2u = 0, which the field above solves exactly.
    const residual = (d: L.FieldDerivatives): Value => sub(d.laplacian as Value, mul(neg(d.value), 2))
    expect(item(L.pdeResidualLoss(u, points, residual))).toBeCloseTo(0, 14)
    const v = tensor([1, -0.5, 0.7])
    const net = (w: Value) => (x: Value) => matmul(tanh(matmul(x, w)), v)
    const w0 = tensor([
      [0.5, -1, 0.3],
      [0.2, 0.8, -0.4],
    ])
    expectConformant((w: Value) => L.pdeResidualLoss(net(w), points, (d) => sub(d.laplacian as Value, 1)), [w0])
    expectConformant(
      (w: Value) =>
        L.boundaryLoss(net(w), points, [0, 1, 0], {
          kind: 'neumann',
          normals: [
            [1, 0],
            [0, 1],
            [0.6, 0.8],
          ],
        }),
      [w0],
    )
    expectConformant((w: Value) => L.initialConditionLoss(net(w), points, [0, 0.5, 1], { derivative: 1 }), [w0])
    expect(() => L.boundaryLoss(u, points, [0, 0, 0], { kind: 'neumann' })).toThrow()
  })

  it('checks boundary and initial conditions', () => {
    expect(item(L.boundaryLoss(u, [[0, 0.3]], [0]))).toBe(0)
    // du/dt of u(x, t) = sin(x) cos(t) at t = 0 is zero.
    expect(item(L.initialConditionLoss(u, [[0.4, 0]], [0], { derivative: 1 }))).toBeCloseTo(0, 15)
  })

  it('balances loss terms by gradient norms and NTK traces', () => {
    const w = tensor([0.5, -1, 2])
    const terms = [(p: Value) => mul(matmul(p, tensor([1, 1, 1])), 3), (p: Value) => matmul(p, tensor([1, 0, 0]))]
    // grad L0 = (3, 3, 3), max 3; grad L1 = (1, 0, 0), mean 1/3: weight 9.
    expect(L.gradientNormWeights(terms, w, { alpha: 1 })).toEqual([1, 9])
    expect(L.gradientNormWeights(terms, w, { previous: [1, 1] })).toEqual([1, expect.closeTo(0.1 + 0.9 * 9, 14)])
    const pointwise = [(p: Value) => mul(p, 2), (p: Value) => sliceAxis(p, 0, 0, 1)]
    // Traces: 4 * 3 = 12 and 1; total 13.
    expect(L.ntkWeights(pointwise, w)).toEqual([13 / 12, 13])
  })
})
