import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import {
  add,
  cos,
  exp,
  mul,
  neg,
  sin,
  sliceAxis,
  squeeze,
  stack,
  sub,
  sum,
  tensor,
  toArray,
  toFlat,
  type Value,
  zerosLike,
} from '../tensor/index.ts'
import * as F from './index.ts'

const c = (p: Value, i: number): Value => squeeze(sliceAxis(p, -1, i, i + 1), -1)
const potential: F.ScalarField = (p) => add(mul(c(p, 0), c(p, 1)), sin(c(p, 2)))
const swirl: F.VectorField = (p) => stack([neg(c(p, 1)), c(p, 0), mul(c(p, 0), c(p, 2))], -1)
const pts = tensor([
  [0.3, -0.7, 1.1],
  [1.2, 0.4, -0.5],
])

describe('derivatives of fields', () => {
  it('the curl of a gradient and the divergence of a curl vanish', () => {
    const gradField: F.VectorField = (p) => F.gradientAt(potential, p)
    for (const v of toFlat(F.curl(gradField, pts))) expect(v).toBeCloseTo(0, 13)
    for (const v of toFlat(F.divergence((p) => F.curl(swirl, p), pts))) expect(v).toBeCloseTo(0, 13)
  })

  it('the Hessian is symmetric and the Jacobian of the gradient', () => {
    const h = toArray(F.hessianAt(potential, pts)) as number[][][]
    for (const m of h)
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(m[i]?.[j]).toBeCloseTo(m[j]?.[i] ?? 0, 14)
    const directional = toFlat(F.directionalDerivative(potential, pts, [0, 0, 2]))
    const grads = toArray(F.gradientAt(potential, pts)) as number[][]
    for (const [i, d] of directional.entries()) expect(d).toBeCloseTo(grads[i]?.[2] ?? 0, 14)
    expect(() => F.curl((p) => p, [[1, 2, 3, 4]])).toThrow()
  })

  it('local models agree with the function to first and second order', () => {
    const q = F.quadraticApproximation(potential, [0.2, 0.1, 0.4])
    const near = (h: number): number[] => [0.2 + h, 0.1 - h, 0.4 + 2 * h]
    const err = (h: number, model: 'linear' | 'quadratic'): number =>
      Math.abs((toFlat(potential(tensor([near(h)])))[0] ?? 0) - (toFlat(q[model]([near(h)]))[0] ?? 0))
    expect(err(1e-2, 'linear') / err(1e-3, 'linear')).toBeCloseTo(100, -1)
    expect(err(1e-2, 'quadratic') / err(1e-3, 'quadratic')).toBeCloseTo(1000, -2)
    const plane = F.graphTangentPlane((p) => exp(sum(p, -1)), [0, 0])
    expect(plane.point).toEqual([0, 0, 1])
    expect(Math.hypot(...plane.normal)).toBeCloseTo(1, 15)
  })

  it('classifies critical points by the Hessian', () => {
    const sq = (p: Value): Value => mul(p, p)
    expect(F.classifyCriticalPoint((p) => sum(sq(p), -1), [0, 0]).kind).toBe('minimum')
    expect(F.classifyCriticalPoint((p) => neg(sum(sq(p), -1)), [0, 0, 0]).kind).toBe('maximum')
    expect(F.classifyCriticalPoint((p) => sum(mul(sq(p), sq(p)), -1), [0, 0]).kind).toBe('degenerate')
  })
})

describe('Lagrange multipliers', () => {
  it('finds the point where the gradients are parallel', () => {
    // Nearest point of the plane x + 2y + 2z = 9 to the origin is (1, 2, 2).
    const r = F.lagrangeMultipliers(
      (p) => sum(mul(p, p), -1),
      [(p) => sub(add(c(p, 0), mul(2, add(c(p, 1), c(p, 2)))), 9)],
      [0.5, 0.5, 0.5],
    )
    expect(r.converged).toBe(true)
    for (const [i, v] of r.x.entries()) expect(v).toBeCloseTo([1, 2, 2][i] ?? 0, 10)
    expect(r.multipliers[0]).toBeCloseTo(2, 10)
    expect(r.value).toBeCloseTo(9, 10)
    expect(() => F.lagrangeMultipliers((p) => sum(p, -1), [], [0])).toThrow()
  })

  it('handles two constraints', () => {
    // Extremise z on the intersection of the sphere and the plane x = y.
    const r = F.lagrangeMultipliers(
      (p) => c(p, 2),
      [(p) => sub(sum(mul(p, p), -1), 1), (p) => sub(c(p, 0), c(p, 1))],
      [0.2, 0.2, 0.9],
    )
    expect(r.x[2]).toBeCloseTo(1, 10)
  })
})

describe('integrals and the integral theorems', () => {
  const circle: F.ParametricCurve = (t) => stack([cos(t), sin(t)], -1)
  const plane: F.VectorField = (p) =>
    stack([sub(mul(c(p, 0), c(p, 0)), c(p, 1)), mul(c(p, 0), mul(c(p, 1), c(p, 1)))], -1)
  const disc = {
    map: F.polarToCartesian,
    box: [
      [0, 1],
      [0, 2 * Math.PI],
    ] as const,
  }

  it("Green's theorem holds for a non-conservative field on the disc", () => {
    const r = F.greensTheorem(plane, { curve: circle, range: [0, 2 * Math.PI] }, disc)
    expect(r.boundary).toBeCloseTo(r.interior, 11)
    expect(Math.abs(r.boundary)).toBeGreaterThan(1)
  })

  it('a gradient field does path-independent work', () => {
    const gradField: F.VectorField = (p) => F.gradientAt(potential, p)
    const straight: F.ParametricCurve = (t) => stack([t, mul(t, 2), mul(t, -1)], -1)
    const bent: F.ParametricCurve = (t) => stack([mul(t, t), mul(2, mul(t, mul(t, t))), neg(t)], -1)
    const end = toFlat(potential(tensor([[1, 2, -1]])))[0] ?? 0
    const start = toFlat(potential(tensor([[0, 0, 0]])))[0] ?? 0
    expect(F.vectorLineIntegral(gradField, straight, 0, 1)).toBeCloseTo(end - start, 11)
    expect(F.vectorLineIntegral(gradField, bent, 0, 1)).toBeCloseTo(end - start, 11)
  })

  it("Stokes' theorem on a hemisphere and the divergence theorem on a ball", () => {
    const rim = { curve: (t: Value) => stack([cos(t), sin(t), zerosLike(t)], -1), range: [0, 2 * Math.PI] as const }
    const s = F.stokesTheorem(swirl, rim, {
      surface: F.sphere(1),
      box: [
        [0, Math.PI / 2],
        [0, 2 * Math.PI],
      ],
    })
    expect(s.difference).toBeLessThan(1e-10)
    expect(s.boundary).toBeCloseTo(2 * Math.PI, 10)
    const ball = {
      map: F.sphericalToCartesian,
      box: [
        [0, 1],
        [0, Math.PI],
        [0, 2 * Math.PI],
      ] as const,
    }
    const d = F.divergenceTheorem(
      swirl,
      {
        surface: F.sphere(1),
        box: [
          [0, Math.PI],
          [0, 2 * Math.PI],
        ],
      },
      ball,
    )
    expect(d.difference).toBeLessThan(1e-10)
    // div(swirl) = x, which integrates to zero over the ball.
    expect(d.interior).toBeCloseTo(0, 10)
  })

  it('outward flux of the identity field through the unit circle is twice its area', () => {
    expect(F.planeFlux((p) => p, circle, 0, 2 * Math.PI)).toBeCloseTo(2 * Math.PI, 12)
    expect(F.scalarLineIntegral((p) => mul(c(p, 0), c(p, 0)), circle, 0, 2 * Math.PI)).toBeCloseTo(Math.PI, 12)
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    expectConformant((p) => F.gradientAt(potential, p), [pts])
    expectConformant((p) => F.jacobianAt(swirl, p), [pts])
    expectConformant((p) => F.divergence(swirl, p), [pts])
    expectConformant((p) => F.curl(swirl, p), [pts])
    expectConformant((p) => F.directionalDerivative(potential, p, [1, 2, 2]), [pts])
  })
})
