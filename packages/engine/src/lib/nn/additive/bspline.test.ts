import { describe, expect, it } from 'vitest'
import { grad, gradCheck, numericalGradient } from '../../autodiff/index.ts'
import { bs, ps } from '../../stats/smoothing/index.ts'
import { asTensor, matmul, mul, reshape, sum, toFlat, type Value } from '../../tensor/index.ts'
import {
  bsplineBasis,
  bsplineBasisByOps,
  bsplineBasisDerivative,
  extendKnots,
  grevilleAbscissae,
  knotLogits,
  knotsFromLogits,
} from './bspline.ts'

const grid = (n: number, lo = 0, hi = 1): number[] =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - (b[i] ?? 0))), 0)

describe('bsplineBasis', () => {
  const interior = [0.2, 0.45, 0.5, 0.8]
  const x = grid(41)

  it('matches lib/stats/smoothing bs() on fixed knots to 1e-10, for several degrees', () => {
    for (const degree of [1, 2, 3, 4]) {
      const reference = bs(x, { knots: interior, degree, intercept: true, boundaryKnots: [0, 1] })
      const ours = bsplineBasis(x, extendKnots(interior, 0, 1, degree), degree)
      expect(ours.shape).toEqual(reference.X.shape)
      expect(maxAbs(toFlat(ours), toFlat(reference.X))).toBeLessThan(1e-10)
    }
  })

  it('is a partition of unity on the closed range, with non-negative entries', () => {
    const t = extendKnots(interior, 0, 1, 3)
    const basis = bsplineBasis(x, t, 3)
    for (const total of toFlat(sum(basis, 1))) expect(total).toBeCloseTo(1, 12)
    expect(Math.min(...toFlat(basis))).toBeGreaterThanOrEqual(0)
    // The right boundary belongs to the last span.
    expect(toFlat(bsplineBasis([1], t, 3)).slice(-1)[0]).toBeCloseTo(1, 12)
  })

  it('has the derivative of lib/stats/smoothing basisDerivative with respect to the input', () => {
    const reference = bs(x.slice(1, -1), { knots: interior, degree: 3, intercept: true, boundaryKnots: [0, 1] })
    const t = extendKnots(interior, 0, 1, 3)
    for (const order of [1, 2] as const) {
      const exact = reference.derivative?.(x.slice(1, -1), order)
      const ours = bsplineBasisDerivative(x.slice(1, -1), t, 3, order)
      expect(maxAbs(toFlat(ours), toFlat(exact as Value))).toBeLessThan(1e-9)
    }
  })

  it('differentiates through the input: the vjp of the basis is the derivative basis', () => {
    const t = extendKnots(interior, 0, 1, 3)
    const points = asTensor([0.13, 0.31, 0.47, 0.66, 0.93])
    const weights = asTensor([1, -2, 0.5, 3, -1, 0.7, 2, -0.4])
    const f = (v: Value): Value => weighted(bsplineBasis(v, t, 3), weights)
    const g = toFlat(grad(f)(points) as Value)
    const d = toFlat(bsplineBasisDerivative(points, t, 3, 1))
    const w = toFlat(weights)
    const expected = Array.from(
      { length: 5 },
      (_, i) => (1 + 0.1 * i) * w.reduce((a, wj, j) => a + (d[i * w.length + j] ?? 0) * wj, 0),
    )
    expect(maxAbs(g, expected)).toBeLessThan(1e-10)
  })

  it('has finite-difference gradients with respect to the interior knots', () => {
    const points = asTensor([0.07, 0.19, 0.33, 0.52, 0.61, 0.74, 0.88, 0.96])
    for (const degree of [2, 3]) {
      const weights = asTensor(Array.from({ length: 3 + degree + 1 }, (_, j) => Math.sin(1 + j)))
      const f = (inner: Value): Value =>
        weighted(bsplineBasis(points, extendKnots(inner, 0, 1, degree), degree), weights)
      const at = asTensor([0.25, 0.5, 0.7])
      const report = gradCheck(f, at, { rtol: 1e-5, atol: 1e-7 })
      expect(report.passed).toBe(true)
      const analytic = toFlat(grad(f)(at) as Value)
      const numeric = toFlat(numericalGradient(f, at, 1e-6))
      expect(maxAbs(analytic, numeric)).toBeLessThan(1e-6)
      expect(analytic.some((v) => Math.abs(v) > 1e-3)).toBe(true)
    }
  })

  it('has finite-difference gradients with respect to the logits of the monotone knot parameterisation', () => {
    const points = asTensor([0.07, 0.19, 0.33, 0.52, 0.61, 0.74, 0.88, 0.96])
    const weights = asTensor(Array.from({ length: 7 }, (_, j) => Math.cos(2 + j)))
    const f = (logits: Value): Value =>
      weighted(bsplineBasis(points, extendKnots(knotsFromLogits(logits, 0, 1), 0, 1, 3), 3), weights)
    const at = asTensor([0.3, -0.2, 0.1, 0.4])
    expect(maxAbs(toFlat(grad(f)(at) as Value), toFlat(numericalGradient(f, at, 1e-6)))).toBeLessThan(1e-6)
  })

  it('handles repeated knots (a kink) without non-finite values or gradients', () => {
    const t = asTensor([0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1])
    const basis = toFlat(bsplineBasis(grid(21), t, 3))
    expect(basis.every(Number.isFinite)).toBe(true)
    const g = toFlat(
      grad((v: Value) => sum(bsplineBasis(grid(21), extendKnots(v, 0, 1, 3), 3)))(asTensor([0.5, 0.5])) as Value,
    )
    expect(g.every(Number.isFinite)).toBe(true)
  })

  it('clamps or continues linearly outside the range', () => {
    const t = extendKnots(interior, 0, 1, 3)
    const edge = toFlat(bsplineBasis([1], t, 3))
    expect(maxAbs(toFlat(bsplineBasis([1.7], t, 3)), edge)).toBeLessThan(1e-14)
    const slope = toFlat(bsplineBasisDerivative([1], t, 3, 1))
    const out = toFlat(bsplineBasis([1.5], t, 3, { outside: 'linear' }))
    expect(
      maxAbs(
        out,
        edge.map((v, j) => v + 0.5 * (slope[j] ?? 0)),
      ),
    ).toBeLessThan(1e-12)
    expect(maxAbs(toFlat(bsplineBasisDerivative([1.5], t, 3, 1, { outside: 'linear' })), slope)).toBeLessThan(1e-12)
  })
})

describe('extended (P-spline) boundary knots', () => {
  it('give the basis of lib/stats/smoothing ps() to 1e-10, with unit sums on the range', () => {
    const x = grid(37, 0.1, 2.6)
    for (const [k, degree] of [
      [8, 3],
      [7, 2],
    ] as const) {
      const reference = ps(x, { k, degree })
      const m = k - degree - 1
      const interior = Array.from({ length: m }, (_, j) => 0.1 + ((2.6 - 0.1) * (j + 1)) / (m + 1))
      const t = extendKnots(interior, 0.1, 2.6, degree, 'extended')
      expect(maxAbs(toFlat(t), toFlat(reference.knots))).toBeLessThan(1e-12)
      const ours = bsplineBasis(x, t, degree, { boundary: 'extended' })
      expect(maxAbs(toFlat(ours), toFlat(reference.X))).toBeLessThan(1e-10)
      for (const total of toFlat(sum(ours, 1))) expect(total).toBeCloseTo(1, 12)
    }
  })
})

describe('knot helpers', () => {
  it('maps logits to strictly increasing interior knots and back', () => {
    const knots = [0.1, 0.35, 0.4, 0.9]
    const logits = knotLogits(knots, 0, 1)
    expect(maxAbs(toFlat(knotsFromLogits(logits, 0, 1)), knots)).toBeLessThan(1e-12)
    const wild = toFlat(knotsFromLogits([5, -9, 3, 0, -4], -2, 6))
    expect(wild.every((v, i) => v > -2 && v < 6 && (i === 0 || v > (wild[i - 1] as number)))).toBe(true)
    // The floor keeps neighbours apart.
    const floored = toFlat(knotsFromLogits([40, -40, 0], 0, 1, 0.05))
    expect((floored[1] as number) - (floored[0] as number)).toBeGreaterThanOrEqual(0.05 - 1e-12)
  })

  it('has Greville abscissae equal to the knot averages', () => {
    const t = extendKnots([0.2, 0.6], 0, 1, 3)
    const g = toFlat(grevilleAbscissae(t, 3))
    expect(g).toHaveLength(6)
    expect(g[0]).toBeCloseTo(0, 12)
    expect(g[5]).toBeCloseTo(1, 12)
    expect(g[2]).toBeCloseTo((0 + 0.2 + 0.6) / 3, 12)
  })
})

describe('the kernel basis and the tensor-operation reference', () => {
  const interior = [0.2, 0.45, 0.5, 0.8]
  const points = asTensor([0, 0.05, 0.2, 0.33, 0.5, 0.71, 0.8, 0.99, 1])
  const weights = asTensor([1, -2, 0.5, 3, -1, 0.7, 2, -0.4])

  it('agree on values, and on the gradients in the points and in the knots', () => {
    for (const degree of [1, 2, 3]) {
      const t = extendKnots(interior, 0, 1, degree)
      expect(
        maxAbs(toFlat(bsplineBasis(points, t, degree)), toFlat(bsplineBasisByOps(points, t, degree))),
      ).toBeLessThan(1e-14)
      const w = asTensor(Array.from({ length: interior.length + degree + 1 }, (_, j) => Math.sin(1 + 2 * j)))
      for (const make of [bsplineBasis, bsplineBasisByOps]) void make
      const inX =
        (basis: typeof bsplineBasis) =>
        (v: Value): Value =>
          weighted(basis(v, t, degree), w)
      const inT =
        (basis: typeof bsplineBasis) =>
        (v: Value): Value =>
          weighted(basis(points, extendKnots(v, 0, 1, degree), degree), w)
      const at = asTensor(interior)
      expect(
        maxAbs(toFlat(grad(inX(bsplineBasis))(points) as Value), toFlat(grad(inX(bsplineBasisByOps))(points) as Value)),
      ).toBeLessThan(1e-10)
      expect(
        maxAbs(toFlat(grad(inT(bsplineBasis))(at) as Value), toFlat(grad(inT(bsplineBasisByOps))(at) as Value)),
      ).toBeLessThan(1e-10)
    }
    expect(weights.shape).toEqual([8])
  })

  it('agree with batched knots, one vector per feature', () => {
    const t = asTensor([extendKnots([0.3, 0.6], 0, 1, 3), extendKnots([0.1, 0.9], 0, 1, 3)].map((k) => toFlat(k)))
    const x = asTensor([
      [0.2, 0.95],
      [0.7, 0.05],
      [1, 0],
    ])
    const fast = bsplineBasis(x, t, 3)
    expect(fast.shape).toEqual([3, 2, 6])
    expect(maxAbs(toFlat(fast), toFlat(bsplineBasisByOps(x, t, 3)))).toBeLessThan(1e-14)
  })
})

/** A scalar of the basis whose gradient exercises every column: the sum of `(basis w) * (1 + row / 10)`. */
function weighted(basis: Value, w: Value): Value {
  const rows = basis.shape[0] ?? 0
  const scale = asTensor(Array.from({ length: rows }, (_, i) => 1 + 0.1 * i))
  return sum(mul(reshape(matmul(basis, reshape(w, [w.shape[0] ?? 0, 1])), [rows]), scale))
}
