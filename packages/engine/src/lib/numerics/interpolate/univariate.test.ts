import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { grad } from '../../autodiff/index.ts'
import { maxOf } from '../../base/index.ts'
import { exp, item, sum, type Tensor, type TensorLike, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as I from './index.ts'

const BOUNDARIES: Readonly<Record<string, I.SplineBoundary>> = {
  'not-a-knot': 'not-a-knot',
  natural: 'natural',
  clamped: 'clamped',
  periodic: 'periodic',
  explicit: [
    [1, 0.5],
    [2, -1],
  ],
}

describe('one-dimensional interpolation against scipy.interpolate', () => {
  for (const c of loadFixture('numerics/interpolate/univariate').cases) {
    it(c.name, () => {
      const inputs = c.inputs.map((v) => decode(v) as Tensor)
      const [a, b, q, d, e] = inputs as [Tensor, Tensor, Tensor, Tensor, Tensor]
      const p = c.params
      const bc = BOUNDARIES[(p['bc'] as string | undefined) ?? 'not-a-knot'] as I.SplineBoundary
      const actual: Value | number[] =
        c.fn === 'cubicSpline'
          ? I.cubicSpline(a, b, bc).evaluate(q, p['nu'] as number)
          : c.fn === 'cubicSplineIntegrate'
            ? I.cubicSpline(a, b, bc).integrate(0.3, 4.1)
            : c.fn === 'cubicSplineAnti'
              ? I.cubicSpline(a, b, bc).antiderivative().evaluate(q)
              : c.fn === 'cubicSplineRoots'
                ? I.ppoly(I.cubicSpline(a, b).coefficients, a, false).roots()
                : c.fn === 'cubicHermite'
                  ? I.cubicHermiteSpline(a, b, q).evaluate(d)
                  : c.fn === 'pchip'
                    ? I.pchip(a, b).evaluate(q)
                    : c.fn === 'akima' || c.fn === 'makima'
                      ? I.akima(a, b, c.fn).evaluate(q)
                      : c.fn === 'bspline'
                        ? I.bspline(a, b, 3).evaluate(q, p['nu'] as number)
                        : c.fn === 'bsplineK'
                          ? I.bspline(a, b, p['k'] as number, p['extrapolate'] as boolean).evaluate(q)
                          : c.fn === 'bsplineBasisElement'
                            ? I.bsplineBasisElement(a, p['extrapolate'] as boolean).evaluate(b)
                            : c.fn === 'bsplineIntegrate'
                              ? I.bspline(a, b, 3).integrate(0.4, 4.4)
                              : c.fn === 'bsplineDesign'
                                ? I.bspline(a, tensor(new Array<number>(toFlat(a).length - 4).fill(0)), 3).designMatrix(
                                    b,
                                  )
                                : c.fn === 'makeInterpSpline'
                                  ? I.makeInterpSpline(a, b, p['k'] as number).evaluate(q)
                                  : c.fn === 'makeLsqSpline'
                                    ? I.makeLsqSpline(a, b, toFlat(q), 3, d).evaluate(e)
                                    : c.fn === 'smoothing'
                                      ? I.makeSmoothingSpline(a, b, { lam: p['lam'] as number }).spline.evaluate(q)
                                      : c.fn === 'smoothingGcv'
                                        ? I.makeSmoothingSpline(a, b).spline.evaluate(q)
                                        : c.fn === 'univariateS0'
                                          ? I.univariateSpline(a, b, { s: 0 }).spline.evaluate(q)
                                          : c.fn === 'barycentric'
                                            ? I.barycentric(a, b).evaluate(q)
                                            : I.lagrange(toFlat(a), toFlat(b))
      expectMatches(Array.isArray(actual) ? tensor(actual) : actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('interpolant properties', () => {
  const xs = [0, 0.7, 1.1, 2, 2.6, 3.3, 4]
  const ys = xs.map((v) => Math.sin(v) + 0.1 * v * v)

  it('every interpolant passes through the data', () => {
    const fits = [
      I.cubicSpline(xs, ys),
      I.pchip(xs, ys),
      I.akima(xs, ys),
      I.makeInterpSpline(xs, ys),
      I.barycentric(xs, ys),
    ]
    for (const f of fits) for (const [i, v] of toFlat(f.evaluate(xs)).entries()) expect(v).toBeCloseTo(ys[i] ?? 0, 11)
  })

  it('PCHIP preserves monotonicity where a cubic spline overshoots', () => {
    const x = [0, 1, 2, 3, 4, 5]
    const y = [0, 0.1, 0.2, 5, 5.1, 5.2]
    const fine = Array.from({ length: 101 }, (_, i) => i / 20)
    const p = toFlat(I.pchip(x, y).evaluate(fine))
    for (let i = 1; i < p.length; i++) expect(p[i] ?? 0).toBeGreaterThanOrEqual((p[i - 1] ?? 0) - 1e-12)
    const s = toFlat(I.cubicSpline(x, y).evaluate(fine))
    expect(s.some((v, i) => i > 0 && v < (s[i - 1] ?? 0) - 1e-6)).toBe(true)
  })

  it('splines are differentiable in the data and the query point', () => {
    // A not-a-knot spline is linear in y: the derivative of the value at q in y_j is the cardinal spline at q.
    const q = 1.7
    const dy = toFlat(grad((y: Value) => I.cubicSpline(xs, y).evaluate(q))(tensor(ys)) as Value)
    const cardinal = xs.map((_, j) =>
      item(
        I.cubicSpline(
          xs,
          xs.map((_, i) => (i === j ? 1 : 0)),
        ).evaluate(q),
      ),
    )
    for (const [j, v] of dy.entries()) expect(v).toBeCloseTo(cardinal[j] ?? 0, 12)
    const dx = item(grad((t: Value) => I.cubicSpline(xs, ys).evaluate(t))(tensor(q)) as Value)
    expect(dx).toBeCloseTo(item(I.cubicSpline(xs, ys).evaluate(q, 1)), 12)
    const db = item(grad((t: Value) => I.makeInterpSpline(xs, ys).evaluate(t))(tensor(q)) as Value)
    expect(db).toBeCloseTo(item(I.makeInterpSpline(xs, ys).evaluate(q, 1)), 10)
  })

  it('B-splines need 2 k + 2 knots, as scipy; a lone basis function comes from bsplineBasisElement (regression)', () => {
    // bspline([0, 1, 2, 3], [1], 2) has no base interval (t_k = 2 > t_n = 1) and evaluated the [2, 3] piece
    // everywhere (1.125 at 1.5); scipy refuses it, and its basis_element gives 0.125, 0.75, 0.125.
    expect(() => I.bspline([0, 1, 2, 3], [1], 2)).toThrow(/at least 6 knots/)
    expect(() => I.bspline([0, 0, 0, 0], [1, 2, 3], 0)).toThrow(/positive length/)
    const b = I.bsplineBasisElement([0, 1, 2, 3])
    expect(toFlat(b.evaluate([0.5, 1.5, 2.5, 3])).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([
      0.125, 0.75, 0.125, 0,
    ])
    expect(item(b.integrate(0, 3))).toBeCloseTo(1, 12)
  })

  it('derivatives, antiderivatives and integrals agree', () => {
    const s = I.cubicSpline(xs, ys)
    expect(item(s.integrate(0.5, 3.5))).toBeCloseTo(
      item(s.antiderivative().evaluate(3.5)) - item(s.antiderivative().evaluate(0.5)),
      12,
    )
    const b = I.makeInterpSpline(xs, ys)
    expect(item(b.integrate(0.5, 3.5))).toBeCloseTo(item(s.integrate(0.5, 3.5)), 10)
  })

  it('smoothing splines trade fit for smoothness and report degrees of freedom', () => {
    const x = Array.from({ length: 25 }, (_, i) => i / 4)
    const y = x.map((v, i) => Math.sin(v) + 0.2 * Math.sin(37 * i))
    const loose = I.makeSmoothingSpline(x, y, { lam: 100 })
    const tight = I.makeSmoothingSpline(x, y, { lam: 1e-6 })
    expect(loose.degreesOfFreedom).toBeLessThan(tight.degreesOfFreedom)
    expect(loose.degreesOfFreedom).toBeGreaterThan(2 - 1e-6)
    const gcv = I.makeSmoothingSpline(x, y)
    expect(gcv.gcv).toBeLessThanOrEqual(loose.gcv + 1e-12)
    expect(gcv.gcv).toBeLessThanOrEqual(tight.gcv + 1e-12)
    const u = I.univariateSpline(x, y, { s: 0.5 })
    expect(u.residual).toBeLessThanOrEqual(0.5)
    expect(u.interiorKnots.length).toBeLessThan(x.length - 4)
  })

  it('the node and weight functions take tensor-likes and return tensors (migration, D-009)', () => {
    const calls: [string, (v: TensorLike) => number[]][] = [
      ['barycentricWeights', (v) => toFlat(I.barycentricWeights(v))],
      ['barycentric weights', (v) => toFlat(I.barycentric([0, 1, 2, 3], [1, 2, 4, 8], v).weights)],
      ['lagrange', (v) => toFlat(I.lagrange(v, [1, 2, 4, 8]))],
      ['lebesgueConstant', (v) => [I.lebesgueConstant(v, 50)]],
      ['notAKnot', (v) => toFlat(I.notAKnot(v, 3))],
      ['realRoots', (v) => toFlat(I.realRoots(v))],
      [
        'makeInterpSpline knots',
        (v) => toFlat(I.makeInterpSpline([0, 1, 2, 3, 4], [1, 2, 0, 5, 4], 1, v).evaluate([0.5])),
      ],
      ['makeLsqSpline knots', (v) => toFlat(I.makeLsqSpline([0, 1, 2, 3, 4], [1, 2, 0, 5, 4], v, 1).evaluate([0.5]))],
      [
        'makeSmoothingSpline weights',
        (v) =>
          toFlat(
            I.makeSmoothingSpline([0, 1, 2, 3, 4, 5], [1, 2, 0, 5, 4, 3], { w: v, lam: 1 }).spline.evaluate([0.5]),
          ),
      ],
    ]
    const vectors: Record<string, number[]> = {
      'makeInterpSpline knots': [0, 0, 1, 2, 3, 4, 4],
      'makeLsqSpline knots': [0, 0, 1, 2, 3, 4, 4],
      'makeSmoothingSpline weights': [1, 1, 2, 1, 1, 2],
    }
    for (const [name, f] of calls) {
      const v = vectors[name] ?? [0, 1, 2, 4]
      const expected = f(v)
      expect(f(Float64Array.from(v)), name).toEqual(expected)
      expect(f(tensor(v)), name).toEqual(expected)
      expect(() => f(tensor([v, v])), name).toThrow(expect.objectContaining({ code: 'shape' }))
    }
    expect(I.chebyshevNodes(5).shape).toEqual([5])
    expect(I.equispacedNodes(5).shape).toEqual([5])
  })

  it('Chebyshev nodes tame the Runge phenomenon; equispaced nodes do not', () => {
    const fine = Array.from({ length: 401 }, (_, i) => -1 + i / 200)
    const truth = toFlat(I.runge(fine))
    const error = (nodes: TensorLike): number => {
      const p = toFlat(I.barycentric(nodes, I.runge(nodes)).evaluate(fine))
      return maxOf(p.map((v, i) => Math.abs(v - (truth[i] ?? 0))))
    }
    expect(error(I.equispacedNodes(21))).toBeGreaterThan(10)
    expect(error(I.chebyshevNodes(21))).toBeLessThan(0.02)
    expect(I.lebesgueConstant(I.equispacedNodes(15))).toBeGreaterThan(100)
    expect(I.lebesgueConstant(I.chebyshevNodes(15))).toBeLessThan(2.8)
    const cheb = I.chebyshevInterpolant((v) => exp(v), 20)
    expect(Math.abs(item(cheb.evaluate(0.77)) - Math.exp(0.77))).toBeLessThan(1e-14)
    expect(item(sum(cheb.weights))).toBeCloseTo(0, 12)
  })
})
