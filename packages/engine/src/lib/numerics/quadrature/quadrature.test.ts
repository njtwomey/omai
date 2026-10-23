import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { grad } from '../../autodiff/index.ts'
import { type Key, key, normal } from '../../random/index.ts'
import {
  abs,
  add,
  cos,
  exp,
  item,
  log,
  log1p,
  mul,
  neg,
  pow,
  prod,
  reciprocal,
  reshape,
  sin,
  sliceAxis,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  tensor,
  type Value,
} from '../../tensor/index.ts'
import * as Q from './index.ts'

const INTEGRANDS: Readonly<Record<string, readonly [(x: Value) => Value, number, number]>> = {
  poly: [(x) => add(sub(pow(x, 3), mul(x, 2)), 1), 0, 2],
  exp: [(x) => exp(x), 0, 1],
  gauss: [(x) => exp(neg(square(x))), -Infinity, Infinity],
  decay: [(x) => exp(neg(x)), 0, Infinity],
  lorentz: [(x) => reciprocal(add(square(x), 1)), -Infinity, 0],
  sqrt: [(x) => sqrt(x), 0, 1],
  log: [(x) => log(x), 0, 1],
  oscillatory: [(x) => mul(x, cos(mul(x, 20))), 0, Math.PI],
  kink: [(x) => abs(sub(x, 0.3)), 0, 1],
}
const EXPECTATIONS: Readonly<Record<string, (x: Value) => Value>> = {
  square: (x) => square(x),
  cos: (x) => cos(x),
  softplus: (x) => log1p(exp(x)),
}

const rule = (r: Q.QuadratureRule): Value => stack([r.nodes, r.weights])

describe('quadrature against numpy and scipy', () => {
  for (const c of loadFixture('numerics/quadrature/quadrature').cases) {
    it(c.name, () => {
      const p = c.params
      const n = p['n'] as number
      const name = p['fn_name'] as string
      const [f, a, b] = INTEGRANDS[name] ?? [(x: Value) => x, 0, 1]
      const spacing = {
        ...(c.inputs[1] === undefined ? {} : { x: decode(c.inputs[1]) as Tensor }),
        ...(p['dx'] === undefined ? {} : { dx: p['dx'] as number }),
        ...(p['axis'] === undefined ? {} : { axis: p['axis'] as number }),
      }
      const actual: Value | number =
        c.fn === 'gaussLegendre'
          ? rule(Q.gaussLegendre(n))
          : c.fn === 'gaussHermite'
            ? rule(Q.gaussHermite(n))
            : c.fn === 'gaussLaguerre'
              ? rule(Q.gaussLaguerre(n))
              : c.fn === 'gaussJacobi'
                ? rule(Q.gaussJacobi(n, p['alpha'] as number, p['beta'] as number))
                : c.fn === 'trapezoid'
                  ? Q.trapezoid(decode(c.inputs[0]) as Tensor, spacing)
                  : c.fn === 'simpson'
                    ? Q.simpson(decode(c.inputs[0]) as Tensor, spacing)
                    : c.fn === 'cumulativeTrapezoid'
                      ? Q.cumulativeTrapezoid(decode(c.inputs[0]) as Tensor, { ...spacing, initial: 0 })
                      : c.fn === 'quad'
                        ? Q.quad(f, a, b, {
                            epsabs: 1e-12,
                            epsrel: 1e-12,
                            limit: 200,
                            ...(name === 'kink' ? { points: [0.3] } : {}),
                          }).value
                        : c.fn === 'fixedQuad'
                          ? Q.fixedQuad(f, a, b, 8)
                          : c.fn === 'tanhSinh'
                            ? Q.tanhSinh(f, a, b).value
                            : c.fn === 'adaptiveSimpson'
                              ? Q.adaptiveSimpson(f, a, b, { tol: 1e-11 }).value
                              : c.fn === 'normalExpectation'
                                ? Q.normalExpectation(EXPECTATIONS[name] as (x: Value) => Value, 0.5, 1.5, 60)
                                : c.fn === 'dblquad'
                                  ? Q.dblquad(
                                      (y, x) => add(mul(x, square(y)), sin(x)),
                                      0,
                                      2,
                                      (x) => x / 2,
                                      (x) => 1 + x,
                                      { epsabs: 1e-12, epsrel: 1e-12 },
                                    )
                                  : c.fn === 'tplquad'
                                    ? Q.tplquad(
                                        (z, y, x) => add(x, mul(y, z)),
                                        0,
                                        1,
                                        0,
                                        (x) => 1 - x,
                                        0,
                                        (x, y) => 1 - x - y,
                                      )
                                    : Q.nquad(
                                        (x0, x1, x2) => exp(neg(add(add(square(x0), square(x1)), square(x2)))),
                                        [
                                          [0, 1],
                                          [0, 1],
                                          [0, 1],
                                        ],
                                        { epsabs: 1e-12, epsrel: 1e-12 },
                                      )
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('rules and estimators', () => {
  it('Gauss-Legendre with n points integrates degree 2n - 1 exactly', () => {
    for (const n of [2, 4, 7]) {
      expect(item(Q.fixedQuad((x) => pow(x, 2 * n - 1), -1, 2, n))).toBeCloseTo((2 ** (2 * n) - 1) / (2 * n), 10)
    }
  })

  it('quad reports intervals that tile the range and a small error', () => {
    const r = Q.quad((x) => mul(x, cos(mul(x, 20))), 0, Math.PI)
    expect(r.converged).toBe(true)
    expect(r.error).toBeLessThan(1e-8)
    const sorted = [...r.intervals].sort((u, v) => u.a - v.a)
    expect(sorted[0]?.a).toBe(0)
    expect(sorted.at(-1)?.b).toBe(Math.PI)
    for (const [i, iv] of sorted.slice(1).entries()) expect(iv.a).toBe(sorted[i]?.b)
  })

  it('Romberg extrapolation converges on smooth integrands', () => {
    const r = Q.romberg((x) => exp(x), 0, 1, { tol: 1e-13, rtol: 1e-13 })
    expect(r.converged).toBe(true)
    expect(r.value).toBeCloseTo(Math.E - 1, 12)
  })

  it('sparse grids are exact for low total degree with fewer points than the product', () => {
    const box = [
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1],
    ] as const
    const sparse = Q.sparseGridRule(box, 3)
    const full = Q.productGaussRule(box, 5)
    expect(sparse.points.shape[0] ?? 0).toBeLessThan(full.points.shape[0] ?? 0)
    // x0^2 x1 + x2^3 over the unit 4-cube: 1/6 + 1/4.
    const f = (p: Value): Value => {
      const col = (k: number): Value => reshape(sliceAxis(p, 1, k, k + 1), [-1])
      return add(mul(square(col(0)), col(1)), pow(col(2), 3))
    }
    expect(item(Q.cubature(f, sparse))).toBeCloseTo(1 / 6 + 1 / 4, 12)
    expect(item(Q.cubature((p) => prod(p, -1), full))).toBeCloseTo(1 / 16, 12)
  })

  it('Monte Carlo and quasi-Monte Carlo estimates fall within their standard errors, QMC far tighter', () => {
    const f = (p: Value): Value => sum(square(p), -1)
    const mc = Q.monteCarlo(
      f,
      [
        [0, 1],
        [0, 1],
        [0, 1],
      ],
      key(3),
      4096,
    )
    const qmc = Q.quasiMonteCarlo(
      f,
      [
        [0, 1],
        [0, 1],
        [0, 1],
      ],
      key(3),
      256,
      16,
    )
    expect(Math.abs(item(mc.value) - 1)).toBeLessThan(4 * mc.standardError)
    expect(Math.abs(item(qmc.value) - 1)).toBeLessThan(Math.max(4 * qmc.standardError, 1e-6))
    expect(qmc.standardError).toBeLessThan(mc.standardError / 5)
  })

  it('importance sampling reports its effective sample size', () => {
    const proposal = {
      sample: (k: Key, n: number) => normal(k, [n, 1]),
      logDensity: (p: Value) => sub(mul(sum(square(p), -1), -0.5), 0.9189385332046727),
    }
    const r = Q.importanceIntegral((p) => exp(neg(sum(square(p), -1))), proposal, key(2), 20000)
    expect(Math.abs(item(r.value) - Math.sqrt(Math.PI))).toBeLessThan(4 * r.standardError)
    expect(r.effectiveSampleSize).toBeGreaterThan(10000)
  })
})

describe('differentiable quadrature', () => {
  it('quad differentiates in the integrand parameters: d/da of the integral of exp(-a x^2) is -sqrt(pi) / (2 a^1.5)', () => {
    const integral = (a: Value): Value => Q.quad((x) => exp(neg(mul(a, square(x)))), -Infinity, Infinity).value
    expect(item(grad(integral)(tensor(2)) as Value)).toBeCloseTo(-Math.sqrt(Math.PI) / (2 * 2 ** 1.5), 8)
  })

  it('fixedQuad differentiates in its limits (fundamental theorem of calculus)', () => {
    const g = grad((b: Value) => Q.fixedQuad((x) => square(x), 0, b, 5))
    expect(item(g(tensor(1.5)) as Value)).toBeCloseTo(2.25, 12)
  })

  it('normal expectations differentiate in mean and standard deviation', () => {
    // E[X^2] = mu^2 + sigma^2.
    const e = (s: Value): Value => Q.normalExpectation((x) => square(x), 1, s)
    expect(item(grad(e)(tensor(0.7)) as Value)).toBeCloseTo(1.4, 12)
  })
})
