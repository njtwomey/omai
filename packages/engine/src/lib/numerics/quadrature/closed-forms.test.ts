import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import {
  abs,
  add,
  div,
  exp,
  item,
  log,
  mul,
  neg,
  sin,
  sqrt,
  square,
  sub,
  sum,
  tensor,
  toArray,
  type Value,
} from '../../tensor/index.ts'
import * as Q from './index.ts'

// Integrals with closed forms that are hard on one rule or another: endpoint singularities, infinite ranges, a sharp
// peak, an oscillation, kinks, reversed and empty ranges, and the Gauss rules on the polynomials they integrate exactly.
type Integrand = (x: Value) => Value
const PI = Math.PI
const near = (got: number, want: number, rtol: number, atol = 0): boolean =>
  Math.abs(got - want) <= rtol * Math.abs(want) + atol

// [name, integrand, a, b, quad options, exact value, absolute tolerance]
const integrals: readonly (readonly [string, Integrand, number, number, Q.QuadOptions, number, number])[] = [
  ['x^2 on [0, 3]', (x) => square(x), 0, 3, {}, 9, 0],
  ['sin on [0, pi]', sin, 0, PI, {}, 2, 0],
  ['1 / sqrt(x) on [0, 1] (endpoint singularity)', (x) => div(1, sqrt(x)), 0, 1, {}, 2, 0],
  ['log x on [0, 1]', log, 0, 1, {}, -1, 0],
  [
    'exp(-x^2) on the real line',
    (x) => exp(neg(square(x))),
    Number.NEGATIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    {},
    Math.sqrt(PI),
    0,
  ],
  ['1 / (1 + x^2) on [0, inf)', (x) => div(1, add(1, square(x))), 0, Number.POSITIVE_INFINITY, {}, PI / 2, 0],
  ['x^3 exp(-x) on [0, inf)', (x) => mul(mul(square(x), x), exp(neg(x))), 0, Number.POSITIVE_INFINITY, {}, 6, 0],
  ['exp(x) on (-inf, 0]', exp, Number.NEGATIVE_INFINITY, 0, {}, 1, 0],
  ['1 / x^2 on [1, inf)', (x) => div(1, square(x)), 1, Number.POSITIVE_INFINITY, {}, 1, 0],
  ['sin(100 x) on [0, pi] (50 periods)', (x) => sin(mul(x, 100)), 0, PI, { limit: 200 }, 0, 1e-10],
  [
    'a Lorentzian of width 0.01 centred at 0.3, with the peak given',
    (x) => div(1, add(1e-4, square(sub(x, 0.3)))),
    0,
    1,
    { points: [0.3], limit: 200 },
    (Math.atan(0.7 / 0.01) + Math.atan(0.3 / 0.01)) / 0.01,
    0,
  ],
  ['x^2 on [3, 0] (reversed)', (x) => square(x), 3, 0, {}, -9, 0],
  ['x^2 on [2, 2] (empty)', (x) => square(x), 2, 2, {}, 0, 1e-300],
  ['|x - 1/3| on [0, 1] (a kink)', (x) => abs(sub(x, 1 / 3)), 0, 1, { points: [1 / 3] }, 5 / 18, 0],
]

describe('adaptive and tanh-sinh quadrature against closed forms', () => {
  it('quad (QUADPACK qags and qagi) reaches 1e-10 relative with tight tolerances and reports it converged', () => {
    const failures: string[] = []
    for (const [name, f, a, b, options, exact, atol] of integrals) {
      const r = Q.quad(f, a, b, { epsabs: 1e-13, epsrel: 1e-12, limit: 100, ...options })
      const v = item(r.value)
      if (!near(v, exact, 1e-10, atol)) failures.push(`${name}: ${v}, exact ${exact}`)
      if (!r.converged || r.status !== 'converged') failures.push(`${name}: status ${r.status}`)
      if (!(r.error >= 0) || !(r.evaluations >= 0))
        failures.push(`${name}: error ${r.error}, evaluations ${r.evaluations}`)
    }
    expect(failures).toEqual([])
  })

  it('tanhSinh integrates endpoint singularities to 1e-9, reports non-convergence, and its double-exponential rules sum to the same', () => {
    // [name, integrand, a, b, exact]: singularities at the lower end point (an upper one is lost to the rounding of 1 - x), smooth ones.
    const rows: readonly (readonly [string, Integrand, number, number, number])[] = [
      ['log x on [0, 1]', log, 0, 1, -1],
      ['1 / sqrt(x) on [0, 1]', (x) => div(1, sqrt(x)), 0, 1, 2],
      ['sqrt(1 - x^2) on [-1, 1]', (x) => sqrt(sub(1, square(x))), -1, 1, PI / 2],
      ['x^-0.6 on [0, 1]', (x) => exp(mul(log(x), -0.6)), 0, 1, 2.5],
      ['exp(x) on [0, 1]', exp, 0, 1, Math.E - 1],
    ]
    for (const [name, f, a, b, exact] of rows) {
      const r = Q.tanhSinh(f, a, b, { tol: 1e-12 })
      expect(item(r.value), name).toBeCloseTo(exact, 8)
      expect(r.status, name).toBe('converged')
    }
    // Fixed double-exponential rules (tanh-sinh on [-1, 1], exp-sinh on [0, inf), sinh-sinh on the line) as nodes and weights.
    const rules: readonly (readonly [Q.DoubleExponentialRange, Integrand, number])[] = [
      ['finite', (x) => sqrt(sub(1, square(x))), PI / 2],
      ['finite', (x) => square(x), 2 / 3],
      ['halfLine', (x) => exp(neg(x)), 1],
      ['halfLine', (x) => mul(square(x), exp(neg(x))), 2],
      ['real', (x) => exp(neg(square(x))), Math.sqrt(PI)],
      ['real', (x) => div(1, add(1, square(x))), PI],
    ]
    for (const [range, f, exact] of rules) {
      const { nodes, weights } = Q.doubleExponentialRule(range)
      expect(item(sum(mul(weights, f(nodes)))), `${range} rule`).toBeCloseTo(exact, 8)
    }
    expect(() => Q.doubleExponentialRule('finite', 0)).toThrow(expect.objectContaining({ code: 'value' }))
  })
})

describe('Gauss rules, sampled-data rules and cubature on integrands they integrate exactly', () => {
  it('Legendre, Hermite, Laguerre and Jacobi rules are exact to degree 2n - 1 and their weights sum to the measure', () => {
    const moments = (rule: Q.QuadratureRule, power: number): number =>
      toArray(rule.nodes as Value).reduce(
        (s, x, i) => s + (toArray(rule.weights as Value) as number[])[i]! * (x as number) ** power,
        0,
      )
    const factorial = (k: number): number => (k <= 1 ? 1 : k * factorial(k - 1))
    // The double factorial (2m - 1)!! / (2m)!! of the Chebyshev moments, Gamma(k + 1/2) = sqrt(pi) (2k - 1)!! / 2^k.
    const halfGamma = (k: number): number => (k === 0 ? Math.sqrt(PI) : (k - 0.5) * halfGamma(k - 1))
    for (const n of [1, 2, 5, 12, 40]) {
      const legendre = Q.gaussLegendre(n)
      // int_-1^1 x^p dx = 2 / (p + 1) for even p, 0 for odd p, up to p = 2n - 1.
      for (let p = 0; p <= 2 * n - 1; p += Math.max(1, Math.floor(n / 2))) {
        expect(moments(legendre, p), `Legendre n=${n} x^${p}`).toBeCloseTo(p % 2 === 0 ? 2 / (p + 1) : 0, 12)
      }
      const hermite = Q.gaussHermite(n)
      // int x^(2k) exp(-x^2) = Gamma(k + 1/2), to degree 2n - 1.
      for (let k = 0; 2 * k <= 2 * n - 1; k += Math.max(1, Math.floor(n / 3))) {
        expect(moments(hermite, 2 * k) / halfGamma(k) - 1, `Hermite n=${n} x^${2 * k}`).toBeCloseTo(0, 11)
      }
      const laguerre = Q.gaussLaguerre(n)
      // int x^k exp(-x) = k!, to degree 2n - 1 (the factorials to 12! stay exact in double).
      for (let k = 0; k <= Math.min(2 * n - 1, 12); k += 3) {
        expect(moments(laguerre, k) / factorial(k) - 1, `Laguerre n=${n} x^${k}`).toBeCloseTo(0, 11)
      }
    }
    // Jacobi (alpha, beta): the weights sum to 2^(alpha + beta + 1) B(alpha + 1, beta + 1); Chebyshev (-1/2, -1/2) has pi / 2^m moments.
    const chebyshev = Q.gaussJacobi(4, -0.5, -0.5)
    expect([moments(chebyshev, 0), moments(chebyshev, 2), moments(chebyshev, 4), moments(chebyshev, 6)]).toEqual(
      [PI, PI / 2, (3 * PI) / 8, (5 * PI) / 16].map((e) => expect.closeTo(e, 12)),
    )
    const lag = Q.gaussJacobi(6, 2, 1)
    // 2^4 B(3, 2) = 16 * (2! 1! / 4!) = 4/3.
    expect(moments(lag, 0)).toBeCloseTo(4 / 3, 12)
    expect(() => Q.gaussJacobi(3, -1, 0)).toThrow(expect.objectContaining({ code: 'value' }))
    expect(() => Q.gaussLegendre(0)).toThrow(expect.objectContaining({ code: 'value' }))
    // fixedQuad on a reversed range, and the expectation of x^2 under N(mu, sigma^2) is mu^2 + sigma^2.
    expect(item(Q.fixedQuad(square, 3, 0, 4))).toBeCloseTo(-9, 12)
    expect(item(Q.normalExpectation(square, 1.5, 0.7, 6))).toBeCloseTo(1.5 ** 2 + 0.7 ** 2, 12)
  })

  it('trapezoid, Simpson, Romberg and the cumulative sums on sampled data; dblquad, tplquad, nquad, product and sparse-grid cubature', () => {
    const grid = (n: number, a: number, b: number): number[] =>
      Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
    const cubes = (xs: number[]): number[] => xs.map((x) => x * x * x)
    // Simpson is exact for cubics on an odd number of uniform points (4) and for quadratics on unequal steps; with an even
    // number of points the last interval gets Cartwright's correction, which is not exact for cubics: scipy's values.
    const xs = [0, 0.3, 0.7, 1.2, 1.5, 2]
    const odd = [0, 0.4, 1, 1.1, 2]
    expect(item(Q.simpson(cubes(grid(5, 0, 2)), { dx: 0.5 }))).toBeCloseTo(4, 12)
    expect(
      item(
        Q.simpson(
          odd.map((x) => x * x),
          { x: odd },
        ),
      ),
    ).toBeCloseTo(8 / 3, 12)
    expect(item(Q.simpson(cubes(odd), { x: odd }))).toBeCloseTo(4.083333333333334, 12)
    expect(item(Q.simpson(cubes(grid(4, 0, 3)), { dx: 1 }))).toBeCloseTo(20.5, 12)
    expect(item(Q.simpson(cubes(grid(6, 0, 5)), { dx: 1 }))).toBeCloseTo(156.5, 12)
    expect(item(Q.simpson(cubes(xs), { x: xs }))).toBeCloseTo(4.0057833333333335, 12)
    expect(item(Q.simpson(cubes([0, 0.5, 1.5, 2]), { x: [0, 0.5, 1.5, 2] }))).toBeCloseTo(4.166666666666666, 12)
    expect(
      item(
        Q.trapezoid(
          xs.map((x) => 3 * x + 1),
          { x: xs },
        ),
      ),
    ).toBeCloseTo(8, 12)
    expect(toArray(Q.cumulativeTrapezoid([1, 1, 1, 1], { dx: 0.5, initial: 0 }))).toEqual([0, 0.5, 1, 1.5])
    expect(item(Q.trapezoid([5]))).toBe(0)
    const r = Q.romberg(sin, 0, PI)
    expect([r.value, r.converged]).toEqual([expect.closeTo(2, 7), true])
    // sqrt has an endpoint singularity in its derivative, so the Richardson tableau converges slowly: the flag says so.
    const hard = Q.romberg(sqrt, 0, 1, { divmax: 4 })
    expect([hard.converged, hard.status]).toEqual([false, 'completed'])
    expect(hard.value).toBeCloseTo(2 / 3, 2)
    // int_0^1 int_0^x x y dy dx = 1/8; int x y z over the unit cube = 1/8; nquad on three nested ranges = 1/48.
    expect(
      item(
        Q.dblquad(
          (y, x) => mul(x, y),
          0,
          1,
          0,
          (x) => x,
        ),
      ),
    ).toBeCloseTo(1 / 8, 10)
    expect(item(Q.tplquad((z, y, x) => mul(mul(x, y), z), 0, 1, 0, 1, 0, 1))).toBeCloseTo(1 / 8, 10)
    // The first range is the innermost variable and may depend on the later ones: z in [0, y], y in [0, x], x in [0, 1].
    const nested = Q.nquad((z, y, x) => mul(mul(x, y), z), [(y: number) => [0, y], (x: number) => [0, x], [0, 1]])
    expect(item(nested)).toBeCloseTo(1 / 48, 10)
    // Product Gauss (n^d points) is exact for degree 2n - 1 in each variable; the Smolyak level-2 rule for total degree 3.
    const box = [
      [0, 2],
      [-1, 1],
    ] as const
    const poly = (p: Value): Value => {
      const a = toArray(p) as number[][]
      return tensor(a.map(([x = 0, y = 0]) => x * x * y * y + 3 * x))
    }
    const product = Q.productGaussRule(box, 3)
    expect(item(sum(mul(product.weights, poly(product.points))))).toBeCloseTo((8 / 3) * (2 / 3) + 3 * 2 * 2, 12)
    const sparse = Q.sparseGridRule(
      [
        [0, 1],
        [0, 1],
        [0, 1],
      ],
      3,
    )
    const w = toArray(sparse.weights) as number[]
    expect(w.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 12)
    const pts = toArray(sparse.points) as number[][]
    expect(pts.reduce((s, [x = 0, y = 0, z = 0], i) => s + (w[i] ?? 0) * (x * y + z * z), 0)).toBeCloseTo(
      0.25 + 1 / 3,
      12,
    )
    // Quasi-Monte Carlo with a fixed key: the integral of x y over the unit square is 1/4, within five standard errors.
    const qmc = Q.quasiMonteCarlo(
      (p) => mul(sum(p, -1), 0.5),
      [
        [0, 1],
        [0, 1],
      ],
      key(3),
      512,
    )
    expect(Math.abs(item(qmc.value) - 0.5)).toBeLessThanOrEqual(5 * qmc.standardError + 1e-3)
  })
})
